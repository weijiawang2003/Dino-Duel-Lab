import json

import numpy as np
import pytest

from dino_duel.env import DuelEnv, EnvConfig, N_ATTACKS, OBS_DIM, RUNNER_DIM
from dino_duel.teacher import scripted_attacks, teacher_actions


def controlled_env(**kwargs):
    return DuelEnv(dict(natural_obstacles=False, acceleration=0.0, **kwargs))


def obstacle(kind=1, x=0.5):
    from dino_duel.env import OBSTACLE_SPECS
    width, y, height = OBSTACLE_SPECS[kind]
    return dict(id=100, kind=kind, x=x, y=y, width=width, height=height, source="fixture")


def put_obstacles(env, left=(), right=()):
    state = env.get_state()
    state["tracks"] = [list(left), list(right)]
    state["next_obstacle_id"] = 101
    env.set_state(state)


def test_shapes_and_finite_observations():
    env = DuelEnv()
    obs, _ = env.reset(seed=23)
    assert obs.shape == (2, OBS_DIM)
    assert env.runner_observations().shape == (2, RUNNER_DIM)
    assert env.attack_masks().shape == (2, N_ATTACKS)
    assert np.isfinite(obs).all()
    assert obs.dtype == np.float32
    np.testing.assert_array_equal(obs[0, :24], obs[1, 24:48])
    assert env.attack_masks()[:, 0].all()


def test_seeded_instances_and_restore_reproduce_future():
    first, second = DuelEnv(), DuelEnv()
    first.reset(seed=74)
    second.reset(seed=74)
    for _ in range(25):
        actions = np.stack([teacher_actions(first.runner_observations()), [0, 0]], axis=1)
        one, two = first.step(actions), second.step(actions)
        np.testing.assert_array_equal(one[0], two[0])
        assert one[2:] == two[2:]
    state = json.loads(json.dumps(first.get_state(), allow_nan=False))
    resumed = DuelEnv()
    resumed.set_state(state)
    rng = np.random.default_rng(100)
    for _ in range(120):
        actions = np.stack([teacher_actions(first.runner_observations()),
                            scripted_attacks(first.observe(), first.attack_masks(), rng)], axis=1)
        original, restored = first.step(actions), resumed.step(actions)
        np.testing.assert_array_equal(original[0], restored[0])
        np.testing.assert_array_equal(original[1], restored[1])
        assert original[2:] == restored[2:]
        assert first.get_state() == resumed.get_state()
        if original[2] or original[3]:
            break


def test_purchases_simultaneous_wallet_conservation_and_cooldown():
    env = controlled_env(initial_coins=20.0, action_repeat=1)
    assert env.attack_masks()[:, 1].all()
    _, _, _, _, info = env.step([[0, 1], [0, 1]])
    gain = env.config.initial_speed * env.config.dt
    assert env.players[0]["coins"] == pytest.approx(10 + gain)
    assert env.players[1]["coins"] == pytest.approx(10 + gain)
    assert env.tracks[0][0]["source"] == "player_1"
    assert env.tracks[1][0]["source"] == "player_0"
    assert info["purchases"] == [1, 1]
    assert not env.attack_masks()[:, 1:].any()
    assert env.tracks[0][0]["x"] > env.config.max_speed * env.config.reaction_time


def test_invalid_attack_is_wait_and_never_spends_or_spawns():
    env = controlled_env(initial_coins=0.0, action_repeat=1)
    _, _, _, _, info = env.step([[0, 4], [0, 2]])
    assert not env.tracks[0] and not env.tracks[1]
    assert info["executed_actions"] == [[0, 0], [0, 0]]
    assert info["invalid_attacks"] == [1, 1]
    assert info["purchases"] == [0, 0]
    assert all(p["coins"] > 0 for p in env.players)


def test_spacing_and_visible_capacity_are_in_public_mask():
    env = controlled_env(initial_coins=60, attack_cooldown=0.0, action_repeat=1)
    env.step([[0, 1], [0, 1]])
    assert env.attack_masks()[0, 1]
    env.step([[0, 1], [0, 1]])
    for track in env.tracks:
        assert track[1]["x"] - track[0]["x"] - track[0]["width"] >= env.config.max_speed * env.config.min_gap_seconds - 1e-8
        assert all(o["x"] + o["width"] <= 40 for o in track)
    assert not env.attack_masks()[:, 1:].any()


def test_simultaneous_collision_is_draw_not_seat_zero_loss():
    env = controlled_env(action_repeat=1)
    put_obstacles(env, [obstacle()], [obstacle()])
    _, reward, term, trunc, info = env.step([[0, 0], [0, 0]])
    assert term and not trunc
    assert info["winner"] is None
    assert info["reason"] == "simultaneous_collision"
    assert not any(p["alive"] for p in env.players)
    np.testing.assert_array_equal(reward, [0, 0])
    with pytest.raises(RuntimeError):
        env.step([[0, 0], [0, 0]])


def test_collision_payoff_is_zero_sum():
    env = controlled_env(action_repeat=1)
    put_obstacles(env, [obstacle()], [])
    _, reward, term, trunc, info = env.step([[0, 0], [0, 0]])
    assert term and not trunc and info["winner"] == 1
    np.testing.assert_array_equal(reward, [-1, 1])


def test_duck_clears_low_bird_and_standing_clears_high_bird():
    low = controlled_env(action_repeat=1)
    put_obstacles(low, [obstacle(3)], [obstacle(3)])
    _, reward, term, _, info = low.step([[2, 0], [0, 0]])
    assert term and info["winner"] == 0
    np.testing.assert_array_equal(reward, [1, -1])
    high = controlled_env(action_repeat=1)
    put_obstacles(high, [obstacle(4)], [obstacle(4)])
    _, _, term, _, _ = high.step([[0, 0], [0, 0]])
    assert not term


def test_teacher_jump_clears_cactus_and_no_air_jump():
    env = controlled_env(max_seconds=1.4)
    put_obstacles(env, [obstacle(2, x=3.6)], [obstacle(1, x=3.6)])
    jumped = False
    while True:
        motors = teacher_actions(env.runner_observations())
        jumped |= bool(np.any(motors == 1))
        _, _, term, trunc, info = env.step(np.stack([motors, [0, 0]], axis=1))
        assert not term
        if trunc:
            break
    assert jumped and info["passed"] == [1, 1]
    env = controlled_env()
    env.step([[1, 0], [1, 0]])
    before = env.players[0]["vy"]
    _, _, _, _, info = env.step([[1, 0], [1, 0]])
    assert env.players[0]["vy"] < before
    assert info["executed_actions"][0][0] == 0


def test_time_limit_is_truncation_and_exact():
    env = controlled_env(max_seconds=0.105)
    while True:
        _, reward, term, trunc, info = env.step([[0, 0], [0, 0]])
        if term or trunc:
            break
    assert trunc and not term and info["reason"] == "time_limit"
    assert env.time == pytest.approx(0.105)
    np.testing.assert_array_equal(reward, [0, 0])
    json.dumps(env.snapshot(), allow_nan=False)


def test_teacher_survives_default_natural_tracks():
    for seed in range(5):
        env = DuelEnv({"max_seconds": 20.0})
        env.reset(seed=seed)
        while True:
            motors = teacher_actions(env.runner_observations())
            _, _, term, trunc, _ = env.step(np.stack([motors, [0, 0]], axis=1))
            assert not term, f"teacher failed default natural tracks at seed {seed}"
            if trunc:
                break


@pytest.mark.parametrize("config", [
    {"attack_costs": [0, -1, 2, 3, 4]}, {"dt": 0.1}, {"gravity": 0},
    {"action_repeat": 2.5}, {"max_speed": 30}, {"wallet_cap": 0},
    {"reaction_time": 0.001}, {"spawn_distance": 42}, {"max_seconds": float("nan")},
])
def test_invalid_configs_rejected(config):
    with pytest.raises(ValueError):
        EnvConfig.from_dict(config)


def test_invalid_action_rejected_and_wrong_restore_config_rejected():
    env = controlled_env()
    for bad in ([[0, 0]], [[3, 0], [0, 0]], [[0, 1.5], [0, 0]], [[0, 5], [0, 0]]):
        with pytest.raises(ValueError):
            env.step(bad)
    other = controlled_env(max_seconds=1)
    with pytest.raises(ValueError):
        other.set_state(env.get_state())


def test_reachability_gate_rejects_trap_but_allows_clearable_sequence():
    env = controlled_env(initial_coins=60, min_gap_seconds=0.1, ensure_reachable=True)
    put_obstacles(env, [], [obstacle(1, x=20.5)])
    # A second small cactus can be cleared in the same jump. A high bird just
    # after it leaves insufficient time to descend and is an unavoidable trap.
    masks = env.attack_masks()
    assert masks[0, 1] and not masks[0, 4]
    assert env._placement_x(1, 4) is not None  # geometric space alone allows it
    env.step([[0, 4], [0, 0]])
    assert env.purchases[0] == 0
    assert len(env.tracks[1]) == 1
    comparison = controlled_env(initial_coins=60, min_gap_seconds=0.1, ensure_reachable=False)
    put_obstacles(comparison, [], [obstacle(1, x=20.5)])
    assert comparison.attack_masks()[0, 4]


@pytest.mark.parametrize("kind,x,airborne", [(1, 1.2, False), (1, 1.8, False),
                                          (3, 0.7, False), (4, 0.7, True)])
def test_reachability_dp_matches_exhaustive_engine_rollouts(kind, x, airborne):
    from itertools import product
    config = dict(natural_obstacles=False, acceleration=0.0, max_seconds=0.20,
                  ensure_reachable=True)
    env = DuelEnv(config)
    if airborne:
        state = env.get_state()
        state["players"][0].update(y=1.0, vy=0.0)
        env.set_state(state)
    predicted = env._known_queue_reachable(0, kind, x)
    start = env.get_state()
    start["tracks"][0] = [obstacle(kind, x)]
    survived = False
    # Three decisions cover this horizon; no external planner approximation is
    # used in this check, only the actual environment's physics and collision.
    for motors in product(range(3), repeat=3):
        trial = DuelEnv(config)
        trial.set_state(start)
        term, trunc = False, False
        for motor in motors:
            _, _, term, trunc, _ = trial.step([[motor, 0], [0, 0]])
            if term or trunc:
                break
        if trunc and not term:
            survived = True
            break
    assert predicted == survived


def test_gated_natural_spawns_happen_at_decision_boundaries():
    env = DuelEnv(dict(ensure_reachable=True, max_seconds=5,
                       natural_interval_min=0.2, natural_interval_max=0.4))
    spawn_times = []
    original_spawn = env._spawn

    def record_spawn(*args, **kwargs):
        spawn_times.append(env.time)
        return original_spawn(*args, **kwargs)

    env._spawn = record_spawn
    while True:
        motors = teacher_actions(env.runner_observations())
        _, _, term, trunc, _ = env.step(np.stack([motors, [0, 0]], axis=1))
        if term or trunc:
            break
    assert spawn_times
    interval = env.config.dt * env.config.action_repeat
    assert all(abs(t / interval - round(t / interval)) < 1e-7 for t in spawn_times)


@pytest.mark.parametrize("ensure_reachable", [False, True])
def test_attack_only_game_never_creates_unsolicited_obstacles(ensure_reachable):
    env = DuelEnv(dict(natural_obstacles=False, ensure_reachable=ensure_reachable,
                       min_gap_seconds=0.30, max_seconds=8.0))
    rng_before = env.get_state()["track_rngs"]
    while True:
        assert env.tracks == [[], []]
        assert not env.runner_observations()[:, [6, 12, 18]].any()
        _, _, term, trunc, _ = env.step([[0, 0], [0, 0]])
        assert not term
        if trunc:
            break
    assert env.next_obstacle_id == 0 and env.purchases == [0, 0]
    assert env.get_state()["track_rngs"] == rng_before


def test_purchase_windows_preserve_motor_control_and_do_not_spend_off_window():
    env = controlled_env(initial_coins=30, attack_cooldown=0,
                         attack_every_n_steps=3)
    assert env.attack_masks()[0, 1]
    env.step([[0, 0], [0, 0]])
    assert not env.attack_masks()[:, 1:].any()
    coins = env.players[0]["coins"]
    _, _, _, _, info = env.step([[1, 1], [0, 0]])
    assert env.players[0]["y"] > 0  # jump remains legal between purchase windows
    assert info["executed_actions"][0] == [1, 0]
    assert env.purchases[0] == 0 and not env.tracks[1]
    assert env.players[0]["coins"] > coins
    assert not env.attack_masks()[:, 1:].any()
    env.step([[0, 0], [0, 0]])
    assert env.attack_masks()[0, 1]
    env.step([[0, 1], [0, 0]])
    assert env.purchases[0] == 1 and len(env.tracks[1]) == 1


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_purchase_window_rejected(interval):
    with pytest.raises(ValueError):
        EnvConfig(attack_every_n_steps=interval)


def test_seeded_initial_speed_jitter_is_shared_visible_and_restorable():
    config = dict(natural_obstacles=False, initial_speed=7.5, initial_speed_jitter=0.5)
    first, second = DuelEnv(config), DuelEnv(config)
    first.reset(seed=123)
    second.reset(seed=123)
    assert first.speed == second.speed
    assert 7.0 <= first.speed <= 8.0
    np.testing.assert_array_equal(first.observe(), second.observe())
    assert first.runner_observations()[0, 4] == first.runner_observations()[1, 4]
    speed = first.speed
    first.reset(seed=124)
    assert first.speed != speed
    restored = DuelEnv(config)
    restored.set_state(json.loads(json.dumps(second.get_state())))
    # The reset stream, including its jitter draw, survives a state round-trip.
    second.reset()
    restored.reset()
    assert second.speed == restored.speed
    assert second.get_state() == restored.get_state()


@pytest.mark.parametrize("jitter", [-0.1, 7.0, 6.1])
def test_invalid_speed_jitter_rejected(jitter):
    with pytest.raises(ValueError):
        EnvConfig(initial_speed_jitter=jitter)
