import json
from types import SimpleNamespace

import numpy as np
import pytest

from dino_duel import evaluation
from dino_duel.evaluation import evaluate, paired_bootstrap_interval, play_episode
from dino_duel.replay import make_replay, write_replay
from dino_duel.report import make_report


class FixedPolicy:
    def __init__(self, motor=0, attack=0, name="fixed"):
        self.motor = motor
        self.attack = attack
        self.name = name
        self.reset_seeds = []

    def reset(self, seed):
        self.reset_seeds.append(seed)

    def act(self, obs, masks, runner_obs, deterministic=True):
        return np.tile([self.motor, self.attack], (len(obs), 1))


class OneStepEnv:
    """Controlled seat bias tests evaluator outcome mapping, not game mechanics."""
    def __init__(self, config=None):
        self.config = SimpleNamespace(max_seconds=1, dt=1, action_repeat=1)
        self.step_count = 0

    def reset(self, seed=None):
        self.seed = seed
        return np.zeros((2, 53), np.float32), {}

    def attack_masks(self):
        return np.tile([True, False, False, False, False], (2, 1))

    def runner_observations(self):
        return np.zeros((2, 24), np.float32)

    def step(self, actions):
        self.step_count += 1
        return np.zeros((2, 53)), np.array([1, -1]), True, False, {"winner": 0, "reason": "collision"}

    def snapshot(self):
        return {"time": self.step_count, "speed": 8, "distance": [8, 8], "horizon": 40,
                "players": [{"x": 0, "y": 0, "vy": 0, "duck": False, "alive": True,
                             "coins": 3, "width": .7, "height": 1.25} for _ in range(2)],
                "tracks": [[], []], "winner": 0 if self.step_count else None, "reason": "collision"}


def test_seat_swap_is_scored_for_logical_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "DuelEnv", OneStepEnv)
    left, right = FixedPolicy(name="left"), FixedPolicy(name="right")
    result = evaluate(left, right, None, [101, 102], tmp_path, record_replay=True)
    assert result["n_games"] == 4
    assert result["wins"] == result["losses"] == 2
    assert result["left_mean_score"] == .5
    assert result["left_score_ci95"] == [.5, .5]
    assert result["left_score_by_seat"] == {"0": 1., "1": 0.}
    assert left.reset_seeds[0] == left.reset_seeds[1]
    assert right.reset_seeds[0] == right.reset_seeds[1]
    assert (tmp_path / "episodes.csv").read_text().count("\n") == 5
    assert json.loads((tmp_path / "summary.json").read_text())["schema"] == "dino_duel.evaluation.v1"
    assert (tmp_path / "replay.html").is_file()


def test_draws_are_not_counted_as_losses(monkeypatch):
    class DrawEnv(OneStepEnv):
        def step(self, actions):
            self.step_count += 1
            return np.zeros((2, 53)), [0, 0], False, True, {"winner": None, "reason": "time_limit"}
    monkeypatch.setattr(evaluation, "DuelEnv", DrawEnv)
    result = evaluate(FixedPolicy(), FixedPolicy(), None, [99])
    assert result["draws"] == 2 and result["wins"] == result["losses"] == 0
    assert result["left_mean_score"] == .5
    assert result["left_decisive_win_rate"] is None
    assert result["left_score_ci95"] is None
    assert result["draw_reasons"] == {"time_limit": 2}


def test_duplicate_seeds_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(FixedPolicy(), FixedPolicy(), None, [7, 7])


def test_invalid_masked_action_rejected(monkeypatch):
    monkeypatch.setattr(evaluation, "DuelEnv", OneStepEnv)
    with pytest.raises(ValueError, match="forbidden"):
        play_episode(FixedPolicy(attack=1), FixedPolicy(), None, 0)


def test_policy_cannot_mutate_mask_to_enable_illegal_attack(monkeypatch):
    class MutatingPolicy(FixedPolicy):
        def act(self, obs, masks, runner_obs, deterministic=True):
            masks[:, 1] = True
            return np.tile([0, 1], (len(obs), 1))
    monkeypatch.setattr(evaluation, "DuelEnv", OneStepEnv)
    with pytest.raises(ValueError, match="forbidden"):
        play_episode(MutatingPolicy(), FixedPolicy(), None, 0)


def test_bootstrap_reproducible_and_clustered():
    a = paired_bootstrap_interval([0, .25, .5, .75, 1], 1000, 3)
    b = paired_bootstrap_interval([0, .25, .5, .75, 1], 1000, 3)
    assert a == b
    assert 0 <= a[0] <= .5 <= a[1] <= 1
    assert paired_bootstrap_interval([.75]) is None


def test_replay_contains_real_frames_and_escapes_policy_names(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "DuelEnv", OneStepEnv)
    malicious = "</script><script>throw new Error('bad')</script>"
    result = make_replay(FixedPolicy(name=malicious), FixedPolicy(), None, 4, tmp_path / "watch.html")
    text = (tmp_path / "watch.html").read_text()
    assert result["frames"] == 2
    assert malicious not in text
    assert "\\u003c/script>" in text
    assert '"seed":4' in text
    assert '<script src=' not in text


def test_replay_frame_budget_keeps_terminal_state(monkeypatch, tmp_path):
    class LongerEnv(OneStepEnv):
        def __init__(self, config=None):
            super().__init__(config)
            self.config.max_seconds = 25

        def step(self, actions):
            self.step_count += 1
            return np.zeros((2, 53)), [0, 0], self.step_count == 25, False, {"winner": None, "reason": "test"}
    monkeypatch.setattr(evaluation, "DuelEnv", LongerEnv)
    row, frames = play_episode(FixedPolicy(), FixedPolicy(), None, 0, record=True, max_frames=6)
    assert len(frames) <= 6
    assert frames[0]["step"] == 0
    assert frames[-1]["step"] == row["steps"] == 25


def test_report_uses_only_recorded_data(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "DuelEnv", OneStepEnv)
    evaluate(FixedPolicy(name="A"), FixedPolicy(name="B"), None, [42, 43], tmp_path / "eval")
    run = tmp_path / "baseline" / "seed_0"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        json.dumps({"update": 1, "global_step": 128, "agent_steps": 256,
                    "policy_loss": .1, "entropy": 1.2, "elapsed_seconds": .5}) + "\n")
    output = make_report(tmp_path)
    text = output.read_text()
    assert "A vs B" in text
    assert "128" in text and "0.500" in text
    assert "<svg" in text and "<polyline" in text
    assert "environment-seed pairs" in text
    assert "exclude uncertainty from training seeds" in text


def test_empty_report_is_valid(tmp_path):
    text = make_report(tmp_path).read_text()
    assert "No completed evaluation" in text
    assert "No training logs" in text


def test_report_distinguishes_training_seed_intervals(tmp_path):
    (tmp_path / "comparison.json").write_text(json.dumps({
        "schema_version": 1,
        "comparisons": [{"left": "jepa", "right": "baseline", "training_seeds": [3, 4],
                         "per_training_seed_scores": [.7, .4], "mean_score": .55,
                         "hierarchical_ci95": [.3, .8], "n_training_seeds": 2,
                         "n_eval_seed_pairs_per_training_seed": 25, "note": "Exploratory"}],
    }))
    text = make_report(tmp_path).read_text()
    assert "Comparison across training seeds" in text
    assert "jepa vs baseline" in text
    assert "0.300–0.800" in text
    assert "3: 0.700, 4: 0.400" in text
