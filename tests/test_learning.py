"""Learning correctness tests; tiny budgets test code paths, not skill."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dino_duel.env import DuelEnv, N_ATTACKS, OBS_DIM, RUNNER_DIM
from dino_duel.learning import (generalized_advantage, load_policy, train,
                                _load_checkpoint)
from dino_duel.models import (ActorCritic, RunnerNet, export_runner_npz,
                              load_runner_npz, masked_categorical)


def tiny_config(updates=2):
    return {
        "env": {"max_seconds": 1.0, "initial_coins": 30.0},
        "model": {"hidden": 16},
        "training": {"updates": updates, "num_envs": 2, "rollout_steps": 16,
                     "epochs": 2, "minibatch_size": 16, "device": "cpu",
                     "snapshot_interval": 1, "checkpoint_interval": 1,
                     "opponent_pool_size": 2, "historical_opponent_prob": 1.0},
    }


def test_action_mask_forbids_invalid_sampling_and_keeps_gradients():
    logits = torch.tensor([[1.0, 100.0, 3.0, 200.0, -4.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False, False]])
    dist = masked_categorical(logits, mask)
    draws = dist.sample((100,))
    assert set(draws.reshape(-1).tolist()) <= {0, 2}
    assert torch.isfinite(dist.entropy()).all()
    (-dist.log_prob(torch.tensor([0]))).sum().backward()
    assert logits.grad[0, 1] == 0 and logits.grad[0, 3] == 0
    assert logits.grad[0, 0] != 0
    with pytest.raises(ValueError, match="at least one"):
        masked_categorical(logits, torch.zeros_like(mask))


def test_gae_terminal_vs_truncated_and_no_reset_leak():
    rewards = np.array([[[2.0, 2.0]], [[100.0, 100.0]]], dtype=np.float32)
    values = np.ones_like(rewards)
    next_values = np.full_like(rewards, 4.0)
    terminal = np.array([[True], [False]])
    truncated = np.array([[False], [True]])
    adv, returns = generalized_advantage(rewards, values, next_values, terminal, truncated, 0.9, 0.95)
    assert np.allclose(adv[0], 1.0)  # True terminal: no value bootstrap.
    assert np.allclose(returns[0], 2.0)
    terminal[0] = False
    truncated[0] = True
    adv, _ = generalized_advantage(rewards, values, next_values, terminal, truncated, 0.9, 0.95)
    assert np.allclose(adv[0], 4.6)  # Time limit: value bootstrap, no next-episode GAE.
    adv, _ = generalized_advantage(rewards, values, next_values, terminal, truncated,
                                  0.9, 0.95, bootstrap_time_limits=False)
    assert np.allclose(adv[0], 1.0)  # A scored finite-match draw has zero terminal value.


def test_baseline_and_jepa_start_with_identical_acting_weights():
    torch.manual_seed(11)
    baseline = ActorCritic(hidden=16, variant="baseline")
    torch.manual_seed(11)
    jepa = ActorCritic(hidden=16, variant="jepa")
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, jepa.state_dict()[key]), key


def test_jepa_backpropagates_to_online_encoder_but_not_target():
    torch.manual_seed(1)
    model = ActorCritic(hidden=16, variant="jepa")
    obs = torch.randn(32, OBS_DIM)
    nxt = obs + 0.1 * torch.randn_like(obs)
    actions = torch.stack([torch.randint(3, (32,)), torch.randint(N_ATTACKS, (32,)),
                           torch.randint(3, (32,)), torch.randint(N_ATTACKS, (32,))], -1)
    loss, metrics = model.auxiliary_loss(obs, actions, nxt, torch.ones(32, dtype=torch.bool))
    assert torch.isfinite(loss) and metrics["latent_std"] > 0.05
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    old_target = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.encoder.parameters()).add_(0.1)
    model.update_target(0.5)
    assert torch.allclose(next(model.target_encoder.parameters()), old_target + 0.05)


def test_runner_export_has_exact_numpy_inference(tmp_path):
    torch.manual_seed(12)
    runner = RunnerNet(hidden=16)
    sample = np.random.default_rng(1).normal(size=(10, RUNNER_DIM)).astype(np.float32)
    path = export_runner_npz(runner, tmp_path / "runner.npz", {"test": True})
    numpy_runner = load_runner_npz(path)
    with torch.no_grad():
        expected = runner(torch.tensor(sample)).numpy()
    assert np.allclose(expected, numpy_runner.logits(sample), atol=1e-6)
    assert numpy_runner.metadata["test"] is True


def _assert_tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tensor_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tensor_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("variant", ["baseline", "jepa", "jepa_shuffled"])
def test_resume_matches_uninterrupted_with_historical_opponents(tmp_path, variant):
    full = tmp_path / "full"
    split = tmp_path / "split"
    train(tiny_config(4), full, variant, 13)
    train(tiny_config(2), split, variant, 13)
    resumed = train(tiny_config(4), split, variant, 13, resume=True)
    assert resumed["update"] == 4
    a, b = _load_checkpoint(full / "latest.pt"), _load_checkpoint(split / "latest.pt")
    for key in ("model_state", "optimizer_state", "rng", "env_states", "opponent_pool",
                "active_opponents", "global_step", "agent_steps", "episode_returns"):
        _assert_tensor_tree_equal(a[key], b[key])
    assert len((split / "metrics.jsonl").read_text().splitlines()) == 4
    assert a["opponent_pool"]
    assert any(item is not None for item in a["active_opponents"])
    policy = load_policy(split / "policy.pt")
    env = DuelEnv()
    obs, _ = env.reset(seed=2)
    action = policy.act(obs, env.attack_masks(), env.runner_observations())
    assert action.shape == (2, 2)
    assert env.attack_masks()[np.arange(2), action[:, 1]].all()


def test_train_joint_motor_embeds_runner_and_changes_weights(tmp_path):
    runner = RunnerNet(hidden=16)
    runner_path = export_runner_npz(runner, tmp_path / "runner.npz")
    config = tiny_config(2)
    config["training"]["train_motor"] = True
    out = tmp_path / "joint"
    train(config, out, "jepa", 3, runner_path=runner_path)
    state = _load_checkpoint(out / "latest.pt")
    assert state["runner"]["kind"] == "numpy"
    assert state["model_config"]["runner_hidden"] == 16
    runner_path.unlink()
    policy = load_policy(out)
    env = DuelEnv()
    assert policy.act(env.observe(), env.attack_masks(), env.runner_observations()).shape == (2, 2)
    assert not torch.equal(state["model_state"]["runner.net.0.weight"], runner.net[0].weight)


def test_stop_file_and_reject_incompatible_resume(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "STOP").write_text("stop")
    result = train(tiny_config(), out, "baseline", 2)
    assert result["status"] == "stopped" and result["update"] == 0
    (out / "STOP").unlink()
    config = tiny_config()
    config["env"]["max_seconds"] = 2
    with pytest.raises(ValueError, match="configuration"):
        train(config, out, "baseline", 2, resume=True)
    with pytest.raises(FileExistsError):
        train(tiny_config(), out, "baseline", 2)


def test_external_stop_file_is_operational_and_resumable(tmp_path):
    external = tmp_path / "experiment-STOP"
    external.write_text("stop at a safe boundary")
    config = tiny_config(1)
    config["training"]["stop_file"] = str(external)
    out = tmp_path / "external-stop"
    result = train(config, out, "baseline", 5)
    assert result["status"] == "stopped" and result["update"] == 0
    config["training"]["stop_file"] = ""
    resumed = train(config, out, "baseline", 5, resume=True)
    assert resumed["status"] == "complete" and resumed["update"] == 1


def test_already_complete_resume_does_not_rewrite_any_artifact(tmp_path):
    out = tmp_path / "complete"
    train(tiny_config(1), out, "jepa", 21)
    paths = [out / name for name in ("latest.pt", "policy.pt", "config.json",
                                     "metrics.jsonl", "training_summary.json")]
    before = {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
              for path in paths}
    result = train(tiny_config(1), out, "jepa", 21, resume=True)
    after = {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
             for path in paths}
    assert result["already_complete"] and result["update"] == 1
    assert before == after


def test_training_episode_seeds_stay_in_reserved_training_domain(tmp_path, monkeypatch):
    original = DuelEnv.reset
    seen = []
    def recorded_reset(self, seed=None):
        if seed is not None:
            seen.append(int(seed))
        return original(self, seed=seed)
    monkeypatch.setattr(DuelEnv, "reset", recorded_reset)
    train(tiny_config(1), tmp_path / "large-model-seed", "baseline", 2**32 + 31)
    assert len(seen) >= 4
    assert all(0 <= seed < 2**31 for seed in seen)
