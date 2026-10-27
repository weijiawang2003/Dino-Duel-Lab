"""Decision-normalized PPO checks; tiny runs verify gradients, not game skill."""
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from dino_duel.learning import (
    OPERATIONAL_KEYS, _actor_terms, _load_checkpoint,
    _normalize_actor_advantages, _resume_signature, resolve_config, train,
)
from dino_duel.models import RunnerNet, export_runner_npz


def _no_attack_config(updates=1):
    return {
        "env": {"max_seconds": 1.0, "natural_obstacles": False,
                "initial_coins": 0.0, "coins_per_meter": 0.0},
        "model": {"hidden": 16},
        "training": {"updates": updates, "num_envs": 2, "rollout_steps": 16,
                     "epochs": 2, "minibatch_size": 16, "device": "cpu",
                     "checkpoint_interval": 1, "decision_only_actor": True},
    }


def test_advantage_normalization_ignores_forced_waits_and_handles_empty_set():
    advantages = torch.tensor([1.0, 3.0, 1000.0, -1000.0])
    active = torch.tensor([True, True, False, False])
    actual = _normalize_actor_advantages(advantages, active)
    assert torch.allclose(actual, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
    empty = _normalize_actor_advantages(advantages, torch.zeros_like(active))
    assert torch.equal(empty, torch.zeros_like(advantages))
    single = _normalize_actor_advantages(advantages, torch.tensor([True, False, False, False]))
    assert torch.equal(single, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_actor_terms_match_decision_subbatch_without_dilution():
    logprob = torch.tensor([0.1, -0.2, 1000.0, -1000.0], requires_grad=True)
    old = torch.zeros(4)
    advantages = torch.tensor([1.0, -1.0, 200.0, -200.0])
    entropy = torch.tensor([0.5, 0.8, 0.0, 0.0], requires_grad=True)
    active = torch.tensor([True, True, False, False])
    actual = _actor_terms(logprob, old, advantages, entropy, active, 0.2)
    reference = _actor_terms(logprob[:2], old[:2], advantages[:2], entropy[:2],
                             torch.ones(2, dtype=torch.bool), 0.2)
    for got, expected in zip(actual, reference):
        assert torch.allclose(got, expected)
        assert torch.isfinite(got)
    (actual[0] - 0.01 * actual[1]).backward()
    assert torch.all(logprob.grad[:2].abs() > 0)
    assert torch.equal(logprob.grad[2:], torch.zeros(2))
    assert torch.equal(entropy.grad[2:], torch.zeros(2))


def test_no_decision_update_trains_critic_and_preserves_initial_policy(tmp_path):
    output = tmp_path / "forced-waits"
    train(_no_attack_config(), output, "baseline", 31)
    initial_path = output / "initial_policy.pt"
    initial = _load_checkpoint(initial_path)
    trained = _load_checkpoint(output / "policy.pt")
    assert initial["update"] == 0 and trained["update"] == 1
    for name in ("attack_head.weight", "attack_head.bias"):
        assert torch.equal(initial["model_state"][name], trained["model_state"][name])
    assert not torch.equal(initial["model_state"]["value_head.weight"],
                           trained["model_state"]["value_head.weight"])
    row = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    assert row["effective_decisions"] == row["actor_samples"] == 0
    assert row["decision_fraction"] == 0
    for name in ("policy_loss", "entropy", "approx_kl", "clip_fraction"):
        assert row[name] == 0.0
    assert row["value_loss"] > 0.0
    before = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    train(_no_attack_config(updates=2), output, "baseline", 31, resume=True)
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == before


def test_joint_motor_keeps_actor_samples_when_attack_is_forced_wait(tmp_path):
    torch.manual_seed(7)
    runner_path = export_runner_npz(RunnerNet(hidden=16), tmp_path / "runner.npz")
    config = _no_attack_config()
    config["training"]["train_motor"] = True
    output = tmp_path / "joint"
    train(config, output, "baseline", 31, runner_path=runner_path)
    row = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    assert row["effective_decisions"] == 0
    assert row["actor_samples"] == 2 * 2 * 16
    assert row["entropy"] > 0


def test_default_signature_stays_compatible_with_original_release():
    config = resolve_config({})
    assert config["training"]["decision_only_actor"] is False
    old_relevant = {
        "env": {k:v for k,v in config["env"].items() if k not in {"wide_cactus_width","wide_cactus_height"}}, "model": {k:config["model"][k] for k in ("hidden","runner_hidden")},
        "training": {key: value for key, value in config["training"].items()
                     if key not in OPERATIONAL_KEYS and key not in {"decision_only_actor", "scripted_opponent_prob", "scripted_opponent_styles", "motor_entropy_coef", "motor_anchor_coef", "aux_batch_size", "warm_start_heads"}},
    }
    old_signature = hashlib.sha256(json.dumps(old_relevant, sort_keys=True).encode()).hexdigest()
    assert _resume_signature(config) == old_signature
    config["training"]["decision_only_actor"] = True
    assert _resume_signature(config) != old_signature
    with pytest.raises(ValueError, match="decision_only_actor"):
        resolve_config({"training": {"decision_only_actor": 1}})
