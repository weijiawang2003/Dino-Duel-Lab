"""Deployment uses reproducible attack sampling without randomizing locomotion."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dino_duel.env import N_ATTACKS, OBS_DIM, RUNNER_DIM
from dino_duel.learning import TorchPolicy
from dino_duel.models import ActorCritic
from dino_duel.policies import (StochasticAttackPolicy, TeacherPolicy,
                                with_stochastic_attacks)


def _loaded_policy(train_motor=False):
    torch.manual_seed(37)
    model = ActorCritic(hidden=16, runner_hidden=16, train_motor=train_motor)
    return TorchPolicy(model, torch.device("cpu"), {"kind": "teacher"},
                       {"env": {"max_seconds": 60}}, "test policy")


def test_seed_reset_is_reproducible_and_forced_wait_uses_no_randomness():
    policy = _loaded_policy()
    sampled = StochasticAttackPolicy(policy)
    obs = np.zeros((64, OBS_DIM), dtype=np.float32)
    runners = obs[:, :RUNNER_DIM]
    legal = np.ones((64, N_ATTACKS), dtype=bool)
    sampled.reset(8675309)
    first = sampled.act(obs, legal, runners)
    sampled.reset(8675309)
    # Neither other players' random draws nor Torch global sampling may alter
    # the policy's own attack stream.
    other = StochasticAttackPolicy(policy, seed=13)
    other.act(obs, legal, runners)
    torch.rand(100)
    forced = np.zeros_like(legal)
    forced[:, 0] = True
    assert np.all(sampled.act(obs, forced, runners)[:, 1] == 0)
    torch_before = torch.get_rng_state().clone()
    second = sampled.act(obs, legal, runners, deterministic=True)
    assert np.array_equal(first, second)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert len(np.unique(first[:, 1])) > 1
    assert sampled.config is policy.config
    script = TeacherPolicy("random")
    assert with_stochastic_attacks(script) is script


def test_sampled_attacks_obey_masks_and_leave_joint_motor_actions_greedy():
    policy = _loaded_policy(train_motor=True)
    sampled = StochasticAttackPolicy(policy, seed=74)
    rng = np.random.default_rng(72)
    obs = rng.normal(size=(96, OBS_DIM)).astype(np.float32)
    runners = obs[:, :RUNNER_DIM]
    masks = np.zeros((96, N_ATTACKS), dtype=bool)
    masks[:, 0] = True
    masks[::2, 2] = True
    masks[1::2, 3] = True
    greedy = policy.act(obs, masks, runners, deterministic=True)
    actions = sampled.act(obs, masks, runners, deterministic=True)
    assert masks[np.arange(len(actions)), actions[:, 1]].all()
    assert np.array_equal(actions[:, 0], greedy[:, 0])
    assert np.any(actions[:, 1] != greedy[:, 1])
    with pytest.raises(ValueError, match="at least one legal"):
        sampled.act(obs, np.zeros_like(masks), runners)
