"""Small inference policies. Importing this module never imports PyTorch."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .teacher import teacher_actions, scripted_attacks


class NumpyRunner:
    """A safe, portable two-hidden-layer tanh network saved without pickle."""

    def __init__(self, path: str | Path | None = None, arrays: dict | None = None):
        if (path is None) == (arrays is None):
            raise ValueError("Supply either path or arrays.")
        if path is not None:
            with np.load(path, allow_pickle=False) as source:
                arrays = {k: source[k].copy() for k in source.files}
        self.arrays = arrays
        for name in ("w0", "b0", "w1", "b1", "w2", "b2"):
            value = np.asarray(arrays[name], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite runner parameter {name}")
            setattr(self, name, value)
        from .env import RUNNER_DIM
        if (self.w0.ndim != 2 or self.w0.shape[1] != RUNNER_DIM or
            self.b0.shape != (self.w0.shape[0],) or
            self.w1.shape != (len(self.b1), len(self.b0)) or
            self.w2.shape != (3, len(self.b1)) or self.b2.shape != (3,)):
            raise ValueError("Runner shape does not match the v1 Dino Duel observation contract.")
        meta = arrays.get("metadata", np.array("{}"))
        self.metadata = json.loads(str(meta.item() if hasattr(meta, "item") else meta))

    def logits(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        x = np.tanh(x @ self.w0.T + self.b0)
        x = np.tanh(x @ self.w1.T + self.b1)
        return x @ self.w2.T + self.b2

    def __call__(self, obs):
        return self.logits(obs).argmax(axis=-1)


class TeacherPolicy:
    def __init__(self, style: str = "aggressive", seed: int = 0,
                 runner_path: str | Path | None = None):
        if style not in {"random", "save", "aggressive", "none", "burst", "cycle"}:
            raise ValueError(f"Unknown scripted attack style: {style}")
        self.style = style
        self.name = f"{'bc' if runner_path else 'teacher'}:{style}"
        self.runner = NumpyRunner(runner_path) if runner_path is not None else teacher_actions
        self.reset(seed)

    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, masks, runner_obs, deterministic=True):
        obs, masks = np.atleast_2d(obs), np.atleast_2d(masks)
        runner_obs = np.atleast_2d(runner_obs)
        motor = np.asarray(self.runner(runner_obs), dtype=np.int64).reshape(-1)
        attack = np.asarray(scripted_attacks(obs, masks, self.rng, style=self.style),
                            dtype=np.int64).reshape(-1)
        return np.column_stack((motor, attack))


def sample_masked_attacks(logits, masks, rng: np.random.Generator) -> np.ndarray:
    """Sample legal attacks with a caller-owned RNG; forced moves use no draw.

    Float64 normalization matches the validation evaluator. The subtraction is
    performed at the logits' original precision before converting to float64.
    """
    logits = np.asarray(logits)
    masks = np.asarray(masks, dtype=bool)
    if logits.ndim != 2 or logits.shape != masks.shape:
        raise ValueError("Attack logits and masks must have the same two-dimensional shape")
    if not masks.any(axis=-1).all():
        raise ValueError("Every mask must contain at least one legal attack")
    if not np.isfinite(logits[masks]).all():
        raise ValueError("Legal attack logits must be finite")
    masked = np.where(masks, logits, -np.inf)
    probabilities = np.exp((masked - masked.max(axis=-1, keepdims=True)).astype(np.float64))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.asarray([
        int(np.flatnonzero(mask)[0]) if mask.sum() == 1
        else int(rng.choice(logits.shape[1], p=prob))
        for mask, prob in zip(masks, probabilities)
    ], dtype=np.int64)


class StochasticAttackPolicy:
    """A frozen TorchPolicy deployed with sampled attacks and greedy motors.

    This explicitly overrides the evaluator's ``deterministic=True`` request
    for the attack head only. Its own resettable NumPy RNG keeps each player's
    samples independent of Torch's global RNG and of the other player's draws.
    NumPy inference uses a snapshot of the loaded MLP's current weights, matching
    ``scripts/evaluate_trained.py`` without sampling or retraining motor actions.
    """

    def __init__(self, policy, seed: int = 0):
        from .learning import TorchPolicy
        if not isinstance(policy, TorchPolicy):
            raise TypeError("StochasticAttackPolicy requires a loaded TorchPolicy")
        self.policy = policy
        self.config = policy.config
        self.variant = policy.variant
        self.name = f"{policy.name} [sampled attacks]"
        self.runner_spec = policy.runner_spec
        state = policy.model.state_dict()

        def array(key):
            return np.asarray(state[key].detach().cpu().numpy(), dtype=np.float32).copy()

        self.layers = [(array(f"encoder.{i}.weight"), array(f"encoder.{i}.bias"))
                       for i in (0, 2)]
        self.head = (array("attack_head.weight"), array("attack_head.bias"))
        self.motor_residual = ((array("motor_residual_head.weight"),array("motor_residual_head.bias"))
                               if policy.model.residual_motor else None)
        self.motor_logit_scale = policy.model.motor_logit_scale
        self.sampling_mode = "attack"

        if policy.model.train_motor:
            self.motor = NumpyRunner(arrays={
                f"{kind}{j}": array(f"runner.net.{i}.{parameter}")
                for j, i in enumerate((0, 2, 4))
                for kind, parameter in (("w", "weight"), ("b", "bias"))
            })
        else:
            self.motor = policy.motor_actions
        self.reset(seed)

    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, masks, runner_obs, deterministic=True):
        from .env import N_ATTACKS, OBS_DIM, RUNNER_DIM
        z = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        legal = np.asarray(masks, dtype=bool).reshape(-1, N_ATTACKS)
        for weight, bias in self.layers:
            z = np.tanh(z @ weight.T + bias)
        logits = z @ self.head[0].T + self.head[1]
        attacks = (np.where(legal,logits,-np.inf).argmax(-1) if self.sampling_mode == "greedy"
                   else sample_masked_attacks(logits, legal, self.rng))
        robs = np.asarray(runner_obs,dtype=np.float32).reshape(-1,RUNNER_DIM)
        if self.policy.model.train_motor:
            motor_logits=self.motor.logits(robs)*self.motor_logit_scale
            if self.motor_residual is not None:
                motor_logits=motor_logits + z@self.motor_residual[0].T+self.motor_residual[1]
            motors=(sample_masked_attacks(motor_logits,np.ones_like(motor_logits,dtype=bool),self.rng)
                    if self.sampling_mode == "joint" else motor_logits.argmax(-1))
        else:
            motors = self.motor(robs)
        return np.column_stack((motors, attacks)).astype(np.int64)


def with_stochastic_attacks(policy):
    """Enable sampled learned attacks; scripted policies retain their behavior."""
    if isinstance(policy, (TeacherPolicy, StochasticAttackPolicy)):
        return policy
    return StochasticAttackPolicy(policy)


def resolve_policy(spec: str | Path, device: str = "cpu", runner_path=None):
    text = str(spec)
    if text in {"teacher", "none", "random", "save", "aggressive"}:
        style = "aggressive" if text == "teacher" else text
        return TeacherPolicy(style, runner_path=runner_path)
    if text.startswith("teacher:"):
        return TeacherPolicy(text.split(":", 1)[1], runner_path=runner_path)
    path = Path(spec)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")
    from .learning import load_policy
    return load_policy(path, device=device)

