"""Small, original policies and an action-conditioned latent prediction model.

This is a structured-state JEPA-style experiment, not an implementation of the
published image/video JEPA architectures. All persisted neural weights are plain
state dictionaries; the numpy runner format never needs pickle.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import N_ATTACKS, OBS_DIM, RUNNER_DIM


def _linear_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(module.bias)


class RunnerNet(nn.Module):
    """A portable MLP for basic running, jumping, and ducking."""

    def __init__(self, input_dim: int = RUNNER_DIM, hidden: int = 64):
        super().__init__()
        self.input_dim, self.hidden = int(input_dim), int(hidden)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3),
        )
        self.apply(_linear_init)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


RUNNER_KEYS = ("w0", "b0", "w1", "b1", "w2", "b2")


def runner_arrays(model: RunnerNet) -> dict[str, np.ndarray]:
    """Return weights in torch's [output, input] convention."""
    result: dict[str, np.ndarray] = {}
    for i, layer in enumerate((model.net[0], model.net[2], model.net[4])):
        result[f"w{i}"] = layer.weight.detach().cpu().numpy().copy()
        result[f"b{i}"] = layer.bias.detach().cpu().numpy().copy()
    return result


def validate_runner_arrays(weights: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {k: np.asarray(weights[k], dtype=np.float32) for k in RUNNER_KEYS}
    h = result["w0"].shape[0] if result["w0"].ndim == 2 else -1
    shapes = {"w0": (h, RUNNER_DIM), "b0": (h,), "w1": (h, h),
              "b1": (h,), "w2": (3, h), "b2": (3,)}
    for key, expected in shapes.items():
        if result[key].shape != expected or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid runner array {key}: expected finite {expected}.")
    if h < 1 or h > 4096:
        raise ValueError("Runner hidden size must be in [1,4096].")
    return result


def export_runner_npz(model: RunnerNet, path: str | Path,
                      metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = runner_arrays(model)
    data["metadata"] = np.asarray(json.dumps({
        "format": "dino-duel-runner-v1", "input_dim": model.input_dim,
        "hidden": model.hidden, **(metadata or {}),
    }, sort_keys=True))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    return path


class NumpyRunner:
    def __init__(self, weights: dict[str, Any], metadata: dict | None = None):
        self.weights = validate_runner_arrays(weights)
        self.metadata = metadata or {}

    def logits(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=np.float32)
        if x.shape[-1] != RUNNER_DIM:
            raise ValueError(f"Runner expects {RUNNER_DIM} features.")
        w = self.weights
        x = np.tanh(x @ w["w0"].T + w["b0"])
        x = np.tanh(x @ w["w1"].T + w["b1"])
        return x @ w["w2"].T + w["b2"]

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return self.logits(observations).argmax(axis=-1).astype(np.int64)

    def act(self, observations: np.ndarray) -> np.ndarray:
        return self(observations)


def load_runner_npz(path: str | Path) -> NumpyRunner:
    with np.load(Path(path), allow_pickle=False) as archive:
        weights = {key: archive[key].copy() for key in RUNNER_KEYS}
        metadata = json.loads(str(archive["metadata"].item())) if "metadata" in archive else {}
    return NumpyRunner(weights, metadata)


def initialize_runner(model: RunnerNet, weights: dict[str, Any]) -> None:
    data = validate_runner_arrays(weights)
    if model.hidden != data["w0"].shape[0]:
        raise ValueError("Runner hidden size does not match the shared checkpoint.")
    with torch.no_grad():
        for i, layer in enumerate((model.net[0], model.net[2], model.net[4])):
            layer.weight.copy_(torch.as_tensor(data[f"w{i}"], device=layer.weight.device))
            layer.bias.copy_(torch.as_tensor(data[f"b{i}"], device=layer.bias.device))


class ActorCritic(nn.Module):
    """PPO backbone; JEPA variants add a training-only future-latent predictor."""

    VARIANTS = {"baseline", "baseline_large", "jepa", "jepa_shuffled"}

    def __init__(self, obs_dim: int = OBS_DIM, n_attacks: int = N_ATTACKS,
                 hidden: int = 128, variant: str = "baseline",
                 train_motor: bool = False, runner_hidden: int = 64,
                 residual_motor: bool = False, motor_logit_scale: float = 1.0,
                 prediction_horizons: tuple[int, ...] = (1,),
                 predictor_fusion: str = "linear"):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose {sorted(self.VARIANTS)}.")
        self.obs_dim, self.n_attacks = int(obs_dim), int(n_attacks)
        self.hidden, self.variant = int(hidden), variant
        self.train_motor, self.runner_hidden = bool(train_motor), int(runner_hidden)
        self.residual_motor = bool(residual_motor)
        self.motor_logit_scale = float(motor_logit_scale)
        self.prediction_horizons = tuple(int(h) for h in prediction_horizons)
        if not self.prediction_horizons or any(h < 1 for h in self.prediction_horizons):
            raise ValueError("prediction_horizons must contain positive integers")
        self.multistep = self.prediction_horizons != (1,)
        if predictor_fusion not in {"linear", "nonlinear"}:
            raise ValueError("predictor_fusion must be linear or nonlinear")
        self.predictor_fusion = predictor_fusion
        if self.residual_motor and not self.train_motor:
            raise ValueError("residual_motor requires train_motor")
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.attack_head = nn.Linear(hidden, n_attacks)
        self.value_head = nn.Linear(hidden, 1)
        self.runner = RunnerNet(RUNNER_DIM, runner_hidden)
        self.encoder.apply(_linear_init)
        nn.init.orthogonal_(self.attack_head.weight, gain=0.01)
        nn.init.zeros_(self.attack_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        self.runner.requires_grad_(train_motor and not self.residual_motor)
        if self.residual_motor:
            self.motor_residual_head = nn.Linear(hidden, 3)
            nn.init.zeros_(self.motor_residual_head.weight)
            nn.init.zeros_(self.motor_residual_head.bias)
        else:
            self.motor_residual_head = None
        self.has_auxiliary = variant.startswith("jepa")
        if self.has_auxiliary:
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
            action_dim = 2 * (3 + n_attacks)
            if self.multistep:
                self.predictor = nn.GRU(action_dim, 32, batch_first=True)
                self.prediction_head = (nn.Linear(hidden + 32, hidden)
                                        if predictor_fusion == "linear" else
                                        nn.Sequential(nn.Linear(hidden + 32, hidden), nn.Tanh(),
                                                      nn.Linear(hidden, hidden)))
                self.prediction_head.apply(_linear_init)
            else:
                self.predictor = nn.Sequential(nn.Linear(hidden + action_dim, hidden),
                                               nn.Tanh(), nn.Linear(hidden, hidden))
                self.predictor.apply(_linear_init)
        else:
            self.target_encoder, self.predictor = None, None

    @property
    def model_config(self) -> dict:
        return {"obs_dim": self.obs_dim, "n_attacks": self.n_attacks,
                "hidden": self.hidden, "variant": self.variant,
                "train_motor": self.train_motor, "runner_hidden": self.runner_hidden,
                "residual_motor": self.residual_motor, "motor_logit_scale": self.motor_logit_scale,
                "prediction_horizons": self.prediction_horizons,
                "predictor_fusion": self.predictor_fusion}

    def forward(self, obs: torch.Tensor,
                runner_obs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.encoder(obs)
        motor_logits = self.runner(obs[..., :RUNNER_DIM] if runner_obs is None else runner_obs) * self.motor_logit_scale
        if self.motor_residual_head is not None:
            motor_logits = motor_logits + self.motor_residual_head(z)
        return {"attack_logits": self.attack_head(z),
                "motor_logits": motor_logits,
                "value": self.value_head(z).squeeze(-1), "z": z}

    def auxiliary_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                       next_obs: torch.Tensor, valid: torch.Tensor,
                       shuffle: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        zero = obs.sum() * 0.0
        metrics = {"aux_loss": 0.0, "prediction_loss": 0.0,
                   "variance_loss": 0.0, "latent_std": 0.0}
        if self.multistep:
            return self.sequence_auxiliary_loss(obs, actions, next_obs, valid, shuffle)
        if not self.has_auxiliary or int(valid.sum().item()) < 2:
            return zero, metrics
        current, following, action = obs[valid], next_obs[valid], actions[valid].long()
        z = self.encoder(current)
        with torch.no_grad():
            target = self.target_encoder(following)
            if shuffle:
                target = target[torch.randperm(target.shape[0], device=target.device)]
        encoded_action = torch.cat([
            F.one_hot(action[:, 0], 3), F.one_hot(action[:, 1], self.n_attacks),
            F.one_hot(action[:, 2], 3), F.one_hot(action[:, 3], self.n_attacks),
        ], dim=-1).to(z.dtype)
        predicted = self.predictor(torch.cat([z, encoded_action], dim=-1))
        prediction_loss = F.smooth_l1_loss(predicted, target)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        # The Tanh encoder is bounded: a 0.5 spread target is achievable.
        variance_loss = F.relu(0.5 - std).mean()
        loss = prediction_loss + 0.1 * variance_loss
        metrics = {"aux_loss": float(loss.detach()),
                   "prediction_loss": float(prediction_loss.detach()),
                   "variance_loss": float(variance_loss.detach()),
                   "latent_std": float(std.mean().detach())}
        return loss, metrics

    def sequence_auxiliary_loss(self, obs, actions, next_obs, valid, shuffle=False):
        """Teacher-forced factual joint-action sequences; NOT inference-time foresight.

        obs: [B,D], actions: [B,Hmax,4], next_obs: [B,K,D], valid: [B,K].
        Each horizon ends at most at a terminal; none crosses an episode reset.
        """
        zero = obs.sum() * 0.0
        metrics = {"aux_loss": 0.0, "prediction_loss": 0.0,
                   "variance_loss": 0.0, "latent_std": 0.0}
        if not self.has_auxiliary or not bool(valid.any()):
            return zero, metrics
        z = self.encoder(obs)
        actions = actions.long()
        encoded = torch.cat([F.one_hot(actions[...,0],3), F.one_hot(actions[...,1],self.n_attacks),
                             F.one_hot(actions[...,2],3), F.one_hot(actions[...,3],self.n_attacks)], -1).to(z.dtype)
        trajectory, _ = self.predictor(encoded)
        action_context = trajectory[:, [h-1 for h in self.prediction_horizons]]
        prediction = self.prediction_head(torch.cat([z[:,None,:].expand(-1,len(self.prediction_horizons),-1), action_context],-1))
        with torch.no_grad():
            targets = self.target_encoder(next_obs)
            if shuffle:
                # Shuffle only among valid examples for each horizon.
                targets = targets.clone()
                for k in range(len(self.prediction_horizons)):
                    idx = torch.nonzero(valid[:,k], as_tuple=True)[0]
                    targets[idx,k] = targets[idx[torch.randperm(len(idx),device=idx.device)],k]
        terms = []
        for k,h in enumerate(self.prediction_horizons):
            mask = valid[:,k]
            if bool(mask.any()):
                term = F.smooth_l1_loss(prediction[mask,k],targets[mask,k])
                terms.append(term)
                metrics[f"prediction_h{h}"] = float(term.detach())
        loss_prediction = torch.stack(terms).mean()
        active_z = z[valid.any(-1)]
        std = torch.sqrt(active_z.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(0.5-std).mean()
        loss = loss_prediction + 0.1*variance
        metrics.update(aux_loss=float(loss.detach()), prediction_loss=float(loss_prediction.detach()),
                       variance_loss=float(variance.detach()), latent_std=float(std.mean().detach()))
        return loss, metrics

    @torch.no_grad()
    def update_target(self, tau: float = 0.99) -> None:
        if self.target_encoder is not None:
            for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target.mul_(tau).add_(online, alpha=1.0 - tau)

    def parameter_counts(self) -> dict[str, int]:
        acting = sum(p.numel() for module in (self.encoder, self.attack_head, self.runner)
                     for p in module.parameters())
        if self.motor_residual_head is not None:
            acting += sum(p.numel() for p in self.motor_residual_head.parameters())
        return {"parameter_count": sum(p.numel() for p in self.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "acting_parameter_count": acting}


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    if logits.shape != mask.shape:
        raise ValueError("Action mask shape must equal logits shape.")
    mask = mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Every action mask must permit at least one action.")
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
