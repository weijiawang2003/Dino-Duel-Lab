"""Masked PPO, online self-play, latent prediction, and resumable training.

An environment step advances BOTH players once. Logs distinguish joint
environment steps from learner-agent transitions. Checkpoints are written only
after complete PPO updates, and contain environment and random-number states.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .env import DuelEnv, EnvConfig, N_ATTACKS, OBS_DIM, RUNNER_DIM
from .teacher import scripted_attacks, teacher_actions
from .models import (ActorCritic, NumpyRunner, initialize_runner, load_runner_npz,
                     masked_categorical)


FORMAT_VERSION = 1
DEFAULTS = {
    "updates": 300, "num_envs": 8, "rollout_steps": 128,
    "epochs": 4, "minibatch_size": 256, "lr": 3e-4, "gamma": 0.995,
    "gae_lambda": 0.95, "clip_coef": 0.2, "entropy_coef": 0.01,
    "value_coef": 0.5, "max_grad_norm": 0.5, "aux_coef": 0.1,
    "ema_tau": 0.99, "checkpoint_interval": 10, "snapshot_interval": 10,
    "opponent_pool_size": 4, "historical_opponent_prob": 0.35,
    "opponent_style": "selfplay", "device": "auto", "torch_threads": 1,
    "train_motor": False, "target_kl": 0.03, "lr_decay_updates": 0,
    "max_wall_seconds": 0.0, "stop_file": "", "bootstrap_time_limits": False,
    "decision_only_actor": False,
    "warm_start": "", "warm_start_heads": True, "scripted_opponent_prob": 0.0,
    "scripted_opponent_styles": ["random", "aggressive", "save"],
    "motor_entropy_coef": None, "motor_anchor_coef": 0.0,
    "aux_batch_size": 64,

}
OPERATIONAL_KEYS = {"updates", "device", "torch_threads", "checkpoint_interval",
                    "max_wall_seconds", "stop_file", "warm_start"}


def choose_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation has no available CUDA device.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available on this machine.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices: auto, cpu, cuda[:index], mps.")
    return device


def resolve_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    requested = result.get("training", {})
    unknown = set(requested) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    result["training"] = {**DEFAULTS, **requested}
    result.setdefault("model", {})
    model_unknown = set(result["model"]) - {"hidden", "runner_hidden", "prediction_horizons", "residual_motor", "motor_logit_scale", "predictor_fusion"}
    if model_unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(model_unknown)}")
    result["model"] = {"hidden": 128, "runner_hidden": 64, "prediction_horizons": [1],
                       "residual_motor": False, "motor_logit_scale": 1.0, "predictor_fusion": "linear", **result["model"]}
    result["env"] = EnvConfig.from_dict(result.get("env", {})).to_dict()
    t = result["training"]
    for key in ("updates", "num_envs", "rollout_steps", "epochs", "minibatch_size",
                "checkpoint_interval", "snapshot_interval", "torch_threads", "aux_batch_size"):
        if isinstance(t[key], bool) or int(t[key]) != t[key] or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer.")
        t[key] = int(t[key])
    for key in ("opponent_pool_size", "lr_decay_updates"):
        if isinstance(t[key], bool) or int(t[key]) != t[key] or t[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer.")
        t[key] = int(t[key])
    for key in ("gamma", "gae_lambda", "ema_tau", "historical_opponent_prob", "scripted_opponent_prob"):
        if not 0 <= t[key] <= 1:
            raise ValueError(f"training.{key} must be in [0,1].")
    for key in ("lr", "max_grad_norm", "clip_coef"):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"training.{key} must be positive and finite.")
    for key in ("entropy_coef", "value_coef", "aux_coef", "target_kl", "max_wall_seconds", "motor_anchor_coef"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"training.{key} must be nonnegative and finite.")
    if t["opponent_style"] not in {"selfplay", "random", "aggressive", "save", "none"}:
        raise ValueError("opponent_style must be selfplay/random/aggressive/save/none.")
    if not isinstance(t["stop_file"], str):
        raise ValueError("training.stop_file must be a path string (or empty).")
    for key in ("train_motor", "bootstrap_time_limits", "decision_only_actor"):
        if not isinstance(t[key], bool):
            raise ValueError(f"training.{key} must be true or false.")
    for key in ("hidden", "runner_hidden"):
        value = result["model"][key]
        if isinstance(value, bool) or int(value) != value or not 8 <= value <= 2048:
            raise ValueError(f"model.{key} must be an integer in [8,2048].")
    horizons = result["model"]["prediction_horizons"]
    if not horizons or any(isinstance(h,bool) or int(h)!=h or not 1<=h<=256 for h in horizons):
        raise ValueError("model.prediction_horizons: positive integers <= 256 required")
    if sorted(set(horizons)) != list(horizons):
        raise ValueError("prediction_horizons must be sorted and unique")
    if max(horizons) > t["rollout_steps"]:
        raise ValueError("rollout_steps must cover the largest prediction horizon")
    if result["model"]["predictor_fusion"] not in {"linear", "nonlinear"}:
        raise ValueError("predictor_fusion must be linear or nonlinear")
    if not isinstance(result["model"]["residual_motor"],bool):
        raise ValueError("residual_motor must be bool")
    if not 0 < result["model"]["motor_logit_scale"] <= 10:
        raise ValueError("motor_logit_scale must be in (0,10]")
    if t["motor_entropy_coef"] is not None and not 0 <= t["motor_entropy_coef"] <= 1:
        raise ValueError("motor_entropy_coef must be null or in [0,1]")
    if not isinstance(t["warm_start_heads"],bool):
        raise ValueError("warm_start_heads must be bool")
    if not isinstance(t["warm_start"],str):
        raise ValueError("warm_start must be a path string")
    if not t["scripted_opponent_styles"] or any(x not in {"random","aggressive","save","none","burst","cycle"} for x in t["scripted_opponent_styles"]):
        raise ValueError("invalid scripted_opponent_styles")
    return result


def build_sequence_targets(batch: dict, horizons: tuple[int, ...]):
    """Contiguous targets formed BEFORE PPO shuffling; no reset leakage.

    A terminal endpoint is permitted. An intervening terminal/truncation is not.
    Future joint actions are factual training labels, never actor inputs.
    """
    obs, joint = batch["obs"], batch["joint_actions"]
    T,N,S,D = obs.shape
    H = max(horizons)
    sequence = np.zeros((T,N,S,H,4),dtype=np.int64)
    targets = np.zeros((T,N,S,len(horizons),D),dtype=np.float32)
    valid = np.zeros((T,N,S,len(horizons)),dtype=bool)
    done = batch["terminated"] | batch["truncated"]
    for k in range(H):
        sequence[:T-k,:,:,k] = joint[k:]
    for j,h in enumerate(horizons):
        length = T-h+1
        if length <= 0:
            continue
        targets[:length,:,:,j] = batch["next_obs"][h-1:]
        no_reset = np.ones((length,N),dtype=bool)
        for k in range(h-1):
            no_reset &= ~done[k:k+length]
        valid[:length,:,:,j] = no_reset[:,:,None]
    return sequence, targets, valid


def _resume_signature(config: dict) -> str:
    # The original release had no decision_only_actor setting. Its disabled
    # default preserves exactly that objective and its existing signatures.
    model_signature = {k:v for k,v in config["model"].items()
                       if not (k=="prediction_horizons" and list(v)==[1])
                       and not (k=="residual_motor" and not v)
                       and not (k=="motor_logit_scale" and v==1.0)
                       and not (k=="predictor_fusion" and v=="linear")}
    new_defaults = {"scripted_opponent_prob":0.0, "scripted_opponent_styles":["random","aggressive","save"],
                    "motor_entropy_coef":None, "motor_anchor_coef":0.0, "aux_batch_size":64, "warm_start_heads":True}
    env_signature = {k:v for k,v in config["env"].items()
                     if not (k=="wide_cactus_width" and v==1.4)
                     and not (k=="wide_cactus_height" and v==0.9)}
    relevant = {"env": env_signature, "model": model_signature,
                "training": {k: v for k, v in config["training"].items()
                             if k not in OPERATIONAL_KEYS
                             and not (k == "decision_only_actor" and not v)
                             and not (k in new_defaults and v == new_defaults[k])}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoint(path: str | Path) -> dict:
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
        raise ValueError("This is not a supported Dino Duel checkpoint.")
    if data.get("project") != "dino-duel-lab":
        raise ValueError("Checkpoint is from another project; no implicit conversion is supported.")
    return data


def _rng_state(rng: np.random.Generator) -> dict:
    old_numpy = np.random.get_state()
    result = {"python": random.getstate(), "numpy_generator": copy.deepcopy(rng.bit_generator.state),
              "numpy_global": [old_numpy[0], old_numpy[1].tolist(), old_numpy[2], old_numpy[3], old_numpy[4]],
              "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state") and torch.backends.mps.is_available():
        result["torch_mps"] = torch.mps.get_rng_state()
    return _cpu_tree(result)


def _restore_rng(state: dict, rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    rng.bit_generator.state = copy.deepcopy(state["numpy_generator"])
    old = state["numpy_global"]
    np.random.set_state((old[0], np.asarray(old[1], dtype=np.uint32), old[2], old[3], old[4]))
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        states = state["torch_cuda"]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)
    if "torch_mps" in state and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"])


def generalized_advantage(rewards: np.ndarray, values: np.ndarray,
                          next_values: np.ndarray, terminated: np.ndarray,
                          truncated: np.ndarray, gamma: float,
                          gae_lambda: float,
                          bootstrap_time_limits: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Never carry GAE across resets; optionally bootstrap external time limits.

    The reusable helper defaults to the usual continuing-task convention. This
    project's trainer explicitly passes False for its finite, scored matches:
    a time-cap draw has terminal value zero, not a continuation estimate.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros_like(rewards[0], dtype=np.float32)
    for index in reversed(range(len(rewards))):
        no_bootstrap = terminated[index] | (truncated[index] if not bootstrap_time_limits else False)
        terminal = no_bootstrap[..., None]
        boundary = (terminated[index] | truncated[index])[..., None]
        delta = rewards[index] + gamma * (~terminal) * next_values[index] - values[index]
        running = delta + gamma * gae_lambda * (~boundary) * running
        advantages[index] = running
    return advantages, advantages + values


def _normalize_actor_advantages(advantages: torch.Tensor,
                                active: torch.Tensor) -> torch.Tensor:
    """Normalize over actual actor samples; forced waits cannot set the scale."""
    result = torch.zeros_like(advantages)
    chosen = advantages[active]
    if chosen.numel() > 1:
        chosen = (chosen - chosen.mean()) / (chosen.std(unbiased=False) + 1e-8)
    result[active] = chosen
    return result


def _actor_terms(logprob: torch.Tensor, old_logprob: torch.Tensor,
                 advantages: torch.Tensor, entropy: torch.Tensor,
                 active: torch.Tensor, clip_coef: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped PPO terms, with finite zero actor loss for forced-wait batches.

    Value and auxiliary losses are computed separately on every transition.
    Masking here also makes KL early stopping reflect real policy decisions.
    """
    if not bool(active.any()):
        zero = logprob.sum() * 0.0
        return zero, entropy.sum() * 0.0, zero.detach(), zero.detach()
    logratio = logprob[active] - old_logprob[active]
    ratio = logratio.exp()
    unclipped = -advantages[active] * ratio
    clipped = -advantages[active] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    policy_loss = torch.maximum(unclipped, clipped).mean()
    with torch.no_grad():
        kl = ((ratio - 1) - logratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean()
    return policy_loss, entropy[active].mean(), kl, clip_fraction


class TorchPolicy:
    """The evaluator's policy interface; runner weights travel with the policy."""

    def __init__(self, model: ActorCritic, device: torch.device,
                 runner: dict, config: dict, name: str | None = None):
        self.model, self.device, self.config = model, device, config
        self.name = name or model.variant
        self.variant = model.variant
        self.runner_spec = runner
        self.numpy_runner = None
        if runner.get("kind") == "numpy":
            weights = {k: np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v)
                       for k, v in runner["weights"].items()}
            self.numpy_runner = NumpyRunner(weights, runner.get("metadata", {}))

    def motor_actions(self, runner_obs: np.ndarray) -> np.ndarray:
        if self.numpy_runner is not None:
            return self.numpy_runner(runner_obs)
        return np.asarray(teacher_actions(runner_obs), dtype=np.int64)

    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray, runner_obs: np.ndarray,
            deterministic: bool = True) -> np.ndarray:
        observation = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        runners = np.asarray(runner_obs, dtype=np.float32).reshape(-1, RUNNER_DIM)
        mask = np.asarray(masks, dtype=bool).reshape(-1, N_ATTACKS)
        self.model.eval()
        result = self.model(torch.as_tensor(observation, device=self.device),
                            torch.as_tensor(runners, device=self.device))
        attack = masked_categorical(result["attack_logits"], torch.as_tensor(mask, device=self.device))
        attacks = attack.logits.argmax(-1) if deterministic else attack.sample()
        if self.model.train_motor:
            motor = torch.distributions.Categorical(logits=result["motor_logits"])
            motors = motor.logits.argmax(-1) if deterministic else motor.sample()
            motors = motors.cpu().numpy()
        else:
            motors = self.motor_actions(runners)
        return np.stack([motors, attacks.cpu().numpy()], axis=-1).astype(np.int64)


def load_policy(path: str | Path, device: str = "cpu") -> TorchPolicy:
    path = Path(path)
    if path.is_dir():
        path = path / ("policy.pt" if (path / "policy.pt").exists() else "latest.pt")
    data = _load_checkpoint(path)
    selected = choose_device(device)
    model = ActorCritic(**data["model_config"]).to(selected)
    model.load_state_dict(data["model_state"])
    model.eval()
    return TorchPolicy(model, selected, data["runner"], data["config"],
                       f"{data['variant']} seed={data['seed']}")


def _trim_metrics(path: Path, update: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # Only a torn trailing write is expected after interruption.
        if item.get("update", -1) <= update:
            retained.append(json.dumps(item, allow_nan=False))
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    os.replace(temporary, path)


def train(config: dict, out_dir: str | Path, variant: str, seed: int,
          resume: bool = False, runner_path: str | Path | None = None) -> dict:
    """Train to a TOTAL update budget; resume=True never adds an implicit budget."""
    config = resolve_config(config)
    t = config["training"]
    if variant not in ActorCritic.VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.pt"
    existing = _load_checkpoint(latest) if resume and latest.exists() else None
    if resume and existing is None:
        raise FileNotFoundError(f"Cannot resume: {latest} does not exist.")
    if not resume and (latest.exists() or (out_dir / "metrics.jsonl").exists()):
        raise FileExistsError(f"Training output already exists: {out_dir}. Use --resume or a new output path.")
    if existing is not None:
        if existing["variant"] != variant or int(existing["seed"]) != int(seed):
            raise ValueError("Resume seed and variant must match the checkpoint.")
        if _resume_signature(config) != existing["resume_signature"]:
            raise ValueError("Resume configuration changes model, environment, or learning settings. "
                             "Only updates/device/threads/checkpoint interval/wall budget may change.")
        # Completed experiment stages are true read-only no-ops. Re-saving an
        # unchanged model would change file hashes and invalidate cached held-out
        # evaluations, even though no training took place.
        if int(existing["update"]) >= t["updates"]:
            weights = existing["model_state"]
            counts = {
                "parameter_count": sum(value.numel() for value in weights.values()),
                "acting_parameter_count": sum(value.numel() for key, value in weights.items()
                                              if key.startswith(("encoder.", "attack_head.", "runner.", "motor_residual_head."))),
                "trainable_parameter_count": sum(value.numel() for key, value in weights.items()
                                                 if not key.startswith("target_encoder.")
                                                 and ((existing["model_config"]["train_motor"]
                                                       and not existing["model_config"].get("residual_motor",False))
                                                      or not key.startswith("runner."))),
            }
            policy_path = out_dir / "policy.pt"
            return {"status": "complete", "already_complete": True,
                    "variant": variant, "seed": seed,
                    "update": int(existing["update"]), "target_updates": t["updates"],
                    "global_step": int(existing["global_step"]),
                    "agent_steps": int(existing["agent_steps"]),
                    "checkpoint": str(latest),
                    "policy": str(policy_path if policy_path.exists() else latest),
                    "elapsed_seconds": float(existing["elapsed_seconds"]),
                    "device": existing["device"], **counts}
    torch.set_num_threads(t["torch_threads"])
    device = choose_device(t["device"])
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    runner_spec: dict = {"kind": "teacher", "metadata": {"source": "built-in observable-state teacher"}}
    if existing is not None:
        runner_spec = existing["runner"]
    elif runner_path is not None:
        runner = load_runner_npz(runner_path)
        runner_spec = {"kind": "numpy", "weights": _cpu_tree(runner.weights),
                       "metadata": runner.metadata,
                       "sha256": hashlib.sha256(Path(runner_path).read_bytes()).hexdigest()}
    if t["train_motor"] and runner_spec["kind"] != "numpy":
        raise ValueError("Joint motor training needs a shared runner .npz; run bootstrap first.")
    runner_hidden = (int(runner_spec["weights"]["w0"].shape[0])
                     if runner_spec["kind"] == "numpy" else config["model"]["runner_hidden"])
    hidden = config["model"]["hidden"] * (2 if variant == "baseline_large" else 1)
    model_config = {"obs_dim": OBS_DIM, "n_attacks": N_ATTACKS, "hidden": hidden,
                    "variant": variant, "train_motor": t["train_motor"], "runner_hidden": runner_hidden,
                    "prediction_horizons": tuple(config["model"]["prediction_horizons"]),
                    "residual_motor": config["model"]["residual_motor"],
                    "motor_logit_scale": config["model"]["motor_logit_scale"],
                    "predictor_fusion": config["model"]["predictor_fusion"]}
    model = ActorCritic(**model_config).to(device)
    if runner_spec["kind"] == "numpy":
        initialize_runner(model.runner, {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                                        for k, v in runner_spec["weights"].items()})
    warm_start_provenance = None
    if existing is None and t["warm_start"]:
        warm_path = Path(t["warm_start"])
        old = _load_checkpoint(warm_path)
        state = model.state_dict()
        copied=[]
        for key,value in old["model_state"].items():
            if key.startswith(("encoder.","attack_head.","value_head.") if t["warm_start_heads"] else ("encoder.",)) and key in state:
                if state[key].shape != value.shape:
                    raise ValueError(f"warm-start shape mismatch: {key}")
                state[key] = value.to(device); copied.append(key)
        model.load_state_dict(state)
        if model.target_encoder is not None:
            model.target_encoder.load_state_dict(model.encoder.state_dict())
        warm_start_provenance = {"sha256":hashlib.sha256(warm_path.read_bytes()).hexdigest(),
                                 "source_seed":old["seed"], "source_variant":old["variant"],
                                 "source_steps":old["global_step"], "copied_keys":copied}
    elif existing is not None:
        warm_start_provenance = existing.get("warm_start_provenance")
    anchor_runner = copy.deepcopy(model.runner).requires_grad_(False).eval()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=t["lr"], eps=1e-5)
    live_policy = TorchPolicy(model, device, runner_spec, config)
    environments = [DuelEnv(config["env"]) for _ in range(t["num_envs"])]
    for index, env in enumerate(environments):
        # Reserve >=2**32 for held-out evaluation. Even a large model seed
        # cannot accidentally initialize an episode in that reserved domain.
        env.reset(seed=int((seed + 10007 * (index + 1)) % (2**31)))
    episode_returns = np.zeros((len(environments), 2), dtype=np.float64)
    episode_lengths = np.zeros(len(environments), dtype=np.int64)
    active: list[dict | None] = [None] * len(environments)
    active_policies: list[TorchPolicy | None] = [None] * len(environments)
    pool: list[dict] = ([{"update":0, "model_state":_cpu_tree(model.state_dict())}]
                        if t["opponent_pool_size"] and config["model"]["residual_motor"] else [])
    completed_update = global_step = agent_steps = 0
    elapsed_before = 0.0
    started = time.monotonic()
    stop_requested = [False]

    def should_stop() -> bool:
        external = bool(t["stop_file"]) and Path(t["stop_file"]).expanduser().exists()
        return stop_requested[0] or (out_dir / "STOP").exists() or bool(external)

    def create_opponent(spec: dict) -> TorchPolicy | None:
        if spec.get("kind") != "snapshot":
            return None
        opponent = ActorCritic(**model_config).to(device)
        opponent.load_state_dict(spec["model_state"])
        opponent.requires_grad_(False).eval()
        return TorchPolicy(opponent, device, runner_spec, config, f"snapshot-{spec['update']}")

    def select_opponent(index: int) -> None:
        style = t["opponent_style"]
        if style == "selfplay" and t["scripted_opponent_prob"] > 0 and rng.random() < t["scripted_opponent_prob"]:
            active[index] = {"kind": "scripted", "style": str(rng.choice(t["scripted_opponent_styles"])), "seat": int(rng.integers(2))}
        elif style != "selfplay":
            active[index] = {"kind": "scripted", "style": style, "seat": int(rng.integers(2))}
        elif pool and rng.random() < t["historical_opponent_prob"]:
            selected = pool[int(rng.integers(len(pool)))]
            active[index] = {"kind": "snapshot", "update": selected["update"],
                             "model_state": selected["model_state"], "seat": int(rng.integers(2))}
        else:
            active[index] = None
        active_policies[index] = create_opponent(active[index]) if active[index] else None

    if existing is not None:
        if len(existing["env_states"]) != len(environments):
            raise ValueError("num_envs cannot change on resume.")
        model.load_state_dict(existing["model_state"])
        optimizer.load_state_dict(existing["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        for env, state in zip(environments, existing["env_states"]):
            env.set_state(state)
        episode_returns[:] = np.asarray(existing["episode_returns"])
        episode_lengths[:] = np.asarray(existing["episode_lengths"])
        pool = existing["opponent_pool"]
        active = existing["active_opponents"]
        active_policies = [create_opponent(spec) if spec else None for spec in active]
        completed_update = int(existing["update"])
        global_step, agent_steps = int(existing["global_step"]), int(existing["agent_steps"])
        elapsed_before = float(existing["elapsed_seconds"])
        _trim_metrics(out_dir / "metrics.jsonl", completed_update)
        _restore_rng(existing["rng"], rng)  # Restore AFTER model construction.
    else:
        for index in range(len(environments)):
            select_opponent(index)
        # Identical baseline/JEPA acting weights and initial action-sampling RNG.
        torch.manual_seed(seed + 17)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 17)

    def payload(full: bool = True) -> dict:
        result = {"format_version": FORMAT_VERSION, "project": "dino-duel-lab",
                  "variant": variant, "seed": int(seed), "config": config,
                  "resume_signature": _resume_signature(config), "model_config": model_config,
                  "model_state": _cpu_tree(model.state_dict()), "runner": runner_spec,
                  "update": completed_update, "global_step": global_step, "agent_steps": agent_steps,
                  "elapsed_seconds": elapsed_before + time.monotonic() - started,
                  "torch_version": str(torch.__version__), "device": str(device),
                  "warm_start_provenance": warm_start_provenance}
        if full:
            result.update({"optimizer_state": _cpu_tree(optimizer.state_dict()),
                           "env_states": [env.get_state() for env in environments],
                           "episode_returns": episode_returns.tolist(),
                           "episode_lengths": episode_lengths.tolist(),
                           "opponent_pool": _cpu_tree(pool), "active_opponents": _cpu_tree(active),
                           "rng": _rng_state(rng)})
        return result

    def save_checkpoint(numbered: bool = False) -> None:
        data = payload()
        atomic_torch_save(data, latest)
        atomic_torch_save({k: v for k, v in data.items() if k not in {
            "optimizer_state", "env_states", "episode_returns", "episode_lengths",
            "opponent_pool", "active_opponents", "rng"}}, out_dir / "policy.pt")
        if numbered:
            atomic_torch_save(data, out_dir / "checkpoints" / f"update_{completed_update:06d}.pt")

    def request_stop(signum, frame):
        stop_requested[0] = True

    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signal_id in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signal_id] = signal.getsignal(signal_id)
            signal.signal(signal_id, request_stop)
    atomic_json({**config, "variant": variant, "seed": seed,
                 "runner": {k: v for k, v in runner_spec.items() if k != "weights"},
                 "effective_model": model_config}, out_dir / "config.json")
    if existing is None:
        save_checkpoint()
        atomic_torch_save(payload(full=False), out_dir / "initial_policy.pt")
    status = "complete"
    try:
        for update in range(completed_update + 1, t["updates"] + 1):
            if should_stop():
                status = "stopped"
                break
            if t["max_wall_seconds"] and time.monotonic() - started >= t["max_wall_seconds"]:
                status = "wall_budget"
                break
            if t["lr_decay_updates"]:
                fraction = max(0.1, 1 - (update - 1) / t["lr_decay_updates"])
                for group in optimizer.param_groups:
                    group["lr"] = t["lr"] * fraction
            records: dict[str, list] = {key: [] for key in (
                "obs", "runner_obs", "mask", "actions", "joint_actions", "logprob", "value",
                "next_value", "next_obs", "reward", "terminated", "truncated", "learn")}
            finished: list[dict] = []
            model.eval()
            for _ in range(t["rollout_steps"]):
                obs = np.stack([env.observe() for env in environments])
                runners = np.stack([env.runner_observations() for env in environments])
                masks = np.stack([env.attack_masks() for env in environments])
                with torch.no_grad():
                    output = model(torch.as_tensor(obs.reshape(-1, OBS_DIM), device=device),
                                   torch.as_tensor(runners.reshape(-1, RUNNER_DIM), device=device))
                    attack_dist = masked_categorical(output["attack_logits"],
                                                     torch.as_tensor(masks.reshape(-1, N_ATTACKS), device=device))
                    attacks_t = attack_dist.sample()
                    logprob = attack_dist.log_prob(attacks_t)
                    if t["train_motor"]:
                        motor_dist = torch.distributions.Categorical(logits=output["motor_logits"])
                        motors_t = motor_dist.sample()
                        logprob = logprob + motor_dist.log_prob(motors_t)
                        motors = motors_t.cpu().numpy()
                    else:
                        motors = live_policy.motor_actions(runners.reshape(-1, RUNNER_DIM))
                    actions = np.stack([motors, attacks_t.cpu().numpy()], axis=-1).reshape(-1, 2, 2)
                    old_logprob = logprob.cpu().numpy().reshape(-1, 2)
                    values = output["value"].cpu().numpy().reshape(-1, 2)
                learn = np.ones((len(environments), 2), dtype=bool)
                for index, opponent in enumerate(active):
                    if opponent is None:
                        continue
                    seat = opponent["seat"]
                    learn[index, seat] = False
                    if opponent["kind"] == "snapshot":
                        actions[index, seat] = active_policies[index].act(
                            obs[index, seat:seat+1], masks[index, seat:seat+1],
                            runners[index, seat:seat+1], deterministic=False)[0]
                    else:
                        chosen = scripted_attacks(obs[index, seat:seat+1], masks[index, seat:seat+1],
                                                  rng, style=opponent["style"])
                        actions[index, seat, 0] = live_policy.motor_actions(runners[index, seat:seat+1])[0]
                        actions[index, seat, 1] = int(np.asarray(chosen).reshape(-1)[0])
                next_observations, rewards, terminal, truncated, executed = [], [], [], [], []
                for index, env in enumerate(environments):
                    next_obs, reward, term, trunc, info = env.step(actions[index])
                    next_observations.append(next_obs)
                    rewards.append(reward)
                    terminal.append(term)
                    truncated.append(trunc)
                    executed.append(info.get("executed_actions", actions[index]))
                    episode_returns[index] += reward
                    episode_lengths[index] += 1
                    if term or trunc:
                        finished.append({"returns": episode_returns[index].copy(),
                                         "length": int(episode_lengths[index]),
                                         "winner": info.get("winner"), "truncated": bool(trunc)})
                        episode_returns[index] = 0
                        episode_lengths[index] = 0
                        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                        select_opponent(index)
                next_obs = np.asarray(next_observations, dtype=np.float32)
                with torch.no_grad():
                    next_value = model(torch.as_tensor(next_obs.reshape(-1, OBS_DIM), device=device))["value"]
                    next_value = next_value.cpu().numpy().reshape(-1, 2)
                joint = np.asarray(executed, dtype=np.int64)
                perspective_actions = np.stack([joint.reshape(-1, 4), joint[:, ::-1, :].reshape(-1, 4)], axis=1)
                entries = {"obs": obs, "runner_obs": runners, "mask": masks, "actions": actions,
                           "joint_actions": perspective_actions, "logprob": old_logprob,
                           "value": values, "next_value": next_value, "next_obs": next_obs,
                           "reward": np.asarray(rewards, dtype=np.float32),
                           "terminated": np.asarray(terminal, dtype=bool),
                           "truncated": np.asarray(truncated, dtype=bool), "learn": learn}
                for key, value in entries.items():
                    records[key].append(value)
                global_step += len(environments)
                agent_steps += int(learn.sum())
            batch_np = {key: np.asarray(value) for key, value in records.items()}
            advantages, returns = generalized_advantage(
                batch_np["reward"], batch_np["value"], batch_np["next_value"],
                batch_np["terminated"], batch_np["truncated"], t["gamma"], t["gae_lambda"],
                bootstrap_time_limits=t["bootstrap_time_limits"])
            selected = batch_np["learn"].reshape(-1)
            flat: dict[str, torch.Tensor] = {}
            for key in ("obs", "runner_obs", "mask", "actions", "joint_actions", "next_obs"):
                array = batch_np[key].reshape(-1, batch_np[key].shape[-1])[selected]
                flat[key] = torch.as_tensor(array, device=device)
            for key, array in (("logprob", batch_np["logprob"]), ("old_value", batch_np["value"]),
                               ("advantage", advantages), ("returns", returns)):
                flat[key] = torch.as_tensor(array.reshape(-1)[selected], device=device)
            valid = np.repeat((~batch_np["terminated"])[..., None], 2, axis=-1).reshape(-1)[selected]
            flat["valid"] = torch.as_tensor(valid, device=device)
            if model.has_auxiliary and model.multistep:
                seq, targets, sequence_valid = build_sequence_targets(batch_np, model.prediction_horizons)
                flat["joint_actions"] = torch.as_tensor(seq.reshape(-1,*seq.shape[3:])[selected],device=device)
                flat["next_obs"] = torch.as_tensor(targets.reshape(-1,*targets.shape[3:])[selected],device=device)
                flat["valid"] = torch.as_tensor(sequence_valid.reshape(-1,sequence_valid.shape[-1])[selected],device=device)
            decision_mask = flat["mask"].sum(dim=-1) > 1
            # A frozen runner leaves only attack choices to learn. Five out of
            # six default motor steps cannot purchase anything, so their forced
            # waits should not dilute this optional decision-normalized actor.
            # Joint motor policies still make a motor decision at every step.
            flat["actor_active"] = (decision_mask if t["decision_only_actor"] and not t["train_motor"]
                                    else torch.ones_like(decision_mask))
            flat["advantage"] = _normalize_actor_advantages(flat["advantage"], flat["actor_active"])
            samples = len(flat["advantage"])
            effective_decisions = int(decision_mask.sum().item())
            actor_samples = int(flat["actor_active"].sum().item())
            if samples == 0:
                raise RuntimeError("No learner transitions were collected.")
            sums: dict[str, float] = {}
            optimized = 0
            stopped_early = False
            model.train()
            for epoch in range(t["epochs"]):
                order = torch.randperm(samples, device=device)
                for start in range(0, samples, t["minibatch_size"]):
                    ix = order[start:start + t["minibatch_size"]]
                    minibatch = {key: value[ix] for key, value in flat.items()}
                    result = model(minibatch["obs"], minibatch["runner_obs"])
                    attack_dist = masked_categorical(result["attack_logits"], minibatch["mask"])
                    logprob = attack_dist.log_prob(minibatch["actions"][:, 1])
                    entropy = attack_dist.entropy()
                    motor_entropy = torch.zeros_like(entropy)
                    motor_anchor_loss = entropy.sum()*0.0
                    if t["train_motor"]:
                        motor_dist = torch.distributions.Categorical(logits=result["motor_logits"])
                        logprob = logprob + motor_dist.log_prob(minibatch["actions"][:, 0])
                        motor_entropy = motor_dist.entropy()
                        if t["motor_entropy_coef"] is None:
                            entropy = entropy + motor_entropy
                        if t["motor_anchor_coef"]:
                            with torch.no_grad():
                                anchor = torch.distributions.Categorical(logits=anchor_runner(minibatch["runner_obs"]) * model.motor_logit_scale)
                            # KL in log space: Categorical.kl can call tiny softmax
                            # probabilities exactly zero and report +inf despite
                            # finite logits. Both policies have full support.
                            log_p = F.log_softmax(anchor.logits,dim=-1)
                            log_q = F.log_softmax(result["motor_logits"],dim=-1)
                            motor_anchor_loss = (log_p.exp()*(log_p-log_q)).sum(-1).mean()
                    policy_loss, actor_entropy, kl, clip_fraction = _actor_terms(
                        logprob, minibatch["logprob"], minibatch["advantage"], entropy,
                        minibatch["actor_active"], t["clip_coef"])
                    value_loss = 0.5 * F.mse_loss(result["value"], minibatch["returns"])
                    aux_count = min(t["aux_batch_size"],len(ix)) if model.multistep else len(ix)
                    aux_loss, aux_metrics = model.auxiliary_loss(
                        minibatch["obs"][:aux_count], minibatch["joint_actions"][:aux_count], minibatch["next_obs"][:aux_count],
                        minibatch["valid"][:aux_count], shuffle=variant == "jepa_shuffled")
                    loss = policy_loss + t["value_coef"] * value_loss - t["entropy_coef"] * actor_entropy + t["aux_coef"] * aux_loss
                    if t["motor_entropy_coef"] is not None:
                        loss = loss - t["motor_entropy_coef"] * motor_entropy.mean()
                    loss = loss + t["motor_anchor_coef"] * motor_anchor_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite PPO loss: actor={policy_loss.item()} value={value_loss.item()} aux={aux_loss.item()} anchor={motor_anchor_loss.item()}; last complete checkpoint available.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("Non-finite gradient; last complete checkpoint remains available.")
                    optimizer.step()
                    model.update_target(t["ema_tau"])
                    values_to_log = {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                                     "entropy": float(actor_entropy.detach()), "motor_entropy":float(motor_entropy.mean().detach()),
                                     "motor_anchor_kl":float(motor_anchor_loss.detach()), "approx_kl": float(kl),
                                     "clip_fraction": float(clip_fraction), "grad_norm": float(grad_norm), **aux_metrics}
                    for key, value in values_to_log.items():
                        sums[key] = sums.get(key, 0.0) + value
                    optimized += 1
                    if t["target_kl"] and float(kl) > t["target_kl"]:
                        stopped_early = True
                        break
                if stopped_early:
                    break
            completed_update = update
            if t["opponent_pool_size"] and update % t["snapshot_interval"] == 0:
                pool.append({"update": update, "model_state": _cpu_tree(model.state_dict())})
                pool = ([pool[0]] + pool[-(t["opponent_pool_size"]-1):]
                        if t["opponent_pool_size"] > 1 else pool[-1:])
            elapsed = elapsed_before + time.monotonic() - started
            row = {"update": update, "global_step": global_step, "agent_steps": agent_steps,
                   "physics_steps": global_step * config["env"].get("action_repeat", 1),
                   "elapsed_seconds": elapsed, "variant": variant, "seed": int(seed),
                   "device": str(device), "learning_rate": optimizer.param_groups[0]["lr"],
                   "minibatches": optimized, "kl_early_stop": stopped_early,
                   "completed_episodes": len(finished), "opponent_pool_size": len(pool),
                   "effective_decisions": effective_decisions,
                   "decision_fraction": effective_decisions / samples,
                   "actor_samples": actor_samples,
                   **model.parameter_counts(),
                   **{key: value / max(1, optimized) for key, value in sums.items()}}
            if finished:
                row.update({"mean_episode_return": float(np.mean([item["returns"].mean() for item in finished])),
                            "mean_abs_episode_return": float(np.mean([np.abs(item["returns"]).mean() for item in finished])),
                            "episode_length": float(np.mean([item["length"] for item in finished])),
                            "win_rate_seat0": float(np.mean([item["winner"] == 0 for item in finished])),
                            "draw_rate": float(np.mean([item["winner"] is None and not item["truncated"] for item in finished])),
                            "time_limit_rate": float(np.mean([item["truncated"] for item in finished]))})
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
            print(f"[{variant} seed={seed}] update {update}/{t['updates']} "
                  f"steps={global_step} policy={row['policy_loss']:.4f} "
                  f"aux={row['aux_loss']:.4f} episodes={len(finished)}", flush=True)
            if update % t["checkpoint_interval"] == 0 or update == t["updates"] or should_stop():
                save_checkpoint(numbered=update % t["checkpoint_interval"] == 0)
            if should_stop():
                status = "stopped"
                break
        save_checkpoint()
    finally:
        for signal_id, handler in old_handlers.items():
            signal.signal(signal_id, handler)
    result = {"status": status, "variant": variant, "seed": seed,
              "update": completed_update, "target_updates": t["updates"],
              "global_step": global_step, "agent_steps": agent_steps,
              "checkpoint": str(latest), "policy": str(out_dir / "policy.pt"),
              "elapsed_seconds": elapsed_before + time.monotonic() - started,
              "device": str(device), **model.parameter_counts()}
    atomic_json(result, out_dir / "training_summary.json")
    return result
