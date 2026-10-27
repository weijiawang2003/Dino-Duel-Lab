"""Build a shared basic runner by imitation; never starts with RL exploration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np

from .env import DuelEnv, EnvConfig
from .teacher import teacher_actions, scripted_attacks
from .utils import atomic_json, file_sha256, runtime_info


def synthetic_data(env_config, samples: int, seed: int):
    """Decorrelate obstacle count/type from timing, using teacher-labelled states.

    These are explicit synthetic supervision states, not claimed game rollouts.
    Ground/air phases are physically consistent; nearest hazard distances are
    oversampled so a high overall accuracy cannot hide a missed jump boundary.
    """
    from .env import RUNNER_DIM, OBSTACLE_SPECS
    c = EnvConfig.from_dict(env_config)
    rng = np.random.default_rng(seed)
    x = np.zeros((samples, RUNNER_DIM), dtype=np.float32)
    for i in range(samples):
        speed = rng.uniform(c.initial_speed, c.max_speed)
        grounded = rng.random() < 0.75
        phase = 0.0 if grounded else rng.uniform(0.02, 2 * c.jump_velocity / c.gravity - 0.02)
        y = max(0.0, c.jump_velocity * phase - 0.5 * c.gravity * phase * phase)
        vy = 0.0 if grounded else c.jump_velocity - c.gravity * phase
        x[i, :6] = (y / 3, vy / 12, grounded, grounded and rng.random() < 0.2, speed / 20, 1)
        count = int(rng.choice([0, 1, 2, 3], p=[0.03, 0.37, 0.40, 0.20]))
        distance = rng.uniform(-0.25, 5.0) if rng.random() < 0.8 else rng.uniform(5, 20)
        for j in range(count):
            kind = int(rng.integers(1, 5))
            width, bottom, height = OBSTACLE_SPECS[kind]
            x[i, 6 + j * 6:12 + j * 6] = (1, distance / 40, width / 3, bottom / 3, height / 3, kind / 4)
            distance += width + rng.uniform(2.0, 12.0)
    return x, np.asarray(teacher_actions(x), dtype=np.int64)


def collect_data(env_config, samples: int, seed: int, student=None,
                 student_fraction: float = 0.0):
    """Labels always come from the common observable-state teacher.

    Student rollouts (DAgger) expose the teacher to errors made by the student.
    A small legal action perturbation expands state coverage. No privileged
    future random state, collision override or reset is given to the teacher.
    """
    env = DuelEnv(env_config)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    xs, ys = [], []
    episodes = 0
    while len(xs) < samples:
        runner = env.runner_observations()
        target = np.asarray(teacher_actions(runner), dtype=np.int64)
        motor = target.copy()
        if student is not None:
            proposed = np.asarray(student(runner), dtype=np.int64)
            use = rng.random(2) < student_fraction
            motor[use] = proposed[use]
        # Deterministic balance of teacher/perturbed trajectories, not labels.
        perturb = rng.random(2) < 0.04
        motor[perturb] = rng.integers(0, 3, size=int(perturb.sum()))
        for x, y in zip(runner, target):
            xs.append(x.copy())
            ys.append(int(y))
        attack = scripted_attacks(obs, env.attack_masks(), rng, style="random")
        obs, _, terminated, truncated, _ = env.step(np.column_stack((motor, attack)))
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset(seed=seed + episodes)
    return np.asarray(xs[:samples], dtype=np.float32), np.asarray(ys[:samples], dtype=np.int64)


def bootstrap(config: dict, out_path: str | Path, *, seed: int | None = None,
              overwrite: bool = False) -> dict:
    import torch
    from torch import nn
    from .models import RunnerNet, export_runner_npz
    from .policies import NumpyRunner

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Runner exists: {out_path}. Use --overwrite to deliberately replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    options = config.get("bootstrap", {})
    seed = int(options.get("seed", 20261025) if seed is None else seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("training", {}).get("torch_threads", 1)))
    rng = np.random.default_rng(seed)
    hidden = int(options.get("hidden", 64))
    samples, epochs = int(options.get("samples", 24000)), int(options.get("epochs", 30))
    batch_size = int(options.get("batch_size", 512))
    rounds = int(options.get("dagger_rounds", 2))
    dagger_steps = int(options.get("dagger_steps", 4000))
    if samples < 100 or epochs < 1 or batch_size < 1 or rounds < 0:
        raise ValueError("Bootstrap requires samples>=100, epochs>=1, batch_size>=1, rounds>=0.")
    env_config = config.get("env", {})
    x, y = collect_data(env_config, samples, seed)
    synthetic_samples = int(options.get("synthetic_samples", 60000))
    if synthetic_samples:
        sx, sy = synthetic_data(env_config, synthetic_samples, seed + 200000)
        x, y = np.concatenate((x, sx)), np.concatenate((y, sy))
    # Validation rollouts use disjoint seeds; no random row split of one trajectory.
    vx, vy = collect_data(env_config, min(max(samples // 5, 500), 5000), seed + 100000)
    if synthetic_samples:
        sx, sy = synthetic_data(env_config, min(synthetic_samples // 5, 10000), seed + 300000)
        vx, vy = np.concatenate((vx, sx)), np.concatenate((vy, sy))
    net = RunnerNet(hidden=hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(options.get("lr", 0.001)))
    tx, ty, tvx, tvy = map(torch.as_tensor, (x, y, vx, vy))
    history = []
    for round_index in range(rounds + 1):
        counts = np.bincount(ty.numpy(), minlength=3).clip(1)
        # Square-root balancing avoids an always-noop fit without making a rare
        # jump command arbitrarily more important than all other actions.
        weights = np.sqrt(counts.sum() / counts).astype(np.float32)
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
        best_loss, best_state = float("inf"), None
        for epoch in range(epochs):
            net.train()
            order = rng.permutation(len(tx))
            total = 0.0
            for offset in range(0, len(order), batch_size):
                ids = order[offset:offset + batch_size]
                loss = criterion(net(tx[ids]), ty[ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            net.eval()
            with torch.no_grad():
                logits = net(tvx)
                validation_loss = float(criterion(logits, tvy))
                accuracy = float((logits.argmax(-1) == tvy).float().mean())
            history.append({"round": round_index, "epoch": epoch + 1,
                            "train_loss": total / len(tx), "validation_loss": validation_loss,
                            "validation_accuracy": accuracy})
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
        with torch.no_grad():
            selected_accuracy = float((net(tvx).argmax(-1) == tvy).float().mean())
        metadata = {"format_version": 1, "source": "original behavior cloning + DAgger",
                    "seed": seed, "hidden": hidden, "env_config": EnvConfig.from_dict(env_config).to_dict(),
                    "round": round_index, "training_rows": len(tx),
                    "validation_seed_start": seed + 100000,
                    "validation_accuracy": selected_accuracy,
                    "synthetic_supervision_rows": synthetic_samples,
                    "description": "Basic motor policy only; no trained attack or JEPA claims."}
        export_runner_npz(net, out_path, metadata=metadata)
        print(f"Runner round {round_index + 1}/{rounds + 1}: validation accuracy "
              f"{history[-1]['validation_accuracy']:.3f}, rows={len(tx)}", flush=True)
        if round_index < rounds:
            nx, ny = collect_data(env_config, dagger_steps, seed + (round_index + 1) * 10000,
                                  student=NumpyRunner(out_path), student_fraction=0.75)
            tx = torch.cat((tx, torch.from_numpy(nx)))
            ty = torch.cat((ty, torch.from_numpy(ny)))
    summary = {"runner": out_path.name, "sha256": file_sha256(out_path), "seed": seed,
               "config": options, "env_config": EnvConfig.from_dict(env_config).to_dict(),
               "history": history, "runtime": runtime_info(),
               "note": "Action-label accuracy is not a survival benchmark. See runner_evaluation for closed-loop performance."}
    atomic_json(out_path.with_suffix(".json"), summary)
    return summary

