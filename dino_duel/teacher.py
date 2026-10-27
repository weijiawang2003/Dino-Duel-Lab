"""Small deterministic motor expert and public-state attack baselines.

The expert observes the same 24 runner features as a learned controller. It has
no access to the engine object, future obstacle generation or opponent actions.
It is a warm start for default physics, not an optimal adversarial policy.
"""
from __future__ import annotations

import numpy as np

from .env import N_ATTACKS, OBS_DIM, RUNNER_DIM


def teacher_actions(runner_obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(runner_obs, dtype=np.float32)
    scalar = obs.ndim == 1
    if scalar:
        obs = obs[None, :]
    if obs.ndim != 2 or obs.shape[1] != RUNNER_DIM:
        raise ValueError(f"runner observations must have shape (...,{RUNNER_DIM})")
    actions = np.zeros(len(obs), dtype=np.int64)
    for i, row in enumerate(obs):
        if row[5] < 0.5:
            continue
        speed = max(float(row[4]) * 20.0, 0.01)
        grounded = row[2] > 0.5
        for j in range(3):
            obstacle = row[6 + 6 * j:12 + 6 * j]
            if obstacle[0] < 0.5:
                continue
            x, width, bottom, height = (float(obstacle[k]) * scale
                                        for k, scale in ((1, 40), (2, 3), (3, 3), (4, 3)))
            if x + width <= 0:
                continue
            if bottom >= 1.5:
                # A high bird is safe when running; do not jump into it.
                continue
            if bottom > 0.55:
                if x < 0.7 + speed * 0.35:
                    actions[i] = 2
                    break
                continue
            if grounded and x < 0.7 + speed * 0.26:
                actions[i] = 1
                break
            # A near cactus is the first active hazard. Future low birds must
            # not pre-empt its jump before they themselves enter range.
    return actions[0] if scalar else actions


def scripted_attacks(obs: np.ndarray, masks: np.ndarray, rng: np.random.Generator,
                     style: str = "random") -> np.ndarray:
    """Choose only allowed attacks. 'save' waits for costly attacks, 'none' idles."""
    observations = np.asarray(obs)
    legal = np.asarray(masks, dtype=bool)
    scalar = legal.ndim == 1
    if scalar:
        legal = legal[None, :]
        observations = observations[None, :] if observations.ndim == 1 else observations
    if legal.ndim != 2 or legal.shape[1] != N_ATTACKS:
        raise ValueError(f"masks must have shape (...,{N_ATTACKS})")
    if style not in {"random", "save", "aggressive", "none", "burst", "cycle"}:
        raise ValueError(f"unknown attack style: {style}")
    actions = np.zeros(len(legal), dtype=np.int64)
    for i, mask in enumerate(legal):
        options = np.flatnonzero(mask[1:]) + 1
        if style == "none" or not len(options):
            continue
        if style == "random":
            if rng.random() < 0.20:
                actions[i] = int(rng.choice(options))
        elif style == "burst":
            row = observations[i]
            queued = any(row[24+6+6*j] > .5 and abs(row[24+11+6*j]-.25)<.05 for j in range(3))
            if mask[1] and (row[48] >= 1/3 - 1e-6 or queued):
                actions[i] = 1
        elif style == "cycle":
            row=observations[i]
            kinds=[round(float(row[24+11+6*j])*4) for j in range(3) if row[24+6+6*j]>.5]
            preferred = {1:3,3:2,2:4,4:1}.get(kinds[-1] if kinds else 4,1)
            if mask[preferred]:
                actions[i]=preferred
            elif rng.random()<.05:
                actions[i]=int(rng.choice(options))
        elif style == "aggressive":
            # Spend promptly, with stochastic types to avoid a deterministic
            # attack sequence tied to the wallet's cheapest price.
            actions[i] = int(rng.choice(options))
        elif style == "save":
            preferred = [kind for kind in (3, 2) if mask[kind]]
            if preferred:
                actions[i] = preferred[0]
    return actions[0] if scalar else actions
