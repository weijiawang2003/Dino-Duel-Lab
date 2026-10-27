"""Paired, seat-swapped evaluations of frozen policies.

Uncertainty is bootstrapped over complete seed pairs, never over simulation
frames. These intervals describe environment-seed uncertainty conditional on the
particular trained policies; they are not intervals across training seeds.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from .env import DuelEnv


class Policy(Protocol):
    def act(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        runner_obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray: ...


def _name(policy: Policy, fallback: str) -> str:
    return str(getattr(policy, "name", fallback))


def _reset_policy(policy: Policy, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset(seed=seed)


def _validated_actions(policy, obs, masks, runner_obs) -> np.ndarray:
    # The environment owns its arrays. A policy must not be able to change the
    # other seat's observation or the mask used to validate its own decision.
    actions = np.asarray(policy.act(obs.copy(), masks.copy(), runner_obs.copy(), deterministic=True))
    if actions.shape != (len(obs), 2):
        raise ValueError(f"Policy must return shape ({len(obs)},2); got {actions.shape}")
    if not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
        raise ValueError("Policy actions must be finite integers")
    if not np.equal(actions, np.floor(actions)).all():
        raise ValueError("Policy actions must be integers")
    actions = actions.astype(np.int64)
    if np.any((actions[:, 0] < 0) | (actions[:, 0] > 2)):
        raise ValueError("Motor action must be 0, 1, or 2")
    if np.any((actions[:, 1] < 0) | (actions[:, 1] >= masks.shape[1])):
        raise ValueError("Attack action is out of range")
    if not np.all(masks[np.arange(len(actions)), actions[:, 1]]):
        raise ValueError("Policy selected an attack forbidden by the public action mask")
    return actions


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _snapshot(env, step: int, actions=None) -> dict:
    snap = copy.deepcopy(env.snapshot())
    snap["step"] = step
    if actions is not None:
        snap["actions"] = np.asarray(actions, dtype=int).tolist()
    return snap


def _step_limit(env) -> int:
    config = getattr(env, "config", None)
    if config is not None:
        get = config.get if isinstance(config, dict) else lambda k, d: getattr(config, k, d)
        duration = get("max_seconds", None)
        dt = get("dt", None)
        repeat = get("action_repeat", None)
        if duration is not None and dt is not None and repeat is not None:
            return int(math.ceil(float(duration) / (float(dt) * int(repeat)))) + 4
    return 1_000_000


def play_episode(
    left: Policy,
    right: Policy,
    env_config: Any,
    seed: int,
    left_seat: int = 0,
    record: bool = False,
    max_frames: int = 3000,
    safety_seconds: float = 300.0,
) -> tuple[dict, list[dict]]:
    """Play one game, with outcomes always expressed for the logical left policy.

Policy reset seeds depend on policy identity and environment seed, not seat.
Policies without a reset method must be stateless. Recorded frames are thinned
uniformly as needed, including the terminal frame, to bound replay size.
"""
    if left_seat not in (0, 1):
        raise ValueError("left_seat must be 0 or 1")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(safety_seconds) or safety_seconds <= 0:
        raise ValueError("safety_seconds must be finite and positive")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    seed = int(seed)
    if left is right:
        right = copy.deepcopy(right)
    _reset_policy(left, seed * 2 + 1009)
    _reset_policy(right, seed * 2 + 1010)
    env = DuelEnv(env_config)
    obs, _ = env.reset(seed=seed)
    seats = [left, right] if left_seat == 0 else [right, left]
    frames = [_snapshot(env, 0)] if record else []
    stride = 1
    command_counts = np.zeros((2, 3), dtype=np.int64)
    attack_commands = np.zeros(2, dtype=np.int64)
    total_reward = np.zeros(2, dtype=np.float64)
    terminated = truncated = False
    info: dict = {}
    elimination = getattr(env.config,"max_seconds",0) is None
    limit = int(math.ceil(safety_seconds/(env.config.dt*env.config.action_repeat))) if elimination else _step_limit(env)
    censored = False
    for step in range(1, limit + 1):
        masks = env.attack_masks()
        runner_obs = env.runner_observations()
        actions = np.concatenate([
            _validated_actions(
                seats[seat], obs[seat:seat + 1], masks[seat:seat + 1],
                runner_obs[seat:seat + 1],
            )
            for seat in range(2)
        ])
        for seat in range(2):
            command_counts[seat, actions[seat, 0]] += 1
            attack_commands[seat] += int(actions[seat, 1] != 0)
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += np.asarray(rewards, dtype=np.float64)
        if record and step % stride == 0:
            frames.append(_snapshot(env, step, actions))
            if len(frames) >= max_frames:
                frames = frames[::2]
                stride *= 2
        if terminated or truncated:
            break
    else:
        if not elimination:
            raise RuntimeError("Environment did not finish within its configured episode limit")
        censored = True
        # External observation ends; the game itself has NOT ended.
        info = {**info,"reason":"evaluation_budget","winner":None}

    final = _snapshot(env, step, actions)
    if record and (not frames or frames[-1]["step"] != step):
        frames.append(final)
    if censored:
        final["reason"] = "evaluation_budget"
        if frames:
            frames[-1]["reason"] = "evaluation_budget"
    winner = info.get("winner", final.get("winner"))
    if winner is not None:
        winner = int(winner)
        if winner not in (0, 1):
            raise ValueError(f"Unexpected winner value {winner}")
    reason = str(info.get("reason", final.get("reason", "unknown")))
    distances = final.get("distance", info.get("distance", [0.0, 0.0]))
    if np.isscalar(distances):
        distances = [distances, distances]
    score = None if censored else (0.5 if winner is None else float(winner == left_seat))
    record_row = {
        "seed": seed,
        "left_seat": left_seat,
        "left_name": _name(left, "left"),
        "right_name": _name(right, "right"),
        "left_score": score,
        "winner_seat": winner,
        "winner_policy": "censored" if censored else ("draw" if winner is None else ("left" if winner == left_seat else "right")),
        "censored": censored,
        "left_collision_source": info.get("collision_sources",[None,None])[left_seat],
        "right_collision_source": info.get("collision_sources",[None,None])[1-left_seat],
        "reason": reason,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": step,
        "seconds": float(final.get("time", 0.0)),
        "left_distance": float(distances[left_seat]),
        "right_distance": float(distances[1 - left_seat]),
        "left_return": float(total_reward[left_seat]),
        "right_return": float(total_reward[1 - left_seat]),
        "left_attack_commands": int(attack_commands[left_seat]),
        "right_attack_commands": int(attack_commands[1 - left_seat]),
        "left_jump_commands": int(command_counts[left_seat, 1]),
        "right_jump_commands": int(command_counts[1 - left_seat, 1]),
        "left_duck_commands": int(command_counts[left_seat, 2]),
        "right_duck_commands": int(command_counts[1 - left_seat, 2]),
    }
    purchase_counts=info.get("purchase_counts", [[0]*5,[0]*5])
    for label,seat in (("left",left_seat),("right",1-left_seat)):
        counts=purchase_counts[seat]
        for kind,name in enumerate(("wait","cactus","wide_cactus","low_bird","high_bird")):
            if kind:record_row[f"{label}_{name}_purchases"]=int(counts[kind])
        costs=getattr(env.config,"attack_costs",(0,10,16,18,12))
        record_row[f"{label}_total_spent"]=float(sum(x*y for x,y in zip(counts,costs)))
        players=final.get("players",[{},{}])
        record_row[f"{label}_wallet_end"]=float(players[seat].get("coins",0.0))
    return record_row, frames


def paired_bootstrap_interval(
    pair_scores: Iterable[float], samples: int = 2000, seed: int = 1729,
) -> list[float] | None:
    """Percentile 95% interval for mean pair scores; unavailable with one pair."""
    values = np.asarray(list(pair_scores), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("pair_scores must be a nonempty finite one-dimensional sequence")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("pair_scores must lie in [0,1]")
    if samples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Keep temporary memory bounded for large evaluation suites.
    batch = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, samples, batch):
        end = min(samples, start + batch)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def evaluate(
    left: Policy,
    right: Policy,
    env_config: Any,
    seeds: Iterable[int],
    out_dir: str | Path | None = None,
    record_replay: bool = False,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 1729,
    safety_seconds: float = 300.0,
) -> dict:
    """Evaluate both seat assignments for every unique held-out environment seed."""
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one evaluation seed is required")
    if any(isinstance(s, bool) or int(s) != s or s < 0 for s in seed_list):
        raise ValueError("All evaluation seeds must be nonnegative integers")
    seed_list = [int(s) for s in seed_list]
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("Duplicate seeds would create pseudoreplication; provide unique seeds")
    if record_replay and out_dir is None:
        raise ValueError("record_replay requires an output directory")
    rows = []
    first_frames = []
    for seed in seed_list:
        for seat in (0, 1):
            row, frames = play_episode(
                left, right, env_config, seed, left_seat=seat,
                record=record_replay and not rows, safety_seconds=safety_seconds,
            )
            rows.append(row)
            if frames:
                first_frames = frames
    pair_scores = [None if rows[i]["left_score"] is None or rows[i+1]["left_score"] is None
                   else (rows[i]["left_score"]+rows[i+1]["left_score"])/2
                   for i in range(0,len(rows),2)]
    censored_count = sum(bool(row.get("censored")) for row in rows)
    resolved_scores = [row["left_score"] for row in rows if row["left_score"] is not None]
    lower = np.mean([0.0 if x["left_score"] is None else x["left_score"] for x in rows])
    upper = np.mean([1.0 if x["left_score"] is None else x["left_score"] for x in rows])
    wins = sum(row["left_score"] == 1 for row in rows)
    losses = sum(row["left_score"] == 0 for row in rows)
    draws = len(rows) - wins - losses - censored_count
    draw_reasons: dict[str, int] = {}
    for row in rows:
        if row["left_score"] == 0.5:
            draw_reasons[row["reason"]] = draw_reasons.get(row["reason"], 0) + 1
    # Resolve defaults now, so a partial caller configuration remains auditable
    # even if simulator defaults change in a later project version.
    resolved_config = getattr(DuelEnv(env_config), "config", env_config)
    if hasattr(resolved_config, "to_dict"):
        saved_config = resolved_config.to_dict()
    elif isinstance(resolved_config, dict):
        saved_config = dict(resolved_config)
    elif resolved_config is not None and hasattr(resolved_config, "__dict__"):
        saved_config = dict(vars(resolved_config))
    else:
        saved_config = None
    summary = {
        "schema": "dino_duel.evaluation.v2" if saved_config and saved_config.get("max_seconds") is None else "dino_duel.evaluation.v1",
        "censored": censored_count,
        "censor_rate": censored_count / len(rows),
        "safety_seconds": safety_seconds,
        "left_score_identification_bounds": [float(lower),float(upper)],
        "left_resolved_mean_score": float(np.mean(resolved_scores)) if resolved_scores else None,
        "left_name": _name(left, "left"),
        "right_name": _name(right, "right"),
        "n_seed_pairs": len(seed_list),
        "n_games": len(rows),
        "seeds": seed_list,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "draw_rate": draws / len(rows),
        "draw_reasons": draw_reasons,
        "left_mean_score": float(np.mean(pair_scores)) if not censored_count else None,
        "left_score_ci95": paired_bootstrap_interval(pair_scores, bootstrap_samples, bootstrap_seed) if not censored_count else None,
        "left_decisive_win_rate": wins / (wins + losses) if wins + losses else None,
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
        "mean_left_distance": float(np.mean([row["left_distance"] for row in rows])),
        "mean_right_distance": float(np.mean([row["right_distance"] for row in rows])),
        "left_score_by_seat": {
            str(seat): (float(np.mean([r["left_score"] for r in rows if r["left_seat"] == seat]))
                       if not any(r["censored"] for r in rows if r["left_seat"] == seat) else None)
            for seat in (0, 1)
        },
        "pair_scores": pair_scores,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "interval_unit": "complete environment-seed pair (both seats)",
        "uncertainty_scope": "environment seeds conditional on these frozen policies; excludes training-seed variation",
        "small_sample_warning": len(seed_list) < 20,
        "policy_selection": "Caller is responsible for keeping these seeds separate from training and model selection.",
        "env": saved_config,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        temporary = out_dir / "episodes.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(out_dir / "episodes.csv")
        _json_write(out_dir / "summary.json", summary)
        if record_replay:
            from .replay import write_replay
            write_replay(first_frames, rows[0], out_dir / "replay.html")
    return summary
