#!/usr/bin/env python3
"""Evaluate frozen Dino checkpoints on disjoint validation or final seed ranges.

Examples (run from the project directory)::

    python scripts/evaluate_trained.py --candidate trained=runs/main/policy.pt \
        --reference initial=runs/initial/policy.pt --split validation \
        --num-seeds 12 --out runs/validation --stochastic-attacks
    python scripts/evaluate_trained.py --candidate selected=runs/main/policy.pt \
        --reference initial=runs/initial/policy.pt --split final \
        --num-seeds 50 --out runs/final --stochastic-attacks

Select checkpoints using validation only. The final suite is for reporting the
already selected checkpoint, not for selecting or retuning it. Both seats are
played for every seed. The confidence intervals are conditional on the frozen
checkpoints and do not quantify variability across training runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path

# Small matrix inference is faster and more predictable with one BLAS thread.
# Set these before importing NumPy or PyTorch, including when invoked directly.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_var] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dino_duel.env import DuelEnv, N_ATTACKS, OBS_DIM, RUNNER_DIM
from dino_duel.evaluation import evaluate
from dino_duel.learning import _load_checkpoint
from dino_duel.models import NumpyRunner
from dino_duel.policies import TeacherPolicy
from dino_duel.teacher import teacher_actions
from dino_duel.utils import atomic_json, file_sha256

torch.set_num_threads(1)

SEED_STARTS = {"validation": 2**33, "final": 2**34}


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).copy()


def _runner_fingerprint(spec):
    """Hash the actual common initial runner, not an arbitrary checkpoint label."""
    if spec["kind"] == "teacher":
        return "teacher:" + file_sha256(ROOT / "dino_duel" / "teacher.py")
    digest = hashlib.sha256()
    for key, value in sorted(spec["weights"].items()):
        a = _array(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(a.shape).encode("ascii"))
        digest.update(a.tobytes())
    return "numpy:" + digest.hexdigest()


class CheckpointPolicy:
    """CPU inference matching the checkpoint MLP with an independent RNG.

    NumPy avoids constructing Torch distributions at every physics decision.
    With stochastic_attacks=True only the attack head is sampled. Motor actions
    stay greedy, including in checkpoints that were trained jointly. Each policy
    gets a separate reproducible RNG through evaluation.play_episode.reset().
    Sampling is from the learned masked softmax, not from a scripted heuristic.
    """

    def __init__(self, path, name=None, stochastic_attacks=False):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path /= "policy.pt" if (path / "policy.pt").is_file() else "latest.pt"
        self.path = path
        self.sha256 = file_sha256(path)
        data = _load_checkpoint(path)
        self.config = data["config"]
        self.runner_spec = data["runner"]
        self.runner_fingerprint = _runner_fingerprint(self.runner_spec)
        self.variant = data["variant"]
        self.name = name or f"{self.variant} seed={data['seed']} update={data['update']}"
        self.stochastic_attacks = bool(stochastic_attacks)
        self.train_motor = bool(data["model_config"]["train_motor"])
        self.metadata = {
            "checkpoint": str(path), "checkpoint_sha256": self.sha256,
            "variant": data["variant"], "training_seed": int(data["seed"]),
            "update": int(data["update"]), "global_step": int(data["global_step"]),
            "agent_steps": int(data["agent_steps"]),
            "training_elapsed_seconds": float(data["elapsed_seconds"]),
            "attack_inference": "sampled_masked_softmax" if stochastic_attacks else "argmax",
            "motor_inference": "argmax", "train_motor": self.train_motor,
            "common_initial_runner_fingerprint": self.runner_fingerprint,
        }
        state = data["model_state"]
        self.layers = [(_array(state[f"encoder.{i}.weight"]),
                        _array(state[f"encoder.{i}.bias"])) for i in (0, 2)]
        self.head = (_array(state["attack_head.weight"]),
                     _array(state["attack_head.bias"]))
        self.initial_runner = None
        if self.runner_spec["kind"] == "numpy":
            self.initial_runner = NumpyRunner(
                {k: _array(v) for k, v in self.runner_spec["weights"].items()},
                self.runner_spec.get("metadata", {}),
            )
        if self.train_motor:
            self.runner = NumpyRunner({
                f"{kind}{j}": _array(state[f"runner.net.{i}.{parameter}"])
                for j, i in enumerate((0, 2, 4))
                for kind, parameter in (("w", "weight"), ("b", "bias"))
            })
        else:
            self.runner = self.initial_runner or teacher_actions
        self.reset()

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def logits(self, obs):
        z = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        for weight, bias in self.layers:
            z = np.tanh(z @ weight.T + bias)
        return z @ self.head[0].T + self.head[1]

    def act(self, obs, masks, runner_obs, deterministic=True):
        masks = np.asarray(masks, dtype=bool).reshape(-1, N_ATTACKS)
        logits = self.logits(obs)
        masked = np.where(masks, logits, -np.inf)
        if not masks.any(axis=-1).all():
            raise ValueError("Every mask must contain at least one legal attack")
        if self.stochastic_attacks:
            # Use float64 to make categorical normalization robust. A forced
            # no-op does not consume randomness; opportunity decisions do.
            probabilities = np.exp((masked - masked.max(axis=-1, keepdims=True)).astype(np.float64))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attacks = np.asarray([
                int(np.flatnonzero(mask)[0]) if mask.sum() == 1
                else int(self.rng.choice(N_ATTACKS, p=prob))
                for mask, prob in zip(masks, probabilities)
            ], dtype=np.int64)
        else:
            attacks = masked.argmax(axis=-1)
        motors = self.runner(np.asarray(runner_obs, dtype=np.float32).reshape(-1, RUNNER_DIM))
        return np.column_stack((motors, attacks)).astype(np.int64)


class BurstPolicy(TeacherPolicy):
    """Handwritten public-observation baseline: save for two small cacti.

    It reads only the public wallet ratio, legal attack mask and runner features.
    Prices and wallet capacity are public game rules supplied at construction.
    No environment object, future actions or hidden state is available here.
    """

    def __init__(self, env_config):
        self.wallet_cap = float(env_config["wallet_cap"])
        self.required_coins = 2 * float(env_config["attack_costs"][1])
        super().__init__("none")
        self.name = "teacher:burst"

    def reset(self, seed=0):
        super().reset(seed)
        self.remaining = np.zeros(0, dtype=np.int64)

    def act(self, obs, masks, runner_obs, deterministic=True):
        observations = np.atleast_2d(obs)
        legal = np.atleast_2d(masks)
        actions = super().act(observations, legal, runner_obs, deterministic)
        if len(self.remaining) != len(observations):
            self.remaining = np.zeros(len(observations), dtype=np.int64)
        for i, row in enumerate(observations):
            wallet = float(row[48]) * self.wallet_cap
            if self.remaining[i] == 0 and wallet + 1e-6 >= self.required_coins:
                self.remaining[i] = 2
            if self.remaining[i] and legal[i, 1]:
                actions[i, 1] = 1
                self.remaining[i] -= 1
        return actions


def make_scripted_opponents(styles, initial_policy, env):
    result = []
    for style in styles:
        policy = BurstPolicy(env) if style == "burst" else TeacherPolicy(style)
        if initial_policy.initial_runner is not None:
            policy.runner = initial_policy.initial_runner
            policy.name = f"bc:{style}"
        result.append(policy)
    return result


def _named_path(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use NAME=CHECKPOINT_PATH")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use a nonempty NAME and CHECKPOINT_PATH")
    return name.strip(), path.strip()


def _slug(name):
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:60] or "policy"
    return clean + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]


def _write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _html_report(out, report):
    cells = []
    for row in report["matchups"]:
        interval = row["score_ci95"]
        ci = "unavailable" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        cells.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["candidate"], row["opponent"], row["games"], row["wins"],
            row["draws"], row["losses"], f"{row['mean_score']:.3f}", ci,
        )) + f'<td><a href="{html.escape(row["result_dir"], quote=True)}/summary.json">JSON</a></td></tr>')
    mode = report["attack_inference"]
    title = f"Dino Duel — {report['split']} evaluation"
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{color-scheme:dark}}body{{font:15px/1.5 system-ui,sans-serif;background:#0c121b;color:#e9f0f8;margin:0;padding:28px}}
main{{max-width:1200px;margin:auto}}h1{{font-size:30px}}.muted{{color:#a8b8cb}}a{{color:#5de1c4}}.scroll{{overflow:auto}}
table{{width:100%;border-collapse:collapse;background:#131d2a}}th,td{{padding:10px 12px;border-bottom:1px solid #29394d;text-align:left;white-space:nowrap}}
th{{color:#5de1c4}}.box{{padding:15px 20px;border:1px solid #29394d;border-radius:10px;margin:22px 0}}code{{color:#ffb37a}}
</style><main><h1>{html.escape(title)}</h1>
<p class="muted">Actual simulated matches · {report['seed_pairs']} environment seeds × both seats per matchup · attacks: {mode} · motors: greedy</p>
<div class="box">Score = win 1, draw 0.5, loss 0. The 95% intervals bootstrap complete environment-seed pairs.
They describe these frozen policies; they do not include variation across training seeds.
{('Validation results may guide selection; do not report them as an untouched final test.' if report['split'] == 'validation' else 'Final seeds are reserved for reporting checkpoints already chosen on validation; do not retune or reselect using this table.')}
</div><div class="scroll"><table><thead><tr><th>Candidate</th><th>Opponent</th><th>Games</th><th>Wins</th><th>Draws</th><th>Losses</th><th>Score</th><th>95% CI</th><th>Raw results</th></tr></thead>
<tbody>{''.join(cells)}</tbody></table></div>
<p><a href="matrix.json">Full JSON and provenance</a> · <a href="matrix.csv">Table CSV</a></p>
<p class="muted">Fixed scripted opponents share the checkpoint's original runner. The attack policy is the tested component when the runner was frozen.
The burst opponent is a handwritten baseline: save enough for two small cacti and buy them at successive legal opportunities. It is not a learned result.
Random and aggressive scripts retain their own seeded randomness. Learned stochastic attack sampling also uses a separate reproducible policy RNG.
No training happens in this report, and one frozen run cannot establish superiority of an algorithm.</p></main></html>"""
    (out / "report.html").write_text(document, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=_named_path, action="append", required=True)
    parser.add_argument("--reference", type=_named_path, action="append", default=[])
    parser.add_argument("--opponents", nargs="*", choices=["random", "aggressive", "save", "burst", "none"],
                        default=["random", "aggressive", "save", "burst"])
    parser.add_argument("--split", choices=SEED_STARTS, required=True)
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Defaults to 12 validation or 50 final seed pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stochastic-attacks", action="store_true",
                        help="Sample learned attack softmax; motor actions stay greedy")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--record", action="store_true", help="Record the first game of each matchup")
    args = parser.parse_args(argv)
    count = args.num_seeds if args.num_seeds is not None else (12 if args.split == "validation" else 50)
    if count < 1 or args.bootstrap_samples < 100:
        parser.error("Need at least 1 seed and at least 100 bootstrap samples")
    labels = [name for name, _ in args.candidate + args.reference]
    if len(labels) != len(set(labels)):
        parser.error("Candidate and reference names must all be unique")
    if not args.opponents and not args.reference:
        parser.error("Choose at least one scripted opponent or reference checkpoint")
    if len(args.opponents) != len(set(args.opponents)):
        parser.error("Scripted opponents must be unique")
    seeds = list(range(SEED_STARTS[args.split], SEED_STARTS[args.split] + count))
    candidates = [CheckpointPolicy(path, name, args.stochastic_attacks) for name, path in args.candidate]
    references = [CheckpointPolicy(path, name, args.stochastic_attacks) for name, path in args.reference]
    first = candidates[0]
    env = DuelEnv(first.config["env"]).config.to_dict()
    for policy in candidates + references:
        if DuelEnv(policy.config["env"]).config.to_dict() != env:
            parser.error(f"Checkpoint {policy.name!r} has different game rules; use a separate suite")
        if policy.runner_fingerprint != first.runner_fingerprint:
            parser.error(f"Checkpoint {policy.name!r} used a different common initial runner")
    scripts = make_scripted_opponents(args.opponents, first, env)
    if set(labels) & {p.name for p in scripts}:
        parser.error("Candidate/reference names must differ from scripted opponent names")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    code = {name: file_sha256(ROOT / name) for name in (
        "dino_duel/env.py", "dino_duel/evaluation.py", "dino_duel/teacher.py",
        "scripts/evaluate_trained.py",
    )}
    mode = "sampled_masked_softmax" if args.stochastic_attacks else "argmax"
    report = {
        "schema": "dino_duel.trained_evaluation.v1", "split": args.split,
        "seed_pairs": count, "seeds": seeds, "attack_inference": mode,
        "motor_inference": "argmax", "env": env, "source_sha256": code,
        "candidates": {p.name: p.metadata for p in candidates},
        "references": {p.name: p.metadata for p in references},
        "fixed_opponents": [p.name for p in scripts],
        "handwritten_burst_baseline": "Public-observation script: save for two small cacti, then purchase at successive legal opportunities. Not trained.",
        "common_initial_runner_fingerprint": first.runner_fingerprint,
        "uncertainty_scope": "environment seeds conditional on frozen policies; not across training seeds",
        "selection_rule": "Use validation only for checkpoint selection; report selected checkpoints on final once",
        "matchups": [],
    }
    for candidate in candidates:
        for opponent in scripts + references:
            relative = Path("matchups") / f"{_slug(candidate.name)}__vs__{_slug(opponent.name)}"
            target = out / relative
            task = {
                "candidate_sha256": candidate.sha256,
                "opponent": opponent.metadata if isinstance(opponent, CheckpointPolicy) else opponent.name,
                "common_initial_runner": first.runner_fingerprint,
                "attack_inference": mode, "seeds": seeds, "env": env,
                "bootstrap_samples": args.bootstrap_samples, "source_sha256": code,
                "record": args.record,
            }
            old = target / "task.json"
            if old.is_file() and json.loads(old.read_text()) != task:
                raise ValueError(f"Existing evaluation identity differs at {target}; use a new --out directory")
            if old.is_file() and (target / "summary.json").is_file() and (target / "episodes.csv").is_file():
                result = json.loads((target / "summary.json").read_text())
                cached = True
            else:
                atomic_json(old, task)
                result = evaluate(candidate, opponent, env, seeds, out_dir=target,
                                  record_replay=args.record, bootstrap_samples=args.bootstrap_samples)
                cached = False
            row = {
                "candidate": candidate.name, "opponent": opponent.name,
                "games": result["n_games"], "wins": result["wins"], "draws": result["draws"],
                "losses": result["losses"], "mean_score": result["left_mean_score"],
                "score_ci95": result["left_score_ci95"],
                "mean_seconds": result["mean_seconds"], "result_dir": relative.as_posix(),
            }
            report["matchups"].append(row)
            atomic_json(out / "matrix.json", report)
            print(json.dumps({**row, "cached": cached}, ensure_ascii=False), flush=True)
    csv_rows = [{**r, "score_ci95": json.dumps(r["score_ci95"])} for r in report["matchups"]]
    _write_csv(out / "matrix.csv", csv_rows)
    _html_report(out, report)
    print(f"Report: {out / 'report.html'}", flush=True)
    return report


if __name__ == "__main__":
    main()
