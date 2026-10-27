#!/usr/bin/env python3
"""Compare matched-seed ordinary PPO and JEPA-PPO checkpoints fairly.

Example:
  python scripts/compare_trained.py --runs runs/training_20261025 \
    --out runs/comparison_final --expected-update 300 \
    --training-seeds 11 23 47 --num-seeds 50 --workers 3

The primary attack mode is sampled masked softmax; motors remain greedy. Use
--attack-mode greedy in a separate output folder for a secondary diagnostic.
Method intervals use a crossed bootstrap: resample training rows and a single
shared set of environment-seed columns, retaining the common test conditions.
Automatic discovery includes only baseline/jepa runs trained with selfplay and
excludes diagnostic runs trained against a different opponent. Explicit paths
can instead be given as --baseline 11=PATH --jepa 11=PATH (repeat per seed).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import json
import multiprocessing
import os
from pathlib import Path
import sys

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_var] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
sys.path.insert(0, str(ROOT / "scripts")) if str(ROOT / "scripts") not in sys.path else None

import numpy as np
import torch

from evaluate_trained import CheckpointPolicy, SEED_STARTS, make_scripted_opponents
from dino_duel.env import DuelEnv
from dino_duel.evaluation import evaluate
from dino_duel.experiment import OutputLock
from dino_duel.learning import _load_checkpoint
from dino_duel.utils import atomic_json, file_sha256

torch.set_num_threads(1)
VARIANTS = ("baseline", "jepa")


def hierarchical_summary(values, training_seeds, samples=5000, seed=92711):
    """Cross-resample training seeds and shared complete environment-seed pairs.

    Input is [training seed, environment seed]. A fixed-opponent macro first
    averages the prespecified opponent set within each cell. For method gaps,
    subtract paired methods before resampling to retain pairing. Every selected
    training row uses the SAME sampled environment columns: these conditions
    were shared across training runs, rather than independently drawn per run.

    The function name and hierarchical_ci95 key remain for file/API compatibility;
    crossed_ci95, bootstrap_design and resampling_unit state the actual design.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(training_seeds) or not np.isfinite(values).all():
        raise ValueError("Expected finite [training seed, environment seed] scores")
    if not values.size or samples < 100:
        raise ValueError("Need nonempty scores and at least 100 resamples")
    per_seed = values.mean(axis=1)
    interval = None
    if len(training_seeds) > 1:
        rng = np.random.default_rng(seed)
        resampled = np.empty(samples, dtype=np.float64)
        for sample in range(samples):
            chosen_runs = rng.integers(0, values.shape[0], size=values.shape[0])
            chosen_environments = rng.integers(0, values.shape[1], size=values.shape[1])
            resampled[sample] = values[chosen_runs[:, None], chosen_environments[None, :]].mean()
        interval = [float(x) for x in np.quantile(resampled, [0.025, 0.975])]
    return {
        "mean": float(per_seed.mean()), "crossed_ci95": interval,
        "hierarchical_ci95": interval,
        "per_training_seed": {str(s): float(v) for s, v in zip(training_seeds, per_seed)},
        "n_training_seeds": len(training_seeds),
        "n_environment_seed_pairs_per_training_seed": int(values.shape[1]),
        "training_seed_sample_std": float(np.std(per_seed, ddof=1)) if len(training_seeds) > 1 else None,
        "bootstrap_samples": samples, "bootstrap_seed": seed,
        "bootstrap_design": "crossed_training_and_shared_environment_seeds",
        "resampling_unit": "matched training-seed rows and shared environment-seed columns; each column contains both seat assignments and is reused across all selected rows",
        "warning": "Exploratory percentile interval. Three training seeds are a small sample; no multiple-comparison correction or general algorithm-superiority claim.",
    }


def _seed_path(text):
    try:
        seed, path = text.split("=", 1)
        seed = int(seed)
        if seed < 0 or not path.strip():
            raise ValueError
        return seed, Path(path).expanduser().resolve()
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Use nonnegative TRAINING_SEED=CHECKPOINT_PATH")


def _discover(args):
    found = {v: {} for v in VARIANTS}
    ignored = []
    if args.runs:
        root = Path(args.runs).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Run directory does not exist: {root}")
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            numbered = folder / "checkpoints" / f"update_{args.expected_update:06d}.pt"
            path = numbered if numbered.is_file() else folder / "policy.pt"
            if not path.is_file():
                continue
            data = _load_checkpoint(path)
            variant, seed = data["variant"], int(data["seed"])
            style = data["config"]["training"].get("opponent_style", "selfplay")
            if variant not in VARIANTS or seed not in args.training_seeds or style != "selfplay":
                ignored.append({"path": str(path), "variant": variant, "seed": seed, "opponent_style": style})
                continue
            if seed in found[variant]:
                raise ValueError(f"Duplicate {variant} seed {seed}; provide explicit --baseline/--jepa paths")
            found[variant][seed] = path
    else:
        for variant in VARIANTS:
            for seed, path in getattr(args, variant):
                if seed in found[variant]:
                    raise ValueError(f"Duplicate {variant} seed {seed}")
                found[variant][seed] = path
    for variant in VARIANTS:
        if set(found[variant]) != set(args.training_seeds):
            raise ValueError(f"{variant}: found seeds {sorted(found[variant])}, expected {sorted(args.training_seeds)}")
    return found, ignored


def _initial_path(path):
    run = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    return run / "initial_policy.pt"


def _check_fairness(policies, args):
    first = policies[("baseline", args.training_seeds[0])]
    env = DuelEnv(first.config["env"]).config.to_dict()
    training = first.config["training"]
    model = first.config["model"]
    expected_steps = args.expected_update * int(training["num_envs"]) * int(training["rollout_steps"])
    for (variant, seed), policy in policies.items():
        meta = policy.metadata
        if meta["variant"] != variant or meta["training_seed"] != seed:
            raise ValueError(f"Metadata does not match label {variant} seed {seed}")
        if meta["update"] != args.expected_update or meta["global_step"] != expected_steps:
            raise ValueError(f"{variant} seed {seed} is at update {meta['update']}, {meta['global_step']} steps; expected {args.expected_update}, {expected_steps}. Wait until training completes.")
        if policy.config["training"] != training or policy.config["model"] != model:
            raise ValueError("Training hyperparameters or acting architecture differ; comparison would not isolate JEPA")
        if DuelEnv(policy.config["env"]).config.to_dict() != env:
            raise ValueError("Game rules differ between checkpoints")
        if policy.runner_fingerprint != first.runner_fingerprint:
            raise ValueError("Checkpoints used different common initial runners")
    if training.get("opponent_style", "selfplay") != "selfplay":
        raise ValueError("This primary comparison requires selfplay for both methods")
    initial_audit = {}
    for seed in args.training_seeds:
        paths = [_initial_path(policies[(variant, seed)].path) for variant in VARIANTS]
        if not all(path.is_file() for path in paths):
            initial_audit[str(seed)] = {"verified": False, "reason": "initial_policy.pt unavailable"}
            continue
        left, right = [_load_checkpoint(path) for path in paths]
        keys = [key for key in left["model_state"]
                if key.startswith(("encoder.", "attack_head.", "value_head.", "runner."))]
        equal = all(torch.equal(left["model_state"][key], right["model_state"][key]) for key in keys)
        if not equal or left["update"] != 0 or right["update"] != 0:
            raise ValueError(f"Initial acting/value weights differ for matched training seed {seed}")
        initial_audit[str(seed)] = {"verified": True, "tensor_count": len(keys),
                                   "sha256": [file_sha256(path) for path in paths]}
    return env, {
        "equal_training_configuration": True, "equal_acting_architecture": True,
        "equal_environment_steps_per_run": expected_steps,
        "common_initial_runner_fingerprint": first.runner_fingerprint,
        "initial_weights_by_training_seed": initial_audit,
        "training": training, "model": model,
        "budget_note": "Matched environment-step budgets. JEPA has additional auxiliary parameters and computation; wall-clock budgets are not matched.",
    }


def _run_match(task):
    path = Path(task["out_dir"])
    identity = {k: v for k, v in task.items() if k != "out_dir"}
    old = path / "task.json"
    if old.is_file() and json.loads(old.read_text()) != identity:
        raise ValueError(f"Existing evaluation identity differs: {path}; use a new output folder")
    if old.is_file() and (path / "summary.json").is_file() and (path / "episodes.csv").is_file():
        return task["key"], json.loads((path / "summary.json").read_text()), True
    left = CheckpointPolicy(task["left"]["path"], task["left"]["name"], task["stochastic_attacks"])
    if left.sha256 != task["left"]["sha256"]:
        raise ValueError("Left checkpoint changed during evaluation")
    if "style" in task["right"]:
        right = make_scripted_opponents([task["right"]["style"]], left, task["env"])[0]
    else:
        right = CheckpointPolicy(task["right"]["path"], task["right"]["name"], task["stochastic_attacks"])
        if right.sha256 != task["right"]["sha256"]:
            raise ValueError("Right checkpoint changed during evaluation")
    atomic_json(old, identity)
    result = evaluate(left, right, task["env"], task["seeds"], out_dir=path,
                      record_replay=task["record"], bootstrap_samples=task["bootstrap_samples"])
    return task["key"], result, False


def _count(summary):
    return {k: summary[k] for k in ("n_games", "wins", "draws", "losses")}


def _aggregate(results, args):
    seeds = args.training_seeds
    h2h = np.asarray([results[f"head_to_head/seed_{s}"]["pair_scores"] for s in seeds])
    head = {
        "left": "baseline", "right": "jepa",
        "baseline_score": hierarchical_summary(h2h, seeds, args.bootstrap_samples),
        "jepa_score": hierarchical_summary(1 - h2h, seeds, args.bootstrap_samples),
        "per_training_seed": {str(s): {**_count(results[f"head_to_head/seed_{s}"]),
                                       "baseline_score": float(h2h[i].mean())}
                              for i, s in enumerate(seeds)},
    }
    head["descriptive_pooled_counts_for_baseline"] = {
        key: sum(row[key] for row in head["per_training_seed"].values())
        for key in ("n_games", "wins", "draws", "losses")
    }
    arrays = {}
    fixed = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for style in args.opponents:
            array = np.asarray([results[f"fixed/{variant}/seed_{s}/{style}"]["pair_scores"] for s in seeds])
            arrays[(variant, style)] = array
            fixed[variant][style] = hierarchical_summary(array, seeds, args.bootstrap_samples)
        macro = np.mean([arrays[(variant, style)] for style in args.opponents], axis=0)
        arrays[(variant, "macro")] = macro
        fixed[variant]["macro"] = hierarchical_summary(macro, seeds, args.bootstrap_samples)
    gaps = {}
    for style in args.opponents + ["macro"]:
        gaps[style] = hierarchical_summary(arrays[("jepa", style)] - arrays[("baseline", style)],
                                            seeds, args.bootstrap_samples)
        gaps[style]["direction"] = "JEPA score minus baseline score; positive favors JEPA"
    return {"head_to_head": head, "fixed_opponents": fixed,
            "paired_fixed_score_differences": gaps,
            "macro_definition": "Equal weight to each prespecified fixed opponent; pair methods and environment seeds before taking differences."}


def _write_report(out, report, rows):
    with (out / "matchups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def interval(item):
        ci = item["crossed_ci95"]
        return "unavailable" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    def table(headers, items):
        return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in items) + '</tbody></table></div>'
    comparison = report["comparison"]
    h2h = comparison["head_to_head"]
    direct = table(["Training seed", "PPO wins", "Draws", "JEPA wins", "PPO score"],
                   [[seed, row["wins"], row["draws"], row["losses"], f"{row['baseline_score']:.3f}"]
                    for seed, row in h2h["per_training_seed"].items()])
    fixed_rows = []
    for style in report["fixed_opponents"] + ["macro"]:
        baseline = comparison["fixed_opponents"]["baseline"][style]
        jepa = comparison["fixed_opponents"]["jepa"][style]
        gap = comparison["paired_fixed_score_differences"][style]
        fixed_rows.append([style, f"{baseline['mean']:.3f}", interval(baseline), f"{jepa['mean']:.3f}",
                           interval(jepa), f"{gap['mean']:+.3f}", interval(gap)])
    fixed = table(["Opponent", "PPO score", "PPO 交叉 95% CI", "JEPA score", "JEPA 交叉 95% CI", "JEPA − PPO", "分差交叉 95% CI"], fixed_rows)
    raw = table(["Matchup", "Games", "Wins", "Draws", "Losses", "Left score"],
                [[r["matchup"], r["games"], r["wins"], r["draws"], r["losses"], f"{r['left_score']:.3f}"] for r in rows])
    base_score = h2h["baseline_score"]
    jepa_score = h2h["jepa_score"]
    doc = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEPA 与普通 PPO：实测对比</title><style>:root{{color-scheme:dark}}body{{font:15px/1.6 system-ui,sans-serif;background:#0c121b;color:#e9f0f8;margin:0;padding:28px}}main{{max-width:1240px;margin:auto}}h1{{font-size:30px}}a{{color:#5de1c4}}.muted{{color:#a8b8cb}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:#131d2a;margin:18px 0 26px}}th,td{{padding:10px 12px;border-bottom:1px solid #29394d;text-align:left;white-space:nowrap}}th{{color:#5de1c4}}.box{{padding:16px 20px;border:1px solid #344960;border-radius:10px;margin:20px 0}}h2{{margin-top:34px}}</style><main>
<h1>JEPA 与普通 PPO：实测对比</h1><p class="muted">{html.escape(report['split'])} · {len(report['training_seeds'])} 个训练种子 · 每个对局组合 {len(report['evaluation_seeds'])} 个环境种子 × 双座位 · update {report['expected_update']} · attack mode: {report['attack_mode']}</p>
<div class="box">每个 seed 的模型用相同环境步数、相同基础跑跳模型和 PPO 参数训练。JEPA 额外训练动作条件的未来 latent 预测器；推理阶段不使用它做规划。计算量和实际运行时间并未匹配。
分数为胜 1 / 平 0.5 / 负 0。置信区间使用交叉 bootstrap：重采样训练种子行，同时重采样一套共同的环境种子列；所有选中的训练组共用这套环境列，每列保留换边对局。这样保留共享测试条件的相关性，不把所有游戏当成独立训练实验。
只有 {len(report['training_seeds'])} 个训练种子，结果具有探索性，不能据此宣称 JEPA 在一般任务上优于 PPO。</div>
<h2>1. 相同训练 seed 的直接对战</h2><p>普通 PPO 平均分 <strong>{base_score['mean']:.3f}</strong>，交叉 bootstrap 95% CI {interval(base_score)}；JEPA 平均分 <strong>{jepa_score['mean']:.3f}</strong>，交叉 bootstrap 95% CI {interval(jepa_score)}。</p>{direct}
<h2>2. 相同固定对手</h2><p>Macro 对预先指定的各对手等权平均。分差先在相同训练 seed、环境 seed 和对手内计算，再做训练行与共享环境列的交叉重采样。Burst 是手写强基线：存够两个小仙人掌的费用，再于两个合法时机连发；它不是训练成果。</p>{fixed}
<h2>3. 原始对局汇总</h2>{raw}
<p><a href="comparison.json">完整统计与可复现记录</a> · <a href="matchups.csv">CSV 汇总</a> · <a href="protocol.json">冻结评估配置</a></p>
<p class="muted">每个 matchup 目录另存逐局 episodes.csv、summary.json 和所用模型/代码 SHA256。若开启 --record，首局回放也在同一目录。Final 用于报告预先固定的 update，不用于挑 checkpoint 或调整方法。</p></main></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", help="Parent directory containing each independent training-run folder")
    parser.add_argument("--baseline", action="append", type=_seed_path, default=[])
    parser.add_argument("--jepa", action="append", type=_seed_path, default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-update", type=int, default=300)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--split", choices=SEED_STARTS, default="final")
    parser.add_argument("--attack-mode", choices=["stochastic", "greedy"], default="stochastic")
    parser.add_argument("--opponents", nargs="+", choices=["random", "aggressive", "save", "burst", "none"],
                        default=["random", "aggressive", "save", "burst"])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    if args.runs and (args.baseline or args.jepa):
        parser.error("Use --runs or explicit --baseline/--jepa paths, not both")
    if not args.runs and (not args.baseline or not args.jepa):
        parser.error("Provide --runs or both sets of checkpoint paths")
    if min(args.expected_update, args.num_seeds, args.workers) < 1 or args.bootstrap_samples < 100:
        parser.error("Updates, seed count and workers must be positive; bootstrap samples >=100")
    if len(set(args.training_seeds)) != len(args.training_seeds) or min(args.training_seeds) < 0:
        parser.error("Training seeds must be unique nonnegative integers")
    if len(set(args.opponents)) != len(args.opponents):
        parser.error("Fixed opponents must be unique")
    paths, ignored = _discover(args)
    policies = {(variant, seed): CheckpointPolicy(path, f"{variant} seed={seed}", args.attack_mode == "stochastic")
                for variant, items in paths.items() for seed, path in items.items()}
    env, audit = _check_fairness(policies, args)
    eval_seeds = list(range(SEED_STARTS[args.split], SEED_STARTS[args.split] + args.num_seeds))
    out = Path(args.out).expanduser().resolve()
    code = {name: file_sha256(ROOT / name) for name in (
        "dino_duel/env.py", "dino_duel/teacher.py", "dino_duel/evaluation.py",
        "dino_duel/learning.py", "dino_duel/models.py", "scripts/evaluate_trained.py", "scripts/compare_trained.py",
    )}
    protocol = {"schema": "dino_duel.matched_method_comparison.v1", "split": args.split,
                "expected_update": args.expected_update, "training_seeds": args.training_seeds,
                "evaluation_seeds": eval_seeds, "attack_mode": args.attack_mode,
                "motor_inference": "greedy", "fixed_opponents": args.opponents,
                "bootstrap_samples": args.bootstrap_samples, "env": env,
                "bootstrap_design": "crossed_training_and_shared_environment_seeds",
                "source_sha256": code, "fairness_audit": audit,
                "checkpoints": {f"{v}/seed_{s}": p.metadata for (v, s), p in policies.items()},
                "excluded_diagnostic_runs": ignored,
                "fixed_checkpoint_rule": "All methods and training seeds evaluated at the prespecified update; no final-test checkpoint selection",
                "handwritten_burst": "Public wallet and mask only; save for two small cacti then buy at successive legal opportunities. Not learned."}
    def spec(policy):
        return {"path": str(policy.path), "name": policy.name, "sha256": policy.sha256}
    tasks = []
    for seed in args.training_seeds:
        tasks.append({"key": f"head_to_head/seed_{seed}", "left": spec(policies[("baseline", seed)]),
                      "right": spec(policies[("jepa", seed)])})
        for variant in VARIANTS:
            for style in args.opponents:
                tasks.append({"key": f"fixed/{variant}/seed_{seed}/{style}",
                              "left": spec(policies[(variant, seed)]), "right": {"style": style}})
    for task in tasks:
        task.update(out_dir=str(out / task["key"]), env=env, seeds=eval_seeds,
                    stochastic_attacks=args.attack_mode == "stochastic", source_sha256=code,
                    bootstrap_samples=args.bootstrap_samples, record=args.record)
    results = {}
    with OutputLock(out):
        previous = out / "protocol.json"
        if previous.is_file() and json.loads(previous.read_text()) != protocol:
            raise ValueError("Existing protocol differs; use a new output folder")
        atomic_json(previous, protocol)
        def receive(completed):
            key, result, cached = completed
            results[key] = result
            print(json.dumps({"matchup": key, **_count(result), "left_score": result["left_mean_score"],
                              "cached": cached}, ensure_ascii=False), flush=True)
        if args.workers == 1:
            for task in tasks:
                receive(_run_match(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(_run_match, task) for task in tasks]
                for future in as_completed(futures):
                    receive(future.result())
        comparison = _aggregate(results, args)
        report = {**protocol, "comparison": comparison,
                  "n_games": sum(result["n_games"] for result in results.values()),
                  "n_matchups": len(results), "matchups": results}
        atomic_json(out / "comparison.json", report)
        rows = [{"matchup": key, "left": result["left_name"], "right": result["right_name"],
                 "games": result["n_games"], "wins": result["wins"], "draws": result["draws"],
                 "losses": result["losses"], "left_score": result["left_mean_score"],
                 "conditional_environment_ci95": json.dumps(result["left_score_ci95"])}
                for key, result in sorted(results.items())]
        _write_report(out, report, rows)
        print(f"Report: {out / 'report.html'}", flush=True)
        return report


if __name__ == "__main__":
    main()
