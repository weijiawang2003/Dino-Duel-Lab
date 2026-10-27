from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

from .utils import PROJECT_ROOT, atomic_json, file_sha256, load_config, runtime_info


def _shared(parser, *, config="cpu", out=None):
    parser.add_argument("--config", default=config, help="Preset smoke/cpu/standard/joint/tight or a JSON path")
    if out is not None:
        parser.add_argument("--out", default=out)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)


def _config(args):
    config = load_config(args.config)
    # Explicit JSON configurations can carry a runner relative to themselves.
    # In particular an experiment output remains portable after moving folders.
    source = Path(args.config).expanduser()
    if source.is_file():
        choice = config.get("experiment", {}).get("runner", "bundled")
        if choice not in {"bundled", "teacher"}:
            runner = Path(choice).expanduser()
            if not runner.is_absolute():
                config["experiment"]["runner"] = str((source.resolve().parent / runner).resolve())
    if getattr(args, "device", None):
        config["training"]["device"] = args.device
    if getattr(args, "updates", None) is not None:
        config["training"]["updates"] = args.updates
    return config


def doctor():
    info = runtime_info()
    info["project_root"] = str(PROJECT_ROOT)
    info["bundled_runner_present"] = (PROJECT_ROOT / "assets" / "runner_bc.npz").is_file()
    info["python_supported"] = (3, 11) <= sys.version_info[:2] < (3, 14)
    info["training_ready"] = bool(info["torch"] and info["bundled_runner_present"])
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not info["torch"]:
        print("Replay/demo need only NumPy. For training: python setup_env.py --backend cpu")
    if not info["python_supported"]:
        print("This release was tested on Python 3.13 and targets Python 3.11–3.13.")
    return info


def benchmark(args):
    import numpy as np
    from .env import DuelEnv
    from .policies import TeacherPolicy
    config = _config(args)
    env = DuelEnv(config["env"])
    policy = TeacherPolicy("random")
    obs, _ = env.reset(seed=args.seed)
    started, episodes, steps = time.perf_counter(), 0, 0
    for steps in range(1, args.steps + 1):
        actions = policy.act(obs, env.attack_masks(), env.runner_observations())
        obs, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset(seed=args.seed + episodes)
    elapsed = time.perf_counter() - started
    result = {"joint_environment_steps": steps, "seconds": elapsed,
              "steps_per_second": steps / elapsed, "episodes_completed": episodes,
              "note": "Simulator+scripted policy only; excludes neural training, replay and evaluation.",
              "env": env.config.to_dict(), "runtime": runtime_info()}
    if args.out:
        atomic_json(args.out, result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dino-duel",
        description="Complete Dino duel training, paired evaluation and offline replay.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Python, dependencies, hardware and bundled assets")

    demo = sub.add_parser("demo", help="Generate a complete offline replay; no torch required")
    _shared(demo, out="runs/demo/replay.html")
    demo.add_argument("--left", default="teacher:random")
    demo.add_argument("--right", default="teacher:aggressive")
    demo.add_argument("--runner", default="bundled", help="bundled, teacher, or an NPZ path")
    demo.add_argument("--seed", type=int, default=2026)
    demo.add_argument("--seconds", type=float, default=30)
    demo.add_argument("--stochastic-attacks", action="store_true",
                      help="Sample learned attack heads reproducibly; keep motor actions greedy")

    boot = sub.add_parser("bootstrap", help="Rebuild shared basic runner with BC and DAgger")
    _shared(boot, config="bootstrap", out="runs/bootstrap/runner_bc.npz")
    boot.add_argument("--samples", type=int)
    boot.add_argument("--epochs", type=int)
    boot.add_argument("--seed", type=int)
    boot.add_argument("--overwrite", action="store_true")

    training = sub.add_parser("train", help="Train one treatment and seed")
    _shared(training, out="runs/single")
    training.add_argument("--variant", choices=["baseline", "jepa", "jepa_shuffled", "baseline_large"], default="baseline")
    training.add_argument("--seed", type=int, default=11)
    training.add_argument("--updates", type=int)
    training.add_argument("--runner", default=None, help="bundled, teacher, or NPZ path; preset default if omitted")
    training.add_argument("--resume", action="store_true")

    experiment = sub.add_parser("experiment", help="Run all training seeds, matchups and reports")
    _shared(experiment, out="runs/experiment")
    experiment.add_argument("--updates", type=int)
    experiment.add_argument("--runner", default=None)
    experiment.add_argument("--seeds", type=int, nargs="+")
    experiment.add_argument("--variants", nargs="+", choices=["baseline", "jepa", "jepa_shuffled", "baseline_large"])
    experiment.add_argument("--resume", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Paired held-out-seed evaluation with both seat assignments")
    _shared(evaluate, out="runs/evaluation")
    evaluate.add_argument("--left", required=True)
    evaluate.add_argument("--right", default="teacher:aggressive")
    evaluate.add_argument("--runner", default="bundled", help="Runner for scripted opponents")
    evaluate.add_argument("--num-seeds", type=int)
    evaluate.add_argument("--seed-start", type=int)
    evaluate.add_argument("--record", action="store_true")
    evaluate.add_argument("--safety-seconds", type=float, default=300., help="External observation budget; not a draw in elimination mode")
    evaluate.add_argument("--sampling-mode", choices=["joint","attack","greedy"], help="Explicit frozen policy execution mode")
    evaluate.add_argument("--stochastic-attacks", action="store_true",
                          help="Sample learned attack heads reproducibly; keep motor actions greedy")
    evaluate.add_argument("--use-checkpoint-env", action="store_true",
                          help="Use the left checkpoint's original simulator configuration")

    replay = sub.add_parser("replay", help="Watch a matchup by generating offline HTML")
    _shared(replay, out="runs/replay.html")
    replay.add_argument("--left", required=True)
    replay.add_argument("--right", default="teacher:aggressive")
    replay.add_argument("--runner", default="bundled")
    replay.add_argument("--seed", type=int, default=4294967296)
    replay.add_argument("--seconds", type=float, help="External observation budget for elimination mode; duration for legacy timed mode")
    replay.add_argument("--sampling-mode", choices=["joint","attack","greedy"], help="Use joint to match elimination training")
    replay.add_argument("--stochastic-attacks", action="store_true",
                        help="Sample learned attack heads reproducibly; keep motor actions greedy")
    replay.add_argument("--use-checkpoint-env", action="store_true")

    report = sub.add_parser("report", help="Regenerate an offline report from existing logs")
    report.add_argument("--runs", required=True)
    report.add_argument("--out")

    stop = sub.add_parser("stop", help="Request a graceful stop at the next PPO update boundary")
    stop.add_argument("--out", required=True, help="Active single-run or experiment directory")

    bench = sub.add_parser("benchmark", help="Measure simulator throughput on this machine")
    _shared(bench)
    bench.add_argument("--steps", type=int, default=10000)
    bench.add_argument("--seed", type=int, default=19)
    bench.add_argument("--out")

    upstream = sub.add_parser("upstream", help="Optional original pretrained Dino model workflow")
    upstream.add_argument("operation", choices=["info", "fetch", "run"])
    upstream.add_argument("--out", default="external/aome510")
    upstream.add_argument("--episodes", type=int, default=5)
    upstream.add_argument("--max-steps", type=int, default=10000)
    upstream.add_argument("--show", action="store_true", help="Open original game window when a display is available")

    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "benchmark":
            if args.steps < 1:
                raise ValueError("--steps must be positive")
            benchmark(args)
        elif args.command == "report":
            from .report import make_report
            print(make_report(args.runs, args.out))
        elif args.command == "stop":
            target = Path(args.out)
            if not target.is_dir():
                raise FileNotFoundError(f"Run directory does not exist: {target}")
            (target / "STOP").touch()
            print(f"Stop requested: {target / 'STOP'}. Remove STOP before resuming.")
        elif args.command == "upstream":
            from .upstream import upstream_command
            upstream_command(args.operation, args.out, episodes=args.episodes,
                             max_steps=args.max_steps, show=args.show)
        else:
            config = _config(args)
            if args.command == "bootstrap":
                from .bootstrap import bootstrap
                for key in ("samples", "epochs"):
                    if getattr(args, key) is not None:
                        config["bootstrap"][key] = getattr(args, key)
                result = bootstrap(config, args.out, seed=args.seed, overwrite=args.overwrite)
                print(f"Saved shared runner: {args.out}; SHA256={result['sha256']}")
            elif args.command == "train":
                from .learning import train, _load_checkpoint
                from .experiment import runner_for_config, OutputLock
                # A resumed checkpoint embeds its original runner. Do not make
                # continuation depend on a custom source NPZ still existing.
                runner = None if args.resume and args.runner is None else runner_for_config(config, args.runner)
                with OutputLock(args.out):
                    if args.resume and args.runner is not None:
                        existing = _load_checkpoint(Path(args.out) / "latest.pt")
                        original = existing["runner"]
                        expected = original.get("sha256") if original["kind"] == "numpy" else "teacher"
                        requested = file_sha256(runner) if runner is not None else "teacher"
                        if expected != requested:
                            raise ValueError("Explicit --runner does not match the runner embedded in this checkpoint.")
                    result = train(config, Path(args.out), args.variant, args.seed,
                                   resume=args.resume, runner_path=runner)
                print(json.dumps(result, indent=2))
            elif args.command == "experiment":
                from .experiment import run_experiment
                if args.seeds:
                    config["experiment"]["seeds"] = args.seeds
                if args.variants:
                    config["experiment"]["variants"] = args.variants
                run_experiment(config, args.out, resume=args.resume, runner_override=args.runner)
            elif args.command in {"demo", "evaluate", "replay"}:
                from .experiment import runner_for_config, policy_uses_runner
                from .policies import resolve_policy, with_stochastic_attacks
                needs_runner = policy_uses_runner(args.left) or policy_uses_runner(args.right)
                runner = runner_for_config(config, args.runner) if needs_runner else None
                device = config["training"].get("device", "auto")
                left = resolve_policy(args.left, device, runner)
                right = resolve_policy(args.right, device, runner)
                mode=getattr(args,"sampling_mode",None)
                if mode and args.stochastic_attacks:
                    raise ValueError("Use --sampling-mode or --stochastic-attacks, not both")
                if args.stochastic_attacks or mode:
                    left = with_stochastic_attacks(left)
                    right = with_stochastic_attacks(right)
                    if mode:
                        from .policies import StochasticAttackPolicy
                        for pol in (left,right):
                            if isinstance(pol,StochasticAttackPolicy):
                                pol.sampling_mode=mode
                                pol.name=pol.name.replace("[sampled attacks]",f"[{mode}]")
                env_config = copy.deepcopy(config["env"])
                if getattr(args, "use_checkpoint_env", False):
                    if not hasattr(left, "config"):
                        raise ValueError("--use-checkpoint-env requires a checkpoint on the left")
                    env_config = copy.deepcopy(left.config["env"])
                if args.command == "evaluate":
                    from .evaluation import evaluate
                    count = args.num_seeds if args.num_seeds is not None else config["evaluation"]["num_seeds"]
                    first = args.seed_start if args.seed_start is not None else config["evaluation"]["seed_start"]
                    result = evaluate(left, right, env_config, range(first, first + count),
                                      out_dir=args.out, record_replay=args.record,
                                      bootstrap_samples=config["evaluation"]["bootstrap_samples"],
                                      safety_seconds=args.safety_seconds)
                    print(json.dumps(result, indent=2))
                else:
                    from .replay import make_replay
                    if args.seconds is not None and env_config.get("max_seconds") is not None:
                        env_config["max_seconds"] = args.seconds
                    result = make_replay(left, right, env_config, args.seed, args.out,
                                         safety_seconds=args.seconds if args.seconds is not None else 300.)
                    print(f"Replay: {Path(args.out).resolve()}")
                    print(json.dumps(result, indent=2))
        return 0
    except ImportError as error:
        print(f"Missing dependency: {error}. Run python setup_env.py --backend cpu, then use its printed Python command.", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed checkpoints/evaluations remain available; use --resume.", file=sys.stderr)
        return 130

