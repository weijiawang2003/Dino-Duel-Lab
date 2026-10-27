#!/usr/bin/env python3
"""Train and evaluate the complete matched-seed PPO / JEPA comparison.

Run from the extracted project with its environment's Python:

    python scripts/reproduce_comparison.py --out runs/reproduced --workers 2
    python scripts/reproduce_comparison.py --out runs/reproduced --workers 2 --resume

The default is 300 TOTAL updates for each of six runs, not 300 extra updates on
resume. Ctrl+C requests checkpointing at update boundaries. Final evaluation is
started only when every requested run has finished its full budget.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline", "jepa")


def _process_group_options():
    # Trainers receive the shared STOP file rather than a terminal's Ctrl+C.
    # The evaluation process and its worker pool can be interrupted together.
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _interrupt_evaluation(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGINT)
    process.wait()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="runs/reproduced_comparison")
    parser.add_argument("--workers", type=int, default=2,
                        help="Maximum simultaneous training runs and evaluation workers")
    parser.add_argument("--updates", type=int, default=300, help="Total updates per run, including earlier resumed updates")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-eval-seeds", type=int, default=50,
                        help="Environment seeds; each is evaluated in both seats")
    parser.add_argument("--training-seeds", "--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--config", default=str(ROOT / "configs" / "trained_strength.json"),
                        help="Shared preset / JSON; the default reproduces this release's experiment")
    args = parser.parse_args(argv)
    if min(args.workers, args.updates, args.num_eval_seeds) < 1:
        parser.error("workers, updates and num-eval-seeds must be positive")
    if any(seed < 0 for seed in args.training_seeds) or len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("training-seeds must be nonnegative and unique")
    return args


def main(argv=None):
    args = _parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Keep CPU work bounded per worker; multiple small models use processes.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from dino_duel.experiment import OutputLock, runner_for_config
    from dino_duel.learning import _load_checkpoint, _resume_signature, resolve_config
    from dino_duel.utils import atomic_json, file_sha256, load_config

    output = Path(args.out).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    config = resolve_config(load_config(config_path))
    if config["training"]["opponent_style"] != "selfplay":
        raise ValueError("This comparison requires selfplay for both variants; use train for a separate fixed-opponent diagnostic")
    config["training"]["updates"] = args.updates
    config["training"]["stop_file"] = str(output / "STOP")
    config["experiment"]["variants"] = list(VARIANTS)
    config["experiment"]["seeds"] = args.training_seeds
    expected_steps = args.updates * config["training"]["num_envs"] * config["training"]["rollout_steps"]
    identity = {"training_seeds": args.training_seeds, "variants": list(VARIANTS),
                "training_signature": _resume_signature(config)}
    interrupted = [False]
    old_handlers = {}

    def request_stop(signum, frame):
        interrupted[0] = True

    def complete(run_dir, variant, seed):
        if not (run_dir / "initial_policy.pt").is_file():
            return False
        for name in ("latest.pt", "policy.pt"):
            path = run_dir / name
            if not path.is_file():
                return False
            checkpoint = _load_checkpoint(path)
            if (checkpoint["variant"] != variant or int(checkpoint["seed"]) != seed
                    or int(checkpoint["update"]) != args.updates
                    or int(checkpoint["global_step"]) != expected_steps):
                return False
        return True

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)
    try:
        with OutputLock(output):
            manifest_path = output / "reproduction.json"
            previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
            if previous is not None and not args.resume:
                raise FileExistsError(f"Output already exists: {output}. Use --resume or another --out")
            if previous is not None and previous["identity"] != identity:
                raise ValueError("Resume changed training seeds, variants, game rules or learning settings")
            if previous is not None and (output / "final").is_dir() and (
                    previous["updates"] != args.updates or previous["num_eval_seeds"] != args.num_eval_seeds):
                raise ValueError("An existing final evaluation has a different budget or seed count. Preserve it and choose a new output folder")

            shared_runner = output / "shared_runner.npz"
            if args.resume and previous is not None:
                if previous["runner_sha256"] == "teacher":
                    runner_hash = "teacher"
                    config["experiment"]["runner"] = "teacher"
                else:
                    if not shared_runner.is_file() or file_sha256(shared_runner) != previous["runner_sha256"]:
                        raise ValueError("The saved common runner is missing or has changed")
                    runner_hash = previous["runner_sha256"]
                    config["experiment"]["runner"] = "shared_runner.npz"
            else:
                choice = config["experiment"].get("runner", "bundled")
                if choice not in {"bundled", "teacher"} and config_path.is_file() and not Path(choice).is_absolute():
                    config["experiment"]["runner"] = str((config_path.resolve().parent / choice).resolve())
                source = runner_for_config(config)
                runner_hash = "teacher" if source is None else file_sha256(source)
                if source is not None:
                    if source.resolve() != shared_runner.resolve():
                        shutil.copyfile(source, shared_runner)
                    if file_sha256(shared_runner) != runner_hash:
                        raise ValueError("Common runner changed during copying")
                config["experiment"]["runner"] = "teacher" if source is None else "shared_runner.npz"

            shared_config = output / "comparison_config.json"
            atomic_json(shared_config, config)
            stop_path = output / "STOP"
            if stop_path.exists():
                if args.resume:
                    stop_path.unlink()
                    print("Resuming: cleared the comparison's STOP request.", flush=True)
                else:
                    raise RuntimeError(f"A stop request exists at {stop_path}; use --resume when ready")
            logs = output / "logs"
            logs.mkdir(exist_ok=True)
            jobs = deque()
            for seed in args.training_seeds:
                for variant in VARIANTS:
                    run_dir = output / "runs" / f"{variant}_seed{seed}"
                    if (run_dir / "STOP").exists():
                        raise RuntimeError(f"Remove the individual run's explicit STOP request before continuing: {run_dir / 'STOP'}")
                    exists = (run_dir / "latest.pt").is_file()
                    if (exists or (run_dir / "metrics.jsonl").exists()) and not args.resume:
                        raise FileExistsError(f"Training output exists: {run_dir}; use --resume")
                    if args.resume and (run_dir / "metrics.jsonl").exists() and not exists:
                        raise RuntimeError(f"Metrics exist without a resumable checkpoint: {run_dir}")
                    command = [sys.executable, str(ROOT / "run.py"), "train", "--config", str(shared_config),
                               "--out", str(run_dir), "--variant", variant, "--seed", str(seed),
                               "--updates", str(args.updates)]
                    if args.resume and exists:
                        command.append("--resume")
                    jobs.append({"variant": variant, "seed": seed, "run_dir": run_dir,
                                 "command": command, "log_path": logs / f"{variant}_seed{seed}.log"})
            manifest = {"schema": "dino_duel.reproduction.v1", "identity": identity,
                        "updates": args.updates, "joint_environment_steps_per_run": expected_steps,
                        "num_eval_seeds": args.num_eval_seeds, "runner_sha256": runner_hash,
                        "config": shared_config.name, "workers": args.workers, "status": "training",
                        "attack_mode": "stochastic", "completed_runs": [], "failed_runs": []}
            atomic_json(manifest_path, manifest)
            active = []
            failure = None
            stop_announced = False
            while jobs or active:
                if interrupted[0] or failure:
                    if not stop_announced:
                        stop_path.touch()
                        print("Stop requested. Active trainers will finish the current update and save; evaluation will not run.", flush=True)
                        stop_announced = True
                    jobs.clear()
                while jobs and len(active) < args.workers and not interrupted[0] and not failure:
                    job = jobs.popleft()
                    handle = job["log_path"].open("ab")
                    handle.write(("\n" + json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
                                                       "command": job["command"]}) + "\n").encode())
                    handle.flush()
                    try:
                        process = subprocess.Popen(job["command"], cwd=ROOT, stdout=handle,
                                                   stderr=subprocess.STDOUT, env=os.environ.copy(),
                                                   **_process_group_options())
                    except OSError as error:
                        handle.close()
                        failure = f"Could not launch {job['variant']} seed={job['seed']}: {error}"
                        break
                    active.append({**job, "process": process, "handle": handle})
                    print(f"Started {job['variant']} seed={job['seed']}; log: {job['log_path']}", flush=True)
                for job in active[:]:
                    code = job["process"].poll()
                    if code is None:
                        continue
                    job["handle"].close()
                    active.remove(job)
                    try:
                        finished = code == 0 and complete(job["run_dir"], job["variant"], job["seed"])
                    except (OSError, ValueError, RuntimeError) as error:
                        finished = False
                        failure = f"Could not validate {job['run_dir']}: {error}"
                    row = {"variant": job["variant"], "seed": job["seed"], "returncode": code,
                           "run_dir": str(job["run_dir"].relative_to(output)),
                           "log": str(job["log_path"].relative_to(output))}
                    manifest["completed_runs" if finished else "failed_runs"].append(row)
                    if not finished and not interrupted[0]:
                        failure = f"Incomplete training: {job['variant']} seed={job['seed']}; inspect {job['log_path']}"
                    atomic_json(manifest_path, manifest)
                    print(f"{'Complete' if finished else 'Incomplete'}: {job['variant']} seed={job['seed']}", flush=True)
                if active:
                    time.sleep(0.2)
            if interrupted[0] or failure:
                manifest["status"] = "interrupted" if interrupted[0] else "incomplete"
                manifest["reason"] = failure or "User stop request"
                atomic_json(manifest_path, manifest)
                print("Checkpoints are retained. Repeat the same command with --resume to continue.", flush=True)
                return 130 if interrupted[0] else 1

            command = [sys.executable, str(ROOT / "scripts" / "compare_trained.py"),
                       "--runs", str(output / "runs"), "--out", str(output / "final"),
                       "--expected-update", str(args.updates), "--training-seeds",
                       *[str(seed) for seed in args.training_seeds],
                       "--num-seeds", str(args.num_eval_seeds), "--workers", str(args.workers),
                       "--attack-mode", "stochastic"]
            manifest["status"] = "evaluating"
            manifest["evaluation_command"] = command
            atomic_json(manifest_path, manifest)
            print("All training budgets completed. Starting the fixed final comparison.", flush=True)
            with (logs / "final_evaluation.log").open("ab") as handle:
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           env=os.environ.copy(), **_process_group_options())
                while process.poll() is None:
                    if interrupted[0]:
                        # Training is already saved. Evaluation outputs are cached
                        # atomically and can be regenerated on the next resume.
                        _interrupt_evaluation(process)
                        break
                    time.sleep(0.2)
            manifest["status"] = "complete" if process.returncode == 0 and not interrupted[0] else "evaluation_incomplete"
            manifest["evaluation_returncode"] = process.returncode
            atomic_json(manifest_path, manifest)
            if manifest["status"] != "complete":
                print(f"Evaluation did not finish. See {logs / 'final_evaluation.log'}; --resume reuses completed training.", flush=True)
                return 130 if interrupted[0] else 1
            print(f"Completed comparison: {output / 'final' / 'report.html'}", flush=True)
            return 0
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
