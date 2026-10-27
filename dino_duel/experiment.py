"""Complete, restartable multi-seed experiment orchestration."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import socket
import shutil
import time
from pathlib import Path

import numpy as np

from .utils import PROJECT_ROOT, atomic_json, file_sha256, runtime_info


class OutputLock:
    """An OS advisory lock; RUNNING.json is metadata, never a PID-based lock.

    The file is deliberately retained after release. Unlinking it would let a
    third process lock a new inode while another process holds the old one.
    """

    def __init__(self, out_dir):
        self.path = Path(out_dir) / "RUNNING.json"
        self._handle = None
        self._metadata = None

    def _unlock(self):
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def _write_metadata(self):
        self._handle.seek(0)
        self._handle.write((json.dumps(self._metadata, indent=2) + "\n").encode("utf-8"))
        self._handle.truncate()
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("OutputLock is not reentrant.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT without O_TRUNC leaves an active owner's metadata intact.
        handle = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644), "r+b")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                # Windows permits locking a byte beyond EOF on a new file.
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError(f"Cannot acquire the output lock for {self.path.parent}; "
                               "another command may still be using it. Do not delete RUNNING.json "
                               "while a command is active.") from error
        self._handle = handle
        self._metadata = {"pid": os.getpid(), "host": socket.gethostname(),
                          "started_at_unix": time.time(), "status": "running"}
        try:
            self._write_metadata()
        except BaseException:
            try:
                self._unlock()
            finally:
                self._handle.close()
                self._handle = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is None:
            return
        try:
            self._metadata.update(status="released", released_at_unix=time.time())
            self._write_metadata()
        finally:
            try:
                self._unlock()
            finally:
                self._handle.close()
                self._handle = None


def runner_for_config(config: dict, override: str | Path | None = None) -> Path | None:
    choice = override if override is not None else config.get("experiment", {}).get("runner", "bundled")
    if choice == "teacher":
        return None
    path = PROJECT_ROOT / "assets" / "runner_bc.npz" if choice == "bundled" else Path(choice).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Basic runner not found: {path}. Restore the complete archive or run bootstrap.")
    return path.resolve()


def policy_uses_runner(spec: str | Path) -> bool:
    """Only scripted policies use an external common runner; checkpoints embed it."""
    text = str(spec)
    return text in {"teacher", "none", "random", "save", "aggressive"} or text.startswith("teacher:")


def _experiment_runner(config: dict, out_dir: Path, previous: dict | None,
                       override=None) -> Path | None:
    """Keep a byte-identical, portable copy of the experiment's shared runner."""
    shared = out_dir / "shared_runner.npz"
    expected = previous.get("identity", {}).get("runner_sha256") if previous else None
    if previous is not None and override is None:
        if expected == "teacher":
            return None
        if shared.is_file():
            if file_sha256(shared) != expected:
                raise ValueError("Saved shared_runner.npz does not match this experiment's recorded SHA256.")
            return shared
    source = runner_for_config(config, override)
    digest = file_sha256(source) if source is not None else "teacher"
    if expected is not None and digest != expected:
        raise ValueError("Cannot resume after changing the shared runner checkpoint.")
    if source is None:
        return None
    if shared.is_file():
        if previous is not None and file_sha256(shared) != expected:
            raise ValueError("Saved shared_runner.npz does not match this experiment's recorded SHA256.")
        if file_sha256(shared) == digest:
            return shared
    if source.resolve() != shared.resolve():
        temporary = shared.with_suffix(".npz.tmp")
        try:
            shutil.copyfile(source, temporary)
            if file_sha256(temporary) != digest:
                raise ValueError("Shared runner changed while being copied; retry from a stable source file.")
            os.replace(temporary, shared)
        finally:
            temporary.unlink(missing_ok=True)
    return shared


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _evaluate_cached(left_spec, right_spec, config, out_dir, runner_path, resume):
    from .evaluation import evaluate
    from .policies import resolve_policy
    from .env import EnvConfig
    out_dir = Path(out_dir)
    e = config["evaluation"]
    seeds = list(range(int(e["seed_start"]), int(e["seed_start"]) + int(e["num_seeds"])))
    if not seeds:
        raise ValueError("evaluation.num_seeds must be positive")
    fingerprints = []
    for spec in (left_spec, right_spec):
        path = Path(spec)
        fingerprints.append(file_sha256(path) if path.is_file() else str(spec))
    needs_runner = policy_uses_runner(left_spec) or policy_uses_runner(right_spec)
    task = {"policies": fingerprints, "env": EnvConfig.from_dict(config.get("env", {})).to_dict(),
            "seeds": seeds, "bootstrap_samples": e["bootstrap_samples"],
            "fixed_runner_sha256": (file_sha256(runner_path) if runner_path else "teacher") if needs_runner else "embedded"}
    task_path = out_dir / "task.json"
    if resume and task_path.exists() and (out_dir / "summary.json").exists():
        if json.loads(task_path.read_text()) == task:
            return json.loads((out_dir / "summary.json").read_text())
    left = resolve_policy(left_spec, device=config["training"].get("device", "auto"), runner_path=runner_path)
    right = resolve_policy(right_spec, device=config["training"].get("device", "auto"), runner_path=runner_path)
    result = evaluate(left, right, task["env"], seeds, out_dir=out_dir,
                      record_replay=False, bootstrap_samples=int(e["bootstrap_samples"]))
    atomic_json(task_path, task)
    return result


def hierarchical_comparison(left: str, right: str, seed_summaries: list,
                            resamples: int = 2000) -> dict:
    """Resample trained seed pairs and whole environment seed pairs, in order."""
    seed_ids = [int(seed) for seed, _ in seed_summaries]
    if any(any(x is None for x in summary["pair_scores"]) for _, summary in seed_summaries):
        return {"status":"censored", "note":"Unresolved games are not draws; use elimination suite bounds.",
                "n_training_seeds":len(seed_summaries)}
    scores = [np.asarray(summary["pair_scores"], dtype=float) for _, summary in seed_summaries]
    means = [float(s.mean()) for s in scores]
    ci = None
    if len(scores) >= 2:
        rng = np.random.default_rng(60925)
        boot = np.empty(resamples)
        for k in range(resamples):
            sampled_seeds = rng.integers(0, len(scores), size=len(scores))
            boot[k] = np.mean([rng.choice(scores[j], len(scores[j]), replace=True).mean()
                               for j in sampled_seeds])
        ci = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
    return {"left": left, "right": right, "training_seeds": seed_ids,
            "per_training_seed_scores": means,
            "mean_score": float(np.mean(means)), "hierarchical_ci95": ci,
            "n_training_seeds": len(scores),
            "n_eval_seed_pairs_per_training_seed": len(scores[0]),
            "note": "Exploratory percentile interval. Resamples training seed pairs, then paired environment seeds. "
                    "Few training seeds give unreliable uncertainty; no multiple-comparison correction or superiority claim."}


def run_experiment(config: dict, out_dir: str | Path, *, resume=False,
                   runner_override=None) -> dict:
    from .learning import train
    from .policies import resolve_policy
    from .replay import make_replay
    from .report import make_report

    out_dir = Path(out_dir).resolve()
    config = copy.deepcopy(config)
    variants = config["experiment"]["variants"]
    seeds = config["experiment"]["seeds"]
    if not variants or len(set(variants)) != len(variants) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment variants and seeds must be nonempty and unique.")
    if any(x not in {"baseline", "jepa", "jepa_shuffled", "baseline_large"} for x in variants):
        raise ValueError("Unknown experiment variant.")
    if any(isinstance(x, bool) or int(x) != x or x < 0 for x in seeds):
        raise ValueError("Training seeds must be nonnegative integers.")
    e = config["evaluation"]
    for key, minimum in (("num_seeds", 1), ("bootstrap_samples", 100), ("seed_start", 2 ** 31)):
        value = e.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"evaluation.{key} must be an integer >= {minimum}; episode seeds below 2**31 are reserved for training.")
    for opponent in config["experiment"].get("fixed_opponents", []):
        if opponent not in {"teacher:none", "teacher:random", "teacher:aggressive", "teacher:save"}:
            raise ValueError(f"Unsupported fixed opponent {opponent!r}.")
    manifest_path = out_dir / "experiment.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Experiment exists: {out_dir}. Use --resume or choose a new directory.")
    if (out_dir / "STOP").exists():
        raise RuntimeError(f"Remove {out_dir / 'STOP'} before starting/resuming this experiment.")
    with OutputLock(out_dir):
        # Recheck under the lock: another process may have created the manifest
        # while this process waited to open the output directory.
        if manifest_path.exists() and not resume:
            raise FileExistsError(f"Experiment exists: {out_dir}. Use --resume or choose a new directory.")
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        runner = _experiment_runner(config, out_dir, previous, runner_override)
        identity = {"variants": variants, "seeds": seeds,
                    "runner_sha256": file_sha256(runner) if runner else "teacher"}
        if previous is not None and previous.get("identity") != identity:
            raise ValueError("Cannot resume after changing treatment seeds, variants, or shared runner.")
        # A flat configuration file is accepted directly by --config. Its
        # runner path is relative to this file, so the entire output can move.
        config["experiment"]["runner"] = "shared_runner.npz" if runner else "teacher"
        atomic_json(out_dir / "experiment_config.json", config)
        manifest = {"schema_version": 1, "identity": identity, "config": config,
                    "runtime": runtime_info(), "status": "running", "completed_runs": [],
                    "updated_at_unix": time.time()}
        atomic_json(manifest_path, manifest)
        checkpoints = {}
        for variant in variants:
            for seed in seeds:
                if (out_dir / "STOP").exists():
                    manifest["status"] = "stopped"
                    atomic_json(manifest_path, manifest)
                    return manifest
                run_dir = out_dir / "runs" / variant / f"seed_{seed}"
                this_config = copy.deepcopy(config)
                this_config["training"]["stop_file"] = str(out_dir / "STOP")
                exists = (run_dir / "latest.pt").is_file()
                print(f"\n=== {variant}, seed {seed} ===", flush=True)
                summary = train(this_config, run_dir, variant=variant, seed=int(seed),
                                resume=bool(resume and exists), runner_path=runner)
                checkpoint = run_dir / "policy.pt"
                if not checkpoint.exists():
                    checkpoint = run_dir / "latest.pt"
                checkpoints[(variant, seed)] = checkpoint
                manifest["completed_runs"].append({"variant": variant, "seed": seed,
                                                    "summary": summary})
                manifest["updated_at_unix"] = time.time()
                atomic_json(manifest_path, manifest)
                # A graceful trainer stop must not silently start the next run.
                if summary.get("status") in {"stopped", "interrupted", "wall_budget"} or summary.get("stopped", False) or (out_dir / "STOP").exists():
                    manifest["status"] = "stopped"
                    atomic_json(manifest_path, manifest)
                    make_report(out_dir)
                    return manifest
        manifest["status"] = "evaluating"
        atomic_json(manifest_path, manifest)
        for (variant, seed), checkpoint in checkpoints.items():
            for opponent in config["experiment"].get("fixed_opponents", []):
                if (out_dir / "STOP").exists():
                    manifest["status"] = "stopped"
                    atomic_json(manifest_path, manifest)
                    return manifest
                slug = opponent.replace(":", "_")
                folder = out_dir / "evaluations" / "fixed" / variant / f"seed_{seed}" / slug
                print(f"Evaluate {variant}/{seed} vs {opponent}", flush=True)
                _evaluate_cached(checkpoint, opponent, config, folder, runner, resume)
        comparisons = []
        for left, right in itertools.combinations(variants, 2):
            summaries = []
            for seed in seeds:
                if (out_dir / "STOP").exists():
                    manifest["status"] = "stopped"
                    atomic_json(manifest_path, manifest)
                    return manifest
                folder = out_dir / "evaluations" / "head_to_head" / f"{left}_vs_{right}" / f"seed_{seed}"
                print(f"Head-to-head {left} vs {right}, training seed {seed}", flush=True)
                result = _evaluate_cached(checkpoints[(left, seed)], checkpoints[(right, seed)],
                                          config, folder, runner, resume)
                summaries.append((seed, result))
            comparisons.append(hierarchical_comparison(left, right, summaries,
                                  int(config["evaluation"]["bootstrap_samples"])))
        atomic_json(out_dir / "comparison.json", {"schema_version": 1, "comparisons": comparisons})
        if len(variants) >= 2:
            left = resolve_policy(checkpoints[(variants[0], seeds[0])], device="cpu")
            right = resolve_policy(checkpoints[(variants[1], seeds[0])], device="cpu")
            make_replay(left, right, config["env"], int(config["evaluation"]["seed_start"]),
                        out_dir / "replay.html")
        manifest["status"] = "complete"
        manifest["updated_at_unix"] = time.time()
        atomic_json(manifest_path, manifest)
        report = make_report(out_dir)
        print(f"\nExperiment complete. Report: {report}", flush=True)
        return manifest

