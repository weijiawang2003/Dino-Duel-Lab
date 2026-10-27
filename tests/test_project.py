"""Integration regressions for the packaged asset, CLI, and experiment guardrails."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from dino_duel.env import DuelEnv, EnvConfig
from dino_duel.policies import NumpyRunner, TeacherPolicy
from dino_duel.utils import PROJECT_ROOT, load_config


def test_packaged_runner_provenance_and_safe_numpy_loading():
    path = PROJECT_ROOT / "assets" / "runner_bc.npz"
    record = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    runner = NumpyRunner(path)
    env = DuelEnv()
    logits = runner.logits(env.runner_observations())
    assert logits.shape == (2, 3)
    assert np.isfinite(logits).all()
    assert runner.metadata["source"] == "original behavior cloning + DAgger"


def test_preset_main_matches_attack_only_research_contract():
    config = load_config("cpu")
    env = EnvConfig.from_dict(config["env"])
    assert not env.natural_obstacles
    assert env.ensure_reachable
    assert env.initial_speed_jitter > 0
    assert env.attack_every_n_steps == 6
    assert env.attack_cooldown < env.dt * env.action_repeat * env.attack_every_n_steps
    assert config["evaluation"]["seed_start"] >= 2 ** 31
    assert config["experiment"]["runner"] == "bundled"


def test_core_demo_does_not_import_torch(tmp_path):
    code = """
import sys
from dino_duel.cli import main
assert 'torch' not in sys.modules
assert main(['demo','--runner','teacher','--seconds','1','--out',sys.argv[1]])==0
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "demo.html")],
                            cwd=PROJECT_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    text = (tmp_path / "demo.html").read_text()
    assert 'id="replay-data"' in text


def test_rule_policy_never_selects_forbidden_purchase():
    env = DuelEnv(load_config("cpu")["env"])
    obs, _ = env.reset(seed=123)
    policy = TeacherPolicy("random", runner_path=PROJECT_ROOT / "assets" / "runner_bc.npz")
    for _ in range(150):
        masks = env.attack_masks()
        actions = policy.act(obs, masks, env.runner_observations())
        assert masks[np.arange(2), actions[:, 1]].all()
        obs, _, terminal, truncated, _ = env.step(actions)
        if terminal or truncated:
            break


def test_evaluation_config_fails_before_any_training(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import dino_duel.learning as learning
    from dino_duel.experiment import run_experiment
    def forbidden_train(*args, **kwargs):
        pytest.fail("Training should not run with invalid evaluation settings")
    monkeypatch.setattr(learning, "train", forbidden_train)
    for key, bad in [("num_seeds", 0), ("bootstrap_samples", 99), ("seed_start", 0)]:
        config = load_config("smoke")
        config["evaluation"][key] = bad
        with pytest.raises(ValueError):
            run_experiment(config, tmp_path / key)


def test_hierarchical_uncertainty_respects_training_seed_unit():
    from dino_duel.experiment import hierarchical_comparison
    rows = [(11, {"pair_scores": [1, 1, 1]}), (23, {"pair_scores": [0, 0, 0]})]
    report = hierarchical_comparison("A", "B", rows, resamples=500)
    assert report["mean_score"] == 0.5
    assert report["hierarchical_ci95"] == [0.0, 1.0]
    single = hierarchical_comparison("A", "B", rows[:1])
    assert single["hierarchical_ci95"] is None


def test_resume_pipeline_skips_unchanged_cached_evaluation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from dino_duel.experiment import _evaluate_cached
    import dino_duel.evaluation as evaluation
    config = load_config("smoke")
    config["env"]["max_seconds"] = 0.2
    config["evaluation"]["num_seeds"] = 2
    runner = PROJECT_ROOT / "assets" / "runner_bc.npz"
    result = _evaluate_cached("teacher:none", "teacher:none", config, tmp_path, runner, False)
    def fail(*args, **kwargs):
        pytest.fail("Cache hit must not simulate games again")
    monkeypatch.setattr(evaluation, "evaluate", fail)
    repeat = _evaluate_cached("teacher:none", "teacher:none", config, tmp_path, runner, True)
    assert result == repeat


def test_optional_upstream_refuses_changed_archive_before_extraction(tmp_path):
    from dino_duel.upstream import _extract_verified
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not the pinned upstream archive")
    with pytest.raises(ValueError, match="SHA256"):
        _extract_verified(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_output_lock_excludes_nested_and_other_process_then_reuses_file(tmp_path):
    from dino_duel.experiment import OutputLock
    path = tmp_path / "RUNNING.json"
    path.write_text(json.dumps({"pid": os.getpid(), "host": "an-old-host-namespace"}))
    code = """
import sys
from dino_duel.experiment import OutputLock
try:
    with OutputLock(sys.argv[1]):
        pass
except RuntimeError:
    raise SystemExit(23)
"""
    with OutputLock(tmp_path):
        with pytest.raises(RuntimeError, match="output lock"):
            with OutputLock(tmp_path):
                pytest.fail("Nested owner acquired the same output lock")
        competing = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                                   cwd=PROJECT_ROOT, text=True, capture_output=True)
        assert competing.returncode == 23, competing.stdout + competing.stderr
    assert path.is_file()  # Never unlink an advisory-lock inode.
    assert json.loads(path.read_text())["status"] == "released"
    with OutputLock(tmp_path):
        pass
    assert path.is_file()
    after = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                           cwd=PROJECT_ROOT, text=True, capture_output=True)
    assert after.returncode == 0, after.stdout + after.stderr


def _portable_tiny_config(custom_runner):
    config = load_config("smoke")
    config["env"].update(max_seconds=0.2, ensure_reachable=False)
    config["model"]["hidden"] = 16
    config["training"].update(updates=1, num_envs=1, rollout_steps=4, epochs=1,
                               minibatch_size=4, checkpoint_interval=1,
                               snapshot_interval=1, opponent_pool_size=0, device="cpu")
    config["experiment"].update(variants=["baseline"], seeds=[11],
                                 fixed_opponents=["teacher:none"], runner=str(custom_runner))
    config["evaluation"].update(num_seeds=1, bootstrap_samples=100)
    return config


def test_experiment_survives_deleted_source_runner_and_moved_output(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from dino_duel.experiment import run_experiment
    from dino_duel.cli import main
    import dino_duel.evaluation as evaluation
    source = tmp_path / "custom-original.npz"
    shutil.copyfile(PROJECT_ROOT / "assets" / "runner_bc.npz", source)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    config = _portable_tiny_config(source)
    original_out = tmp_path / "original-output"
    result = run_experiment(config, original_out)
    assert result["status"] == "complete"
    assert hashlib.sha256((original_out / "shared_runner.npz").read_bytes()).hexdigest() == expected
    flat = json.loads((original_out / "experiment_config.json").read_text())
    assert flat["experiment"]["runner"] == "shared_runner.npz"
    assert "config" not in flat
    checkpoint = original_out / "runs" / "baseline" / "seed_11" / "policy.pt"
    policy_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source.unlink()
    def must_use_cache(*args, **kwargs):
        pytest.fail("Unchanged portable experiment should retain its evaluation cache")
    monkeypatch.setattr(evaluation, "evaluate", must_use_cache)
    # API continuation ignores the vanished source in the original config.
    assert run_experiment(config, original_out, resume=True)["status"] == "complete"
    moved = tmp_path / "moved-output"
    shutil.move(original_out, moved)
    assert main(["experiment", "--config", str(moved / "experiment_config.json"),
                 "--out", str(moved), "--resume"]) == 0
    checkpoint = moved / "runs" / "baseline" / "seed_11" / "policy.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == policy_hash
    # Explicitly changing the common motor skill cannot masquerade as resume.
    with pytest.raises(ValueError, match="shared runner"):
        run_experiment(config, moved, resume=True, runner_override="teacher")
    with (moved / "shared_runner.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        run_experiment(config, moved, resume=True)


def test_cli_resume_and_checkpoint_only_inference_need_no_source_runner(tmp_path):
    pytest.importorskip("torch")
    from dino_duel.cli import main
    from dino_duel.learning import _load_checkpoint
    source = tmp_path / "custom.npz"
    shutil.copyfile(PROJECT_ROOT / "assets" / "runner_bc.npz", source)
    config = _portable_tiny_config(source)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out = tmp_path / "single"
    common = ["train", "--config", str(config_path), "--out", str(out), "--seed", "11"]
    assert main(common) == 0
    source.unlink()
    assert main(common + ["--updates", "2", "--resume"]) == 0
    assert _load_checkpoint(out / "latest.pt")["update"] == 2
    assert main(common + ["--updates", "2", "--resume", "--runner", "teacher"]) == 2
    policy = str(out / "policy.pt")
    # Neither checkpoint policy uses the deliberately missing external runner.
    for command in ("evaluate", "replay"):
        target = tmp_path / ("evaluation" if command == "evaluate" else "replay.html")
        args = [command, "--config", str(config_path), "--left", policy, "--right", policy,
                "--runner", str(source), "--out", str(target)]
        assert main(args) == 0
    assert (tmp_path / "evaluation" / "summary.json").is_file()
    assert (tmp_path / "replay.html").is_file()


def test_teacher_demo_allowed_with_joint_training_preset(tmp_path):
    from dino_duel.cli import main
    assert main(["demo", "--config", "joint", "--runner", "teacher", "--seconds", "0.2",
                 "--out", str(tmp_path / "joint-config-teacher.html")]) == 0

