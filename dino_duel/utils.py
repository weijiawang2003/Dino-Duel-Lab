from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "configs").is_dir():
    # Wheel installations carry a second, release-generated copy of presets
    # and bundled weights; repository commands use the editable top-level copy.
    PROJECT_ROOT = Path(__file__).resolve().parent / "data"


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_info() -> dict:
    import numpy as np
    result = {"python": sys.version, "platform": platform.platform(),
              "numpy": np.__version__, "cpu_count": os.cpu_count()}
    try:
        import torch
        result.update(torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
                      cuda_version=torch.version.cuda,
                      mps_available=bool(hasattr(torch.backends, "mps") and
                                         torch.backends.mps.is_available()))
    except ImportError:
        result["torch"] = None
    return result


def deep_merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        result[key] = deep_merge(result[key], value) if (
            isinstance(value, dict) and isinstance(result.get(key), dict)) else value
    return result


def load_config(path: str | Path | None = None) -> dict:
    default_path = PROJECT_ROOT / "configs" / "cpu.json"
    with default_path.open(encoding="utf-8") as f:
        config = json.load(f)
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            candidate = PROJECT_ROOT / "configs" / (str(path).removesuffix(".json") + ".json")
        with candidate.open(encoding="utf-8") as f:
            config = deep_merge(config, json.load(f))
    allowed = {"schema_version", "description", "env", "training", "model", "evaluation",
               "experiment", "bootstrap"}
    unexpected = set(config) - allowed
    if unexpected:
        raise ValueError(f"Unknown configuration sections: {sorted(unexpected)}")
    if config.get("schema_version") != 1:
        raise ValueError("This release supports schema_version=1 only.")
    return config


def write_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

