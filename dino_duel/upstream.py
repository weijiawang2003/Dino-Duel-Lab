"""Optional fixed-revision access to a genuine public pretrained pixel policy.

The original game stays separate: its observation/physics contract is not our
duel simulator's contract. No third-party weights/source/assets ship in this
project. Nothing is downloaded by the default experiment pipeline.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .utils import atomic_json, file_sha256


REGISTRY = {
    "project": "aome510/chrome-dino-game-rl",
    "homepage": "https://github.com/aome510/chrome-dino-game-rl",
    "commit": "8a536f6f4e39280a08b70a575cf19c4f1f1c7015",
    "archive_url": "https://codeload.github.com/aome510/chrome-dino-game-rl/zip/8a536f6f4e39280a08b70a575cf19c4f1f1c7015",
    "archive_sha256": "8ebff017f4112d8d0c2229e2328abe31d48e3cc3c7115fe5a5866a708fdf89b1",
    "model_sha256": "cd0f44c3aef59fb125e2ce3ff9ee637b9ab07a0c93bfaa4011ff58d8493fde58",
    "model_bytes": 3468319,
    "observation": "4 x 64 x 128 grayscale frames in original environment",
    "actions": ["stand", "jump", "duck"],
    "license_note": "Upstream pyproject.toml declares MIT; no separate LICENSE file at inspected revision. See upstream for original terms and asset provenance.",
    "duel_compatible": False,
}


def _extract_verified(archive: Path, out: Path):
    if file_sha256(archive) != REGISTRY["archive_sha256"]:
        raise ValueError("Upstream archive SHA256 mismatch; refusing changed bytes.")
    with zipfile.ZipFile(archive) as source:
        if sum(i.file_size for i in source.infolist()) > 100 * 1024 * 1024:
            raise ValueError("Upstream archive expands beyond expected size.")
        for info in source.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("Unsafe path in upstream archive.")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symbolic links are not allowed in upstream archives.")
            target = out.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(info))
    if file_sha256(out / "model.pth") != REGISTRY["model_sha256"]:
        raise ValueError("Extracted model checksum mismatch.")
    manifest = {**REGISTRY, "files": {str(p.relative_to(out)): file_sha256(p)
                for p in sorted(out.rglob("*")) if p.is_file() and p.name != "UPSTREAM_MANIFEST.json"}}
    atomic_json(out / "UPSTREAM_MANIFEST.json", manifest)


def fetch(out_dir: str | Path) -> Path:
    out = Path(out_dir).resolve()
    manifest_path = out / "UPSTREAM_MANIFEST.json"
    if manifest_path.exists():
        _verify_existing(out)
        print(f"Verified existing upstream download: {out}")
        return out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty directory for upstream fetch: {out}")
    out.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(REGISTRY["archive_url"], headers={"User-Agent": "Dino-Duel-Lab/1.0"})
    with tempfile.TemporaryDirectory(prefix="dino-upstream-") as scratch:
        archive = Path(scratch) / "upstream.zip"
        with urllib.request.urlopen(request, timeout=45) as response, archive.open("wb") as handle:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 30 * 1024 * 1024:
                    raise ValueError("Upstream download exceeds expected size limit.")
                handle.write(block)
        _extract_verified(archive, out)
    print(f"Fetched and checksum-verified: {out}")
    return out


def _verify_existing(out):
    manifest = json.loads((out / "UPSTREAM_MANIFEST.json").read_text())
    if manifest.get("commit") != REGISTRY["commit"]:
        raise ValueError("Downloaded source revision differs from the pinned adapter revision.")
    for relative, checksum in manifest["files"].items():
        path = out / relative
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"Upstream source was modified: {relative}; use a fresh verified download.")
    if file_sha256(out / "model.pth") != REGISTRY["model_sha256"]:
        raise ValueError("Unrecognized upstream model weights.")


def _load_original_model(path):
    import torch
    import numpy as np
    from torch import nn

    # A local compatibility class, implementing the published layer contract.
    # No model.py is executed and unrestricted pickle loading is never used.
    class DQN(nn.Module):
        def forward(self, x):
            return self.fc(torch.flatten(self.conv(x), 1))
    DQN.__module__, DQN.__name__, DQN.__qualname__ = "model", "DQN", "DQN"
    if file_sha256(path) != REGISTRY["model_sha256"]:
        raise ValueError("Safe adapter accepts only the exact pinned model bytes.")
    def legacy_scalar(dtype, value):
        if np.dtype(dtype).hasobject:
            raise ValueError("Object numpy scalars are not part of this checkpoint.")
        return np.frombuffer(value, dtype=dtype, count=1)[0]
    legacy_scalar.__module__, legacy_scalar.__name__, legacy_scalar.__qualname__ = "numpy.core.multiarray", "scalar", "scalar"
    allowed = [DQN, nn.Sequential, nn.Conv2d, nn.ReLU, nn.Linear, set,
               legacy_scalar, np.dtype, type(np.dtype("int64")), type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        model = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(model, DQN):
        raise ValueError("Unexpected serialized model type.")
    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(1, 4, 64, 128))
    if tuple(prediction.shape) != (1, 3) or not torch.isfinite(prediction).all():
        raise ValueError("Unexpected original model observation/action contract.")
    return model


def run(out_dir: str | Path, *, episodes=5, max_steps=10000, show=False):
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    import torch
    import numpy as np
    out = Path(out_dir).resolve()
    _verify_existing(out)
    model = _load_original_model(out / "model.pth")
    torch.set_num_threads(1)
    if not show:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # Running this explicit optional command imports the pinned, verified
    # original simulator. The default duel pipeline never imports external code.
    spec = importlib.util.spec_from_file_location("dino_upstream_env", out / "envs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = Path.cwd()
    records = []
    try:
        os.chdir(out)  # Original asset loader uses paths relative to this root.
        env = module.Wrapper(module.Env(render_mode="human" if show else None))
        try:
            for episode in range(episodes):
                obs, _ = env.reset(seed=episode + 1000)
                total, terminated, truncated = 0.0, False, False
                for steps in range(1, max_steps + 1):
                    with torch.no_grad():
                        action = int(model(torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)).argmax(-1))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                records.append({"episode": episode, "seed": episode + 1000, "wrapper_steps": steps,
                                "return": total, "terminated": bool(terminated),
                                "max_steps_reached": not (terminated or truncated)})
                print(json.dumps(records[-1]), flush=True)
        finally:
            env.close()
    finally:
        os.chdir(prior)
    result = {"upstream": REGISTRY, "episodes": records,
              "note": "Original single-player pixel-policy demo. Not duel training or evidence of transfer to our simulator."}
    atomic_json(out / "adapter_evaluation.json", result)
    return result


def upstream_command(operation, out_dir, **kwargs):
    if operation == "info":
        print(json.dumps(REGISTRY, indent=2, ensure_ascii=False))
        return REGISTRY
    if operation == "fetch":
        return fetch(out_dir)
    return run(out_dir, **kwargs)

