#!/usr/bin/env python3
"""Cross-platform one-command installation into an isolated local environment."""
import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "mps", "cuda126", "cuda128", "core"], default="cpu")
    parser.add_argument("--venv", default=".venv")
    parser.add_argument("--test", action="store_true", help="Install pytest and execute the suite")
    parser.add_argument("--upstream", action="store_true", help="Also install optional original-Dino environment dependencies")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error("Use Python 3.11–3.13 (tested here: 3.13) for the pinned dependency set.")
    if args.backend == "mps" and platform.system() != "Darwin":
        parser.error("MPS is for Apple Silicon/macOS. Use cpu or a CUDA backend here.")
    if args.backend.startswith("cuda") and platform.system() == "Darwin":
        parser.error("CUDA wheels do not support macOS. Use mps or cpu.")
    root = Path(__file__).resolve().parent
    target = Path(args.venv).resolve()
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(target)
    def pip(*parts):
        subprocess.run([str(python), "-m", "pip", *parts], check=True)
    pip("install", "--upgrade", "pip")
    pip("install", "-r", str(root / "requirements-core.txt"))
    if args.backend != "core":
        parts = ["install", "torch==2.10.0"]
        if platform.system() != "Darwin":
            suffix = {"cpu": "cpu", "cuda126": "cu126", "cuda128": "cu128"}[args.backend]
            parts += ["--index-url", "https://download.pytorch.org/whl/" + suffix]
        pip(*parts)
    if args.upstream:
        pip("install", "-r", str(root / "requirements-upstream.txt"))
    if args.test:
        pip("install", "pytest>=8,<10")
        subprocess.run([str(python), "-m", "pytest", "-q"], cwd=root, check=True)
    subprocess.run([str(python), str(root / "run.py"), "doctor"], cwd=root, check=True)
    print("\nReady. Run these from the extracted project directory:")
    print(f'  "{python}" run.py demo')
    if args.backend != "core":
        print(f'  "{python}" run.py experiment --config smoke --out runs/smoke')
        print(f'  "{python}" scripts/elimination_suite.py evaluate --out elimination --workers 1')
        print(f'  "{python}" scripts/elimination_suite.py train --out elimination --updates 1500 --workers 1')


if __name__ == "__main__":
    main()

