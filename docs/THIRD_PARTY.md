# Sources, dependencies, and provenance

## Original material shipped here

The simulator, teacher, training/evaluation code, original canvas graphics,
documentation, and bundled `runner_bc.npz` were created for this project.
The neural runner was actually trained by behavior cloning and DAgger in the
project's own state-observation environment. Its provenance and calibration
records are included under `assets/`.

No third-party Chrome Dino source code, sprites, or pretrained weights are
redistributed in the default archive. The main project uses the MIT license in
`LICENSE`.

## Optional original-game adapter

- Repository: https://github.com/aome510/chrome-dino-game-rl
- Inspected commit: `8a536f6f4e39280a08b70a575cf19c4f1f1c7015`
- Archive SHA-256: `8ebff017f4112d8d0c2229e2328abe31d48e3cc3c7115fe5a5866a708fdf89b1`
- Original `model.pth` SHA-256: `cd0f44c3aef59fb125e2ce3ff9ee637b9ab07a0c93bfaa4011ff58d8493fde58`
- Original model bytes: 3,468,319.
- Its `pyproject.toml` declares MIT. No separate LICENSE file was present at
  the inspected revision. Its README attributes sprites to another Dino repo;
  consult the original sources for their applicable terms.

`upstream fetch` is explicit and optional. It downloads from the original
repository to a user-selected directory, checks bytes, and records provenance.
The adapter preserves the source environment and visual observation contract;
it does not claim transfer into the duel simulator. The loader uses a narrow
local class/type allowlist and verifies the exact known checkpoint checksum.
It never automatically falls back to `weights_only=False`.

## Research references

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
  https://arxiv.org/abs/1707.06347
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture*, 2023. https://arxiv.org/abs/2301.08243
- Bardes et al., *VICReg: Variance-Invariance-Covariance Regularization for
  Self-Supervised Learning*, 2021. https://arxiv.org/abs/2105.04906
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, 2011. https://arxiv.org/abs/1011.0686

The project uses a variance penalty inspired by representation-learning work;
it does not implement the full VICReg loss. Its JEPA-style objective predicts
multi-horizon structured-state embeddings in v1.2 (one-step in the original release) and is not an official I-JEPA/V-JEPA
reproduction. Its DAgger process aggregates teacher labels on mixed student
trajectories, without claiming the finite-sample guarantees of the paper.

## Offline report appearance

Version1.2 uses system fonts only. No font binary or base64-embedded font is
included in this release; legacy font-embedding report files are omitted.

## Runtime dependencies

Core: NumPy. Learning: PyTorch. Optional external original-game environment:
Gymnasium, Pygame, Pillow. Testing: pytest. Dependencies retain their own
licenses and are installed separately, not bundled as binaries in this archive.
Exact versions used during verification are recorded in `VALIDATION.json`.

