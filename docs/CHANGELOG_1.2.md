# 1.2 changes

- Added `max_seconds=null`, continuous rollouts, external censor-aware evaluation and honest replay labels.
- Added explicit collision attribution and purchase-type/economy metrics.
- Added multi-horizon action-conditioned JEPA with terminal/reset masks and shared motor residual head.
- Added stable log-softmax motor KL (no softmax-underflow `inf`), separate attack/motor entropy, teacher anchor and auxiliary sub-batches.
- Added common viable-action-distilled motor and disclosed teacher privilege.
- Added configurable wide-obstacle geometry; default values preserve legacy rules.
- Added identical within-seed warm starts with optional attack/value-head reinitialization; source hashes embedded.
- Added fixed-checkpoint three-seed suite, execution-mode checks, cross-seed and historical leagues, distribution shifts and defense-only evaluation.
- Added portable resume tests and environment-state compatibility for older default configs.
- Updated install pins to the CPU runtime actually used here, including Python3.13support. Non-CPU backends have not been tested here.
- Retained abandoned development logs, rather than turning failed/incomplete stages into a positive result.

This release does not implement an inference-time planner or official visualJEPA. Exact completed work is recorded in VALIDATION.json and elimination/comparison.json.
