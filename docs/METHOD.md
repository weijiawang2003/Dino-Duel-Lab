> **Version note (1.2):** This document describes legacy presets unless stated otherwise. For the elimination experiment, use `../README.md` and `ELIMINATION_METHOD.md`; time-limit draws in legacy presets are not censored outcomes in v1.2.

# Learning method and what the comparison can establish

## Scope

The main experiment is an **original, structured-state, two-player benchmark**.
It is not a pixel-identical Chrome Dino clone, and its agents do not consume
screenshots. Both players receive the same 53-feature public observation format:
their own runner state, the opponent's runner state, wallets, cooldowns, and the
match clock. Future random obstacles and the other player's not-yet-executed
action are absent. See `GAME_RULES.md` for feature definitions and game physics.

The `jepa` treatment implements **action-conditioned future-latent auxiliary
learning** alongside PPO. It is a small JEPA-style research implementation,
not the official I-JEPA or V-JEPA architecture or pretrained model. It does not
perform latent-space search or model-predictive control when acting. This is an
intentional, concrete hypothesis: does training the policy's representation to
predict the next public state help attack decisions or joint motor adaptation?

The package contains code to test that hypothesis. Small smoke runs verify
execution; they are not evidence that JEPA is superior.

## Shared starting skill

The built-in observable-state teacher can jump and duck immediately. An original
two-hidden-layer Tanh `RunnerNet` can imitate it; the portable checkpoint uses
NumPy `.npz` arrays and does not need pickle. Both treatments load exactly the
same runner file and the training checkpoint embeds its weights and SHA-256.

With `training.train_motor=false`, this controller is frozen. PPO learns only
the attack decision. An imperfect but competent cloned runner can expose useful
attack weaknesses; the deterministic teacher is also a reference opponent.
If both motor controllers are effectively perfect under the game's legal
spacing, a high draw rate is expected. That is a property of the task, not an
implementation failure or evidence against either learning method.

With `training.train_motor=true`, the motor head is a separate `RunnerNet`
initialized from the same `.npz` checkpoint. A common checkpoint is required;
there is no silent random initialization of motor behavior. PPO then updates
both movement and attack decisions. The motor network receives only the own
runner features; the attack and value network receives the full public state.

Changing physics or observation definitions can invalidate a runner checkpoint.
Its small file size does not imply general transfer to an arbitrary Dino game.
Third-party checkpoints are never silently inserted into this simulator.

## PPO and simultaneous self-play

The acting backbone is an MLP with two Tanh hidden layers, an attack-logit head,
and a scalar value head. The attack space is wait, cactus, wide cactus, low bird,
and high bird. At every decision the environment supplies a legal-action mask;
invalid logits are suppressed before sampling **and before calculating the PPO
likelihood ratio**. Wait is always legal. A motor head supplies noop/jump/duck
when joint motor training is enabled.

For frozen motors, the policy log probability is the attack log probability.
For joint training it is the sum of motor and attack log probabilities. The PPO
clipped objective uses these same sampled action commands and the same stored
action masks. An airborne jump can be a physically ignored command; PPO still
uses the probability of the command actually chosen. The dynamics predictor
instead conditions on the environment's recorded effective commands.

The default task reward is the zero-sum result of the match: winner `+1`, loser
`-1`, simultaneous collision or time-cap draw `0` each. Optional explicit
environment shaping is separately configured and recorded. A match ending at
`max_seconds` is a **scored draw**, so by default its continuation value is zero.

For each seat, the implementation computes

\[
\delta_t = r_t + \gamma (1-d_t)V(s_{t+1}) - V(s_t),\qquad
A_t = \delta_t + \gamma\lambda(1-b_t)A_{t+1}.
\]

Here `b_t` is any reset boundary. For the default finite-match objective `d_t`
also includes a time-cap draw. `training.bootstrap_time_limits=true` is available
for experiments using a continuing-task interpretation, but changes the
objective and should be reported; it is not the preset for scored matches. The
generic GAE helper tests both conventions. It never carries advantages across a
reset. Next observations are captured **before** the environment is reset.

The clipped policy objective, squared value error, entropy term, and optional
auxiliary loss are optimized over shuffled rollout minibatches. Gradient norm
clipping, finite-loss checks, and approximate-KL early stopping are implemented.
Constant learning rate is the default. An optional `lr_decay_updates` defines
an absolute schedule, so extending the requested update count on resume does
not silently alter the previous schedule.

All training episode seeds, including the initial reset even when a large model
seed is supplied, are restricted to `[0, 2**31)`. The experiment defaults reserve
seeds starting at `2**32` for held-out evaluation. Keep these seed domains
separate when supplying custom configurations; bootstrap/calibration seeds used
to choose a model or change game balance are not untouched evaluation data.

Self-play uses one shared current policy for both seats. At episode boundaries,
an optional opponent pool replaces one randomly selected seat with a frozen
historical policy. Those opponent transitions never enter the learner's PPO
minibatches. The active opponent remains fixed until the next episode; the pool
is refreshed every `snapshot_interval` completed updates and bounded by
`opponent_pool_size`. Active historical policies remain recoverable even after
they age out of the sampling pool. Fixed scripted opponents can alternatively
be selected with `opponent_style`.

## JEPA-style auxiliary objective

For the learner's perspective, let

\[
z_t=f_\theta(o_t),\quad
\bar z_{t+1}=\operatorname{sg}(f_{\bar\theta}(o_{t+1})),\quad
\hat z_{t+1}=g_\phi(z_t,a_t^{self},a_t^{other}).
\]

The predictor sees the one-hot encodings of the **recorded** motor and attack
actions of both players. This information is used only as a supervised dynamics
training input after the transition occurs. The acting policy never sees the
opponent's future action. No future latent is supplied to the policy at
inference, and no target gradient flows into the EMA encoder.

The implementation predicts one control-step ahead (four 60-Hz physics ticks
under the default environment configuration). The loss is

\[
L_{aux}=\operatorname{SmoothL1}(\hat z_{t+1},\bar z_{t+1})
+0.1\,\operatorname{mean}_j\max(0,0.5-\sqrt{\operatorname{Var}_B(z_{t,j})+10^{-4}}).
\]

The second term penalizes a low-variance representation. The 0.5 spread target
is suitable for a bounded Tanh encoder; it does not mathematically guarantee
that all forms of representational collapse are avoided. The logs include
`latent_std`, `prediction_loss`, and `variance_loss` so collapse can be checked.
The auxiliary term is skipped if fewer than two valid transitions are in a
minibatch. Collision-terminal transitions are excluded. A physically observed
time-cap transition can still train the one-step representation predictor.

The combined optimization loss is

\[
L=L_{PPO}+c_vL_V-c_eH(\pi)+c_aL_{aux}.
\]

After each optimizer step the target encoder updates as

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

`ema_tau` is the old-target coefficient, not the new-online coefficient. The
default is 0.99. The acting backbone of `baseline` and `jepa` is identical and
starts with identical weights for a matched seed. The extra predictor and
target network incur additional training memory and computation, which the
comparison must report. The sampling RNG is reset after model initialization
so creating auxiliary modules does not change the initial sampling seed.

## Included controls and interpretation

| Variant | Acting model | Auxiliary training |
|---|---|---|
| `baseline` | Standard PPO backbone | None |
| `jepa` | Same backbone and observation | Correct next-latent targets |
| `jepa_shuffled` | Same backbone and observation | Next-latent targets randomly permuted within each minibatch |
| `baseline_large` | Twice the configured hidden width | None |

The larger baseline is a capacity sensitivity control; it is **not exact
parameter or FLOP matching**. The shuffled treatment checks whether useful
temporal correspondence matters, but it is an imperfect control when states
within a minibatch are very similar. Report multiple seeds, held-out match
seeds, both seating orders, fixed opponents, and direct cross-play. Do not choose
hyperparameters on the same held-out seeds used for the reported comparison.

Equal update/rollout/environment counts give equal joint environment-interaction
budgets. Historical-opponent selection can change the number of learner-agent
transitions when episodes have different lengths, so `agent_steps` is logged
separately. For exact learner-transition matching use only live self-play
(`historical_opponent_prob=0`), or report the observed difference. Wall time and
parameter counts are also logged; environment-step matching does not equal
compute matching.

`mean_episode_return` averaged over both seats of a zero-sum match is zero by
construction. It is retained for diagnostic completeness, **not a progress
metric**. Held-out win/draw/loss rates, score, match duration, and opponent
coverage are the appropriate outcome measures. A draw-heavy result requires
task/balance diagnosis, not a claim of equivalent intelligence.

## Checkpoints, stopping, and reproducibility

An experiment copies its shared runner to `shared_runner.npz` in the output
directory and records its SHA-256. Resume uses and verifies that copy, so the
original custom runner file can be removed. The flat `experiment_config.json`
can be passed directly to `--config`; its runner path is resolved relative to
that JSON file, allowing the output directory to move. Explicitly substituting
a different runner during resume is rejected. A standalone training checkpoint
already embeds the runner, so `train --resume` without `--runner` does not reopen
the source NPZ. Evaluation and replay between two checkpoints also do not need
an unused external runner file.

Concurrent writers are excluded with an operating-system advisory lock
(`fcntl.flock` on Linux/macOS, `msvcrt.locking` on Windows). The persistent
`RUNNING.json` file contains diagnostic ownership/release metadata; its
existence does not indicate that a run is active. There are no stale-PID or
hostname heuristics. Releasing the handle releases the lock; the file is not
deleted because unlinking a lock file can permit concurrent owners of different
inodes. As with other advisory locks, all writers must use the project commands
and a filesystem that supports the platform's locking semantics.

`latest.pt` contains the policy, value/predictor/EMA weights, optimizer, complete
environment states, current episode accounting, historical pool, active
opponents, Python/NumPy/PyTorch RNG states, configuration, seed, and counters.
`policy.pt` is the lighter inference artifact and embeds the shared runner.
Numbered checkpoints are saved at the configured checkpoint interval. Files are
written to a temporary file, flushed, and atomically replaced.

SIGINT, SIGTERM, a `STOP` file in the run directory, or the optional
`training.stop_file` request a stop at the next complete update boundary. A
wall-time budget is also checked at those boundaries. A hard kill can lose work
since the most recent checkpoint, but will not intentionally overwrite that
checkpoint with an incomplete PPO update. Remove the relevant STOP file before
resuming. The total `updates` target may be increased; changing physics,
architecture, or learning hyperparameters is rejected rather than silently
changing the experiment. Device, thread count, checkpoint interval, stop path,
and wall-time limit are operational exceptions.

An already-completed resume is a read-only no-op: it does not rewrite model,
configuration, metrics, or summary files. Their hashes remain stable, preserving
the validity of checkpoint-keyed evaluation caches.

On resume, any metrics rows after the checkpoint's update are removed before
new rows are appended. This handles a hard stop after a log write but before
the corresponding checkpoint. Source runner files are no longer required after
the first training checkpoint because their data are embedded.

The supplied tests verify **bit-identical CPU update-boundary resume**, including
historical opponents, for baseline, JEPA, and shuffled-target runs. Cross-device
or cross-version numerical identity is not claimed. CUDA/MPS kernel choices can
also affect reproducibility; record the resolved device and dependency versions.
Checkpoint loading uses `torch.load(..., weights_only=True)` on project-owned
state dictionaries. It never enables unrestricted unpickling for an external
download.

## Primary references

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), 2023. Conceptual reference; this package is not a reproduction of its image architecture or experiments.
- [Gymnasium environment API and termination/truncation distinction](https://gymnasium.farama.org/api/env/).
- [PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html).
