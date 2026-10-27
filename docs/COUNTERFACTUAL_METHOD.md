# v1.3: Does forecasting improve attack selection?

## Scope

This release adds an executed counterfactual experiment, not a claim of another
large PPO self-play run. The source is the **69.4 MB v1.2 elimination branch**, ZIP
SHA-256 `b691398d11748cc28cbfa8cfafa971c6fe8629a39332ecf9c2cd223cb64eb9d7`.
The alternative `-elimination-trained.zip` branch is not used. Old trained
policies and optimizer checkpoints remain unchanged.

The game still ends only on collision. The eight-second horizon below is a
prediction/measurement window, **not a time limit added to the game**. Ongoing,
win, loss, and simultaneous collision are four distinct labels. An absorbed
terminal branch never resets into another episode.

## First repair: nonlinear interaction

`ActorCritic(predictor_fusion="nonlinear")` supports a two-layer nonlinear fusion
of current state and action-sequence context. Old checkpoints default to
`"linear"` for exact compatibility. `configs/elimination_nonlinear.json` is a
fresh-training configuration; do not transplant a linear predictor's optimizer
into the changed architecture. Tests explicitly verify that differences between
candidate future predictions can depend on current state.

The offline macro models below are separate predictors. They require only the
current public observation, current legal-purchase mask, and an intended own
attack macro. No actual future motor action, actual opponent action, policy ID,
full simulator snapshot, seed, or branch outcome enters their inputs.

## Frozen source policies and data

The controlled agent is the existing `baseline_seed11` PPO for *every* method.
Its motor is greedy; its default attacks are sampled. Training and in-population
validation/test use frozen opponents baseline/JEPA seeds 11 and 23. Opponents
with seed 47 are held out from fitting the new predictors and tested separately.
They are not unseen game rules or wholly different policy families.

One eligible snapshot is reservoir-sampled from each independently seeded
source episode (up to 24 seconds). Only snapshots at purchase boundaries after
0.8 seconds and with at least one legal purchase enter the experiment. Thus the
population is **attack-eligible snapshots**, not arbitrary states or all games.
Source-episode groups never cross training, validation, or test. The full
snapshot is retained only for replay and offline counterfactual simulation.

The recorded run uses 512 training, 96 validation, 128 test and 128 held-out-
opponent snapshots. Each has the same 12 planned attack macros. Training uses
three independent future random tapes per macro; other splits use eight.
All macros in a snapshot/repetition share exogenous uniforms indexed by absolute
step and seat, including forced actions, so differences in legal masks do not
shift later random draws. Both policies recalculate their actions from the
new state in every branch. They do not repeat logged future actions.

A macro specifies eight purchase opportunities (3.2 seconds) of intended attacks.
An illegal request becomes wait. Token -1 means consult the original PPO *at the
future state*. After eight opportunities, PPO resumes. Targets are public
observations and event classes at 2, 4, and 8 seconds (or the absorbed terminal
state). A wait macro is a genuine candidate. All rules, costs and reachability
checks are unchanged.

## Fitted models

Four conditions share the same data, intended-action inputs and optimization
budget, with downstream fitting seeds 101, 202 and 303:

* `latent_additive`: additive state/action fusion, predicts frozen JEPA features.
* `latent_nonlinear`: nonlinear interaction, predicts frozen JEPA features.
* `state_nonlinear`: nonlinear predictor of actual structured future state.
* `outcome_only`: same architecture as latent_nonlinear without future-feature
  prediction loss; a direct outcome-learning control.

Latent targets come from the **frozen pretrained JEPA seed-11 encoder**. No new
EMA target or online actor is trained here. This is a frozen-representation
JEPA-style forecasting test; it must not be called a newly trained full JEPA
agent. The state-prediction control does not use this JEPA encoder. The latent
conditions inherit its prior training, so this is not an equal total-lifetime-
compute architecture benchmark.

The input and target normalization use training data only. A small outcome
probe is trained on true future target features, selected by validation loss,
and frozen. Predictors minimize normalized Huber future-feature error plus
outcome cross entropy through this probe; `outcome_only` omits the Huber term.
The probe and outcome supervision are explicit task-specific components, not
self-supervision-only training. Different target dimensions entail different
parameter counts, reported in the results. There are no equal-FLOP claims.

Each predictor has a fixed 60-epoch budget; the checkpoint minimizing validation
8-second outcome NLL is selected. Neither test split selects checkpoints,
hyperparameters, model family, or examples for training. Ensembles average the
three fitting seeds' probabilities. These are **not** three new independent PPO
training seeds.

## Action-selection endpoint

Every fitted model ranks the same 12 macros using

    predicted P(win by 8 seconds) - predicted P(loss by 8 seconds).

This is finite-window net decisive-event probability, **not full-game win rate**
and not a score that labels unresolved games as draws. Outcome tables always
report the ongoing class separately. This endpoint need not reward saving for
an opportunity more than eight seconds away.

For an empirical simulator selector, the first four random tapes pick the
best macro; the last four estimate its performance. All primary learned-selector
comparisons also use those last four tapes, paired against original PPO
continuation (candidate 0). Sensitivity results use all eight for selectors that
never saw outcomes. The best fixed macro is chosen from validation, and a
uniform random macro has its exact finite-candidate average evaluated.
The same-sample maximum is labeled an *optimistic ceiling*, not performance.

Uncertainty uses 5,000 paired bootstraps of source-episode roots. Candidate pairs,
branches and random tapes are not independent training repetitions. Intervals
are unadjusted exploratory intervals, conditional on the frozen source policies,
training data and fitted predictors. They do not establish a general architecture
advantage. Macro ranking accuracy only uses candidate pairs whose measured net
outcomes differ by at least 0.25, and is averaged within roots first.

## Optional live control

`PlanningPolicy` is a real policy wrapper, with no simulator or opponent handle.
It evaluates all candidates using only current observation/mask, executes the
first requested attack, and replans at the next 0.4-second opportunity. Running
motors are unchanged. A longer macro commitment is supported as a separate
explicit protocol; the default is one opportunity.

A full-duel pilot uses fresh seeds, both seats, and a fixed common opponent set.
Its external observation budget is separate from game termination; unfinished
matches remain unresolved. Replanning changes the future relative to a committed
macro, and a predictor calibrated on baseline-policy trajectories can fail when
its own decisions change the state distribution. A good offline result therefore
does not establish good closed-loop performance. No model family is silently
selected after seeing these pilot outcomes.

## Artifacts and reproducibility

`PROTOCOL.json` freezes counts, rules, source hashes, candidate plans and intended
analysis. Each root has an `.npz` with labels and a `.json` with exact snapshot
and seeds. Collection safely resumes per root. Models include `best.pt`,
optimizer/RNG-bearing `latest.pt`, validation history, and summaries. All .pt
loads use `weights_only=True`; .npz arrays load without pickle.

Reproducibility scripts are in `scripts/`. Reports keep offline local decisions
and full-duel outcomes separate. Paired replay cases are illustrative and have
explicit selection rules, including a failure case when one exists.

## Observed failure and post-hoc commitment diagnostic

The executed first-action replanning pilot selected wait on every learned-planner
call and made zero purchases. Each of the three learned methods won 0, lost 26,
and remained unresolved in 38 of its 64 games. The unplanned PPO won 22, lost 26,
and was unresolved in 16. These failed results are retained in `online_commit1`.

The offline wait macro has a crucial continuation: wait for 3.2 seconds, THEN
use reactive PPO for 4.8 seconds. Replanning every 0.4 seconds can repeatedly
postpone that continuation. Thus the evaluated plan is not the plan executed.
This temporal inconsistency is an observed deployment failure of this prototype,
not a general theorem about JEPA or model-predictive control.

A separately declared **post-hoc** diagnostic uses new environment seeds
(starting at 2**49), unchanged model weights, and 20 purchase opportunities of
commitment. It executes the 8 explicit macro tokens and then 12 PPO tokens.
Only after that 8-second block does it select another plan. It also compares a
fixed wait/PPO schedule with no forecasting, and ordinary state forecasting.
The original prospective pilot and this repair branch remain separate, including
separate protocols, CSVs and summaries. This is not a new game time limit.

Offline held-out comparisons likewise show that a fixed wait macro matches or
slightly exceeds learned selection. Latent and outcome-only ensembles make the
same choices. This experiment therefore does not establish JEPA-specific
state-dependent attack selection, even where the local metric improves.
