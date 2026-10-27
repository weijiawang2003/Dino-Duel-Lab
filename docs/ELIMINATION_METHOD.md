# Version 1.2: experimental method

## Objective
Compare ordinary PPO and PPO with a JEPA-style action-conditioned latent future prediction auxiliary objective in an untimed, elimination-only adversarial task. A strong common motor prior avoids relearning how to run, jump and duck. The experiment tests representation-assisted attack **and residual defense learning**, not latent-space planning.

## Dynamics and public information
The original 60Hz simulator executes four physics ticks per policy action (15Hz). Both seats act simultaneously. Each policy sees 53 structured features: two24-feature runner observations, wallets, attack cooldowns, and a final time feature. In elimination mode that final feature is zero, not a countdown. Each runner observation exposes at most3 nearest obstacles. Coins accrue with distance. Purchasing sends a visible obstacle to the other track.

Main configuration: speed sampled once per match uniformly in[7,9]m/s and held constant;20 starting coins;60 wallet cap; distance income1coin/m; purchase decision every6policy steps; cooldown0.3s; spawn22m; minimum spacing0.3s times conservative maximum speed13m/s; no environmental obstacles; no speed escalation; no shaping reward. Four purchase costs are10,16,18,12. The wide cactus is2.2m wide/1.2m tall; other geometry is unchanged. Exact values are recorded with every checkpoint.

A deterministic dynamic program rejects purchases for which no safe sequence exists through the currently revealed queue. It uses the actual physics and branches only on action boundaries. Conservative resource guards may reject exotic configurations; they do not admit a known impossible queue. This guard makes hard attacks physically avoidable, not easy for the learned policy. A perfect pair may still survive indefinitely.

## Common motor bootstrapping
`bootstrap_adversarial.py` collects viable-action-set labels on policy rollouts and intervenes when a model action would make the existing queue impossible to clear. The teacher sees the full revealed queue; its student observes the standard24runner features. The soft set-valued objective is `-log sum_{a in viable_set} pi(a|observation)`, rather than penalizing other valid actions. Three rounds of12000joint environment steps produced19850constrained training examples. Those training-fit metrics are not held-out accuracy.

This prior is identical for both treatment arms. A learned residual motor head, fed by the same encoder as the attack head, adds logits to the scaled frozen common motor logits. Thus the learned strategy can alter actual motor decisions without replacing the low-level prior. An anchor KL and a separate small motor entropy coefficient regularize this correction. Attack/value heads are reset in both arms; encoder initialization comes from the same oldbaseline checkpoint within each seed.

## Auxiliary prediction
For h in{8,32,64}, a GRU encodes the factual sequence of both players' motor and attack actions over h steps. A predictor takes current online latent state and this action-sequence representation and predicts the EMA encoder's latent representation of the h-step-ahead observation. The EMA side is stop-gradient. A variance floor term discourages collapse.

Targets are built from contiguous rollout positions before minibatch shuffling. A terminal endpoint is allowed; a target with any intervening termination or truncation is invalid. Unavailable rollout tails are masked, not padded with training targets. Factual future actions are training labels, not available to the policy at the current step. The model does not hallucinate actions at inference and does not plan through the predictor.

Only a random subset of64valid learner samples per PPO minibatch receives auxiliary computation. The baseline uses the same acting model without this loss. Training samples, optimizer steps and wall time are logged; equal environment budgets do not imply equal computation.

## Termination, bootstrapping, and censoring
A PPO rollout is a computational block, not a match. Live game states carry across blocks and across saved/resumed training. An actual collision emits win/loss terminal reward. A simultaneous collision is a true draw. No fixed elapsed game time ends main training episodes.

Evaluation uses an external finite observation budget to keep runs operational. Exhaustion marks the result censored, leaves `terminated=false` and `truncated=false`, assigns no score and no winner. Report complete outcome bounds, not artificially imputed ties. Repeating the same match with a longer budget reproduces its prefix and is not a new independent trial.

## Comparison
Three matched training seeds11,23,47; final update300;8environments×256steps per update=614400new joint environment steps/model. Main execution samples motors and attacks. Other execution modes are reported separately. Test environment seeds start at2^36, disjoint from development calibration seeds. Seat-swapping preserves policy-identity RNG seeds. The final-checkpoint rule is fixed before held-out evaluation; no checkpoint is selected for the most favorableJEPA score.

Additional diagnostics include fixed strong-common-motor opponents, held-out attack programs, oldv1.1models, initial policies, cross-training-seed comparisons, history league, changed speed/economy/spacing, and defense-only tests. They answer different questions and should not be pooled into one overall win rate.

Development changes, including failed numerical work and a saturated original geometry, are disclosed separately. Seed11's first40updates were used as an exploratory same-configuration pilot and retained. The benchmark is not a preregistration and only has three training seeds. No significance claim is justified merely by a large count of evaluation games.

## Sources
- Gymnasium: Handling Time Limits; Farama: terminated/truncated step API. The code's external observation censoring is an additional evaluation convention, not a game terminal.
- V-JEPA2 paper(arXiv2506.09985) and official facebookresearch/vjepa2 repository provide relevant background on latent predictive learning. The small structured-state architecture here is custom; it is not an official V-JEPA2 reproduction.
- See original project attribution in `docs/THIRD_PARTY.md`; main simulator and small neural controllers are local project code, not an imported Chrome Dino asset.

### Replication units and symmetry
The main arena is symmetric and has no independent natural-obstacle streams.
With policy-identity RNG held fixed, swapping seats can produce exactly mirrored
trajectories. Seat swaps are an implementation fairness check, not two independent
samples. Report uncertainty uses the whole pair as one environment-seed unit.
The initial defense-only policy is the same common motor across seeds; its
reported defense benchmark uses only the seed11copy, rather than treating three
identical initial motor controllers as independent replications. Extra recorded
redundant diagnostic rows, if present, are not included in that summary.

### Forecast diagnostic
A separate held-out trajectory set starts at environment seed2^37. Under frozen
policies, record actual future joint actions and observations without online
updates. Sample up to512starts per training seed and mask unavailable horizons.
Compare Smooth-L1 error with target-encoder latent persistence and a permutation
of factual future action sequences. This is an oracle-action conditional
prediction check, not a deployment/causal planning experiment. Report per-seed
metrics in their own latent spaces; overlapping windows are correlated.

`mean_seconds` and duration quantiles are observed durations, including
right-censored observation windows. They must not be described as uncensored
average/full match durations. The legacy `physics_steps` training counter is
a nominal action-repeat multiplier, not an exact count of substeps executed
before collisions within the last action interval.
