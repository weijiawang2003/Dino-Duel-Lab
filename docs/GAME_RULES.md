> **Version note (1.2):** This document describes legacy presets unless stated otherwise. For the elimination experiment, use `../README.md` and `ELIMINATION_METHOD.md`; time-limit draws in legacy presets are not censored outcomes in v1.2.

# Game rules and observation contract

This is an original two-track runner inspired by Chrome Dino. It is **not** a
pixel-perfect recreation and does not use Google's game assets or code. Its
purpose is a controlled resource-spending/self-play benchmark. Default actors
receive structured simulator observations: this is not a visual understanding
benchmark and not a test of learning the rules from language.

Numbers below describe raw `EnvConfig` defaults unless stated otherwise. The
provided experiment presets override some of them; each run saves its resolved
configuration, which is the authoritative record of its actual rules.

## Time, movement, and outcome

- Physics uses metres and seconds. Positive vertical coordinates point upward.
  A dinosaur occupies x=0…0.7 m. Standing height is 1.25 m, crouching height
  0.55 m. Speed starts at 7 m/s, increases by 0.025 m/s², and caps at 13 m/s.
  Optional `initial_speed_jitter` samples starting speed uniformly from
  `initial_speed ± initial_speed_jitter` using the episode seed; the same draw
  applies to both tracks and is directly visible in each runner observation.
- A decision holds for four 1/60 s physics ticks (15 decisions/s). Motors are
  0=run, 1=jump, 2=duck. Jump starts only if grounded at the decision boundary;
  holding jump does not give mid-air jumps or automatically jump on landing.
  Initial upward velocity is 8 m/s and gravity 20 m/s². Duck only lowers the
  hit box on the ground; it does not accelerate a fall.
- Both players buy against the same pre-step state. Both then move and collide
  in every physics tick before an outcome is decided. One surviving player wins
  (+1), the other loses (−1). A collision in the same tick is a draw (0 each).
- Default time limit is 90 s and gives a draw. Collision sets `terminated=True`;
  reaching the time limit sets `truncated=True`. These are distinct for training
  bootstrapping. Optional `shaping_coef` adds a zero-sum difference in newly
  passed obstacles; it is zero by default and changes the learning objective.

## Economy and obstacle placement

One metre travelled earns one coin by default. Wallets start empty and cap at
60. Costs are cactus=10, wide cactus=16, low bird=18, high bird=12; attack 0 is
always free wait. A successful purchase starts a 0.65 s cooldown. The public
action mask incorporates wallet, cooldown, and available room on the other
track. Illegal purchases execute as wait and do not spend coins. Malformed or
out-of-range actions raise errors.

`attack_every_n_steps` controls purchase windows separately from movement. It
defaults to 1; at 6, purchases are allowed every sixth decision (every 0.4 s
under default physics), while motor control remains 15 Hz. The first decision
is a purchase window. Between windows the public mask permits only wait;
attempted purchases spend nothing. Slower purchase windows make it easier for a
stochastic learning policy to wait and accumulate enough coins for expensive
obstacles instead of repeatedly spending at the first moment ten coins accrue.
This changes the decision problem and must be identical across treatments.
Cooldowns are rounded up by purchase windows: with 0.4 s windows, a 0.65 s
cooldown permits purchases no closer than 0.8 s, whereas a 0.30 s cooldown permits
0.4 s spacing, subject to wallet and feasibility checks. Configure these two
settings together. The tighter experiment presets can use a 0.30 s cooldown;
conservative calibration can retain the base 0.65 s setting.

| Attack | Width | Bottom y | Height | Basic response |
|---|---:|---:|---:|---|
| Cactus | 0.60 m | 0 | 0.90 m | Jump |
| Wide cactus | 1.40 m | 0 | 0.90 m | Jump with enough clearance |
| Low bird | 0.90 m | 0.85 m | 0.45 m | Duck |
| High bird | 0.90 m | 1.55 m | 0.45 m | Stay on the ground |

New obstacles start at least 22 m in front, and at least `reaction_time ×
max_speed + player_width`. If a track already contains obstacles, the new
obstacle is appended after the farthest one's trailing edge with at least
`min_gap_seconds × max_speed` clear space (default 11.7 m). It must fit fully
inside the 40 m visible horizon. If it cannot, that purchase is masked out.
Natural obstacles follow the same placement rules, with an independent seeded
generator per track and nominal intervals sampled uniformly from 1.8–3.0 s;
when the queue is full, spawning waits. Natural types have probabilities
0.40/0.25/0.25/0.10 in the table's order.

The experiment's **attack-only** configuration sets `natural_obstacles=false`:
every obstacle is bought by the opponent, and an empty track stays empty if the
opponent waits. Natural obstacles are useful for basic-runner bootstrapping and
calibration. They should not silently be mixed into the main attack-only game:
with a finite queue and shared placement rules, bought obstacles can delay
natural generation, so spending can sometimes make an opponent's natural track
easier. This is a different environment with a different incentive structure.

These rules prevent overlaps and surprise spawning at the dinosaur. They are
**not a formal reachability proof** for arbitrary configurations. Default gaps
are deliberately conservative. Making gaps small, raising action repeat or
changing jump physics increases difficulty and can create physically
unavoidable combinations. Keep rules identical across treatments and record
all configuration changes. A perfect frozen runner on conservative tracks can
produce all draws; report that result as a ceiling rather than a JEPA effect.

### Optional feasibility gate for tight obstacle combinations

Set `ensure_reachable=true` to check each proposed obstacle before allowing it.
The checker starts from the target dinosaur's **current** position and vertical
velocity. It branches over legal future run/jump/duck commands, once per
decision interval, and uses the same sub-tick acceleration, gravity, collision
boxes, and ground-only jump rules as the game. Equal `(y,vy)` motor states merge
in a finite dynamic program; this is not a check that the supplied teacher can
survive. At least one surviving motor sequence must exist through the entire
currently revealed queue plus the proposal, or through the remaining episode
time limit if that arrives first. Unaffordable or geometrically blocked attacks
are rejected before the more expensive check, and results are cached within
the current game state. The gate applies equally to bought and natural
obstacles; natural spawns are deferred to decision boundaries when it is on.

This makes tighter settings such as `min_gap_seconds=0.30` useful: a simple
reactive teacher can fail a combination which an anticipatory controller could
survive. A numerical check of finite revealed queues is still not a proof about
all future games. It assumes no additional unknown future spawns; each later
proposal is checked again from the then-current state. A player can make a bad
decision after a feasible spawn and lose. Small conservative collision padding
(1e-9 m) avoids accepting paths that rely solely on floating-point edge contact.
Pathological searches exceeding 4096 decisions or 1024 simultaneous motor
states are rejected, never accepted without checking. Default physics stays far
below these limits.

Motor commands and purchases are chosen simultaneously. The gate asks whether
**some** legal motor sequence exists from the target's pre-step state; it does
not inspect or protect the actual motor command submitted for that same step.
Consequently, a feasible proposal can become unavoidable after the target
commits to an incompatible jump. This is an intentional timing interaction,
not a promise that every currently selected action remains safe. Neither player
sees the other's unexecuted command through the observation or mask. All already
placed queue entries participate in the physical feasibility check, including
entries beyond the actor's three nearest-obstacle feature slots.

Observed development calibration (five seeds, twenty-second games, the supplied
teacher and aggressive scripted attacks): gaps 0.90 and 0.45 produced five
time-limit draws each; gap 0.30 produced collisions around 5–8 s. This is a small
functional calibration, not evidence of a trained algorithm's superiority.

## What actors see

`runner_observations()` is float32 `[2,24]`. For each player:

| Indices | Contents |
|---|---|
| 0–5 | y/3, vertical velocity/12, grounded, ducking, speed/20, alive |
| 6–11 | nearest obstacle: present, x/40, width/3, bottom/3, height/3, kind/4 |
| 12–17 | second obstacle, same fields |
| 18–23 | third obstacle, same fields |

Absent obstacles have all-zero features. Obstacles are ordered by x. A partly
passed obstacle has negative x until its entire width has passed the dinosaur.
The three nearest visible obstacles are exposed; additional queue entries are
not encoded. In particular, future RNG draws and unrevealed future opponent
commands are not available.

`observe()` is float32 `[2,53]`, re-ordered from each player's perspective:

- 0–23: own runner features;
- 24–47: the opponent's runner features;
- 48,49: own and opponent wallet / wallet cap;
- 50,51: own and opponent cooldown / configured cooldown (zero if inactive);
- 52: elapsed time / episode time limit, capped at one.

The attack mask is a separate `[2,5]` boolean array. The supplied motor teacher
uses only its runner features. It has no access to environment internals. Its
time-to-collision thresholds assume default physical dimensions; changing
gravity, jump velocity or dinosaur dimensions may require re-running the
included bootstrap, validating the teacher, or using a learned motor policy.
It is a practical warm start, not an optimal adversarial controller.

## State and reproducibility

`get_state()` contains the full game state and both track RNGs, plus the reset
RNG, in JSON-compatible values. Restoring requires exactly matching game
configuration. Environments never share a module-global random generator.
`reset(seed)` reproduces an episode; `reset()` advances the instance's reset
stream. Paired evaluation plays both seats for each seed because track streams
are independent and may differ in difficulty.

In attack-only mode the natural track streams are unused. With deterministic
policies and zero starting-speed jitter, changing only the environment seed
would repeat the same game. Experiment presets therefore use a nonzero
`initial_speed_jitter` (for example, 7.5 ± 0.5 m/s) to create genuine seeded
initial-condition variation. This does not make different evaluation seeds
independent trained models: score intervals still condition on the particular
frozen policies. Report both starting-condition and training-seed variation.

Snapshots contain x/y/width/height for each obstacle and dinosaur, in the same
units, plus coins, purchases, actions, distance and outcome. Offline replays
render these snapshots; replay code does not implement a second physics engine.
