"""Original deterministic, display-free two-player Dino simulator.

Coordinates use metres, seconds and an upward vertical axis.  The actor sees
structured *public* state, not pixels.  No game engine, browser, or torch is
needed here.  See docs/GAME_RULES.md for the exact observation and timing rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from copy import deepcopy
import math
from typing import Any

import numpy as np

RUNNER_DIM = 24
OBS_DIM = 53
N_ATTACKS = 5
OBS_HORIZON = 40.0
ACTION_NAMES = ("wait", "cactus", "wide_cactus", "low_bird", "high_bird")
# kind -> width, bottom, height; original geometric objects, not game assets.
OBSTACLE_SPECS = {
    1: (0.60, 0.00, 0.90),
    2: (1.40, 0.00, 0.90),
    3: (0.90, 0.85, 0.45),
    4: (0.90, 1.55, 0.45),
}


@dataclass
class EnvConfig:
    dt: float = 1.0 / 60.0
    action_repeat: int = 4
    initial_speed: float = 7.0
    initial_speed_jitter: float = 0.0
    max_speed: float = 13.0
    acceleration: float = 0.025
    max_seconds: float | None = 90.0
    gravity: float = 20.0
    jump_velocity: float = 8.0
    player_width: float = 0.7
    standing_height: float = 1.25
    duck_height: float = 0.55
    coins_per_meter: float = 1.0
    initial_coins: float = 0.0
    wallet_cap: float = 60.0
    attack_cooldown: float = 0.65
    attack_every_n_steps: int = 1
    attack_costs: tuple[float, ...] = (0.0, 10.0, 16.0, 18.0, 12.0)
    spawn_distance: float = 22.0
    reaction_time: float = 0.8
    min_gap_seconds: float = 0.9
    natural_interval_min: float = 1.8
    natural_interval_max: float = 3.0
    natural_obstacles: bool = True
    shaping_coef: float = 0.0
    ensure_reachable: bool = False
    wide_cactus_width: float = 1.4
    wide_cactus_height: float = 0.9

    def __post_init__(self) -> None:
        self.attack_costs = tuple(float(x) for x in self.attack_costs)
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{f.name} must be finite")
        if not 0 < self.dt <= 1 / 30:
            raise ValueError("dt must be in (0, 1/30] seconds")
        if isinstance(self.action_repeat, bool) or int(self.action_repeat) != self.action_repeat or not 1 <= self.action_repeat <= 12:
            raise ValueError("action_repeat must be an integer from 1 to 12")
        self.action_repeat = int(self.action_repeat)
        if (isinstance(self.attack_every_n_steps, bool)
                or int(self.attack_every_n_steps) != self.attack_every_n_steps
                or self.attack_every_n_steps < 1):
            raise ValueError("attack_every_n_steps must be a positive integer")
        self.attack_every_n_steps = int(self.attack_every_n_steps)
        if not 0 < self.initial_speed <= self.max_speed <= 20:
            raise ValueError("require 0 < initial_speed <= max_speed <= 20")
        if (self.initial_speed_jitter < 0 or self.initial_speed - self.initial_speed_jitter <= 0
                or self.initial_speed + self.initial_speed_jitter > self.max_speed):
            raise ValueError("initial_speed +/- initial_speed_jitter must stay in (0,max_speed]")
        if self.acceleration < 0 or (self.max_seconds is not None and self.max_seconds <= 0):
            raise ValueError("acceleration must be nonnegative; max_seconds must be positive or null (elimination)")
        if not 0 < self.duck_height < 0.85 < self.standing_height < 1.55:
            raise ValueError("heights must permit ducking low birds and standing under high birds")
        if not 0.2 <= self.player_width <= 1.2:
            raise ValueError("player_width must be in [0.2,1.2]")
        if self.gravity <= 0 or self.jump_velocity <= 0 or self.jump_velocity ** 2 / (2 * self.gravity) <= 0.95:
            raise ValueError("jump must clear a 0.9 m cactus")
        if self.jump_velocity > 12 or self.jump_velocity ** 2 / (2 * self.gravity) > 3:
            raise ValueError("jump exceeds the observation's supported vertical range")
        if self.coins_per_meter < 0 or self.wallet_cap <= 0 or not 0 <= self.initial_coins <= self.wallet_cap:
            raise ValueError("invalid coin or wallet settings")
        if len(self.attack_costs) != N_ATTACKS or self.attack_costs[0] != 0 or any(not math.isfinite(v) or v <= 0 for v in self.attack_costs[1:]):
            raise ValueError("attack_costs must contain wait=0 and four positive finite costs")
        if self.attack_cooldown < 0 or self.reaction_time < self.dt * self.action_repeat:
            raise ValueError("cooldown must be nonnegative; reaction_time must cover a decision interval")
        if self.min_gap_seconds < 0.1:
            raise ValueError("min_gap_seconds must be >= 0.1")
        if not self.player_width < self.spawn_distance < OBS_HORIZON - 2:
            raise ValueError("spawn_distance must fit in the public observation horizon")
        if self.max_speed * self.reaction_time + self.player_width >= OBS_HORIZON - 2:
            raise ValueError("reaction-time spawn cannot fit in the visible horizon")
        if not 0 < self.natural_interval_min <= self.natural_interval_max:
            raise ValueError("natural intervals must be positive and ordered")
        if not 0.6 <= self.wide_cactus_width <= 3.0:
            raise ValueError("wide cactus width must be in [0.6,3.0]")
        if not 0.5 <= self.wide_cactus_height < self.jump_velocity**2/(2*self.gravity):
            raise ValueError("wide cactus height must be below jump apex")
        if self.shaping_coef < 0:
            raise ValueError("shaping_coef must be nonnegative")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "EnvConfig":
        return cls(**(value or {}))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["attack_costs"] = list(self.attack_costs)
        return result


class DuelEnv:
    """Two seats step simultaneously; all mutable state belongs to this instance."""

    metadata = {"render_modes": [], "render_fps": 15}

    def __init__(self, config: EnvConfig | dict | None = None):
        self.config = config if isinstance(config, EnvConfig) else EnvConfig.from_dict(config)
        self._reset_rng = np.random.default_rng()
        self._track_rngs = [np.random.default_rng(), np.random.default_rng()]
        self._initialized = False
        self._done = False
        self._reachability_cache: dict[tuple[int, int, float], bool] = {}
        self._motor_transition_cache: dict[tuple, tuple] = {}
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            if isinstance(seed, bool) or int(seed) != seed or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            self._reset_rng = np.random.default_rng(int(seed))
            self.seed = int(seed)
        else:
            self.seed = int(self._reset_rng.integers(0, 2**63 - 1))
        streams = np.random.SeedSequence(self.seed).spawn(2)
        self._track_rngs = [np.random.default_rng(stream) for stream in streams]
        c = self.config
        self.time = 0.0
        self.ticks = 0
        self.speed = (float(self._reset_rng.uniform(c.initial_speed - c.initial_speed_jitter,
                                                    c.initial_speed + c.initial_speed_jitter))
                      if c.initial_speed_jitter else c.initial_speed)
        self.distance = [0.0, 0.0]
        self.players = [dict(y=0.0, vy=0.0, duck=False, alive=True, coins=c.initial_coins,
                             cooldown=0.0, passed=0) for _ in range(2)]
        self.tracks: list[list[dict]] = [[], []]
        self.natural_timers = [0.0, 0.0]
        self.purchases = [0, 0]
        self.purchase_counts = [[0] * N_ATTACKS for _ in range(2)]
        self.invalid_attacks = [0, 0]
        self.next_obstacle_id = 0
        self.collision_sources = [None, None]
        self.winner = None
        self.reason = "running"
        self._done = False
        self._initialized = True
        self.last_actions = [[0, 0], [0, 0]]
        self._reachability_cache.clear()
        if c.natural_obstacles:
            for seat in range(2):
                self._spawn_natural(seat)
        return self.observe(), self._info()

    def _obstacle_spec(self, kind: int):
        if kind == 2:
            return (self.config.wide_cactus_width, 0.0, self.config.wide_cactus_height)
        return OBSTACLE_SPECS[kind]

    def _placement_x(self, target: int, kind: int) -> float | None:
        c = self.config
        x = max(c.spawn_distance, c.max_speed * c.reaction_time + c.player_width)
        # A conservative gap is measured at the maximum possible game speed.
        # It separates hit boxes; it is not a reachability solver.
        if self.tracks[target]:
            end = max(o["x"] + o["width"] for o in self.tracks[target])
            x = max(x, end + c.max_speed * c.min_gap_seconds)
        if x + self._obstacle_spec(kind)[0] > OBS_HORIZON:
            return None
        return float(x)

    def _spawn(self, target: int, kind: int, source: str, x: float) -> None:
        width, y, height = self._obstacle_spec(kind)
        self.tracks[target].append(dict(id=self.next_obstacle_id, kind=kind,
                                       x=float(x), y=y, width=width, height=height,
                                       source=source))
        self.next_obstacle_id += 1
        self.tracks[target].sort(key=lambda o: (o["x"], o["id"]))
        self._reachability_cache.clear()

    def _placement_allowed(self, target: int, kind: int, x: float) -> bool:
        if not self.config.ensure_reachable:
            return True
        key = (target, kind, x)
        if key not in self._reachability_cache:
            self._reachability_cache[key] = self._known_queue_reachable(target, kind, x)
        return self._reachability_cache[key]

    def _known_queue_reachable(self, target: int, kind: int, x: float) -> bool:
        """Existence of a legal future motor sequence for the revealed queue.

        A finite dynamic program branches only at decision boundaries and uses
        exactly the simulator's sub-tick jump/duck integration. Grounded jump
        starts are identical, so equal airborne phases merge exactly: no beam
        pruning, interpolation, or heuristic teacher is used. The test assumes
        no additional future obstacle spawns and continues until the ENTIRE queue has passed, never the episode clock.
        """
        c = self.config
        p = self.players[target]
        if not p["alive"]:
            return False
        width, bottom, height = self._obstacle_spec(kind)
        queue = [(float(o["x"]), float(o["width"]), float(o["y"]), float(o["height"]))
                 for o in self.tracks[target]]
        queue.append((x, width, bottom, height))
        farthest = max(front + w for front, w, _, _ in queue)
        # Duck does not carry a constraint between decisions: every new action
        # explicitly sets it, so (y,vy) is a sufficient motor state here.
        states = {(float(p["y"]), float(p["vy"]))}
        future_speed, future_time, travelled = self.speed, self.time, 0.0
        stable_free = False
        full_interval = (c.dt,) * c.action_repeat
        for _ in range(4096):
            schedule = []
            finished = False
            for _ in range(c.action_repeat):
                dt = c.dt
                future_speed = min(c.max_speed, future_speed + c.acceleration * dt)
                travelled += future_speed * dt
                future_time += dt
                # Tiny conservative contact padding avoids approving a sequence
                # that exists only through floating-point boundary roundoff.
                overlapping = [(y, h) for front, w, y, h in queue
                               if front - travelled < c.player_width + 1e-9
                               and front + w - travelled > -1e-9]
                schedule.append((dt, overlapping))
                if travelled > farthest + 1e-9:
                    finished = True
                    break
            if not schedule:
                return bool(states)
            dts = tuple(dt for dt, _ in schedule)
            collision_ticks = [(j, overlaps) for j, (_, overlaps) in enumerate(schedule) if overlaps]
            # Once all reachable free-flight phases have been attained, more
            # empty full decision intervals leave that finite set unchanged.
            if stable_free and not collision_ticks and dts == full_interval:
                if finished:
                    return True
                continue
            next_states = set()
            for old_y, old_vy in states:
                grounded = old_y <= 1e-9
                for motor in ((0, 1, 2) if grounded else (0, 2)):
                    end_state, trajectory = self._motor_transition(old_y, old_vy, motor, dts)
                    valid = True
                    for j, overlapping in collision_ticks:
                        y, body_height = trajectory[j]
                        if any(y < bottom + height + 1e-9 and y + body_height > bottom - 1e-9
                               for bottom, height in overlapping):
                            valid = False
                            break
                    if valid:
                        next_states.add(end_state)
            if not next_states:
                return False
            if finished:
                return True
            stable_free = not collision_ticks and dts == full_interval and states == next_states
            states = next_states
            # Only pathological non-default physics can reach these resource
            # guards. Reject conservatively rather than accepting an unchecked
            # queue or allowing unbounded placement work.
            if len(states) > 1024:
                return False
        return False

    def viable_motor_mask(self, target: int) -> np.ndarray:
        """Exact viable first actions for the currently revealed queue.

        Used ONLY as a common offline demonstration teacher, not by learned
        policies during evaluation. It has no knowledge of future purchases.

        A finite dynamic program branches only at decision boundaries and uses
        exactly the simulator's sub-tick jump/duck integration. Grounded jump
        starts are identical, so equal airborne phases merge exactly: no beam
        pruning, interpolation, or heuristic teacher is used. The test assumes
        no additional future obstacle spawns and continues until the ENTIRE queue has passed, never the episode clock.
        """
        c = self.config
        p = self.players[target]
        if not p["alive"]:
            return np.zeros(3,dtype=bool)
        queue = [(float(o["x"]), float(o["width"]), float(o["y"]), float(o["height"]))
                 for o in self.tracks[target]]
        if not queue:
            return np.ones(3,dtype=bool)
        farthest = max(front + w for front, w, _, _ in queue)
        # Duck does not carry a constraint between decisions: every new action
        # explicitly sets it, so (y,vy) is a sufficient motor state here.
        states = {(float(p["y"]), float(p["vy"])): 7}
        future_speed, future_time, travelled = self.speed, self.time, 0.0
        stable_free = False
        full_interval = (c.dt,) * c.action_repeat
        for decision in range(4096):
            schedule = []
            finished = False
            for _ in range(c.action_repeat):
                dt = c.dt
                future_speed = min(c.max_speed, future_speed + c.acceleration * dt)
                travelled += future_speed * dt
                future_time += dt
                # Tiny conservative contact padding avoids approving a sequence
                # that exists only through floating-point boundary roundoff.
                overlapping = [(y, h) for front, w, y, h in queue
                               if front - travelled < c.player_width + 1e-9
                               and front + w - travelled > -1e-9]
                schedule.append((dt, overlapping))
                if travelled > farthest + 1e-9:
                    finished = True
                    break
            if not schedule:
                return np.array([any(bits & (1<<a) for bits in states.values()) for a in range(3)],dtype=bool)
            dts = tuple(dt for dt, _ in schedule)
            collision_ticks = [(j, overlaps) for j, (_, overlaps) in enumerate(schedule) if overlaps]
            # Once all reachable free-flight phases have been attained, more
            # empty full decision intervals leave that finite set unchanged.
            if stable_free and not collision_ticks and dts == full_interval:
                if finished:
                    return np.array([any(bits & (1<<a) for bits in states.values()) for a in range(3)],dtype=bool)
                continue
            next_states = {}
            for (old_y, old_vy), origins in states.items():
                grounded = old_y <= 1e-9
                for motor in (0,1,2):
                    end_state, trajectory = self._motor_transition(old_y, old_vy, motor, dts)
                    valid = True
                    for j, overlapping in collision_ticks:
                        y, body_height = trajectory[j]
                        if any(y < bottom + height + 1e-9 and y + body_height > bottom - 1e-9
                               for bottom, height in overlapping):
                            valid = False
                            break
                    if valid:
                        next_states[end_state] = next_states.get(end_state,0) | ((1<<motor) if decision == 0 else origins)
            if not next_states:
                return np.zeros(3,dtype=bool)
            if finished:
                return np.array([any(bits & (1<<a) for bits in next_states.values()) for a in range(3)],dtype=bool)
            stable_free = not collision_ticks and dts == full_interval and states == next_states
            states = next_states
            # Only pathological non-default physics can reach these resource
            # guards. Reject conservatively rather than accepting an unchecked
            # queue or allowing unbounded placement work.
            if len(states) > 1024:
                return np.zeros(3,dtype=bool)
        return np.zeros(3,dtype=bool)

    def _motor_transition(self, y: float, vy: float, motor: int, dts: tuple) -> tuple:
        """Exact vertical transition reused across proposed-obstacle checks."""
        key = (y, vy, motor, dts)
        cached = self._motor_transition_cache.get(key)
        if cached is not None:
            return cached
        c = self.config
        grounded = y <= 1e-9
        duck = bool(motor == 2 and grounded)
        if motor == 1 and grounded:
            vy, duck = c.jump_velocity, False
        trajectory = []
        for dt in dts:
            if y > 0 or vy > 0:
                y += vy * dt - 0.5 * c.gravity * dt * dt
                vy -= c.gravity * dt
                if y <= 0:
                    y, vy, duck = 0.0, 0.0, motor == 2
            trajectory.append((y, c.duck_height if duck else c.standing_height))
        result = ((y, vy), tuple(trajectory))
        if len(self._motor_transition_cache) >= 10_000:
            self._motor_transition_cache.clear()
        self._motor_transition_cache[key] = result
        return result

    def _spawn_natural(self, target: int) -> None:
        rng = self._track_rngs[target]
        # Select the kind only once room is available; failed attempts do not
        # consume random draws, keeping schedules reproducible.
        x = self._placement_x(target, 2)
        if x is None:
            self.natural_timers[target] = 0.1
            return
        kinds, probabilities = [1, 2, 3, 4], [0.40, 0.25, 0.25, 0.10]
        if self.config.ensure_reachable:
            feasible = [(kind, probability) for kind, probability in zip(kinds, probabilities)
                        if self._placement_allowed(target, kind, x)]
            if not feasible:
                self.natural_timers[target] = 0.1
                return
            kinds = [item[0] for item in feasible]
            probabilities = np.array([item[1] for item in feasible], dtype=float)
            probabilities /= probabilities.sum()
        kind = int(rng.choice(kinds, p=probabilities))
        self._spawn(target, kind, "natural", x)
        self.natural_timers[target] = float(rng.uniform(self.config.natural_interval_min,
                                                        self.config.natural_interval_max))

    def attack_masks(self) -> np.ndarray:
        masks = np.zeros((2, N_ATTACKS), dtype=bool)
        masks[:, 0] = True
        if self._done:
            return masks
        if (self.ticks // self.config.action_repeat) % self.config.attack_every_n_steps:
            return masks
        for seat, p in enumerate(self.players):
            if not p["alive"] or p["cooldown"] > 1e-9:
                continue
            for kind in range(1, N_ATTACKS):
                if p["coins"] + 1e-9 < self.config.attack_costs[kind]:
                    continue
                x = self._placement_x(1 - seat, kind)
                masks[seat, kind] = (x is not None and self._placement_allowed(1 - seat, kind, x))
        return masks

    def runner_observations(self) -> np.ndarray:
        out = np.zeros((2, RUNNER_DIM), dtype=np.float32)
        for seat, p in enumerate(self.players):
            out[seat, :6] = (p["y"] / 3.0, p["vy"] / 12.0, p["y"] <= 1e-9,
                             p["duck"], self.speed / 20.0, p["alive"])
            visible = [o for o in self.tracks[seat] if o["x"] < OBS_HORIZON and o["x"] + o["width"] >= 0]
            for j, o in enumerate(visible[:3]):
                out[seat, 6 + j * 6:12 + j * 6] = (1.0, o["x"] / OBS_HORIZON,
                                                      o["width"] / 3.0, o["y"] / 3.0,
                                                      o["height"] / 3.0, o["kind"] / 4.0)
        return out

    def observe(self) -> np.ndarray:
        runner = self.runner_observations()
        out = np.zeros((2, OBS_DIM), dtype=np.float32)
        for seat in range(2):
            other = 1 - seat
            out[seat, :RUNNER_DIM] = runner[seat]
            out[seat, RUNNER_DIM:2 * RUNNER_DIM] = runner[other]
            out[seat, 48:] = (self.players[seat]["coins"] / self.config.wallet_cap,
                               self.players[other]["coins"] / self.config.wallet_cap,
                               self.players[seat]["cooldown"] / max(self.config.attack_cooldown, 1e-6),
                               self.players[other]["cooldown"] / max(self.config.attack_cooldown, 1e-6),
                               (min(self.time / self.config.max_seconds, 1.0)
                                if self.config.max_seconds is not None else 0.0))
        return out

    def _collides(self, p: dict, obstacle: dict) -> bool:
        height = self.config.duck_height if p["duck"] else self.config.standing_height
        return (obstacle["x"] < self.config.player_width
                and obstacle["x"] + obstacle["width"] > 0
                and p["y"] < obstacle["y"] + obstacle["height"]
                and p["y"] + height > obstacle["y"])

    def step(self, actions: np.ndarray | list) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        if self._done:
            raise RuntimeError("episode is over; call reset() before step()")
        raw = np.asarray(actions)
        if raw.shape != (2, 2) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("actions must be a finite integer array of shape (2,2)")
        actions = raw.astype(np.int64)
        if np.any(actions[:, 0] < 0) or np.any(actions[:, 0] > 2) or np.any(actions[:, 1] < 0) or np.any(actions[:, 1] >= N_ATTACKS):
            raise ValueError("motor must be 0..2 and attack 0..4")
        c = self.config
        masks = self.attack_masks()
        executed = actions.copy()
        # Evaluate both purchases against the same pre-step state. Each attacks
        # a different track, so processing order cannot invalidate the other.
        for seat in range(2):
            kind = int(actions[seat, 1])
            if not masks[seat, kind]:
                self.invalid_attacks[seat] += 1
                executed[seat, 1] = 0
            elif kind:
                x = self._placement_x(1 - seat, kind)
                assert x is not None
                self.players[seat]["coins"] -= c.attack_costs[kind]
                self.players[seat]["cooldown"] = c.attack_cooldown
                self.purchases[seat] += 1
                self.purchase_counts[seat][kind] += 1
                self._spawn(1 - seat, kind, f"player_{seat}", x)
        for seat, p in enumerate(self.players):
            motor = int(actions[seat, 0])
            grounded = p["y"] <= 1e-9
            p["duck"] = bool(motor == 2 and grounded)
            if motor == 1:
                if grounded:
                    p["vy"] = c.jump_velocity
                    p["duck"] = False
                else:
                    executed[seat, 0] = 0
        old_passed = [int(p["passed"]) for p in self.players]
        terminated = False
        truncated = False
        for _ in range(c.action_repeat):
            # Include a shorter last tick so the configured time limit is exact.
            dt = c.dt if c.max_seconds is None else min(c.dt, c.max_seconds - self.time)
            if dt <= 1e-12:
                truncated = True
                break
            self.speed = min(c.max_speed, self.speed + c.acceleration * dt)
            dx = self.speed * dt
            self.time += dt
            self.ticks += 1
            for seat, p in enumerate(self.players):
                self.distance[seat] += dx
                p["coins"] = min(c.wallet_cap, p["coins"] + dx * c.coins_per_meter)
                p["cooldown"] = max(0.0, p["cooldown"] - dt)
                if p["y"] > 0 or p["vy"] > 0:
                    p["y"] += p["vy"] * dt - 0.5 * c.gravity * dt * dt
                    p["vy"] -= c.gravity * dt
                    if p["y"] <= 0:
                        p["y"], p["vy"] = 0.0, 0.0
                        p["duck"] = bool(actions[seat, 0] == 2)
                for o in self.tracks[seat]:
                    o["x"] -= dx
                # Do not stop after seat 0: collisions in this tick are simultaneous.
                hits = [o for o in self.tracks[seat] if self._collides(p, o)]
                p["alive"] = not bool(hits)
                if hits:
                    self.collision_sources[seat] = hits[0]["source"]
                passed = [o for o in self.tracks[seat] if o["x"] + o["width"] <= 0]
                p["passed"] += len(passed)
                self.tracks[seat] = [o for o in self.tracks[seat] if o["x"] + o["width"] > 0]
            if not all(p["alive"] for p in self.players):
                terminated = True
                self.reason = "simultaneous_collision" if not any(p["alive"] for p in self.players) else "collision"
                if self.players[0]["alive"] != self.players[1]["alive"]:
                    self.winner = 0 if self.players[0]["alive"] else 1
                break
            if c.max_seconds is not None and self.time >= c.max_seconds - 1e-10:
                truncated = True
                break
            if c.natural_obstacles:
                for seat in range(2):
                    self.natural_timers[seat] -= dt
                    if self.natural_timers[seat] <= 0 and not c.ensure_reachable:
                        self._spawn_natural(seat)
        if truncated:
            self.reason = "time_limit"
        self._done = bool(terminated or truncated)
        self._reachability_cache.clear()
        if c.natural_obstacles and c.ensure_reachable and not self._done:
            # A feasibility test begins at a decision boundary. Defer natural
            # additions until this boundary, when the target may choose again.
            for seat in range(2):
                if self.natural_timers[seat] <= 0:
                    self._spawn_natural(seat)
        rewards = np.zeros(2, dtype=np.float32)
        if self.winner is not None:
            rewards[self.winner], rewards[1 - self.winner] = 1.0, -1.0
        if c.shaping_coef:
            delta = ((self.players[0]["passed"] - old_passed[0])
                     - (self.players[1]["passed"] - old_passed[1]))
            rewards += np.array([delta, -delta], dtype=np.float32) * c.shaping_coef
        self.last_actions = executed.tolist()
        info = self._info()
        info["executed_actions"] = executed.tolist()
        return self.observe(), rewards, terminated, truncated, info

    def _info(self) -> dict:
        return dict(winner=self.winner, reason=self.reason, distance=list(self.distance),
                    time=self.time, purchases=list(self.purchases),
                    collision_sources=list(self.collision_sources),
                    purchase_counts=deepcopy(self.purchase_counts),
                    invalid_attacks=list(self.invalid_attacks),
                    passed=[int(p["passed"]) for p in self.players], seed=self.seed)

    def snapshot(self) -> dict:
        players = []
        for seat, p in enumerate(self.players):
            players.append(dict(p, x=0.0, width=self.config.player_width,
                                height=self.config.duck_height if p["duck"] else self.config.standing_height,
                                distance=self.distance[seat]))
        return dict(time=self.time, ticks=self.ticks, distance=list(self.distance),
                    speed=self.speed, horizon=OBS_HORIZON, players=players,
                    tracks=deepcopy(self.tracks), winner=self.winner, reason=self.reason,
                    purchases=list(self.purchases), actions=deepcopy(self.last_actions),
                    seed=self.seed)

    def get_state(self) -> dict:
        return dict(version=1, config=self.config.to_dict(), seed=self.seed,
                    time=self.time, ticks=self.ticks, speed=self.speed,
                    distance=list(self.distance), players=deepcopy(self.players),
                    tracks=deepcopy(self.tracks), natural_timers=list(self.natural_timers),
                    purchases=list(self.purchases), purchase_counts=deepcopy(self.purchase_counts),
                    invalid_attacks=list(self.invalid_attacks), next_obstacle_id=self.next_obstacle_id,
                    winner=self.winner, reason=self.reason, done=self._done,
                    last_actions=deepcopy(self.last_actions), collision_sources=list(self.collision_sources),
                    reset_rng=deepcopy(self._reset_rng.bit_generator.state),
                    track_rngs=[deepcopy(r.bit_generator.state) for r in self._track_rngs])

    def set_state(self, state: dict) -> None:
        if state.get("version") != 1 or EnvConfig.from_dict(state.get("config")).to_dict() != self.config.to_dict():
            raise ValueError("state version/config does not match this environment")
        state = deepcopy(state)
        for name in ("seed", "time", "ticks", "speed", "distance", "players", "tracks", "natural_timers",
                     "purchases", "purchase_counts", "invalid_attacks", "next_obstacle_id", "winner", "reason", "last_actions"):
            setattr(self, name, state[name])
        self.collision_sources = state.get("collision_sources", [None, None])
        self._done = bool(state["done"])
        self._reset_rng.bit_generator.state = state["reset_rng"]
        for rng, rng_state in zip(self._track_rngs, state["track_rngs"]):
            rng.bit_generator.state = rng_state
        self._initialized = True
        self._reachability_cache.clear()

    def close(self) -> None:
        """Compatibility with resource-owning environments; this one owns none."""

