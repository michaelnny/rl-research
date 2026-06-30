"""Recoverable deterministic continuous-control maze with long-horizon coupling.

This replaces the original ``RecoverablePointMaze-*-v0`` family. The
mission narrowed to continuous-action algorithms (see
``CLAUDE.md``), so the substrate keeps a spatial-control family but
upgrades it so a long horizon is decision-saturated.

What the env tests
------------------

Continuous-control policies that must manage:

  * a global fuel budget (with local recharge);
  * route order through a set of obligations (key/seal regions);
  * timed gates whose open/closed phase changes which corridors are
    available later;
  * heat / damage from aggressive control;
  * a redundant actuator basis with per-actuator energy weights.

A naive PD-to-nearest-target policy should not solve v0/Large
because:
  - it ignores fuel scarcity and gets stranded;
  - it ignores key/seal *order* constraints (some seals require keys
    held; some require visiting a region during an open gate phase);
  - aggressive control runs up heat/damage that lowers terminal
    capacity.

Action space
------------

``Box([-1, 1]^D)`` where D depends on tier. The agent's action is
mapped through a per-world ``A_world ∈ R^{2 × D}`` actuator matrix
into a 2-D force. Different action vectors that produce the same
force differ in per-actuator energy/heat cost (``A`` and the per-
actuator cost weights come from ``world_gen.make_actuator_matrix``
and ``make_actuator_costs``).

Terminal vector (9 components, larger-is-better)
------------------------------------------------

``(success, seal_completion, key_coverage, fuel_margin, neg_damage,
neg_lateness, neg_energy, neg_collision, route_efficiency)``.

Determinism
-----------

``reset(seed=s)`` deterministically generates the entire world
(room layout, key/seal placement, gate phases, actuator matrix and
costs, fuel-station positions) from ``s``. Different seeds → different
worlds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rlh_bench.core import RewardSpec, StepReturn, make_reward, zero_reward
from rlh_bench.spaces import Box, clip_to_box
from rlh_bench.world_gen import make_actuator_costs, make_actuator_matrix


DEFAULT_KEYFUEL_REWARD_SPEC = RewardSpec(
    names=(
        "success",
        "seal_completion",
        "key_coverage",
        "fuel_margin",
        "neg_damage",
        "neg_lateness",
        "neg_energy",
        "neg_collision",
        "route_efficiency",
    ),
    weights=(
        1.0,    # success
        0.35,   # seal_completion
        0.10,   # key_coverage
        0.10,   # fuel_margin
        0.04,   # neg_damage
        0.05,   # neg_lateness
        0.005,  # neg_energy
        0.03,   # neg_collision
        0.10,   # route_efficiency
    ),
)


@dataclass(frozen=True)
class KeyFuelMazeConfig:
    """Configuration for :class:`RecoverableKeyFuelMazeEnv`.

    Defaults give the v0 tier (H=2000, D=32, ~48×48 world, 4 key
    types, 6 seals, 4 timed gates).
    """

    horizon: int = 2000
    action_dim: int = 32

    # Map / world geometry
    world_size: float = 48.0          # square side length in continuous units
    n_rooms: int = 10                  # advisory; used to scale obstacle density
    n_key_types: int = 4
    n_seals: int = 6                   # number of required seals to "complete" the world
    n_gates: int = 3
    n_fuel_stations: int = 5

    # Physics
    dt: float = 0.1
    damping: float = 0.85               # velocity carries with damping
    max_speed: float = 4.0
    max_force: float = 2.0
    agent_radius: float = 0.5

    # Fuel
    initial_fuel: float = 200.0
    fuel_per_distance: float = 0.05     # fuel consumed per unit of distance traveled
    fuel_per_action_energy: float = 0.01  # fuel consumed per unit of integrated action energy
    fuel_station_recharge: float = 60.0
    fuel_station_cooldown: int = 50     # steps between recharges from same station

    # Heat / damage
    heat_buildup_rate: float = 0.002    # heat per action energy unit
    heat_decay_rate: float = 0.001
    heat_damage_threshold: float = 0.8  # heat above this accrues damage
    damage_per_collision: float = 0.01
    max_damage: float = 1.0

    # Region radii (for "agent inside region" tests)
    key_radius: float = 1.5
    seal_radius: float = 1.5
    fuel_radius: float = 1.5
    key_dwell_steps: int = 4            # steps to dwell to collect a key

    # Gate timing (period in steps)
    gate_period_range: tuple[int, int] = (80, 240)
    gate_duty_cycle: float = 0.4         # fraction of period the gate is OPEN

    # Required keys per seal (drawn deterministically from n_key_types)
    seal_key_requirement_max: int = 2    # at most this many key types per seal

    # Final extraction zone
    extraction_radius: float = 2.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.action_dim < 2:
            raise ValueError("action_dim must be >= 2")
        if self.n_seals < 1:
            raise ValueError("n_seals must be >= 1")
        if self.n_key_types < 1:
            raise ValueError("n_key_types must be >= 1")
        if not 0.0 < self.gate_duty_cycle < 1.0:
            raise ValueError("gate_duty_cycle must be in (0, 1)")


class RecoverableKeyFuelMazeEnv:
    """Continuous-control maze with fuel, keys, seals, gates, and a
    redundant actuator basis."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: KeyFuelMazeConfig | None = None,
        reward_spec: RewardSpec = DEFAULT_KEYFUEL_REWARD_SPEC,
        reward_mode: str = "scalar",
    ) -> None:
        self.config = KeyFuelMazeConfig() if config is None else config
        self.reward_spec = reward_spec
        self.reward_dim = reward_spec.dim
        if reward_mode not in {"scalar", "vector"}:
            raise ValueError("reward_mode must be 'scalar' or 'vector'")
        self.reward_mode = reward_mode

        c = self.config
        self.action_space = Box(low=-1.0, high=1.0, shape=(c.action_dim,), dtype=np.float32)

        # Observation contains:
        #   position(2), velocity(2), fuel(1), heat(1), damage(1)         = 7
        #   inventory[n_key_types]                                        = K_t
        #   seal status[n_seals]                                          = S
        #   nearest-3 landmarks: dx, dy, kind_one_hot(3+n_key_types)      = 3 * (2 + 3 + K_t)
        #     where kind one-hot is over [key, seal, fuel, extraction]
        #   gate phases[n_gates]                                          = G
        #   time / horizon                                                = 1
        #   previous action summary (energy)                              = 1
        K_t = c.n_key_types
        landmark_feat_dim = 3 * (2 + 4 + K_t)  # dx, dy + kind one-hot over [key, seal, fuel, extraction], then which key/seal index encoded as offset
        obs_dim = 7 + K_t + c.n_seals + landmark_feat_dim + c.n_gates + 1 + 1
        self.observation_space = Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        # World tensors (filled in reset)
        self._rng = np.random.default_rng(0)
        self._seed: int | None = None
        self._actuator_matrix = np.zeros((2, c.action_dim), dtype=np.float32)
        self._actuator_costs = np.ones(c.action_dim, dtype=np.float32)
        self._key_positions = np.zeros((K_t, 2), dtype=np.float32)
        self._seal_positions = np.zeros((c.n_seals, 2), dtype=np.float32)
        self._fuel_positions = np.zeros((c.n_fuel_stations, 2), dtype=np.float32)
        self._extraction_position = np.zeros(2, dtype=np.float32)
        self._gate_periods = np.zeros(c.n_gates, dtype=np.int64)
        self._gate_phases = np.zeros(c.n_gates, dtype=np.int64)  # offset within period
        self._gate_positions = np.zeros((c.n_gates, 2), dtype=np.float32)
        self._seal_key_requirements: list[tuple[int, ...]] = []
        self._seal_gate_requirements: list[int] = []  # -1 = no gate required; else gate index
        # Deadlines (in env steps), spread across the horizon.
        self._seal_deadlines = np.zeros(c.n_seals, dtype=np.int64)
        # Oracle route length lower bound for route_efficiency
        self._oracle_route_length: float = 1.0
        # Dynamic state
        self._t = 0
        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._fuel = 0.0
        self._heat = 0.0
        self._damage = 0.0
        self._keys_held = np.zeros(K_t, dtype=np.float32)
        self._seal_status = np.zeros(c.n_seals, dtype=np.float32)
        self._seal_completion_times = np.full(c.n_seals, -1, dtype=np.int64)
        self._dwell_counters = np.zeros(K_t, dtype=np.int64)
        self._fuel_cooldowns = np.zeros(c.n_fuel_stations, dtype=np.int64)
        self._total_energy = 0.0
        self._total_distance = 0.0
        self._total_collision = 0.0
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)
        self._prev_action_energy = 0.0
        self._extraction_reached = False
        self._extraction_time: int | None = None

    # ------------------------------------------------------------------
    # Public properties

    @property
    def t(self) -> int:
        return self._t

    @property
    def seed(self) -> int | None:
        """Seed of the current world (set by the last `reset()`)."""
        return self._seed

    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def fuel(self) -> float:
        return float(self._fuel)

    @property
    def keys_held(self) -> np.ndarray:
        return self._keys_held.copy()

    @property
    def seal_status(self) -> np.ndarray:
        return self._seal_status.copy()

    @property
    def damage(self) -> float:
        return float(self._damage)

    @property
    def actuator_matrix(self) -> np.ndarray:
        return self._actuator_matrix.copy()

    # ------------------------------------------------------------------
    # Gym-like API

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is None:
            seed = 0
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        c = self.config
        K_t = c.n_key_types

        # World sampling
        self._actuator_matrix = make_actuator_matrix(
            self._rng, action_dim=c.action_dim, n_force_dims=2, redundancy_bias=0.5
        )
        self._actuator_costs = make_actuator_costs(
            self._rng, action_dim=c.action_dim, base_cost=1.0, spread=0.6
        )
        self._key_positions = self._rng.uniform(
            2.0, c.world_size - 2.0, size=(K_t, 2)
        ).astype(np.float32)
        self._seal_positions = self._rng.uniform(
            2.0, c.world_size - 2.0, size=(c.n_seals, 2)
        ).astype(np.float32)
        self._fuel_positions = self._rng.uniform(
            2.0, c.world_size - 2.0, size=(c.n_fuel_stations, 2)
        ).astype(np.float32)
        self._extraction_position = self._rng.uniform(
            2.0, c.world_size - 2.0, size=(2,)
        ).astype(np.float32)
        self._gate_periods = self._rng.integers(
            c.gate_period_range[0], c.gate_period_range[1] + 1, size=c.n_gates
        )
        self._gate_phases = self._rng.integers(0, self._gate_periods).astype(np.int64)
        self._gate_positions = self._rng.uniform(
            2.0, c.world_size - 2.0, size=(c.n_gates, 2)
        ).astype(np.float32)

        # Seal key requirements: each seal needs a random subset of keys.
        self._seal_key_requirements = []
        for _ in range(c.n_seals):
            n_required = int(self._rng.integers(1, min(c.seal_key_requirement_max, K_t) + 1))
            required = tuple(sorted(int(x) for x in self._rng.choice(K_t, size=n_required, replace=False)))
            self._seal_key_requirements.append(required)

        # Some seals require a gate to be open during completion.
        if c.n_gates > 0:
            self._seal_gate_requirements = [
                int(self._rng.integers(-1, c.n_gates)) for _ in range(c.n_seals)
            ]
            # Convert -1 to "no gate required" sentinel
        else:
            self._seal_gate_requirements = [-1] * c.n_seals

        # Deadlines: spread soft deadlines across the horizon. Last seal
        # by step ~95% of horizon.
        spacing = np.linspace(0.30, 0.95, c.n_seals)
        self._seal_deadlines = np.asarray(
            np.ceil(spacing * c.horizon), dtype=np.int64
        )

        # Oracle route length: sum of nearest-neighbor distances between
        # start, all required regions (keys + seals + extraction).
        start = np.array([c.world_size / 2.0, c.world_size / 2.0], dtype=np.float32)
        waypoints = np.concatenate(
            [self._key_positions, self._seal_positions, self._extraction_position[None, :]],
            axis=0,
        )
        # Greedy nearest-neighbor tour
        visited = np.zeros(len(waypoints), dtype=bool)
        cur = start.copy()
        total = 0.0
        for _ in range(len(waypoints)):
            dists = np.linalg.norm(waypoints - cur, axis=1)
            dists[visited] = np.inf
            nxt = int(np.argmin(dists))
            total += float(dists[nxt])
            cur = waypoints[nxt]
            visited[nxt] = True
        self._oracle_route_length = max(total, 1.0)

        # Dynamic state
        self._t = 0
        self._pos = start.copy()
        self._vel = np.zeros(2, dtype=np.float32)
        self._fuel = float(c.initial_fuel)
        self._heat = 0.0
        self._damage = 0.0
        self._keys_held = np.zeros(K_t, dtype=np.float32)
        self._seal_status = np.zeros(c.n_seals, dtype=np.float32)
        self._seal_completion_times = np.full(c.n_seals, -1, dtype=np.int64)
        self._dwell_counters = np.zeros(K_t, dtype=np.int64)
        self._fuel_cooldowns = np.zeros(c.n_fuel_stations, dtype=np.int64)
        self._total_energy = 0.0
        self._total_distance = 0.0
        self._total_collision = 0.0
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)
        self._prev_action_energy = 0.0
        self._extraction_reached = False
        self._extraction_time = None

        return self._observation(), self._info(np.zeros(self.reward_dim, dtype=np.float32))

    def step(self, action: np.ndarray) -> StepReturn:
        if self._terminated:
            raise RuntimeError("step() called after episode terminated; call reset() first")

        c = self.config
        raw_action = clip_to_box(self.action_space, action)

        # Force via actuator matrix
        force = self._actuator_matrix @ raw_action
        force = np.clip(force, -c.max_force, c.max_force).astype(np.float32)

        # Per-actuator energy: sum of (action[i]^2 * cost[i])
        per_action_energy = float(np.sum(np.square(raw_action) * self._actuator_costs))
        self._total_energy += per_action_energy
        self._prev_action_energy = per_action_energy

        # If fuel exhausted, force is null
        if self._fuel <= 0.0:
            force[:] = 0.0
            per_action_energy = 0.0

        # Dynamics: damped point mass
        old_pos = self._pos.copy()
        self._vel = (c.damping * self._vel + force * c.dt).astype(np.float32)
        # Speed cap
        speed = float(np.linalg.norm(self._vel))
        if speed > c.max_speed:
            self._vel = (self._vel / speed * c.max_speed).astype(np.float32)
        new_pos = self._pos + self._vel * c.dt
        # Boundary collision (soft): clamp and reflect velocity
        collision = False
        for axis in (0, 1):
            if new_pos[axis] < 0.0:
                new_pos[axis] = 0.0
                self._vel[axis] = -0.5 * self._vel[axis]
                collision = True
            elif new_pos[axis] > c.world_size:
                new_pos[axis] = c.world_size
                self._vel[axis] = -0.5 * self._vel[axis]
                collision = True
        if collision:
            self._damage = min(self._damage + c.damage_per_collision, c.max_damage)
            self._total_collision += 1.0

        step_distance = float(np.linalg.norm(new_pos - old_pos))
        self._total_distance += step_distance
        self._pos = new_pos.astype(np.float32)

        # Fuel consumption
        fuel_used = c.fuel_per_distance * step_distance + c.fuel_per_action_energy * per_action_energy
        self._fuel = max(self._fuel - fuel_used, 0.0)

        # Heat dynamics
        self._heat = min(
            max(self._heat + c.heat_buildup_rate * per_action_energy - c.heat_decay_rate, 0.0),
            1.0,
        )
        # Damage accrues if heat above threshold
        if self._heat > c.heat_damage_threshold:
            self._damage = min(self._damage + 0.5 * c.damage_per_collision * (self._heat - c.heat_damage_threshold), c.max_damage)

        # Key collection (dwell mechanic)
        for i in range(c.n_key_types):
            if self._keys_held[i] >= 1.0:
                continue
            dist = float(np.linalg.norm(self._pos - self._key_positions[i]))
            if dist < c.key_radius:
                self._dwell_counters[i] += 1
                if self._dwell_counters[i] >= c.key_dwell_steps:
                    self._keys_held[i] = 1.0
            else:
                self._dwell_counters[i] = 0

        # Seal completion: requires being in seal radius AND having all
        # required keys AND (if gated) the gate is open.
        for i in range(c.n_seals):
            if self._seal_status[i] >= 1.0:
                continue
            dist = float(np.linalg.norm(self._pos - self._seal_positions[i]))
            if dist >= c.seal_radius:
                continue
            required_keys = self._seal_key_requirements[i]
            has_all = all(self._keys_held[k] >= 1.0 for k in required_keys)
            if not has_all:
                continue
            gate_req = self._seal_gate_requirements[i]
            if gate_req >= 0:
                # Check gate open
                period = int(self._gate_periods[gate_req])
                phase = int(self._gate_phases[gate_req])
                pos_in_cycle = (self._t + phase) % period
                open_duration = int(c.gate_duty_cycle * period)
                if pos_in_cycle >= open_duration:
                    continue
            self._seal_status[i] = 1.0
            self._seal_completion_times[i] = self._t

        # Fuel station recharge
        for i in range(c.n_fuel_stations):
            if self._fuel_cooldowns[i] > 0:
                self._fuel_cooldowns[i] -= 1
                continue
            dist = float(np.linalg.norm(self._pos - self._fuel_positions[i]))
            if dist < c.fuel_radius:
                self._fuel = min(self._fuel + c.fuel_station_recharge, c.initial_fuel * 1.5)
                self._fuel_cooldowns[i] = c.fuel_station_cooldown

        # Extraction
        if not self._extraction_reached:
            d = float(np.linalg.norm(self._pos - self._extraction_position))
            if d < c.extraction_radius:
                self._extraction_reached = True
                self._extraction_time = self._t

        self._t += 1

        terminal = self._t >= c.horizon
        self._terminated = terminal
        if terminal:
            vector = self._terminal_reward_vector()
            reward = make_reward(self.reward_mode, self.reward_spec, vector)
            self._last_reward_vector = vector.copy()
        else:
            vector = np.zeros(self.reward_dim, dtype=np.float32)
            reward = zero_reward(self.reward_mode, self.reward_spec)

        return self._observation(), reward, terminal, False, self._info(vector)

    # ------------------------------------------------------------------
    # Observation / info / terminal

    def _gate_phase_features(self) -> np.ndarray:
        c = self.config
        if c.n_gates == 0:
            return np.zeros(0, dtype=np.float32)
        # 1 = open, 0 = closed
        features = np.zeros(c.n_gates, dtype=np.float32)
        for i in range(c.n_gates):
            period = int(self._gate_periods[i])
            phase = int(self._gate_phases[i])
            pos_in_cycle = (self._t + phase) % period
            open_duration = int(c.gate_duty_cycle * period)
            features[i] = 1.0 if pos_in_cycle < open_duration else 0.0
        return features

    def _nearest_landmarks(self, n: int = 3) -> np.ndarray:
        """Return concatenated features for the nearest ``n`` *unfinished* landmarks.

        A landmark is one of: an uncollected key, an uncompleted seal,
        an off-cooldown fuel station, or the extraction zone. Features
        per landmark: dx, dy (relative), then a 4-D one-hot over
        ``[key, seal, fuel, extraction]`` and then a K_t-D one-hot
        over which key it is (zero for non-keys).
        """
        c = self.config
        K_t = c.n_key_types
        candidates: list[tuple[float, np.ndarray]] = []

        for i in range(K_t):
            if self._keys_held[i] >= 1.0:
                continue
            rel = self._key_positions[i] - self._pos
            kind = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            which_key = np.zeros(K_t, dtype=np.float32)
            which_key[i] = 1.0
            feat = np.concatenate([rel, kind, which_key])
            candidates.append((float(np.linalg.norm(rel)), feat))
        for i in range(c.n_seals):
            if self._seal_status[i] >= 1.0:
                continue
            rel = self._seal_positions[i] - self._pos
            kind = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            which_key = np.zeros(K_t, dtype=np.float32)
            feat = np.concatenate([rel, kind, which_key])
            candidates.append((float(np.linalg.norm(rel)), feat))
        for i in range(c.n_fuel_stations):
            if self._fuel_cooldowns[i] > 0:
                continue
            rel = self._fuel_positions[i] - self._pos
            kind = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            which_key = np.zeros(K_t, dtype=np.float32)
            feat = np.concatenate([rel, kind, which_key])
            candidates.append((float(np.linalg.norm(rel)), feat))
        if not self._extraction_reached:
            rel = self._extraction_position - self._pos
            kind = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            which_key = np.zeros(K_t, dtype=np.float32)
            feat = np.concatenate([rel, kind, which_key])
            candidates.append((float(np.linalg.norm(rel)), feat))

        candidates.sort(key=lambda x: x[0])
        feat_dim = 2 + 4 + K_t
        out = np.zeros(n * feat_dim, dtype=np.float32)
        for idx in range(min(n, len(candidates))):
            out[idx * feat_dim : (idx + 1) * feat_dim] = candidates[idx][1]
        return out

    def _observation(self) -> np.ndarray:
        c = self.config
        # Normalize position and velocity by world_size and max_speed
        pos_norm = self._pos / c.world_size
        vel_norm = self._vel / max(c.max_speed, 1e-6)
        fuel_norm = np.asarray([self._fuel / max(c.initial_fuel, 1e-6)], dtype=np.float32)
        heat = np.asarray([self._heat], dtype=np.float32)
        damage = np.asarray([self._damage], dtype=np.float32)
        keys = self._keys_held.copy()
        seals = self._seal_status.copy()
        landmarks = self._nearest_landmarks(n=3)
        gate_features = self._gate_phase_features()
        t_norm = np.asarray([self._t / c.horizon], dtype=np.float32)
        prev_energy = np.asarray([min(self._prev_action_energy / max(c.action_dim, 1), 5.0)], dtype=np.float32)
        obs = np.concatenate(
            [pos_norm, vel_norm, fuel_norm, heat, damage, keys, seals,
             landmarks, gate_features, t_norm, prev_energy]
        ).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _terminal_reward_vector(self) -> np.ndarray:
        c = self.config
        seal_completion = float(np.mean(self._seal_status))
        # All seals + reached extraction = success
        success = float(seal_completion >= 1.0 and self._extraction_reached)

        key_coverage = float(np.mean(self._keys_held))
        fuel_margin = float(self._fuel / max(c.initial_fuel, 1e-6))
        damage = float(self._damage / max(c.max_damage, 1e-6))

        # Lateness: for each seal, how far past deadline
        late = 0.0
        for i in range(c.n_seals):
            if self._seal_status[i] >= 1.0:
                if self._seal_completion_times[i] > self._seal_deadlines[i]:
                    late += float(self._seal_completion_times[i] - self._seal_deadlines[i])
            else:
                # Unfinished seals incur full lateness
                late += float(c.horizon - self._seal_deadlines[i])
        normalized_lateness = late / max(c.horizon * c.n_seals, 1)

        normalized_energy = self._total_energy / max(c.horizon * c.action_dim, 1)
        normalized_collision = self._total_collision / max(c.horizon, 1)

        # Route efficiency: oracle / actual; capped to [0, 1]
        actual = max(self._total_distance, 1.0)
        route_efficiency = float(min(self._oracle_route_length / actual, 1.0))

        return np.asarray(
            [
                success,
                seal_completion,
                key_coverage,
                fuel_margin,
                -damage,
                -normalized_lateness,
                -normalized_energy,
                -normalized_collision,
                route_efficiency,
            ],
            dtype=np.float32,
        )

    def _info(self, reward_vector: np.ndarray) -> dict[str, Any]:
        info = self.diagnostics()
        info.update(
            {
                "reward_vector": np.asarray(reward_vector, dtype=np.float32).copy(),
                "reward_names": self.reward_spec.names,
                "is_success": bool(info["success"]),
            }
        )
        return info

    def diagnostics(self) -> dict[str, Any]:
        seal_completion = float(np.mean(self._seal_status))
        return {
            "t": int(self._t),
            "success": float(seal_completion >= 1.0 and self._extraction_reached),
            "seal_completion": seal_completion,
            "key_coverage": float(np.mean(self._keys_held)),
            "fuel": float(self._fuel),
            "heat": float(self._heat),
            "damage": float(self._damage),
            "total_energy": float(self._total_energy),
            "total_distance": float(self._total_distance),
            "total_collisions": int(self._total_collision),
            "extraction_reached": bool(self._extraction_reached),
        }

    def render(self) -> str:
        diag = self.diagnostics()
        return (
            f"RecoverableKeyFuelMaze(t={self._t}/{self.config.horizon}, "
            f"pos=({self._pos[0]:.1f},{self._pos[1]:.1f}), fuel={self._fuel:.1f}, "
            f"keys={int(np.sum(self._keys_held))}/{self.config.n_key_types}, "
            f"seals={int(np.sum(self._seal_status))}/{self.config.n_seals})"
        )
