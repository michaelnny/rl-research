"""Baseline policy portfolio for the RecoverableKeyFuelMaze family.

Same design as `scheduling.py`: a *portfolio* of cheap baselines
plus a decomposition diagnostic. No single heuristic is the
difficulty signal.

The baseline policies read the observation (no privileged access to
env internals beyond what is exposed through the public env API).
Oracle diagnostics, when present, are listed separately and are not
part of ``MAZE_BASELINES``.
"""

from __future__ import annotations

import numpy as np

from rlh_bench.envs.keyfuel_maze import (
    KeyFuelMazeConfig,
    RecoverableKeyFuelMazeEnv,
)


def _observation_slices(cfg: KeyFuelMazeConfig) -> dict[str, slice]:
    """Observation layout (from the env source):

      pos(2), vel(2), fuel(1), heat(1), damage(1)         = 7
      keys(K_t)                                            = K_t
      seal_status(S)                                       = S
      nearest_landmarks: 3 × (dx, dy, kind_one_hot[4], key_one_hot[K_t]) = 3 * (2 + 4 + K_t)
      gate_phases(G)                                       = G
      t/H                                                  = 1
      prev_action_energy                                   = 1
    """
    K_t, S, G = cfg.n_key_types, cfg.n_seals, cfg.n_gates
    landmark_dim = 2 + 4 + K_t
    cur = 0
    out = {}
    out["pos"] = slice(cur, cur + 2); cur += 2
    out["vel"] = slice(cur, cur + 2); cur += 2
    out["fuel"] = slice(cur, cur + 1); cur += 1
    out["heat"] = slice(cur, cur + 1); cur += 1
    out["damage"] = slice(cur, cur + 1); cur += 1
    out["keys"] = slice(cur, cur + K_t); cur += K_t
    out["seals"] = slice(cur, cur + S); cur += S
    # 3 landmarks
    for i in range(3):
        out[f"landmark_{i}"] = slice(cur, cur + landmark_dim)
        cur += landmark_dim
    out["gate_phases"] = slice(cur, cur + G); cur += G
    out["t_norm"] = slice(cur, cur + 1); cur += 1
    out["prev_energy"] = slice(cur, cur + 1)
    return out


# ----- baseline policies ----------------------------------------------------- #


class MazeZeroPolicy:
    name = "zero"

    def __init__(self, env: RecoverableKeyFuelMazeEnv) -> None:
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.env.action_space.shape, dtype=np.float32)


class MazeRandomActuatorPolicy:
    """Constant per-actuator random pattern, sampled at reset time.

    Useful to detect whether the env rewards *any* fixed pattern of
    actuator use (it should not — the env is not stationary).
    """

    name = "random_constant"

    def __init__(self, env: RecoverableKeyFuelMazeEnv, seed: int = 0) -> None:
        self.env = env
        rng = np.random.default_rng(seed)
        self._pattern = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._pattern


class MazeGreedyLandmarkPolicy:
    """Move toward the nearest visible landmark using top-K aligned actuators.

    Reads landmark_0 (the nearest unfinished landmark in env's
    observation) and picks the top-4 actuators whose force
    contribution aligns with that direction.
    """

    name = "greedy_landmark"

    def __init__(self, env: RecoverableKeyFuelMazeEnv, *, n_actuators: int = 4) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.n_actuators = int(n_actuators)
        # The actuator matrix is per-world; we sample it from the env after reset.
        self._cached_A: np.ndarray | None = None
        self._cached_seed: int | None = None

    def _get_actuator_matrix(self) -> np.ndarray:
        # The env's actuator matrix is regenerated on each reset, so
        # we cache by env._seed. We rely on the public property.
        if self._cached_A is None or self._cached_seed != self.env.seed:
            self._cached_A = self.env.actuator_matrix
            self._cached_seed = self.env.seed
        return self._cached_A

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        landmark = obs[self.slices["landmark_0"]]
        dx, dy = landmark[0], landmark[1]
        direction = np.array([dx, dy], dtype=np.float32)
        if float(np.linalg.norm(direction)) < 1e-6:
            return np.zeros(c.action_dim, dtype=np.float32)
        A = self._get_actuator_matrix()
        alignment = A.T @ direction
        a = np.zeros(c.action_dim, dtype=np.float32)
        top = np.argsort(-alignment)[: self.n_actuators]
        for i in top:
            if alignment[i] > 0:
                a[i] = 1.0
        return a


class MazeOracleRoutePlannerPolicy:
    """Oracle waypoint diagnostic with a PD actuator controller.

    This intentionally reads private generated-world descriptors
    (key/seal/extraction positions and gate requirements).  It is useful as a
    feasibility/control diagnostic, but it is *not* an honest learner-facing
    baseline because those descriptors are not in the public observation.

    The controller collects all keys, visits all seals (waiting through closed
    gate phases when necessary), then extracts. It maps the desired 2-D force
    through the redundant actuator matrix with a clipped least-norm inverse.
    """

    name = "oracle_route_planner"

    def __init__(self, env: RecoverableKeyFuelMazeEnv) -> None:
        self.env = env
        self._cached_seed: int | None = None
        self._pinv: np.ndarray | None = None

    def _actuator_pinv(self) -> np.ndarray:
        seed = self.env.seed
        if self._pinv is None or self._cached_seed != seed:
            A = self.env.actuator_matrix.astype(np.float32)
            self._pinv = (A.T @ np.linalg.pinv(A @ A.T + 1e-4 * np.eye(2, dtype=np.float32))).astype(np.float32)
            self._cached_seed = seed
        return self._pinv

    def _gate_open(self, gate_idx: int) -> bool:
        if gate_idx < 0:
            return True
        c = self.env.config
        period = int(self.env._gate_periods[gate_idx])
        phase = int(self.env._gate_phases[gate_idx])
        return ((self.env.t + phase) % period) < int(c.gate_duty_cycle * period)

    def _target(self) -> np.ndarray:
        c = self.env.config
        pos = self.env.position
        keys = self.env.keys_held
        missing_keys = [i for i in range(c.n_key_types) if keys[i] < 1.0]
        if missing_keys:
            return self.env._key_positions[min(missing_keys, key=lambda i: float(np.linalg.norm(self.env._key_positions[i] - pos)))]

        seals = self.env.seal_status
        open_or_near: list[int] = []
        unfinished = [i for i in range(c.n_seals) if seals[i] < 1.0]
        for i in unfinished:
            # Head to the closest unfinished seal; if its gate is closed, wait
            # inside the region until the phase opens.
            if self._gate_open(self.env._seal_gate_requirements[i]):
                open_or_near.append(i)
        candidates = open_or_near or unfinished
        if candidates:
            return self.env._seal_positions[min(candidates, key=lambda i: float(np.linalg.norm(self.env._seal_positions[i] - pos)))]
        return self.env._extraction_position

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        del obs
        c = self.env.config
        target = self._target().astype(np.float32)
        pos = self.env.position.astype(np.float32)
        err = target - pos
        dist = float(np.linalg.norm(err))
        # Hold still in collection/completion radii to allow key dwell and gate waits.
        if dist < 0.45 * min(c.key_radius, c.seal_radius, c.extraction_radius):
            return np.zeros(c.action_dim, dtype=np.float32)
        vel = getattr(self.env, "_vel", np.zeros(2, dtype=np.float32)).astype(np.float32)
        desired_force = 1.4 * err - 1.8 * vel
        norm = float(np.linalg.norm(desired_force))
        if norm > c.max_force:
            desired_force = desired_force / norm * c.max_force
        a = self._actuator_pinv() @ desired_force
        max_abs = float(np.max(np.abs(a)))
        if max_abs > 1.0:
            a = a / max_abs
        return np.clip(a, -1.0, 1.0).astype(np.float32)


class MazeFuelAwareGreedyPolicy:
    """Greedy nearest-landmark, with a fuel reserve threshold.

    If fuel drops below ``fuel_reserve_fraction``, divert to the nearest
    fuel station instead. Should reduce fuel exhaustion and improve
    fuel_margin terminal component vs naive greedy.
    """

    name = "fuel_aware_greedy"

    def __init__(
        self,
        env: RecoverableKeyFuelMazeEnv,
        *,
        fuel_reserve_fraction: float = 0.2,
        n_actuators: int = 4,
    ) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.fuel_reserve = float(fuel_reserve_fraction)
        self.n_actuators = int(n_actuators)
        self._cached_A: np.ndarray | None = None
        self._cached_seed: int | None = None

    def _get_A(self) -> np.ndarray:
        seed = self.env.seed
        if self._cached_A is None or self._cached_seed != seed:
            self._cached_A = self.env.actuator_matrix
            self._cached_seed = seed
        return self._cached_A

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        fuel = float(obs[self.slices["fuel"]][0])
        # If fuel low, look for a fuel landmark among the 3 nearest.
        if fuel < self.fuel_reserve:
            for i in range(3):
                lm = obs[self.slices[f"landmark_{i}"]]
                # kind one-hot at indices 2..6: [key, seal, fuel, extraction]
                kind = lm[2:6]
                if float(kind[2]) > 0.5:  # fuel
                    direction = lm[:2]
                    A = self._get_A()
                    alignment = A.T @ direction
                    a = np.zeros(c.action_dim, dtype=np.float32)
                    top = np.argsort(-alignment)[: self.n_actuators]
                    for j in top:
                        if alignment[j] > 0:
                            a[j] = 1.0
                    return a
        # Otherwise greedy nearest-landmark
        landmark = obs[self.slices["landmark_0"]]
        direction = landmark[:2]
        if float(np.linalg.norm(direction)) < 1e-6:
            return np.zeros(c.action_dim, dtype=np.float32)
        A = self._get_A()
        alignment = A.T @ direction
        a = np.zeros(c.action_dim, dtype=np.float32)
        top = np.argsort(-alignment)[: self.n_actuators]
        for i in top:
            if alignment[i] > 0:
                a[i] = 1.0
        return a


class MazeEfficientActuatorPolicy:
    """Greedy landmark policy with smaller per-step action energy.

    Like ``MazeGreedyLandmarkPolicy`` but uses fewer, less-intense
    actuators. Trades arrival time for lower fuel/energy cost.
    Should win on neg_energy in the terminal vector but lose on
    seal_completion if it can't arrive in time.
    """

    name = "efficient_actuator"

    def __init__(self, env: RecoverableKeyFuelMazeEnv, *, intensity: float = 0.5) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.intensity = float(intensity)
        self._cached_A: np.ndarray | None = None
        self._cached_seed: int | None = None

    def _get_A(self) -> np.ndarray:
        seed = self.env.seed
        if self._cached_A is None or self._cached_seed != seed:
            self._cached_A = self.env.actuator_matrix
            self._cached_seed = seed
        return self._cached_A

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        landmark = obs[self.slices["landmark_0"]]
        direction = landmark[:2]
        if float(np.linalg.norm(direction)) < 1e-6:
            return np.zeros(c.action_dim, dtype=np.float32)
        A = self._get_A()
        alignment = A.T @ direction
        a = np.zeros(c.action_dim, dtype=np.float32)
        # Single best actuator at lower intensity.
        best = int(np.argmax(alignment))
        if alignment[best] > 0:
            a[best] = self.intensity
        return a


class MazeShortHorizonRolloutPolicy:
    """Decomposition diagnostic: pick a target by short-horizon look-ahead.

    Cheap proxy: rather than really rolling out the env (which would
    be expensive), look at the 3 nearest landmarks and score each by
    ``1 / distance`` adjusted by whether we already have a required
    key (for seals) or fuel reserve (for fuel stations). Move toward
    the highest-scoring landmark.

    If this matches the greedy_landmark policy, the env doesn't
    reward look-ahead — the long-horizon claim fails. If it
    out-performs greedy on seal_completion or fuel_margin, the env
    rewards prioritization.
    """

    name = "short_horizon_lookahead"

    def __init__(self, env: RecoverableKeyFuelMazeEnv, *, n_actuators: int = 4) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.n_actuators = int(n_actuators)
        self._cached_A: np.ndarray | None = None
        self._cached_seed: int | None = None

    def _get_A(self) -> np.ndarray:
        seed = self.env.seed
        if self._cached_A is None or self._cached_seed != seed:
            self._cached_A = self.env.actuator_matrix
            self._cached_seed = seed
        return self._cached_A

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        K_t = c.n_key_types
        keys_held = obs[self.slices["keys"]]
        fuel = float(obs[self.slices["fuel"]][0])
        best_score = -np.inf
        best_dir = np.zeros(2, dtype=np.float32)
        for i in range(3):
            lm = obs[self.slices[f"landmark_{i}"]]
            rel = lm[:2]
            d = float(np.linalg.norm(rel))
            if d < 1e-6:
                continue
            kind = lm[2:6]  # [key, seal, fuel, extraction]
            key_one_hot = lm[6:6 + K_t]
            # Base score: inverse distance.
            score = 1.0 / max(d, 0.5)
            # Bump seals if we likely have keys.
            if kind[1] > 0.5:
                score *= 1.0 + float(np.mean(keys_held)) * 1.5
            # Bump fuel if low.
            if kind[2] > 0.5 and fuel < 0.3:
                score *= 3.0
            # Slightly penalize extraction unless seals are done.
            if kind[3] > 0.5:
                seals_done = float(np.mean(obs[self.slices["seals"]]))
                score *= 0.2 + 0.8 * seals_done
            if score > best_score:
                best_score = score
                best_dir = rel
        if float(np.linalg.norm(best_dir)) < 1e-6:
            return np.zeros(c.action_dim, dtype=np.float32)
        A = self._get_A()
        alignment = A.T @ best_dir
        a = np.zeros(c.action_dim, dtype=np.float32)
        top = np.argsort(-alignment)[: self.n_actuators]
        for i in top:
            if alignment[i] > 0:
                a[i] = 1.0
        return a


# ----- registry -------------------------------------------------------------- #


MAZE_BASELINES = [
    MazeZeroPolicy,
    MazeRandomActuatorPolicy,
    MazeGreedyLandmarkPolicy,
    MazeFuelAwareGreedyPolicy,
    MazeEfficientActuatorPolicy,
    MazeShortHorizonRolloutPolicy,
]
"""All learner-facing maze-family baselines. Ordered approximately by
sophistication: zero/random_constant are sanity checks, the middle
entries are observation-only heuristics, and the last is an
observation-only short-horizon diagnostic."""


MAZE_ORACLE_DIAGNOSTICS = [
    MazeOracleRoutePlannerPolicy,
]
"""Privileged maze diagnostics. These may read generated-world internals
and must not be counted as learner-facing baselines."""

# Backward-compatible alias for external notebooks that imported the old name.
# The policy's public ``name`` and registry placement now make its oracle status
# explicit.
MazeRoutePlannerPolicy = MazeOracleRoutePlannerPolicy
