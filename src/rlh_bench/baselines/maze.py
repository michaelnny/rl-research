"""Baseline policy portfolio for the RecoverableKeyFuelMaze family.

Same design as `scheduling.py`: a *portfolio* of cheap baselines
plus a decomposition diagnostic. No single heuristic is the
difficulty signal.

These policies read the observation (no privileged access to env
internals beyond what is exposed through the public env API).
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
        if self._cached_A is None or self._cached_seed != self.env.seed if hasattr(self.env, "seed") else True:
            self._cached_A = self.env.actuator_matrix
            self._cached_seed = getattr(self.env, "_seed", None)
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
        seed = getattr(self.env, "_seed", None)
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
        seed = getattr(self.env, "_seed", None)
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
        seed = getattr(self.env, "_seed", None)
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
"""All maze-family baselines. Ordered approximately by
sophistication: zero/random_constant are sanity, the next three are
heuristics with increasing structure, the last is a decomposition
diagnostic."""
