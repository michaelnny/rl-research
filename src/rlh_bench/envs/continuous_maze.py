"""Recoverable continuous-control maze environments.

The environment is deterministic, terminal-feedback-only, and intentionally
physics-light. Actions are continuous accelerations. Mistakes such as collisions
are non-terminal: they waste time/energy and appear in the terminal outcome
vector, but the agent can recover and still finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlh_bench.core import RewardSpec, StepReturn, make_reward, zero_reward
from rlh_bench.spaces import Box, clip_to_box


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangular obstacle in unit-square coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if not (self.xmin < self.xmax and self.ymin < self.ymax):
            raise ValueError("invalid rectangle bounds")

    def contains(self, point: np.ndarray, radius: float = 0.0) -> bool:
        x, y = float(point[0]), float(point[1])
        return (self.xmin - radius <= x <= self.xmax + radius) and (
            self.ymin - radius <= y <= self.ymax + radius
        )


@dataclass(frozen=True)
class RecoverableMazeConfig:
    """Configuration for :class:`RecoverablePointMazeEnv`.

    Args:
        horizon: Fixed episode length. Feedback is terminal-only at this step.
        action_dim: Continuous action dimensionality. Must be even. Extra action
            dimensions are redundant actuator channels that combine into a 2D
            acceleration, giving a simple action-size difficulty knob.
        start: Deterministic start position in ``[0, 1]^2``.
        goal: Deterministic goal position in ``[0, 1]^2``.
        goal_radius: Terminal success radius.
        obstacles: Axis-aligned rectangular obstacles. Collisions are soft and
            recoverable; they never terminate the episode.
        waypoints: Optional waypoints used only by the supplied heuristic policy.
        dt: Position integration step size.
        acceleration_scale: Converts normalized action into velocity change.
        damping: Velocity damping factor in ``[0, 1]``.
        max_speed: Maximum velocity norm per step.
        agent_radius: Collision radius.
    """

    horizon: int = 160
    action_dim: int = 2
    start: tuple[float, float] = (0.10, 0.10)
    goal: tuple[float, float] = (0.90, 0.90)
    goal_radius: float = 0.055
    obstacles: tuple[Rectangle, ...] = (Rectangle(0.34, 0.18, 0.66, 0.82),)
    waypoints: tuple[tuple[float, float], ...] = ((0.16, 0.90), (0.90, 0.90))
    dt: float = 1.0
    acceleration_scale: float = 0.025
    damping: float = 0.86
    max_speed: float = 0.042
    agent_radius: float = 0.018

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.action_dim <= 0 or self.action_dim % 2 != 0:
            raise ValueError("action_dim must be a positive even integer")
        if not (0.0 < self.goal_radius < 1.0):
            raise ValueError("goal_radius must be in (0, 1)")
        if not (0.0 <= self.damping <= 1.0):
            raise ValueError("damping must be in [0, 1]")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")


DEFAULT_MAZE_REWARD_SPEC = RewardSpec(
    names=("success", "neg_final_distance", "neg_energy", "neg_collisions", "neg_path_length"),
    weights=(1.0, 0.30, 0.003, 0.03, 0.02),
)


class RecoverablePointMazeEnv:
    """Deterministic, terminal-only, recoverable continuous maze.

    Observation: ``[x, y, vx, vy, goal_x, goal_y, t / H]``.

    Action: a continuous vector in ``[-1, 1]^{action_dim}``. When
    ``action_dim > 2``, pairs of action coordinates act as redundant actuator
    channels and are deterministically averaged into a 2D acceleration.

    Reward: zero until the terminal step. The terminal reward vector is always
    placed in ``info['reward_vector']``. In ``reward_mode='scalar'`` the returned
    reward is the weighted sum from ``reward_spec``. In ``reward_mode='vector'``
    the returned reward is the vector itself.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: RecoverableMazeConfig | None = None,
        reward_spec: RewardSpec = DEFAULT_MAZE_REWARD_SPEC,
        reward_mode: str = "scalar",
    ) -> None:
        self.config = RecoverableMazeConfig() if config is None else config
        self.reward_spec = reward_spec
        self.reward_dim = reward_spec.dim
        if reward_mode not in {"scalar", "vector"}:
            raise ValueError("reward_mode must be 'scalar' or 'vector'")
        self.reward_mode = reward_mode

        self.observation_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.config.action_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self._energy = 0.0
        self._collisions = 0
        self._path_length = 0.0
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)

    @property
    def t(self) -> int:
        return self._t

    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._vel.copy()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = np.asarray(self.config.start, dtype=np.float32).copy()
        self._vel = np.zeros(2, dtype=np.float32)
        self._last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self._energy = 0.0
        self._collisions = 0
        self._path_length = 0.0
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)
        return self._observation(), self._info(np.zeros(self.reward_dim, dtype=np.float32))

    def step(self, action: np.ndarray) -> StepReturn:
        if self._terminated:
            raise RuntimeError("step() called after episode terminated; call reset() first")

        clipped_action = clip_to_box(self.action_space, action)
        old_pos = self._pos.copy()
        acceleration = self._action_to_acceleration(clipped_action)

        self._vel = self.config.damping * self._vel + self.config.acceleration_scale * acceleration
        speed = float(np.linalg.norm(self._vel))
        if speed > self.config.max_speed:
            self._vel = (self._vel / speed * self.config.max_speed).astype(np.float32)

        desired_pos = self._pos + self.config.dt * self._vel
        new_pos, new_vel, collided = self._resolve_collision(old_pos, desired_pos, self._vel)
        self._pos = new_pos.astype(np.float32)
        self._vel = new_vel.astype(np.float32)
        self._last_action = clipped_action
        self._t += 1
        self._energy += float(np.sum(np.square(clipped_action)))
        self._path_length += float(np.linalg.norm(self._pos - old_pos))
        if collided:
            self._collisions += 1

        terminal = self._t >= self.config.horizon
        self._terminated = terminal
        if terminal:
            vector = self._terminal_reward_vector()
            reward = make_reward(self.reward_mode, self.reward_spec, vector)
            self._last_reward_vector = vector.copy()
        else:
            vector = np.zeros(self.reward_dim, dtype=np.float32)
            reward = zero_reward(self.reward_mode, self.reward_spec)

        return self._observation(), reward, terminal, False, self._info(vector)

    def render(self) -> str:
        """Return a simple text rendering useful for debugging."""

        return (
            f"RecoverablePointMaze(t={self._t}, pos={self._pos.round(3).tolist()}, "
            f"goal={list(self.config.goal)}, collisions={self._collisions})"
        )

    def diagnostics(self) -> dict[str, float | int]:
        dist = float(np.linalg.norm(self._pos - np.asarray(self.config.goal, dtype=np.float32)))
        return {
            "t": self._t,
            "success": float(dist <= self.config.goal_radius),
            "final_distance": dist,
            "energy": float(self._energy),
            "collisions": int(self._collisions),
            "path_length": float(self._path_length),
        }

    def _observation(self) -> np.ndarray:
        goal = np.asarray(self.config.goal, dtype=np.float32)
        # Positions and goal are already in [0, 1]. Velocities are scaled to roughly [-1, 1].
        obs = np.asarray(
            [
                self._pos[0],
                self._pos[1],
                self._vel[0] / max(self.config.max_speed, 1e-8),
                self._vel[1] / max(self.config.max_speed, 1e-8),
                goal[0],
                goal[1],
                self._t / self.config.horizon,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _info(self, reward_vector: np.ndarray) -> dict[str, Any]:
        diag = self.diagnostics()
        diag.update(
            {
                "reward_vector": np.asarray(reward_vector, dtype=np.float32).copy(),
                "reward_names": self.reward_spec.names,
                "is_success": bool(diag["success"]),
            }
        )
        return diag

    def _action_to_acceleration(self, action: np.ndarray) -> np.ndarray:
        pairs = action.reshape(-1, 2).astype(np.float32)
        weights = 1.0 / (1.0 + np.arange(pairs.shape[0], dtype=np.float32))
        weighted = (pairs * weights[:, None]).sum(axis=0) / weights.sum()
        norm = float(np.linalg.norm(weighted))
        if norm > 1.0:
            weighted = weighted / norm
        return weighted.astype(np.float32)

    def _valid_position(self, point: np.ndarray) -> bool:
        r = self.config.agent_radius
        x, y = float(point[0]), float(point[1])
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return False
        return not any(rect.contains(point, radius=r) for rect in self.config.obstacles)

    def _resolve_collision(
        self, old_pos: np.ndarray, desired_pos: np.ndarray, desired_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._valid_position(desired_pos):
            return desired_pos, desired_vel, False

        collided = True
        new_pos = old_pos.copy()
        new_vel = desired_vel.copy()

        x_candidate = np.asarray([desired_pos[0], old_pos[1]], dtype=np.float32)
        y_candidate = np.asarray([old_pos[0], desired_pos[1]], dtype=np.float32)

        if self._valid_position(x_candidate):
            new_pos[0] = x_candidate[0]
        else:
            new_vel[0] = -0.20 * new_vel[0]

        if self._valid_position(y_candidate):
            new_pos[1] = y_candidate[1]
        else:
            new_vel[1] = -0.20 * new_vel[1]

        # Final guard against numerical or corner cases.
        if not self._valid_position(new_pos):
            new_pos = old_pos.copy()
            new_vel *= -0.20

        return new_pos.astype(np.float32), new_vel.astype(np.float32), collided

    def _terminal_reward_vector(self) -> np.ndarray:
        dist = float(np.linalg.norm(self._pos - np.asarray(self.config.goal, dtype=np.float32)))
        success = 1.0 if dist <= self.config.goal_radius else 0.0
        return np.asarray(
            [success, -dist, -self._energy, -float(self._collisions), -self._path_length],
            dtype=np.float32,
        )
