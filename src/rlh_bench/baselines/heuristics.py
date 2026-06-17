"""Lightweight heuristic policies for benchmark validation.

These are intentionally simple. Their purpose is not to be a strong research
baseline; they verify that the environments are feasible and recoverable.
"""

from __future__ import annotations

import numpy as np

from rlh_bench.envs.continuous_maze import RecoverablePointMazeEnv
from rlh_bench.envs.resource_allocation import RecoverableResourceAllocationEnv
from rlh_bench.spaces import Box


class MazeWaypointPolicy:
    """PD waypoint-following policy for :class:`RecoverablePointMazeEnv`."""

    def __init__(self, env: RecoverablePointMazeEnv, waypoint_radius: float = 0.07):
        self.env = env
        self.waypoints = [np.asarray(wp, dtype=np.float32) for wp in env.config.waypoints]
        self.goal = np.asarray(env.config.goal, dtype=np.float32)
        if len(self.waypoints) == 0 or np.linalg.norm(self.waypoints[-1] - self.goal) > 1e-6:
            self.waypoints.append(self.goal)
        self.waypoint_radius = waypoint_radius
        self.index = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        pos = obs[0:2].astype(np.float32)
        vel = obs[2:4].astype(np.float32) * self.env.config.max_speed

        while self.index < len(self.waypoints) - 1:
            if np.linalg.norm(self.waypoints[self.index] - pos) <= self.waypoint_radius:
                self.index += 1
            else:
                break

        target = self.waypoints[self.index]
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-8:
            direction = delta / dist
        else:
            direction = np.zeros(2, dtype=np.float32)

        desired_vel = direction * self.env.config.max_speed
        control = 1.7 * direction + 2.0 * (desired_vel - vel) / max(self.env.config.max_speed, 1e-8)
        control = np.clip(control, -1.0, 1.0).astype(np.float32)

        # Repeat the same desired acceleration over redundant actuator pairs.
        action = np.tile(control, self.env.config.action_dim // 2).astype(np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)


class ResourceGreedyPolicy:
    """Allocate budget to the most deficient ready project."""

    def __init__(self, env: RecoverableResourceAllocationEnv):
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        k = self.env.config.num_projects
        ratios = obs[:k]
        readiness = obs[k : 2 * k]
        shortage = np.maximum(1.0 - ratios, 0.0)

        # Prioritize high shortage and high readiness. Add a small upstream bias
        # so the dependency chain is solved from left to right when tied.
        upstream_bias = np.linspace(0.03, 0.0, k, dtype=np.float32)
        raw_score = shortage * (0.25 + readiness) + upstream_bias
        if np.all(shortage <= 1e-6):
            return np.zeros(k, dtype=np.float32)
        score = np.where(shortage > 1e-6, raw_score, -np.inf)

        allocation = np.zeros(k, dtype=np.float32)
        remaining = self.env.config.budget
        # Put safe allocation on the best project first, then use remaining on
        # the next best. This gives a less degenerate vector reward than always
        # maxing one project.
        for idx in np.argsort(score)[::-1]:
            if remaining <= 1e-8:
                break
            if shortage[idx] <= 1e-6:
                continue
            amount = min(remaining, max(self.env.config.safe_allocation, 1e-6))
            allocation[idx] = amount
            remaining -= amount
        if remaining > 1e-8:
            best = int(np.argmax(score))
            allocation[best] += remaining
        return np.clip(allocation, self.env.action_space.low, self.env.action_space.high).astype(np.float32)


def make_heuristic_policy(env):
    """Return the default heuristic for a supported environment."""

    if isinstance(env, RecoverablePointMazeEnv):
        return MazeWaypointPolicy(env)
    if isinstance(env, RecoverableResourceAllocationEnv):
        return ResourceGreedyPolicy(env)
    raise TypeError(f"no heuristic available for {type(env)!r}")
