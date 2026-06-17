"""Evaluation helpers for terminal-reward environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rlh_bench.core import Env


Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class EpisodeResult:
    """Data captured from one rollout."""

    scalar_return: float
    reward_vector: np.ndarray
    length: int
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EvaluationSummary:
    """Aggregate metrics over multiple episodes."""

    mean_return: float
    std_return: float
    mean_reward_vector: np.ndarray
    success_rate: float
    mean_length: float
    episodes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "mean_reward_vector": self.mean_reward_vector,
            "success_rate": self.success_rate,
            "mean_length": self.mean_length,
            "episodes": self.episodes,
        }


def rollout(env: Env, policy: Policy, seed: int | None = None, max_steps: int | None = None) -> EpisodeResult:
    """Run one episode and return scalar/vector terminal outcomes."""

    obs, _ = env.reset(seed=seed)
    total = 0.0
    last_info: dict[str, Any] = {}
    reward_vector = np.zeros(env.reward_dim, dtype=np.float32)
    terminated = truncated = False
    steps = 0

    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if isinstance(reward, np.ndarray):
            # Vector-mode envs are summarized with the environment's default
            # reward_spec when available; otherwise use the vector sum.
            if hasattr(env, "reward_spec"):
                total += env.reward_spec.scalarize(reward)
            else:
                total += float(np.sum(reward))
        else:
            total += float(reward)
        reward_vector = np.asarray(info.get("reward_vector", reward_vector), dtype=np.float32)
        last_info = info
        steps += 1
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            truncated = True
            break

    return EpisodeResult(
        scalar_return=total,
        reward_vector=reward_vector,
        length=steps,
        terminated=terminated,
        truncated=truncated,
        info=last_info,
    )


def evaluate_policy(
    env_factory: Callable[[], Env],
    policy_factory: Callable[[], Policy] | Callable[[Env], Policy],
    episodes: int = 5,
    seed: int = 0,
) -> EvaluationSummary:
    """Evaluate a policy over deterministic or seeded environments."""

    returns: list[float] = []
    vectors: list[np.ndarray] = []
    successes: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        env = env_factory()
        try:
            policy = policy_factory(env)  # type: ignore[misc]
        except TypeError:
            policy = policy_factory()  # type: ignore[operator]
        result = rollout(env, policy, seed=seed + ep)
        returns.append(result.scalar_return)
        vectors.append(result.reward_vector)
        successes.append(float(result.info.get("is_success", result.info.get("success", 0.0))))
        lengths.append(result.length)

    return EvaluationSummary(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_reward_vector=np.mean(np.stack(vectors), axis=0).astype(np.float32),
        success_rate=float(np.mean(successes)),
        mean_length=float(np.mean(lengths)),
        episodes=episodes,
    )


def first_success_episode(successes: list[bool] | np.ndarray) -> int | None:
    """Return 1-indexed first success episode or ``None`` when absent."""

    for idx, value in enumerate(successes, start=1):
        if bool(value):
            return idx
    return None


def pareto_non_dominated(points: np.ndarray) -> np.ndarray:
    """Return a mask selecting non-dominated reward vectors.

    All reward dimensions are assumed to be maximized.
    """

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array")
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if np.any(dominated_by):
            keep[i] = False
    return keep
