"""Cross-Entropy Method policy-search baseline.

CEM is included as a CPU-friendly classic baseline for deterministic continuous
control and structured continuous-action tasks. It optimizes a linear tanh policy
without requiring gradients or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rlh_bench.core import Env
from rlh_bench.metrics import rollout
from rlh_bench.spaces import Box


@dataclass
class LinearPolicy:
    """Linear policy with tanh squashing into a Box action space."""

    params: np.ndarray
    obs_dim: int
    action_space: Box

    def __post_init__(self) -> None:
        if not isinstance(self.action_space, Box):
            raise TypeError("LinearPolicy supports Box action spaces only")
        action_dim = int(np.prod(self.action_space.shape))
        expected = action_dim * (self.obs_dim + 1)
        if self.params.shape != (expected,):
            raise ValueError(f"expected params shape ({expected},), got {self.params.shape}")

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        features = np.concatenate([obs, np.ones(1, dtype=np.float32)])
        weights = self.params.reshape(self.action_dim, self.obs_dim + 1)
        raw = np.tanh(weights @ features)
        low = self.action_space.low.reshape(-1)
        high = self.action_space.high.reshape(-1)
        action = low + (raw + 1.0) * 0.5 * (high - low)
        return action.reshape(self.action_space.shape).astype(self.action_space.dtype)


@dataclass
class CEMResult:
    """Training output for :func:`train_cem`."""

    best_params: np.ndarray
    best_score: float
    mean: np.ndarray
    std: np.ndarray
    history: list[dict[str, float]]
    policy: LinearPolicy


def policy_num_params(env: Env) -> int:
    """Number of linear policy parameters for an environment."""

    if not isinstance(env.action_space, Box):
        raise TypeError("CEM supports Box action spaces only")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return action_dim * (obs_dim + 1)


def evaluate_params(
    env_factory: Callable[[], Env], params: np.ndarray, episodes: int = 1, seed: int = 0
) -> float:
    """Evaluate one parameter vector over one or more episodes."""

    scores = []
    for ep in range(episodes):
        env = env_factory()
        if not isinstance(env.action_space, Box):
            raise TypeError("CEM supports Box action spaces only")
        obs_dim = int(np.prod(env.observation_space.shape))
        policy = LinearPolicy(params=params, obs_dim=obs_dim, action_space=env.action_space)
        scores.append(rollout(env, policy, seed=seed + ep).scalar_return)
    return float(np.mean(scores))


def train_cem(
    env_factory: Callable[[], Env],
    iterations: int = 10,
    population: int = 32,
    elite_frac: float = 0.20,
    init_std: float = 0.7,
    min_std: float = 0.03,
    eval_episodes: int = 1,
    seed: int = 0,
) -> CEMResult:
    """Train a linear policy with the Cross-Entropy Method.

    The function is deliberately small and deterministic under ``seed``. It is
    intended as a benchmark sanity check, not as a state-of-the-art optimizer.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if population <= 1:
        raise ValueError("population must be at least 2")
    if not (0.0 < elite_frac <= 1.0):
        raise ValueError("elite_frac must be in (0, 1]")

    rng = np.random.default_rng(seed)
    probe_env = env_factory()
    if not isinstance(probe_env.action_space, Box):
        raise TypeError("CEM supports Box action spaces only")
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    dim = policy_num_params(probe_env)
    mean = np.zeros(dim, dtype=np.float32)
    std = np.full(dim, init_std, dtype=np.float32)
    elite_n = max(1, int(round(population * elite_frac)))

    best_params = mean.copy()
    best_score = -np.inf
    history: list[dict[str, float]] = []

    for it in range(iterations):
        samples = rng.normal(loc=mean, scale=std, size=(population, dim)).astype(np.float32)
        scores = np.asarray(
            [
                evaluate_params(env_factory, sample, episodes=eval_episodes, seed=seed + 10_000 * it + i)
                for i, sample in enumerate(samples)
            ],
            dtype=np.float32,
        )
        order = np.argsort(scores)[::-1]
        elites = samples[order[:elite_n]]
        mean = elites.mean(axis=0).astype(np.float32)
        std = np.maximum(elites.std(axis=0).astype(np.float32), min_std)

        if float(scores[order[0]]) > best_score:
            best_score = float(scores[order[0]])
            best_params = samples[order[0]].copy()

        history.append(
            {
                "iteration": float(it),
                "mean_score": float(np.mean(scores)),
                "best_score": float(np.max(scores)),
                "elite_mean_score": float(np.mean(scores[order[:elite_n]])),
                "std_mean": float(np.mean(std)),
            }
        )

    policy = LinearPolicy(params=best_params, obs_dim=obs_dim, action_space=probe_env.action_space)
    return CEMResult(
        best_params=best_params,
        best_score=best_score,
        mean=mean,
        std=std,
        history=history,
        policy=policy,
    )
