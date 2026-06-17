"""Core types for RLH Bench environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from rlh_bench.spaces import Space


StepReturn = tuple[np.ndarray, float | np.ndarray, bool, bool, dict[str, Any]]
ResetReturn = tuple[np.ndarray, dict[str, Any]]


class Env(Protocol):
    """Small Gymnasium-like environment protocol.

    Environments implement the post-Gym-0.26 five-element step API:
    ``observation, reward, terminated, truncated, info``. The package is
    self-contained and does not require Gymnasium at runtime, but adapters are
    supplied in :mod:`rlh_bench.wrappers`.
    """

    observation_space: Space
    action_space: Space
    reward_dim: int

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> ResetReturn:
        ...

    def step(self, action: np.ndarray) -> StepReturn:
        ...


@dataclass(frozen=True)
class RewardSpec:
    """Terminal reward vector metadata and scalarization weights.

    ``names`` specify the vector reward components. ``weights`` are used only
    when an environment is in ``reward_mode='scalar'``. Vector rewards are always
    available in ``info['reward_vector']``.
    """

    names: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) == 0:
            raise ValueError("RewardSpec must contain at least one component")
        if len(self.names) != len(self.weights):
            raise ValueError("names and weights must have the same length")

    @property
    def dim(self) -> int:
        return len(self.names)

    def scalarize(self, vector: np.ndarray) -> float:
        vector = np.asarray(vector, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        if vector.shape != weights.shape:
            raise ValueError(f"expected reward vector shape {weights.shape}, got {vector.shape}")
        return float(np.dot(weights, vector))


def make_reward(reward_mode: str, spec: RewardSpec, vector: np.ndarray) -> float | np.ndarray:
    """Return scalar or vector reward according to environment reward mode."""

    if reward_mode == "scalar":
        return spec.scalarize(vector)
    if reward_mode == "vector":
        return np.asarray(vector, dtype=np.float32).copy()
    raise ValueError("reward_mode must be either 'scalar' or 'vector'")


def zero_reward(reward_mode: str, spec: RewardSpec) -> float | np.ndarray:
    """Non-terminal zero reward in scalar or vector mode."""

    if reward_mode == "scalar":
        return 0.0
    if reward_mode == "vector":
        return np.zeros(spec.dim, dtype=np.float32)
    raise ValueError("reward_mode must be either 'scalar' or 'vector'")


PolicyFn = Callable[[np.ndarray], np.ndarray]
