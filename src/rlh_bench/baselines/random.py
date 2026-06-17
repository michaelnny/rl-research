"""Random and constant policies."""

from __future__ import annotations

import numpy as np

from rlh_bench.spaces import Box, Space


class RandomPolicy:
    """Sample each action independently from the environment action space."""

    def __init__(self, action_space: Space, seed: int | None = None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return self.action_space.sample(self.rng)


class ZeroPolicy:
    """Return the zero action when supported by a Box action space."""

    def __init__(self, action_space: Box):
        if not isinstance(action_space, Box):
            raise TypeError("ZeroPolicy currently supports Box action spaces only")
        self.action_space = action_space

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.clip(np.zeros(self.action_space.shape, dtype=self.action_space.dtype), self.action_space.low, self.action_space.high)
