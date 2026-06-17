"""Wrappers and optional Gymnasium interop."""

from __future__ import annotations

from typing import Any

import numpy as np

from rlh_bench.core import Env
from rlh_bench.spaces import Box, Discrete, MultiDiscrete, Space


class ScalarizeRewardWrapper:
    """Convert vector rewards returned by an environment into scalar rewards.

    This wrapper is useful when an environment is constructed with
    ``reward_mode='vector'`` but a single-objective baseline is being run. The
    original vector remains available in ``info['reward_vector']``.
    """

    def __init__(self, env: Env, weights: np.ndarray | list[float] | tuple[float, ...]):
        self.env = env
        self.weights = np.asarray(weights, dtype=np.float32)
        if self.weights.shape != (env.reward_dim,):
            raise ValueError(f"weights must have shape ({env.reward_dim},)")
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_dim = env.reward_dim

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vector = info.get("reward_vector", reward)
        scalar = float(np.dot(self.weights, np.asarray(vector, dtype=np.float32)))
        return obs, scalar, terminated, truncated, info

    def diagnostics(self):
        return self.env.diagnostics()

    def render(self):
        return self.env.render()


class GymnasiumAdapter:
    """Expose an RLH Bench environment as a real ``gymnasium.Env``.

    Gymnasium is an optional dependency. The adapter converts the package's
    small self-contained spaces into ``gymnasium.spaces``.
    """

    def __init__(self, env: Env):
        try:
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "GymnasiumAdapter requires gymnasium. Install with `pip install rlh-bench[gymnasium]`."
            ) from exc

        class _AdaptedEnv(gym.Env):
            metadata = getattr(env, "metadata", {})

            def __init__(self, wrapped: Env):
                super().__init__()
                self.wrapped = wrapped
                self.observation_space = _to_gym_space(wrapped.observation_space)
                self.action_space = _to_gym_space(wrapped.action_space)
                self.reward_dim = wrapped.reward_dim

            def reset(self, seed=None, options=None):
                return self.wrapped.reset(seed=seed, options=options)

            def step(self, action):
                return self.wrapped.step(action)

            def render(self):
                if hasattr(self.wrapped, "render"):
                    return self.wrapped.render()
                return None

            def close(self):
                return None

        self.env = _AdaptedEnv(env)

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def _to_gym_space(space: Space):
    try:
        from gymnasium import spaces
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("gymnasium is required to convert spaces") from exc

    if isinstance(space, Box):
        return spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
    if isinstance(space, Discrete):
        return spaces.Discrete(space.n)
    if isinstance(space, MultiDiscrete):
        return spaces.MultiDiscrete(space.nvec)
    raise TypeError(f"unsupported space type: {type(space)!r}")
