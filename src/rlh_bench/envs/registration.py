"""Lightweight environment registry."""

from __future__ import annotations

from typing import Any, Callable

from rlh_bench.envs.continuous_maze import RecoverableMazeConfig, RecoverablePointMazeEnv
from rlh_bench.envs.resource_allocation import (
    RecoverableResourceAllocationEnv,
    ResourceAllocationConfig,
)


RegistryFn = Callable[..., object]


def _maze_small(**kwargs: Any) -> RecoverablePointMazeEnv:
    return RecoverablePointMazeEnv(config=RecoverableMazeConfig(horizon=120), **kwargs)


def _maze_default(**kwargs: Any) -> RecoverablePointMazeEnv:
    return RecoverablePointMazeEnv(config=RecoverableMazeConfig(), **kwargs)


def _maze_hd(**kwargs: Any) -> RecoverablePointMazeEnv:
    return RecoverablePointMazeEnv(config=RecoverableMazeConfig(action_dim=8, horizon=180), **kwargs)


def _resource_small(**kwargs: Any) -> RecoverableResourceAllocationEnv:
    return RecoverableResourceAllocationEnv(
        config=ResourceAllocationConfig(horizon=60, num_projects=4), **kwargs
    )


def _resource_default(**kwargs: Any) -> RecoverableResourceAllocationEnv:
    return RecoverableResourceAllocationEnv(config=ResourceAllocationConfig(), **kwargs)


def _resource_large(**kwargs: Any) -> RecoverableResourceAllocationEnv:
    return RecoverableResourceAllocationEnv(
        config=ResourceAllocationConfig(horizon=120, num_projects=8), **kwargs
    )


_REGISTRY: dict[str, RegistryFn] = {
    "RecoverablePointMaze-Small-v0": _maze_small,
    "RecoverablePointMaze-v0": _maze_default,
    "RecoverablePointMaze-HD-v0": _maze_hd,
    "RecoverableResourceAllocation-Small-v0": _resource_small,
    "RecoverableResourceAllocation-v0": _resource_default,
    "RecoverableResourceAllocation-Large-v0": _resource_large,
}


def registered_envs() -> tuple[str, ...]:
    """Return available environment IDs."""

    return tuple(sorted(_REGISTRY))


def make_env(env_id: str, **kwargs: Any):
    """Construct an environment by ID.

    Args:
        env_id: One of :func:`registered_envs`.
        **kwargs: Forwarded to the environment constructor. For example,
            ``reward_mode='vector'``.
    """

    if env_id not in _REGISTRY:
        available = ", ".join(registered_envs())
        raise KeyError(f"unknown env_id {env_id!r}; available: {available}")
    return _REGISTRY[env_id](**kwargs)
