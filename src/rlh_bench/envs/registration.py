"""Lightweight environment registry.

After the v2 substrate redesign (2026-06-30), the two registered
families are:

  * ``RecoverableCapacityScheduling-{Small, v0, Large}-v0``
  * ``RecoverableKeyFuelMaze-{Small, v0, Large}-v0``

The previous families (``RecoverablePointMaze``,
``RecoverableResourceAllocation``) are retired. Their classes remain
importable for backward compatibility with any external code that
references them, but they are not in the public registry.

See ``lab/notes/PLAN_substrate_redesign_v2_2026-06-30.md`` for the
rationale.
"""

from __future__ import annotations

from typing import Any, Callable

from rlh_bench.envs.capacity_scheduling import (
    CapacitySchedulingConfig,
    RecoverableCapacitySchedulingEnv,
)
from rlh_bench.envs.continuous_maze import RecoverableMazeConfig, RecoverablePointMazeEnv
from rlh_bench.envs.keyfuel_maze import KeyFuelMazeConfig, RecoverableKeyFuelMazeEnv
from rlh_bench.envs.resource_allocation import (
    RecoverableResourceAllocationEnv,
    ResourceAllocationConfig,
)


RegistryFn = Callable[..., object]


# ----- CapacityScheduling --------------------------------------------------- #


def _scheduling_small(**kwargs: Any) -> RecoverableCapacitySchedulingEnv:
    return RecoverableCapacitySchedulingEnv(
        config=CapacitySchedulingConfig(
            horizon=500, num_projects=16, num_modes=4, num_products=4,
            action_dim=32, n_bundles=2,
        ),
        **kwargs,
    )


def _scheduling_default(**kwargs: Any) -> RecoverableCapacitySchedulingEnv:
    return RecoverableCapacitySchedulingEnv(
        config=CapacitySchedulingConfig(),  # v0 defaults: H=2000, K=48, M=8, P=8
        **kwargs,
    )


def _scheduling_large(**kwargs: Any) -> RecoverableCapacitySchedulingEnv:
    return RecoverableCapacitySchedulingEnv(
        config=CapacitySchedulingConfig(
            horizon=10000, num_projects=128, num_modes=16, num_products=16,
            action_dim=224, n_bundles=24, bundle_size_range=(3, 6),
            demand_peak_width_range=(60, 200),
        ),
        **kwargs,
    )


# ----- KeyFuelMaze ----------------------------------------------------------- #


def _keyfuel_small(**kwargs: Any) -> RecoverableKeyFuelMazeEnv:
    return RecoverableKeyFuelMazeEnv(
        config=KeyFuelMazeConfig(
            horizon=500, action_dim=16, world_size=24.0,
            n_key_types=2, n_seals=2, n_gates=1, n_fuel_stations=2,
        ),
        **kwargs,
    )


def _keyfuel_default(**kwargs: Any) -> RecoverableKeyFuelMazeEnv:
    return RecoverableKeyFuelMazeEnv(
        config=KeyFuelMazeConfig(),  # v0 defaults: H=2000, D=32, 48×48
        **kwargs,
    )


def _keyfuel_large(**kwargs: Any) -> RecoverableKeyFuelMazeEnv:
    return RecoverableKeyFuelMazeEnv(
        config=KeyFuelMazeConfig(
            horizon=10000, action_dim=64, world_size=96.0,
            n_key_types=6, n_seals=12, n_gates=8, n_fuel_stations=8,
        ),
        **kwargs,
    )


_REGISTRY: dict[str, RegistryFn] = {
    "RecoverableCapacityScheduling-Small-v0": _scheduling_small,
    "RecoverableCapacityScheduling-v0": _scheduling_default,
    "RecoverableCapacityScheduling-Large-v0": _scheduling_large,
    "RecoverableKeyFuelMaze-Small-v0": _keyfuel_small,
    "RecoverableKeyFuelMaze-v0": _keyfuel_default,
    "RecoverableKeyFuelMaze-Large-v0": _keyfuel_large,
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
