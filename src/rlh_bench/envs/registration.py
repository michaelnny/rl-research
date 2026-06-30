"""Lightweight environment registry.

Strict-validation policy
------------------------

``registered_envs()`` returns only env IDs that have been
calibrated, peer-reviewed by Codex, and pass the acceptance gates
in the v2 substrate redesign plan
(`lab/notes/PLAN_substrate_redesign_v2_2026-06-30.md`) to a tight
bar. An env that we couldn't validate honestly is removed from
the codebase — not registered with a warning, not kept as
"experimental." If it can't carry the weight of being a testbed,
it doesn't belong here.

Currently registered (validated):
  * ``RecoverableCapacityScheduling-Small-v0``
  * ``RecoverableKeyFuelMaze-Small-v0``

History: the v2 redesign initially shipped three tiers per family
(Small / v0 / Large). Subsequent validation found:

  * Sched-v0: no calibration setting made trivial policies fail
    while smart policies succeed — uniform vs bundle_aware sit on
    opposite sides of every threshold curve. Removed.
  * Sched-Large: dynamics inherit from v0 but never had a baseline
    sweep complete; could harbor latent bugs at K=128 / M=16 /
    H=10000 scale. Removed pending validation.
  * Maze-v0: feasible under oracle (priviledged info), no honest
    observation-only baseline reached success. Could be feasible
    but unreached, could be subtly impossible. Removed pending an
    honest baseline that demonstrates learnability.
  * Maze-Large: same as Sched-Large.

The non-trivial baselines for those tiers (e.g.
``SchedulingBundleAwarePolicy``) remain in the codebase because
they're useful on the Small tier and on future re-validations.

The previous families (``RecoverablePointMaze``,
``RecoverableResourceAllocation``) are retired entirely. Their
classes remain importable for backward compatibility.
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


# ----- KeyFuelMaze ----------------------------------------------------------- #


def _keyfuel_small(**kwargs: Any) -> RecoverableKeyFuelMazeEnv:
    return RecoverableKeyFuelMazeEnv(
        config=KeyFuelMazeConfig(
            horizon=500, action_dim=16, world_size=24.0,
            n_key_types=2, n_seals=2, n_gates=1, n_fuel_stations=2,
        ),
        **kwargs,
    )


_REGISTRY: dict[str, RegistryFn] = {
    "RecoverableCapacityScheduling-Small-v0": _scheduling_small,
    "RecoverableKeyFuelMaze-Small-v0": _keyfuel_small,
}


def registered_envs() -> tuple[str, ...]:
    """Return available environment IDs.

    Only envs that pass strict validation are returned. To add a
    new env: land a commit that satisfies the acceptance gates,
    has a Codex review, and explicitly re-adds it to ``_REGISTRY``
    below.
    """

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
