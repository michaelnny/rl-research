"""Environment families exposed by RLH Bench.

After the v2 substrate redesign, the canonical families are
``RecoverableCapacityScheduling`` and ``RecoverableKeyFuelMaze``.
The legacy families remain importable but are not in
``registered_envs()``.
"""

from rlh_bench.envs.capacity_scheduling import (
    DEFAULT_SCHEDULING_REWARD_SPEC,
    CapacitySchedulingConfig,
    RecoverableCapacitySchedulingEnv,
)
from rlh_bench.envs.continuous_maze import (
    DEFAULT_MAZE_REWARD_SPEC,
    RecoverableMazeConfig,
    RecoverablePointMazeEnv,
    Rectangle,
)
from rlh_bench.envs.keyfuel_maze import (
    DEFAULT_KEYFUEL_REWARD_SPEC,
    KeyFuelMazeConfig,
    RecoverableKeyFuelMazeEnv,
)
from rlh_bench.envs.registration import make_env, registered_envs
from rlh_bench.envs.resource_allocation import (
    DEFAULT_RESOURCE_REWARD_SPEC,
    RecoverableResourceAllocationEnv,
    ResourceAllocationConfig,
)

__all__ = [
    "CapacitySchedulingConfig",
    "DEFAULT_KEYFUEL_REWARD_SPEC",
    "DEFAULT_MAZE_REWARD_SPEC",
    "DEFAULT_RESOURCE_REWARD_SPEC",
    "DEFAULT_SCHEDULING_REWARD_SPEC",
    "KeyFuelMazeConfig",
    "RecoverableCapacitySchedulingEnv",
    "RecoverableKeyFuelMazeEnv",
    "RecoverableMazeConfig",
    "RecoverablePointMazeEnv",
    "RecoverableResourceAllocationEnv",
    "Rectangle",
    "ResourceAllocationConfig",
    "make_env",
    "registered_envs",
]
