"""Environment families exposed by RLH Bench."""

from rlh_bench.envs.continuous_maze import (
    DEFAULT_MAZE_REWARD_SPEC,
    RecoverableMazeConfig,
    RecoverablePointMazeEnv,
    Rectangle,
)
from rlh_bench.envs.registration import make_env, registered_envs
from rlh_bench.envs.resource_allocation import (
    DEFAULT_RESOURCE_REWARD_SPEC,
    RecoverableResourceAllocationEnv,
    ResourceAllocationConfig,
)

__all__ = [
    "DEFAULT_MAZE_REWARD_SPEC",
    "DEFAULT_RESOURCE_REWARD_SPEC",
    "RecoverableMazeConfig",
    "RecoverablePointMazeEnv",
    "RecoverableResourceAllocationEnv",
    "Rectangle",
    "ResourceAllocationConfig",
    "make_env",
    "registered_envs",
]
