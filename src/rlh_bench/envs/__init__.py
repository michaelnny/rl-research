"""Environment families exposed by RLH Bench.

Two registered families, both continuous-action:
  * ``RecoverableCapacityScheduling`` — allocation / scheduling
  * ``RecoverableKeyFuelMaze`` — continuous control with route +
    fuel + key-gated objectives

See ``rlh_bench.envs.registration`` for the full registry.
"""

from rlh_bench.envs.capacity_scheduling import (
    DEFAULT_SCHEDULING_REWARD_SPEC,
    CapacitySchedulingConfig,
    RecoverableCapacitySchedulingEnv,
)
from rlh_bench.envs.keyfuel_maze import (
    DEFAULT_KEYFUEL_REWARD_SPEC,
    KeyFuelMazeConfig,
    RecoverableKeyFuelMazeEnv,
)
from rlh_bench.envs.registration import make_env, registered_envs

__all__ = [
    "CapacitySchedulingConfig",
    "DEFAULT_KEYFUEL_REWARD_SPEC",
    "DEFAULT_SCHEDULING_REWARD_SPEC",
    "KeyFuelMazeConfig",
    "RecoverableCapacitySchedulingEnv",
    "RecoverableKeyFuelMazeEnv",
    "make_env",
    "registered_envs",
]
