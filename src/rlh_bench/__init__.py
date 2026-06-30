"""RLH Bench: recoverable long-horizon sparse-feedback RL environments.

After the v2 substrate redesign (2026-06-30), the canonical env
families are :class:`RecoverableCapacitySchedulingEnv` and
:class:`RecoverableKeyFuelMazeEnv`. The legacy env classes
(:class:`RecoverablePointMazeEnv`,
:class:`RecoverableResourceAllocationEnv`) remain importable but
are not part of the public registry.

See ``docs/SUBSTRATE_MAP.md`` for the one-page substrate API and
``lab/notes/PLAN_substrate_redesign_v2_2026-06-30.md`` for the
rationale behind the redesign.
"""

from rlh_bench.core import RewardSpec
from rlh_bench.envs import (
    DEFAULT_KEYFUEL_REWARD_SPEC,
    DEFAULT_MAZE_REWARD_SPEC,
    DEFAULT_RESOURCE_REWARD_SPEC,
    DEFAULT_SCHEDULING_REWARD_SPEC,
    CapacitySchedulingConfig,
    KeyFuelMazeConfig,
    RecoverableCapacitySchedulingEnv,
    RecoverableKeyFuelMazeEnv,
    RecoverableMazeConfig,
    RecoverablePointMazeEnv,
    RecoverableResourceAllocationEnv,
    Rectangle,
    ResourceAllocationConfig,
    make_env,
    registered_envs,
)
from rlh_bench.metrics import evaluate_policy, pareto_non_dominated, rollout

__version__ = "0.2.0"

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
    "RewardSpec",
    "evaluate_policy",
    "make_env",
    "pareto_non_dominated",
    "registered_envs",
    "rollout",
]
