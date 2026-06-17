"""RLH Bench: recoverable long-horizon sparse-feedback RL environments."""

from rlh_bench.core import RewardSpec
from rlh_bench.envs import (
    DEFAULT_MAZE_REWARD_SPEC,
    DEFAULT_RESOURCE_REWARD_SPEC,
    RecoverableMazeConfig,
    RecoverablePointMazeEnv,
    RecoverableResourceAllocationEnv,
    Rectangle,
    ResourceAllocationConfig,
    make_env,
    registered_envs,
)
from rlh_bench.metrics import evaluate_policy, pareto_non_dominated, rollout

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MAZE_REWARD_SPEC",
    "DEFAULT_RESOURCE_REWARD_SPEC",
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
