"""RLH Bench: recoverable long-horizon sparse-feedback RL environments.

The substrate hosts two continuous-action env families with
terminal-only sparse vector rewards on deterministic, recoverable
worlds. See ``docs/SUBSTRATE_MAP.md`` for the one-page API and
``lab/notes/planning/PLAN_substrate_redesign_v2_2026-06-30.md`` for the
design rationale.

Registered tiers (validated):
  * :class:`RecoverableCapacitySchedulingEnv` — ``RecoverableCapacityScheduling-Small-v0``
  * :class:`RecoverableKeyFuelMazeEnv` — ``RecoverableKeyFuelMaze-Small-v0``

v0 / Large tiers for both families were removed pending honest
validation; see ``lab/notes/strict_registry_outcome_2026-06-30.md``.
"""

from rlh_bench.core import RewardSpec
from rlh_bench.envs import (
    DEFAULT_KEYFUEL_REWARD_SPEC,
    DEFAULT_SCHEDULING_REWARD_SPEC,
    CapacitySchedulingConfig,
    KeyFuelMazeConfig,
    RecoverableCapacitySchedulingEnv,
    RecoverableKeyFuelMazeEnv,
    make_env,
    registered_envs,
)
from rlh_bench.metrics import evaluate_policy, pareto_non_dominated, rollout

__version__ = "0.2.0"

__all__ = [
    "CapacitySchedulingConfig",
    "DEFAULT_KEYFUEL_REWARD_SPEC",
    "DEFAULT_SCHEDULING_REWARD_SPEC",
    "KeyFuelMazeConfig",
    "RecoverableCapacitySchedulingEnv",
    "RecoverableKeyFuelMazeEnv",
    "RewardSpec",
    "evaluate_policy",
    "make_env",
    "pareto_non_dominated",
    "registered_envs",
    "rollout",
]
