"""Baseline policies and small training loops.

Family-specific baseline portfolios:
  * :data:`rlh_bench.baselines.scheduling.SCHEDULING_BASELINES`
  * :data:`rlh_bench.baselines.maze.MAZE_BASELINES`
  * :data:`rlh_bench.baselines.maze.MAZE_ORACLE_DIAGNOSTICS` —
    privileged diagnostics, NOT comparable to learner-facing
    baselines.

Plus generic policies and training loops:
  * :class:`rlh_bench.baselines.random.RandomPolicy` /
    :class:`rlh_bench.baselines.random.ZeroPolicy`
  * :func:`rlh_bench.baselines.cem.train_cem` /
    :class:`rlh_bench.baselines.cem.LinearPolicy`
  * :func:`rlh_bench.baselines.reinforce.train_reinforce`
    (requires the optional ``[torch]`` extra)
"""

from rlh_bench.baselines.cem import CEMResult, LinearPolicy, train_cem
from rlh_bench.baselines.maze import MAZE_BASELINES, MAZE_ORACLE_DIAGNOSTICS
from rlh_bench.baselines.random import RandomPolicy, ZeroPolicy
from rlh_bench.baselines.scheduling import SCHEDULING_BASELINES

__all__ = [
    "CEMResult",
    "LinearPolicy",
    "MAZE_BASELINES",
    "MAZE_ORACLE_DIAGNOSTICS",
    "RandomPolicy",
    "SCHEDULING_BASELINES",
    "ZeroPolicy",
    "train_cem",
]
