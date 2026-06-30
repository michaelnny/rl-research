"""Baseline policies and small training loops.

After the v2 substrate redesign (2026-06-30), the canonical
baselines are the family-specific portfolios in
``rlh_bench.baselines.scheduling`` and ``rlh_bench.baselines.maze``.
The legacy ``MazeWaypointPolicy``, ``ResourceGreedyPolicy``, and
``make_heuristic_policy`` remain importable for backward
compatibility with the legacy env classes.
"""

from rlh_bench.baselines.cem import CEMResult, LinearPolicy, train_cem
from rlh_bench.baselines.heuristics import (
    MazeWaypointPolicy,
    ResourceGreedyPolicy,
    make_heuristic_policy,
)
from rlh_bench.baselines.maze import MAZE_BASELINES, MAZE_ORACLE_DIAGNOSTICS
from rlh_bench.baselines.random import RandomPolicy, ZeroPolicy
from rlh_bench.baselines.scheduling import SCHEDULING_BASELINES

__all__ = [
    "CEMResult",
    "LinearPolicy",
    "MAZE_BASELINES",
    "MAZE_ORACLE_DIAGNOSTICS",
    "MazeWaypointPolicy",
    "RandomPolicy",
    "ResourceGreedyPolicy",
    "SCHEDULING_BASELINES",
    "ZeroPolicy",
    "make_heuristic_policy",
    "train_cem",
]
