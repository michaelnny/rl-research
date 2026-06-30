"""Run the baseline portfolios on the registered env families.

Usage:
    PYTHONPATH=src python examples/run_heuristics.py
"""

import numpy as np

from rlh_bench import make_env, registered_envs
from rlh_bench.baselines import (
    MAZE_BASELINES,
    SCHEDULING_BASELINES,
    ZeroPolicy,
)
from rlh_bench.metrics import rollout


def _portfolio_for(env_id: str):
    """Yield (name, factory) pairs of cheap-to-run learner-facing baselines."""
    yield "zero", lambda env: ZeroPolicy(env.action_space)
    if "Scheduling" in env_id:
        for cls in SCHEDULING_BASELINES:
            if cls.__name__ == "SchedulingZeroPolicy":
                continue
            yield cls.name, lambda env, cls=cls: cls(env)
    elif "KeyFuelMaze" in env_id:
        for cls in MAZE_BASELINES:
            if cls.__name__ == "MazeZeroPolicy":
                continue
            yield cls.name, lambda env, cls=cls: cls(env)


for env_id in registered_envs():
    print(f"\n=== {env_id} ===")
    for name, factory in _portfolio_for(env_id):
        env = make_env(env_id, reward_mode="vector")
        policy = factory(env)
        result = rollout(env, policy, seed=0)
        rv = np.round(result.reward_vector, 3).tolist()
        print(f"  {name:30s} scalar={result.scalar_return:+.3f} vector={rv}")
