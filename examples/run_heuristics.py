"""Run the built-in heuristic policies on both benchmark families.

Usage:
    PYTHONPATH=src python examples/run_heuristics.py
"""

from rlh_bench.baselines import make_heuristic_policy
from rlh_bench.envs import make_env
from rlh_bench.metrics import rollout


for env_id in ["RecoverablePointMaze-v0", "RecoverableResourceAllocation-v0"]:
    env = make_env(env_id)
    policy = make_heuristic_policy(env)
    result = rollout(env, policy, seed=0)
    print(f"\n{env_id}")
    print(f"  scalar_return: {result.scalar_return:.3f}")
    print(f"  reward_vector: {result.reward_vector}")
    print(f"  diagnostics: {result.info}")
