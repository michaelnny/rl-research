"""Train a tiny CEM baseline.

Usage:
    PYTHONPATH=src python examples/train_cem.py
"""

from rlh_bench.baselines import train_cem
from rlh_bench.envs import make_env
from rlh_bench.metrics import rollout


def env_factory():
    return make_env("RecoverablePointMaze-Small-v0")


result = train_cem(env_factory, iterations=3, population=12, elite_frac=0.25, seed=7)
final = rollout(env_factory(), result.policy, seed=123)
print("best_score:", result.best_score)
print("history:")
for row in result.history:
    print(row)
print("final reward vector:", final.reward_vector)
