"""Train a minimal REINFORCE baseline.

Requires torch:
    pip install -e .[torch]

Usage:
    PYTHONPATH=src python examples/train_reinforce.py
"""

from rlh_bench.baselines.reinforce import train_reinforce
from rlh_bench.envs import make_env


def env_factory():
    return make_env("RecoverableResourceAllocation-Small-v0")


result = train_reinforce(env_factory, episodes=5, hidden_size=32, seed=0)
print("returns:", result.returns)
print("losses:", result.losses)
