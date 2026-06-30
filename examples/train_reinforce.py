"""Train a minimal REINFORCE baseline on the registered Maze env.

Requires the optional [torch] extra:
    uv pip install -e ".[torch]"

Usage:
    PYTHONPATH=src python examples/train_reinforce.py
"""

from rlh_bench import make_env
from rlh_bench.baselines.reinforce import train_reinforce


def env_factory():
    return make_env("RecoverableKeyFuelMaze-Small-v0")


result = train_reinforce(env_factory, episodes=5, hidden_size=32, seed=0)
print("returns:", result.returns)
print("losses:", result.losses)
