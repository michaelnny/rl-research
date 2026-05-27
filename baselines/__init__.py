"""Frozen baseline algorithms — used by `scripts/build_baselines.py` to
populate `baselines.json`. Not edited by the agent.

Each module exposes:
    train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn

with the same contract as `train.py`'s `train()`. The build script imports
them, runs each on each panel env, and pins the max score per env into
`baselines.json` as the `strong` baseline; the random module's score is
pinned as the `random` baseline.
"""
