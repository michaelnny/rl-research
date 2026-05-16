"""Frozen baselines.

The single allowed RL algorithm in this project (see `docs/charter.md` §Hard rules,
exception 1). PPO is implemented from primitives, frozen with per-domain
hyperparameters, and used as the yardstick the Curator weighs candidates against.

Do not re-tune per candidate. Do not import from any RL algorithm library.
"""
