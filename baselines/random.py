"""Random-policy baseline.

Trains nothing. Returns a uniform-action policy. The floor every candidate
must beat to earn `n_beat_random > 0` on a panel env.
"""

from __future__ import annotations

import time

import numpy as np

import harness


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed=seed)
    action_space = env.action_space
    env.close()

    rng = np.random.default_rng(seed + 99_999)

    def policy_fn(obs):
        if hasattr(action_space, "n"):
            return int(rng.integers(action_space.n))
        return action_space.sample()

    # Use the time budget — it's the agent contract — even though we don't train.
    end = time.monotonic() + min(time_budget_s, 1)
    while time.monotonic() < end:
        time.sleep(0.05)

    return policy_fn
