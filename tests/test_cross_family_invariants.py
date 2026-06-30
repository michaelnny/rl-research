"""Cross-family substrate invariants — parametrized over all registered envs.

These tests cover acceptance gate properties that the per-family
test files only spot-check at Small tier. By running across the
full registry, we catch the case where a property holds at Small
but breaks at v0 / Large (e.g. a unit dimensionality issue in the
Large config).

Tests here should be cheap. Anything that would take minutes per
env belongs in a probe under `experiments/probes/`, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlh_bench import make_env, registered_envs


@pytest.mark.parametrize("env_id", registered_envs())
def test_reward_is_terminal_only_all_envs(env_id: str) -> None:
    """Gate 2 (terminal-only) on every registered env at every tier.

    Use a small fixed action sequence to keep wall-clock down.
    """
    env = make_env(env_id, reward_mode="vector")
    obs, _ = env.reset(seed=0)
    horizon = env.config.horizon
    action = 0.5 * np.ones(env.action_space.shape, dtype=np.float32)
    action = np.clip(action, env.action_space.low, env.action_space.high)
    nonzero = 0
    for _ in range(horizon):
        obs, r, term, trunc, info = env.step(action)
        if np.any(np.asarray(r) != 0):
            nonzero += 1
        if term:
            break
    assert nonzero == 1, (
        f"{env_id}: terminal-only contract violated; "
        f"got {nonzero} non-zero rewards (expected 1)"
    )


@pytest.mark.parametrize("env_id", registered_envs())
def test_determinism_all_envs(env_id: str) -> None:
    """Gate 1 (determinism) on every registered env."""
    actions = np.random.default_rng(42).uniform(
        -1, 1, size=(20, make_env(env_id).action_space.shape[0])
    ).astype(np.float32)

    def _run():
        env = make_env(env_id, reward_mode="vector")
        obs, _ = env.reset(seed=7)
        last_info = None
        for a in actions:
            obs, r, term, trunc, last_info = env.step(a)
            if term:
                break
        return obs, last_info["reward_vector"]

    obs1, rv1 = _run()
    obs2, rv2 = _run()
    assert np.allclose(obs1, obs2), f"{env_id}: terminal observation not deterministic"
    assert np.allclose(rv1, rv2), f"{env_id}: terminal reward vector not deterministic"


@pytest.mark.parametrize("env_id", registered_envs())
def test_info_contract_all_envs(env_id: str) -> None:
    """``info`` always contains ``reward_vector``, ``reward_names``,
    ``is_success`` — at reset and after every step."""
    env = make_env(env_id, reward_mode="vector")
    obs, info = env.reset(seed=0)
    for key in ("reward_vector", "reward_names", "is_success"):
        assert key in info, f"{env_id}: info missing {key!r} at reset"
    obs, r, term, trunc, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    for key in ("reward_vector", "reward_names", "is_success"):
        assert key in info, f"{env_id}: info missing {key!r} after step"


# The KeyFuelMaze cross-tier reward-normalization test and the
# Maze-v0 idle-tail test were removed when Maze-v0 / Maze-Large
# were deleted from the registry pending re-validation. They'll
# come back when those tiers re-pass validation gates.
