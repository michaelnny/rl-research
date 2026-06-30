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


def test_keyfuel_reward_components_stay_within_3x_cross_tier() -> None:
    """Gate 10 for KeyFuelMaze (Scheduling is covered separately in
    test_capacity_scheduling_env.py).

    Cost components should be comparable across Small / v0 / Large
    under a random policy.
    """
    def _measure(env_id: str, n_seeds: int = 3) -> np.ndarray:
        vectors = []
        for seed in range(n_seeds):
            env = make_env(env_id, reward_mode="vector")
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed + 1000)
            for _ in range(env.config.horizon):
                obs, r, term, trunc, info = env.step(
                    rng.uniform(-1, 1, size=env.action_space.shape[0]).astype(np.float32)
                )
                if term:
                    break
            vectors.append(info["reward_vector"])
        return np.mean(vectors, axis=0)

    small = _measure("RecoverableKeyFuelMaze-Small-v0")
    large = _measure("RecoverableKeyFuelMaze-Large-v0")
    names = make_env("RecoverableKeyFuelMaze-Small-v0").reward_spec.names
    for i, name in enumerate(names):
        s, l = abs(float(small[i])), abs(float(large[i]))
        if s < 1e-3 or l < 1e-3:
            continue
        ratio = max(s, l) / min(s, l)
        assert ratio <= 3.0, (
            f"KeyFuelMaze reward component {name!r} differs by "
            f"{ratio:.2f}x across Small ({s:.3f}) vs Large "
            f"({l:.3f}); must be <= 3x"
        )


def test_idle_tail_maze_v0() -> None:
    """Gate 4 for KeyFuelMaze v0: zeroing the last 25% of trajectory
    should shift terminal vector by at least 0.01 on some component.

    Maze-Small has its own gate 4 test; this fills the v0 gap.
    """
    env_id = "RecoverableKeyFuelMaze-v0"
    def _run(zero_after: float | None) -> np.ndarray:
        env = make_env(env_id, reward_mode="vector")
        obs, _ = env.reset(seed=0)
        horizon = env.config.horizon
        cutoff = int(zero_after * horizon) if zero_after is not None else horizon + 1
        a = 0.5 * np.ones(env.action_space.shape, dtype=np.float32)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        z = np.zeros_like(a)
        last_info = None
        for t in range(horizon):
            obs, r, term, trunc, last_info = env.step(a if t < cutoff else z)
            if term:
                break
        return last_info["reward_vector"]

    normal = _run(None)
    tail_zero = _run(0.75)
    deltas = np.abs(tail_zero - normal)
    n_shifted = int(np.sum(deltas > 0.01))
    assert n_shifted >= 1, (
        f"Maze v0: zeroing the last 25% shifted no components by "
        f">0.01. Deltas: {deltas.tolist()}. Gate 4 says the tail "
        f"must affect terminal vector."
    )
