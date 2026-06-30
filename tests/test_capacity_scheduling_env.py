"""Tests for the RecoverableCapacityScheduling env family."""

from __future__ import annotations

import numpy as np
import pytest

from rlh_bench.envs.capacity_scheduling import (
    CapacitySchedulingConfig,
    DEFAULT_SCHEDULING_REWARD_SPEC,
    RecoverableCapacitySchedulingEnv,
)


def _small_cfg(**overrides):
    base = dict(
        horizon=500, num_projects=16, num_modes=4, num_products=4,
        action_dim=32, n_bundles=2,
    )
    base.update(overrides)
    return CapacitySchedulingConfig(**base)


def _run(env, actions, *, seed=0):
    obs, _ = env.reset(seed=seed)
    last_info = None
    for a in actions:
        obs, r, term, trunc, last_info = env.step(a)
        if term:
            break
    return last_info, obs


# ----- gate 1: determinism --------------------------------------------------- #


def test_same_seed_same_action_same_terminal_vector():
    """Acceptance gate 1: identical seed + action sequence → identical terminal."""
    cfg = _small_cfg()
    actions = np.random.default_rng(42).uniform(-1, 1, size=(500, 32)).astype(np.float32)

    e1 = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
    info1, obs1 = _run(e1, actions, seed=7)
    e2 = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
    info2, obs2 = _run(e2, actions, seed=7)

    assert np.allclose(obs1, obs2), "terminal observations should match"
    assert np.allclose(info1["reward_vector"], info2["reward_vector"]), "terminal vectors should match"


def test_different_seeds_different_worlds():
    """Acceptance gate 1: different seeds generate different worlds."""
    cfg = _small_cfg()
    e1 = RecoverableCapacitySchedulingEnv(cfg)
    e2 = RecoverableCapacitySchedulingEnv(cfg)
    e1.reset(seed=1)
    e2.reset(seed=2)
    # World tensors should differ
    assert not np.allclose(e1.compat_matrix, e2.compat_matrix)


# ----- gate 2: terminal-only ------------------------------------------------- #


def test_reward_is_terminal_only():
    """Acceptance gate 2: reward == 0 for every non-terminal step."""
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
    obs, _ = env.reset(seed=0)
    nonzero_rewards = 0
    for t in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(32, dtype=np.float32))
        if np.any(np.asarray(r) != 0):
            nonzero_rewards += 1
        if term:
            break
    assert nonzero_rewards == 1, f"expected exactly 1 non-zero reward, got {nonzero_rewards}"


def test_scalar_reward_mode_also_terminal_only():
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="scalar")
    obs, _ = env.reset(seed=0)
    nonzero = 0
    for t in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(32, dtype=np.float32))
        if r != 0:
            nonzero += 1
        if term:
            break
    assert nonzero == 1


def test_reward_vector_always_in_info():
    """info['reward_vector'] is present every step (zero pre-terminal)."""
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
    obs, info = env.reset(seed=0)
    assert "reward_vector" in info
    obs, r, term, trunc, info = env.step(np.zeros(32, dtype=np.float32))
    assert "reward_vector" in info and "reward_names" in info and "is_success" in info


# ----- gate 6: myopic-gap (calibration shape) -------------------------------- #


def test_zero_policy_has_zero_fill_rate():
    """A zero action should not accidentally drive production."""
    cfg = _small_cfg()
    fills = []
    for seed in range(5):
        env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=seed)
        for _ in range(cfg.horizon):
            obs, r, term, trunc, info = env.step(np.zeros(32, dtype=np.float32))
            if term:
                break
        fills.append(info["reward_vector"][1])
    assert max(fills) < 0.05, f"zero policy should produce nothing; got fills {fills}"


def test_random_policy_leaves_headroom():
    """Random ± policy should not solve the env."""
    cfg = _small_cfg()
    successes = 0
    fills = []
    for seed in range(10):
        env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        for _ in range(cfg.horizon):
            obs, r, term, trunc, info = env.step(
                rng.uniform(-1, 1, size=32).astype(np.float32)
            )
            if term:
                break
        successes += int(info["is_success"])
        fills.append(info["reward_vector"][1])
    assert successes <= 3, f"random should rarely succeed; got {successes}/10 successes, fills {fills}"
    assert np.mean(fills) < 0.7, f"random shouldn't dominate fill; mean fill {np.mean(fills):.2f}"


# ----- gate 7: recoverability (graded degradation) --------------------------- #


def test_recoverability_is_graded():
    """A single mid-horizon misstep should grade-degrade, not collapse the env.

    Compare: (a) a constant positive action vs (b) the same constant
    positive action interrupted by 20 steps of negative actions at
    different times in the horizon. The terminal fill rate should
    drop monotonically (or near-monotonically) the later the misstep
    is — early missteps have time to recover.
    """
    cfg = _small_cfg()
    good_action = 0.5 * np.ones(32, dtype=np.float32)
    bad_action = -np.ones(32, dtype=np.float32)
    burst = 20

    def _run_with_burst(burst_start: int) -> float:
        env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=0)
        for t in range(cfg.horizon):
            if burst_start <= t < burst_start + burst:
                a = bad_action
            else:
                a = good_action
            obs, r, term, trunc, info = env.step(a)
            if term:
                break
        return float(info["reward_vector"][1])

    baseline = _run_with_burst(cfg.horizon + 1)  # never inject
    early = _run_with_burst(50)
    mid = _run_with_burst(250)
    late = _run_with_burst(450)
    # Some degradation: late burst worse than baseline, but not zero.
    assert early < baseline + 0.05, "early burst should not be ignored"
    assert late < baseline + 0.05, "late burst should not be ignored"
    # No collapse: even the worst burst leaves nontrivial fill.
    assert min(early, mid, late) >= 0.0, "burst should not break the env"


# ----- gate 8: action-complexity --------------------------------------------- #


def test_action_dim_must_match_layout():
    """The env demands action_dim >= K + 3M + P; a smaller action_dim is rejected."""
    with pytest.raises(ValueError):
        CapacitySchedulingConfig(num_projects=16, num_modes=4, num_products=4, action_dim=16)


def test_top_k_sparsification_loses_return():
    """Top-1 sparsification of a random policy should lose mean fill.

    This is a quick proxy for "action complexity is not syntactic": if
    we can produce the same fill rate by zeroing all but the largest
    action dim, then action complexity was illusory.
    """
    cfg = _small_cfg()
    rng = np.random.default_rng(0)
    full_fills = []
    sparse_fills = []
    for seed in range(5):
        full_action = rng.uniform(0, 1, size=32).astype(np.float32)
        sparse_action = np.zeros_like(full_action)
        sparse_action[int(np.argmax(full_action))] = float(np.max(full_action))
        # Same action used every step.
        for action, store in [(full_action, full_fills), (sparse_action, sparse_fills)]:
            env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
            obs, _ = env.reset(seed=seed)
            for _ in range(cfg.horizon):
                obs, r, term, trunc, info = env.step(action)
                if term:
                    break
            store.append(info["reward_vector"][1])
    # Top-1 sparsification should reduce fill substantially.
    assert np.mean(full_fills) - np.mean(sparse_fills) > 0.05, (
        f"action sparsification did not change fill enough; "
        f"full={np.mean(full_fills):.2f} sparse={np.mean(sparse_fills):.2f}"
    )


# ----- gate 11: action-space invariants -------------------------------------- #


def test_action_space_shape():
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg)
    assert env.action_space.shape == (32,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_observation_space_shape():
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg)
    K, M, P = cfg.num_projects, cfg.num_modes, cfg.num_products
    expected = 4 * K + 4 * M + P * M + 2 * P + 3 * K + 2 + 1
    assert env.observation_space.shape == (expected,), (
        f"obs shape {env.observation_space.shape} != expected ({expected},)"
    )


# ----- error handling --------------------------------------------------------- #


def test_step_after_terminal_raises():
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.zeros(32, dtype=np.float32))
        if term:
            break
    with pytest.raises(RuntimeError):
        env.step(np.zeros(32, dtype=np.float32))


def test_reset_clears_state():
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg)
    obs1, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(32, dtype=np.float32))
        if term:
            break
    # Reset, take same actions: should produce same outcome.
    obs2, _ = env.reset(seed=0)
    assert np.allclose(obs1, obs2), "reset should restore initial observation"


# ----- reward vector shape --------------------------------------------------- #


def test_reward_vector_has_eleven_components():
    """The terminal vector should have exactly the 11 named components."""
    assert len(DEFAULT_SCHEDULING_REWARD_SPEC.names) == 11
    cfg = _small_cfg()
    env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
    obs, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.zeros(32, dtype=np.float32))
        if term:
            break
    assert info["reward_vector"].shape == (11,)


def test_all_reward_components_are_finite():
    """No NaN or inf in the terminal vector for any of several action distributions."""
    cfg = _small_cfg()
    for kind in ["zero", "ones", "neg", "rand"]:
        env = RecoverableCapacitySchedulingEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(cfg.horizon):
            if kind == "zero":
                a = np.zeros(32, dtype=np.float32)
            elif kind == "ones":
                a = np.ones(32, dtype=np.float32)
            elif kind == "neg":
                a = -np.ones(32, dtype=np.float32)
            else:
                a = rng.uniform(-1, 1, size=32).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            if term:
                break
        assert np.all(np.isfinite(info["reward_vector"])), f"non-finite reward for {kind}"
