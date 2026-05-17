"""Unit tests for ``rl_research.evaluate``.

Verifies the algorithm-agnostic evaluator works for the gym-classic family
on CartPole-v1 and Pendulum-v1 (the sanity envs). The other families
(gym-atari, mo-minecart, dm-control) are exercised end-to-end by the PPO
baseline; testing the integration there would duplicate that without adding
signal.
"""

from __future__ import annotations

import numpy as np

from rl_research.evaluate import evaluate


def test_evaluate_cartpole_random_policy():
    rng = np.random.default_rng(0)

    def policy(obs):
        return int(rng.integers(0, 2))

    mean, std, pc = evaluate("CartPole-v1", policy, seed=0, n_episodes=5, max_steps_per_episode=500)
    assert pc is None
    # Random CartPole returns ~22 mean. Five episodes is noisy; just sanity-check
    # the value is in a plausible range and finite.
    assert np.isfinite(mean)
    assert mean > 0
    assert mean <= 500.0
    assert std >= 0


def test_evaluate_cartpole_constant_policy():
    """A constant policy should still produce finite returns (>0 since CartPole
    rewards every alive step)."""

    def policy(_obs):
        return 0  # always-left

    mean, _std, pc = evaluate(
        "CartPole-v1", policy, seed=0, n_episodes=3, max_steps_per_episode=500
    )
    assert pc is None
    assert mean > 0


def test_evaluate_pendulum_continuous_policy():
    def policy(_obs):
        return np.array([0.0], dtype=np.float32)

    mean, std, pc = evaluate("Pendulum-v1", policy, seed=0, n_episodes=3, max_steps_per_episode=200)
    assert pc is None
    # Pendulum penalty is bounded; mean must be a finite negative number.
    assert np.isfinite(mean)
    assert mean < 0
    assert std >= 0


def test_evaluate_obs_transform_called():
    seen = []

    def transform(obs):
        seen.append(obs.shape)
        return obs

    rng = np.random.default_rng(0)

    def policy(_obs):
        return int(rng.integers(0, 2))

    evaluate(
        "CartPole-v1",
        policy,
        seed=0,
        n_episodes=2,
        obs_transform=transform,
        max_steps_per_episode=100,
    )
    assert seen, "obs_transform was never invoked"
    # CartPole obs is a 4-vector.
    assert all(s == (4,) for s in seen)


def test_evaluate_seed_determinism():
    rng = np.random.default_rng(0)

    def policy(_obs):
        return int(rng.integers(0, 2))

    # Same seed + same RNG state → same return.
    rng = np.random.default_rng(0)
    a, _, _ = evaluate("CartPole-v1", policy, seed=0, n_episodes=2, max_steps_per_episode=100)
    rng = np.random.default_rng(0)
    b, _, _ = evaluate("CartPole-v1", policy, seed=0, n_episodes=2, max_steps_per_episode=100)
    assert a == b


def test_evaluate_returns_zero_std_for_single_episode():
    def policy(_obs):
        return 0

    _, std, _ = evaluate("CartPole-v1", policy, seed=0, n_episodes=1, max_steps_per_episode=100)
    # Single-episode std is undefined ddof-1; the helper returns 0.0.
    assert std == 0.0
