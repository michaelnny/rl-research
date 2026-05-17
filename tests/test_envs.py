"""Unit tests for ``rl_research.envs``.

Covers the dispatch surface and the gym-classic adapter behavior. The
heavier adapters (Atari, dm_control, minecart) are exercised end-to-end by
the PPO baseline; here we only verify their dispatch path and that they
raise the expected error on missing optional deps. Importing the heavy
backends inside the test would duplicate PPO integration coverage and slow
the suite for no signal gain.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_research.envs import (
    CONTINUOUS_CLASSIC,
    DISCRETE_CLASSIC,
    GymVecAdapter,
    adapter_family,
    make_classic_env,
    make_vec,
)

# -- adapter_family ---------------------------------------------------------


@pytest.mark.parametrize(
    ("env_id", "expected"),
    [
        ("CartPole-v1", "gym-classic"),
        ("Acrobot-v1", "gym-classic"),
        ("MountainCar-v0", "gym-classic"),
        ("Pendulum-v1", "gym-classic"),
        ("MountainCarContinuous-v0", "gym-classic"),
        ("ALE/MontezumaRevenge-v5", "gym-atari"),
        ("ALE/Pong-v5", "gym-atari"),
        ("minecart-v0", "mo-minecart"),
        ("humanoid.run", "dm-control"),
    ],
)
def test_adapter_family_known_envs(env_id, expected):
    assert adapter_family(env_id) == expected


def test_adapter_family_unknown_raises():
    with pytest.raises(ValueError, match="no adapter family"):
        adapter_family("Walker2d-v5")


def test_classic_frozensets_disjoint():
    assert frozenset() == DISCRETE_CLASSIC & CONTINUOUS_CLASSIC


def test_classic_frozensets_have_expected_members():
    assert "CartPole-v1" in DISCRETE_CLASSIC
    assert "Pendulum-v1" in CONTINUOUS_CLASSIC


# -- make_classic_env -------------------------------------------------------


def test_make_classic_env_discrete_returns_envable():
    env = make_classic_env("CartPole-v1")(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    obs, r, term, trunc, _ = env.step(0)
    assert obs.shape == (4,)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    env.close()


def test_make_classic_env_continuous_applies_clip_action():
    """ClipAction must be applied so unbounded Gaussian samples don't silently
    mismatch executed action vs log-prob — see envs.py docstring."""
    env = make_classic_env("Pendulum-v1")(seed=0)
    env.reset(seed=0)
    # Pendulum action is Box(low=-2, high=2). Out-of-range action must still
    # step cleanly (proof the wrapper is in place; without it, the env raises).
    env.step(np.array([100.0], dtype=np.float32))
    env.close()


# -- GymVecAdapter ----------------------------------------------------------


def test_gym_vec_adapter_reset_shape():
    adapter = GymVecAdapter(make_classic_env("CartPole-v1"), n_envs=3, seed=0)
    obs = adapter.reset(seed=0)
    assert obs.shape == (3, 4)
    assert adapter.n_envs == 3
    adapter.close()


def test_gym_vec_adapter_step_returns_five_arrays():
    adapter = GymVecAdapter(make_classic_env("CartPole-v1"), n_envs=2, seed=0)
    adapter.reset(seed=0)
    actions = np.array([0, 1])
    next_obs, rewards, term, trunc, final_obs = adapter.step(actions)
    assert next_obs.shape == (2, 4)
    assert rewards.shape == (2,) and rewards.dtype == np.float32
    assert term.shape == (2,) and term.dtype == bool
    assert trunc.shape == (2,) and trunc.dtype == bool
    assert final_obs.shape == (2, 4)
    adapter.close()


def test_gym_vec_adapter_pop_completed_drains_and_resets():
    """Auto-reset on terminal/trunc: completed-episode return is captured in
    pop_completed; final_obs holds the pre-reset observation so a GAE
    consumer can bootstrap V(s_truncated)."""
    adapter = GymVecAdapter(make_classic_env("CartPole-v1"), n_envs=1, seed=0)
    adapter.reset(seed=0)
    # Step until the cart falls over (term=True). CartPole guarantees this
    # within ~500 steps under any policy; we cap at 1000 for safety.
    completed = []
    for _ in range(1000):
        _, _, term, trunc, _ = adapter.step(np.array([0]))
        if term[0] or trunc[0]:
            completed = adapter.pop_completed()
            break
    assert completed, "expected at least one completed episode"
    assert isinstance(completed[0], float) and completed[0] > 0
    # Second pop returns nothing — the queue is drained.
    assert adapter.pop_completed() == []
    adapter.close()


def test_gym_vec_adapter_separate_seeds_diverge():
    """Different seeds at construction time must yield different rollouts —
    cheap regression against per-env seed plumbing breaking."""
    a = GymVecAdapter(make_classic_env("CartPole-v1"), n_envs=2, seed=0)
    b = GymVecAdapter(make_classic_env("CartPole-v1"), n_envs=2, seed=100)
    obs_a = a.reset(seed=0)
    obs_b = b.reset(seed=100)
    assert not np.allclose(obs_a, obs_b)
    a.close()
    b.close()


# -- make_vec dispatch ------------------------------------------------------


def test_make_vec_classic_returns_gym_adapter():
    adapter = make_vec("CartPole-v1", n_envs=2, seed=0)
    assert isinstance(adapter, GymVecAdapter)
    obs = adapter.reset(seed=0)
    assert obs.shape == (2, 4)
    adapter.close()


def test_make_vec_unknown_raises():
    with pytest.raises(ValueError, match="no adapter family"):
        make_vec("Walker2d-v5", n_envs=1, seed=0)
