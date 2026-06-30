"""Smoke tests for generic baselines (CEM, REINFORCE, random).

The family-specific baseline portfolios (SCHEDULING_BASELINES,
MAZE_BASELINES) have their own dedicated test files; this file
covers the cross-family helpers in ``rlh_bench.baselines``.
"""

import importlib.util

import numpy as np
import pytest

from rlh_bench import make_env
from rlh_bench.baselines import LinearPolicy, RandomPolicy, train_cem
from rlh_bench.baselines.reinforce import train_reinforce
from rlh_bench.metrics import rollout


SMOKE_ENV_ID = "RecoverableCapacityScheduling-Small-v0"


def test_linear_policy_action_bounds():
    """LinearPolicy clips to the env's action space."""
    env = make_env(SMOKE_ENV_ID)
    obs, _ = env.reset(seed=0)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_params = env.action_space.shape[0] * (obs_dim + 1)
    params = np.zeros(n_params, dtype=np.float32)
    policy = LinearPolicy(params=params, obs_dim=obs_dim, action_space=env.action_space)
    action = policy(obs)
    assert env.action_space.contains(action)


def test_cem_smoke_runs_one_iteration():
    """train_cem completes one iteration on the smoke env."""
    def env_factory():
        return make_env(SMOKE_ENV_ID)

    result = train_cem(env_factory, iterations=1, population=4, elite_frac=0.5, seed=0)
    assert len(result.history) == 1
    assert np.isfinite(result.best_score)
    final = rollout(env_factory(), result.policy, seed=1)
    assert final.length == env_factory().config.horizon


def test_random_rollout_smoke():
    """RandomPolicy rolls out for the full horizon and returns a reward vector."""
    env = make_env(SMOKE_ENV_ID)
    result = rollout(env, RandomPolicy(env.action_space, seed=0), seed=0)
    assert result.length == env.config.horizon
    assert result.reward_vector.shape == (env.reward_dim,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_reinforce_smoke_one_episode():
    """train_reinforce completes one episode on the smoke env."""
    def env_factory():
        return make_env(SMOKE_ENV_ID)

    result = train_reinforce(env_factory, episodes=1, hidden_size=8, seed=0)
    assert len(result.returns) == 1
    assert len(result.losses) == 1
