import importlib.util

import numpy as np
import pytest

from rlh_bench.baselines import LinearPolicy, RandomPolicy, train_cem
from rlh_bench.baselines.reinforce import train_reinforce
from rlh_bench.envs import RecoverableMazeConfig, RecoverablePointMazeEnv, make_env
from rlh_bench.metrics import rollout


def test_linear_policy_action_bounds():
    env = RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=3))
    obs, _ = env.reset(seed=0)
    params = np.zeros(env.action_space.shape[0] * (env.observation_space.shape[0] + 1), dtype=np.float32)
    policy = LinearPolicy(params=params, obs_dim=env.observation_space.shape[0], action_space=env.action_space)
    action = policy(obs)
    assert env.action_space.contains(action)


def test_cem_smoke_runs_one_iteration():
    def env_factory():
        return RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=4))

    result = train_cem(env_factory, iterations=1, population=4, elite_frac=0.5, seed=0)
    assert len(result.history) == 1
    assert np.isfinite(result.best_score)
    final = rollout(env_factory(), result.policy, seed=1)
    assert final.length == 4


def test_random_rollout_smoke():
    env = make_env("RecoverableResourceAllocation-Small-v0")
    result = rollout(env, RandomPolicy(env.action_space, seed=0), seed=0)
    assert result.length == env.config.horizon
    assert result.reward_vector.shape == (env.reward_dim,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_reinforce_smoke_one_episode():
    def env_factory():
        return RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=3))

    result = train_reinforce(env_factory, episodes=1, hidden_size=8, seed=0)
    assert len(result.returns) == 1
    assert len(result.losses) == 1
