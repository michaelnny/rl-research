import numpy as np
import pytest

from rlh_bench.envs import make_env, registered_envs
from rlh_bench.metrics import evaluate_policy, first_success_episode, pareto_non_dominated


def test_registry_constructs_all_envs():
    ids = registered_envs()
    assert "RecoverablePointMaze-v0" in ids
    assert "RecoverableResourceAllocation-v0" in ids
    for env_id in ids:
        env = make_env(env_id)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert info["reward_vector"].shape == (env.reward_dim,)


def test_registry_rejects_unknown_env():
    with pytest.raises(KeyError):
        make_env("Nope-v0")


def test_evaluate_policy_summary():
    def env_factory():
        return make_env("RecoverableResourceAllocation-Small-v0")

    def policy_factory(env):
        return lambda obs: np.zeros(env.action_space.shape, dtype=np.float32)

    summary = evaluate_policy(env_factory, policy_factory, episodes=3, seed=0)
    assert summary.episodes == 3
    assert summary.mean_reward_vector.shape == (env_factory().reward_dim,)
    assert summary.mean_length == env_factory().config.horizon


def test_first_success_episode():
    assert first_success_episode([False, False, True]) == 3
    assert first_success_episode([False, False]) is None


def test_pareto_non_dominated():
    points = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.2, 0.2],
        [0.0, 1.0],
    ], dtype=np.float32)
    mask = pareto_non_dominated(points)
    assert mask.tolist() == [True, True, False, True]
