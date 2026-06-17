import numpy as np
import pytest

from rlh_bench.baselines import ResourceGreedyPolicy
from rlh_bench.envs import RecoverableResourceAllocationEnv, ResourceAllocationConfig
from rlh_bench.metrics import rollout


def test_resource_reset_is_deterministic():
    env = RecoverableResourceAllocationEnv()
    obs1, info1 = env.reset(seed=1)
    obs2, info2 = env.reset(seed=2)
    assert np.allclose(obs1, obs2)
    assert np.allclose(info1["reward_vector"], info2["reward_vector"])


def test_resource_terminal_only_scalar_reward():
    env = RecoverableResourceAllocationEnv(ResourceAllocationConfig(horizon=5, num_projects=3))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = np.ones(env.action_space.shape, dtype=np.float32)
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert np.allclose(info["reward_vector"], 0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated
    assert not truncated
    assert isinstance(reward, float)
    assert info["reward_vector"].shape == (env.reward_dim,)
    with pytest.raises(RuntimeError):
        env.step(action)


def test_resource_vector_reward_mode():
    env = RecoverableResourceAllocationEnv(
        ResourceAllocationConfig(horizon=3, num_projects=3), reward_mode="vector"
    )
    env.reset(seed=0)
    for _ in range(2):
        _, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert isinstance(reward, np.ndarray)
        assert reward.shape == (env.reward_dim,)
        assert not terminated
    _, reward, terminated, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    assert terminated
    assert isinstance(reward, np.ndarray)
    assert np.allclose(reward, info["reward_vector"])


def test_resource_budget_projection_and_recoverability():
    env = RecoverableResourceAllocationEnv(ResourceAllocationConfig(horizon=20, num_projects=4))
    env.reset(seed=0)
    # Bad early behavior: allocate to downstream projects before they are ready.
    bad = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(5):
        _, reward, terminated, _, info = env.step(bad)
        assert reward == 0.0
        assert not terminated
        assert np.sum(info["projected_allocation"]) <= env.config.budget + 1e-6
    assert env.diagnostics()["service_level"] > 0.0
    assert not terminated


def test_resource_heuristic_feasible_and_better_than_zero():
    env = RecoverableResourceAllocationEnv()
    heuristic_result = rollout(env, ResourceGreedyPolicy(env), seed=0)
    zero_env = RecoverableResourceAllocationEnv()
    zero_result = rollout(zero_env, lambda obs: np.zeros(zero_env.action_space.shape), seed=0)
    assert heuristic_result.info["service_level"] > zero_result.info["service_level"]
    assert heuristic_result.info["service_level"] > 0.85


def test_resource_large_action_dimension():
    env = RecoverableResourceAllocationEnv(ResourceAllocationConfig(horizon=3, num_projects=8))
    obs, _ = env.reset(seed=0)
    assert env.action_space.shape == (8,)
    obs, reward, terminated, truncated, _ = env.step(np.ones(env.action_space.shape))
    assert env.observation_space.contains(obs)
    assert reward == 0.0
    assert not terminated
    assert not truncated
