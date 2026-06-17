import numpy as np
import pytest

from rlh_bench.baselines import MazeWaypointPolicy, RandomPolicy
from rlh_bench.envs import RecoverableMazeConfig, RecoverablePointMazeEnv
from rlh_bench.metrics import rollout


def test_maze_reset_is_deterministic():
    env = RecoverablePointMazeEnv()
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=456)
    assert np.allclose(obs1, obs2)
    assert np.allclose(info1["reward_vector"], np.zeros(env.reward_dim))


def test_maze_terminal_only_scalar_reward():
    env = RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=5))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for step in range(4):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert np.allclose(info["reward_vector"], 0.0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated
    assert not truncated
    assert isinstance(reward, float)
    assert info["reward_vector"].shape == (env.reward_dim,)
    assert info["reward_names"] == env.reward_spec.names
    with pytest.raises(RuntimeError):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))


def test_maze_vector_reward_mode():
    env = RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=3), reward_mode="vector")
    env.reset(seed=0)
    for _ in range(2):
        _, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert isinstance(reward, np.ndarray)
        assert reward.shape == (env.reward_dim,)
        assert not terminated
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated
    assert isinstance(reward, np.ndarray)
    assert np.allclose(reward, info["reward_vector"])


def test_maze_bad_actions_are_recoverable_nonterminal():
    env = RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=20))
    env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert not terminated
        assert not truncated
        assert reward == 0.0
    assert info["collisions"] > 0
    # Still inside valid unit-square bounds after repeated wall hits.
    assert np.all(env.position >= 0.0) and np.all(env.position <= 1.0)


def test_maze_high_dimensional_action_space():
    env = RecoverablePointMazeEnv(RecoverableMazeConfig(action_dim=8, horizon=4))
    obs, _ = env.reset(seed=0)
    action = np.ones(8, dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert env.action_space.shape == (8,)
    assert env.observation_space.contains(obs)
    assert reward == 0.0
    assert not terminated
    assert not truncated


def test_maze_heuristic_beats_stationary_policy_on_distance():
    env = RecoverablePointMazeEnv()
    heuristic_result = rollout(env, MazeWaypointPolicy(env), seed=0)
    stationary_env = RecoverablePointMazeEnv()
    stationary = rollout(stationary_env, lambda obs: np.zeros(stationary_env.action_space.shape), seed=0)
    assert heuristic_result.info["final_distance"] < stationary.info["final_distance"]
    assert heuristic_result.info["final_distance"] < 0.20


def test_random_policy_valid_actions():
    env = RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=2))
    policy = RandomPolicy(env.action_space, seed=0)
    obs, _ = env.reset()
    action = policy(obs)
    assert env.action_space.contains(action)
