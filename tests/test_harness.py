from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402


def random_policy(env):
    space = env.action_space
    rng = np.random.default_rng(123)

    def policy_fn(_obs):
        if hasattr(space, "n"):
            return int(rng.integers(int(space.n)))
        return space.sample()

    return policy_fn


def test_stage_shape():
    assert harness.STAGES["sparse"] == harness.SPARSE_ENVS
    assert harness.STAGES["vector"] == harness.VECTOR_ENVS
    assert harness.STAGES["core"] == harness.CORE_ENVS
    assert harness.STAGES["all"] == harness.ENVS
    assert set(harness.ENV_TYPE) == set(harness.ENVS)
    assert harness.PANEL == harness.ENVS


@pytest.mark.parametrize("env_id", harness.ENVS)
def test_make_env_step(env_id):
    try:
        env = harness.make_env(env_id, seed=0)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"backend missing for {env_id}: {exc}")

    obs, _ = env.reset(seed=0)
    assert obs is not None
    action = random_policy(env)(obs)
    _obs, reward, _term, _trunc, info = env.step(action)
    assert isinstance(reward, float | int | np.floating | np.integer)
    if harness.ENV_TYPE[env_id] == "vector":
        assert "vector" in info
        assert np.asarray(info["vector"]).ndim == 1
    env.close()


@pytest.mark.parametrize("env_id", harness.ENVS)
def test_evaluate_returns_finite_score(env_id):
    try:
        env = harness.make_env(env_id, seed=0)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"backend missing for {env_id}: {exc}")
    policy = random_policy(env)
    env.close()

    score = harness.evaluate(policy, env_id, seed=0, n_episodes=1)
    assert math.isfinite(score)


def test_hypervolume_2d():
    points = np.array([[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]])
    assert harness.hypervolume(points, np.array([0.0, 0.0])) == pytest.approx(3.0)


def test_hypervolume_3d_smoke():
    points = np.array([[1.0, 1.0, 1.0]])
    assert harness.hypervolume(points, np.array([0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_load_baselines_monotone(tmp_path, monkeypatch):
    path = tmp_path / "baselines.json"
    path.write_text('{"resource-gathering-v0": {"random": 2, "strong_local": 1}}')
    monkeypatch.setattr(harness, "BASELINES_PATH", path)
    loaded = harness.load_baselines()
    assert set(loaded) == set(harness.ENVS)
    assert loaded["resource-gathering-v0"] == {"random": 2.0, "strong": 2.0}


def test_craftax_score_normalization_constant():
    assert harness.CRAFTAX_MAX_RETURN == 226.0
