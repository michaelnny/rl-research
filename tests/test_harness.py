"""Smoke tests for the frozen harness.

Confirms `harness.make_env(env_id, seed=0)` and `harness.evaluate(...)`
work end-to-end against each panel env with a random policy. If a panel
env's backend (MiniGrid / MO-Gymnasium / MuJoCo) is
not installed, the corresponding test is skipped — useful on dev machines
where `uv sync` has not been run yet.

Run:
    uv run pytest tests/test_harness.py
    uv run pytest tests/test_harness.py -k smoke   # smoke subset only
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402


def _random_policy(env):
    space = env.action_space
    rng = np.random.default_rng(123)

    def policy_fn(_obs):
        if hasattr(space, "n"):
            return int(rng.integers(int(space.n)))
        return space.sample()

    return policy_fn


@pytest.mark.parametrize("env_id", harness.PANEL)
def test_smoke_make_env_and_step(env_id):
    """make_env() builds, reset/step works, and vector envs inject info['vector']."""
    try:
        env = harness.make_env(env_id, seed=0)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend for {env_id} not installed: {e}")

    obs, info = env.reset(seed=0)
    assert obs is not None

    policy = _random_policy(env)
    is_vector = harness.PANEL_TYPE[env_id] == "vector"
    saw_vector = False

    for _ in range(5):
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        assert isinstance(r, float | int | np.floating | np.integer), (
            f"reward must be scalar, got {type(r)}"
        )
        if is_vector:
            assert "vector" in info, f"vector env {env_id} missing info['vector']"
            v = np.asarray(info["vector"])
            assert v.ndim == 1 and v.shape[0] >= 1
            saw_vector = True
        if term or trunc:
            obs, _ = env.reset()

    if is_vector:
        assert saw_vector

    env.close()


@pytest.mark.parametrize("env_id", harness.PANEL)
def test_smoke_evaluate(env_id):
    """harness.evaluate returns a finite scalar score for a random policy.

    Uses n_episodes=2 to keep the test fast; the eval protocol is the same.
    """
    try:
        # Build a throwaway env just to grab the action space for the policy.
        env = harness.make_env(env_id, seed=0)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend for {env_id} not installed: {e}")
    policy = _random_policy(env)
    env.close()

    score = harness.evaluate(policy, env_id, seed=0, n_episodes=2)
    assert isinstance(score, float)
    assert math.isfinite(score), f"evaluate returned non-finite score for {env_id}"


def test_hypervolume_2d_basic():
    """2D hypervolume sanity: a single point at (1, 1) vs ref (0, 0) → area 1."""
    pts = np.array([[1.0, 1.0]])
    ref = np.array([0.0, 0.0])
    hv = harness._hypervolume(pts, ref)
    assert hv == pytest.approx(1.0)


def test_hypervolume_2d_pareto_filtering():
    """Dominated points must not contribute. (1,1) dominates (0.5, 0.5)."""
    pts = np.array([[1.0, 1.0], [0.5, 0.5]])
    ref = np.array([0.0, 0.0])
    hv = harness._hypervolume(pts, ref)
    assert hv == pytest.approx(1.0)


def test_hypervolume_2d_two_pareto_points():
    """Two non-dominated points give the union area."""
    pts = np.array([[2.0, 1.0], [1.0, 2.0]])
    ref = np.array([0.0, 0.0])
    hv = harness._hypervolume(pts, ref)
    # Union: rect (0,0)-(2,1) area=2, plus rect (0,1)-(1,2) area=1.
    assert hv == pytest.approx(3.0)


def test_hypervolume_below_ref_zero():
    """A point below the reference contributes zero."""
    pts = np.array([[-1.0, -1.0]])
    ref = np.array([0.0, 0.0])
    hv = harness._hypervolume(pts, ref)
    assert hv == 0.0


def test_load_baselines_default_when_missing(monkeypatch, tmp_path):
    """When baselines.json + baselines_hard.json don't exist, all envs return
    zero floors; keys cover both tiers."""
    monkeypatch.setattr(harness, "BASELINES_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(harness, "BASELINES_HARD_PATH", tmp_path / "missing_hard.json")
    b = harness.load_baselines()
    expected = set(harness.PANEL_SMOKE) | set(harness.PANEL_HARD)
    assert set(b.keys()) == expected
    for env in expected:
        assert b[env]["random"] == 0.0
        assert b[env]["strong"] == 0.0


def test_panel_consistency():
    """PANEL_SMOKE + PANEL_HARD must align with PANEL_TYPE; HV_REF must cover
    every vector env in either tier."""
    all_envs = set(harness.PANEL_SMOKE) | set(harness.PANEL_HARD)
    assert all_envs == set(harness.PANEL_TYPE.keys())
    for env in all_envs:
        if harness.PANEL_TYPE[env] == "vector":
            assert env in harness.HV_REF, f"missing HV_REF for vector env {env}"
    # Back-compat alias
    assert harness.PANEL == harness.PANEL_SMOKE
