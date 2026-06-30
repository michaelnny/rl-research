"""Tests for the RecoverableKeyFuelMaze env family."""

from __future__ import annotations

import numpy as np
import pytest

from rlh_bench.envs.keyfuel_maze import (
    DEFAULT_KEYFUEL_REWARD_SPEC,
    KeyFuelMazeConfig,
    RecoverableKeyFuelMazeEnv,
)


def _small_cfg(**overrides):
    base = dict(
        horizon=500, action_dim=16, world_size=24.0,
        n_key_types=2, n_seals=2, n_gates=1, n_fuel_stations=2,
    )
    base.update(overrides)
    return KeyFuelMazeConfig(**base)


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
    cfg = _small_cfg()
    actions = np.random.default_rng(42).uniform(-1, 1, size=(cfg.horizon, 16)).astype(np.float32)
    e1 = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
    info1, obs1 = _run(e1, actions, seed=7)
    e2 = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
    info2, obs2 = _run(e2, actions, seed=7)
    assert np.allclose(obs1, obs2)
    assert np.allclose(info1["reward_vector"], info2["reward_vector"])


def test_different_seeds_different_worlds():
    cfg = _small_cfg()
    e1 = RecoverableKeyFuelMazeEnv(cfg)
    e2 = RecoverableKeyFuelMazeEnv(cfg)
    e1.reset(seed=1)
    e2.reset(seed=2)
    assert not np.allclose(e1.actuator_matrix, e2.actuator_matrix), "different seeds should sample different actuator matrices"


# ----- gate 2: terminal-only ------------------------------------------------- #


def test_reward_is_terminal_only_vector_mode():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
    obs, _ = env.reset(seed=0)
    nonzero = 0
    for t in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(16, dtype=np.float32))
        if np.any(np.asarray(r) != 0):
            nonzero += 1
        if term:
            break
    assert nonzero == 1


def test_reward_is_terminal_only_scalar_mode():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="scalar")
    obs, _ = env.reset(seed=0)
    nonzero = 0
    for t in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(16, dtype=np.float32))
        if r != 0:
            nonzero += 1
        if term:
            break
    assert nonzero == 1


def test_reward_vector_always_in_info():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    obs, info = env.reset(seed=0)
    assert "reward_vector" in info and "reward_names" in info and "is_success" in info
    obs, r, term, trunc, info = env.step(np.zeros(16, dtype=np.float32))
    assert "reward_vector" in info


# ----- gate 6: myopic-gap ---------------------------------------------------- #


def test_zero_policy_does_not_solve():
    """A zero action should not move the agent or collect anything."""
    cfg = _small_cfg()
    successes = 0
    distances = []
    for seed in range(5):
        env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=seed)
        for _ in range(cfg.horizon):
            obs, r, term, trunc, info = env.step(np.zeros(16, dtype=np.float32))
            if term:
                break
        successes += int(info["is_success"])
        distances.append(info["total_distance"])
    assert successes == 0
    assert max(distances) < 1.0, f"zero policy should not move; got distances {distances}"


def test_random_policy_leaves_seal_headroom():
    cfg = _small_cfg()
    successes = 0
    seals = []
    for seed in range(10):
        env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        for _ in range(cfg.horizon):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=16).astype(np.float32))
            if term:
                break
        successes += int(info["is_success"])
        seals.append(info["reward_vector"][1])
    assert successes == 0, f"random policy should not solve KeyFuelMaze Small in 500 steps; got {successes}/10"
    assert np.mean(seals) < 0.5, f"random shouldn't complete most seals; got mean {np.mean(seals):.2f}"


# ----- gate 7: recoverability ----------------------------------------------- #


def test_fuel_exhaustion_is_recoverable_not_terminal():
    """A fuel-exhausted agent does not terminate the episode; it just
    cannot move until refueled (or until horizon)."""
    cfg = _small_cfg(initial_fuel=5.0)  # almost empty
    env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
    obs, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(16, dtype=np.float32))
        if term:
            break
    # Episode terminates at horizon, not from fuel exhaustion.
    assert info["t"] == cfg.horizon, f"episode should terminate at horizon; got t={info['t']}"


# ----- gate 8: action-complexity --------------------------------------------- #


def test_action_dim_must_be_at_least_2():
    with pytest.raises(ValueError):
        KeyFuelMazeConfig(action_dim=1)


def test_actuator_matrix_is_redundant():
    """Higher action_dim means more redundant actuators, not fewer
    physical force dims. Force is 2-D regardless of action_dim."""
    cfg = _small_cfg(action_dim=32)
    env = RecoverableKeyFuelMazeEnv(cfg)
    env.reset(seed=0)
    assert env.actuator_matrix.shape == (2, 32)


# ----- gate 11: action / obs space invariants -------------------------------- #


def test_action_space_shape():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    assert env.action_space.shape == (16,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_observation_shape_matches_layout():
    """Observation layout: pos(2)+vel(2)+fuel(1)+heat(1)+damage(1)+keys(K)+seals(S)+landmarks(3*(2+4+K))+gates(G)+t(1)+prev_energy(1)."""
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    K_t = cfg.n_key_types
    expected = 7 + K_t + cfg.n_seals + 3 * (2 + 4 + K_t) + cfg.n_gates + 1 + 1
    assert env.observation_space.shape == (expected,)


# ----- error handling -------------------------------------------------------- #


def test_step_after_terminal_raises():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.zeros(16, dtype=np.float32))
        if term:
            break
    with pytest.raises(RuntimeError):
        env.step(np.zeros(16, dtype=np.float32))


def test_reset_clears_state():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    obs1, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(16, dtype=np.float32))
        if term:
            break
    obs2, _ = env.reset(seed=0)
    assert np.allclose(obs1, obs2)


# ----- reward vector --------------------------------------------------------- #


def test_reward_vector_has_nine_components():
    assert len(DEFAULT_KEYFUEL_REWARD_SPEC.names) == 9
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
    obs, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.zeros(16, dtype=np.float32))
        if term:
            break
    assert info["reward_vector"].shape == (9,)


def test_reward_components_are_finite():
    cfg = _small_cfg()
    for kind in ["zero", "ones", "neg", "rand"]:
        env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(cfg.horizon):
            if kind == "zero":
                a = np.zeros(16, dtype=np.float32)
            elif kind == "ones":
                a = np.ones(16, dtype=np.float32)
            elif kind == "neg":
                a = -np.ones(16, dtype=np.float32)
            else:
                a = rng.uniform(-1, 1, size=16).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            if term:
                break
        assert np.all(np.isfinite(info["reward_vector"])), f"non-finite for {kind}"


# ----- physical correctness -------------------------------------------------- #


def test_agent_stays_inside_world():
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(cfg.horizon):
        obs, r, term, trunc, info = env.step(np.ones(16, dtype=np.float32))
        assert 0.0 <= env.position[0] <= cfg.world_size
        assert 0.0 <= env.position[1] <= cfg.world_size
        if term:
            break


def test_fuel_only_decreases_when_moving():
    """Fuel is consumed by distance and action energy; a zero action
    should not consume fuel."""
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    env.reset(seed=0)
    initial_fuel = env.fuel
    env.step(np.zeros(16, dtype=np.float32))
    assert env.fuel == initial_fuel, "zero action should not consume fuel"


# ----- public seed property -------------------------------------------------- #


def test_seed_property_tracks_reset():
    """The public ``seed`` property returns the seed of the current world."""
    cfg = _small_cfg()
    env = RecoverableKeyFuelMazeEnv(cfg)
    env.reset(seed=42)
    assert env.seed == 42
    env.reset(seed=7)
    assert env.seed == 7


# ----- baseline portfolio is honest (observation-only) ----------------------- #


def test_oracle_route_planner_not_in_baseline_portfolio():
    """``MazeOracleRoutePlannerPolicy`` reads privileged env-internal
    state (waypoint coordinates, gate phases). It must NOT be in
    ``MAZE_BASELINES`` — only in ``MAZE_ORACLE_DIAGNOSTICS``.
    """
    from rlh_bench.baselines import MAZE_BASELINES, MAZE_ORACLE_DIAGNOSTICS

    baseline_names = {cls.__name__ for cls in MAZE_BASELINES}
    oracle_names = {cls.__name__ for cls in MAZE_ORACLE_DIAGNOSTICS}
    assert "MazeOracleRoutePlannerPolicy" not in baseline_names, (
        "oracle policy must not appear as an honest baseline"
    )
    assert "MazeOracleRoutePlannerPolicy" in oracle_names, (
        "oracle policy should be registered as a diagnostic"
    )
    # No overlap.
    assert baseline_names.isdisjoint(oracle_names), (
        "baseline portfolio and oracle diagnostics must be disjoint"
    )
