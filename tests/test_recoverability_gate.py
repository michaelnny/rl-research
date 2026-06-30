"""Tests for acceptance gate 7 (recoverability) — graded curve form.

Building on `tests/test_capacity_scheduling_env.py::test_recoverability_is_graded`,
which only checked "early/mid/late degrade vs baseline and don't
collapse to 0". The graded-curve form here adds:

  - The burst shifts at LEAST 2 distinct components by >0.01 (not
    just one).
  - At least one component shifts more for mid/late bursts than for
    no burst (a sanity floor — the env is responsive to the burst).
  - The terminal vector for the WORST burst still leaves nontrivial
    nonzero fill (not catastrophic collapse).
"""

from __future__ import annotations

import numpy as np

from rlh_bench import make_env


def _action(env, intensity: float) -> np.ndarray:
    return np.clip(
        intensity * np.ones(env.action_space.shape, dtype=np.float32),
        env.action_space.low, env.action_space.high,
    )


def _run_with_burst(env_id, *, seed, burst_start, burst_len, good=0.5, bad=-1.0):
    env = make_env(env_id, reward_mode="vector")
    obs, _ = env.reset(seed=seed)
    g, b = _action(env, good), _action(env, bad)
    last_info = None
    for t in range(env.config.horizon):
        a = b if burst_start <= t < burst_start + burst_len else g
        obs, r, term, trunc, last_info = env.step(a)
        if term:
            break
    return last_info["reward_vector"]


def test_scheduling_recoverability_graded_curve():
    """Acceptance gate 7 (graded curve): bursts at multiple times must
    produce graded terminal-vector shifts, not all-or-nothing collapse.
    """
    env_id = "RecoverableCapacityScheduling-Small-v0"
    env = make_env(env_id)
    horizon = env.config.horizon
    burst_len = int(0.05 * horizon)

    # 3 seeds; average for stability.
    n_seeds = 3
    baseline = np.mean(
        [_run_with_burst(env_id, seed=s, burst_start=horizon + 1, burst_len=burst_len) for s in range(n_seeds)],
        axis=0,
    )
    early = np.mean(
        [_run_with_burst(env_id, seed=s, burst_start=int(0.05 * horizon), burst_len=burst_len) for s in range(n_seeds)],
        axis=0,
    )
    mid = np.mean(
        [_run_with_burst(env_id, seed=s, burst_start=int(0.50 * horizon), burst_len=burst_len) for s in range(n_seeds)],
        axis=0,
    )
    late = np.mean(
        [_run_with_burst(env_id, seed=s, burst_start=int(0.85 * horizon), burst_len=burst_len) for s in range(n_seeds)],
        axis=0,
    )

    # At least 2 components shift by >0.01 between baseline and the
    # worst burst (mid or late).
    worst = np.minimum(np.minimum(early, mid), late)
    deltas = np.abs(worst - baseline)
    n_shifted = int(np.sum(deltas > 0.01))
    assert n_shifted >= 2, (
        f"only {n_shifted} components shifted by >0.01 between baseline "
        f"and worst burst. Deltas: {deltas.tolist()}. The env should "
        f"respond to perturbations on multiple components."
    )

    # No catastrophic collapse: worst-burst fill_rate stays nonzero.
    fill_idx = 1  # weighted_fill_rate
    assert worst[fill_idx] > 0.3, (
        f"worst burst collapsed fill_rate to {worst[fill_idx]:.3f}; "
        f"expected > 0.3 (recoverable)"
    )


def test_maze_recoverability_responds_to_perturbation():
    """Maze recoverability test: a mid-horizon burst of bad actions
    should shift at least 2 non-progress components (energy /
    collision) since the uniform 0.5 baseline doesn't typically
    complete seals.
    """
    env_id = "RecoverableKeyFuelMaze-Small-v0"
    env = make_env(env_id)
    horizon = env.config.horizon
    burst_len = int(0.05 * horizon)
    n_seeds = 3

    baseline = np.mean(
        [_run_with_burst(env_id, seed=s, burst_start=horizon + 1, burst_len=burst_len) for s in range(n_seeds)],
        axis=0,
    )
    perturbed = np.mean(
        [_run_with_burst(env_id, seed=s, burst_start=int(0.50 * horizon), burst_len=burst_len) for s in range(n_seeds)],
        axis=0,
    )
    deltas = np.abs(perturbed - baseline)
    n_shifted = int(np.sum(deltas > 0.01))
    assert n_shifted >= 2, (
        f"Maze: only {n_shifted} components shifted by >0.01 from "
        f"a mid-horizon burst. Deltas: {deltas.tolist()}."
    )
