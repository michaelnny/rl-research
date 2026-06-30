"""Tests for acceptance gate 4 (no-idle-tail).

The final 25% of the horizon must affect terminal-vector components
on most worlds. This is enforced by running a uniform-positive
policy normally vs zeroing it after t = 0.75 * H and checking the
terminal vectors shift meaningfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlh_bench import make_env


def _uniform_action(action_space):
    return (0.5 * np.ones(action_space.shape, dtype=np.float32)).clip(
        action_space.low, action_space.high
    )


def _run(env_id: str, seed: int, *, zero_after_fraction: float | None = None):
    env = make_env(env_id, reward_mode="vector")
    obs, _ = env.reset(seed=seed)
    horizon = env.config.horizon
    cutoff = int(zero_after_fraction * horizon) if zero_after_fraction is not None else horizon + 1
    action = _uniform_action(env.action_space)
    zero = np.zeros_like(action)
    last_info = None
    for t in range(horizon):
        obs, r, term, trunc, last_info = env.step(action if t < cutoff else zero)
        if term:
            break
    return last_info["reward_vector"]


@pytest.mark.parametrize(
    "env_id",
    [
        "RecoverableCapacityScheduling-Small-v0",
        "RecoverableCapacityScheduling-v0",
    ],
)
def test_tail_affects_terminal_for_scheduling(env_id: str) -> None:
    """Zeroing the last 25% of the trajectory must change ≥3 components
    by ≥0.05 (averaged over 3 seeds) on the scheduling envs.

    Acceptance gate 4: the final 25% of horizon must affect terminal
    vector components.
    """
    n_components_shifted = 0
    normal_vecs = []
    tail_zero_vecs = []
    for seed in range(3):
        normal_vecs.append(_run(env_id, seed))
        tail_zero_vecs.append(_run(env_id, seed, zero_after_fraction=0.75))
    normal_mean = np.mean(normal_vecs, axis=0)
    tail_mean = np.mean(tail_zero_vecs, axis=0)
    deltas = np.abs(tail_mean - normal_mean)
    n_shifted = int(np.sum(deltas > 0.05))
    assert n_shifted >= 3, (
        f"{env_id}: only {n_shifted} components shifted by >0.05 when "
        f"zeroing the last 25% of horizon. Deltas: {deltas.tolist()}. "
        f"Gate 4 (no-idle-tail) requires the tail to affect terminal vector."
    )


def test_tail_affects_terminal_for_maze() -> None:
    """Same gate 4 check for the maze. Maze uniform-policy doesn't
    typically succeed but its energy / collision / fuel components
    should shift when the tail is zeroed.
    """
    env_id = "RecoverableKeyFuelMaze-Small-v0"
    normal_vecs = []
    tail_zero_vecs = []
    for seed in range(3):
        normal_vecs.append(_run(env_id, seed))
        tail_zero_vecs.append(_run(env_id, seed, zero_after_fraction=0.75))
    normal_mean = np.mean(normal_vecs, axis=0)
    tail_mean = np.mean(tail_zero_vecs, axis=0)
    deltas = np.abs(tail_mean - normal_mean)
    n_shifted = int(np.sum(deltas > 0.01))
    assert n_shifted >= 2, (
        f"Maze-Small: only {n_shifted} components shifted by >0.01 "
        f"when zeroing the last 25% of horizon. Deltas: {deltas.tolist()}."
    )
