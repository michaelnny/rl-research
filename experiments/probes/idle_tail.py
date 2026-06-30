"""Idle-tail probe for the substrate (acceptance gate 4).

A "no idle tail" env means the final 25% of the horizon contributes
to the terminal reward vector. If a baseline trajectory's state
stops evolving in the last quarter of the episode, the horizon is
padding rather than substantive long-horizon coupling.

This probe runs a uniform-positive policy on each env and compares
the terminal reward vector to a counterfactual where the same
policy is replaced by zeros after t = 0.75 * H. If the components
shift meaningfully, the tail is active.

We use uniform-positive (intensity 0.5) rather than random because
random's variance swamps the comparison signal.
"""

from __future__ import annotations

import numpy as np

from rlh_bench import make_env, registered_envs


def _uniform_action(action_space, intensity: float = 0.5) -> np.ndarray:
    """Constant uniform-positive action vector."""
    return (intensity * np.ones(action_space.shape, dtype=np.float32)).clip(
        action_space.low, action_space.high
    )


def _run_with_tail_zero(env_id: str, seed: int, *, zero_after_fraction: float = 0.75):
    """Roll out the uniform action, but zero it out after the cutoff.

    Returns the terminal reward vector under this perturbed trajectory.
    """
    env = make_env(env_id, reward_mode="vector")
    obs, _ = env.reset(seed=seed)
    horizon = env.config.horizon
    cutoff = int(zero_after_fraction * horizon)
    action_normal = _uniform_action(env.action_space, 0.5)
    action_zero = np.zeros_like(action_normal)
    last_info = None
    for t in range(horizon):
        a = action_normal if t < cutoff else action_zero
        obs, r, term, trunc, last_info = env.step(a)
        if term:
            break
    return last_info["reward_vector"]


def _run_normal(env_id: str, seed: int):
    env = make_env(env_id, reward_mode="vector")
    obs, _ = env.reset(seed=seed)
    action = _uniform_action(env.action_space, 0.5)
    last_info = None
    for _ in range(env.config.horizon):
        obs, r, term, trunc, last_info = env.step(action)
        if term:
            break
    return last_info["reward_vector"]


def main() -> None:
    print("Idle-tail probe: does zeroing the last 25% of the trajectory")
    print("meaningfully change the terminal reward vector?")
    print("(If yes, the tail is active. If no, it's padding.)")
    print()
    for env_id in registered_envs():
        print(f"=== {env_id} ===")
        normal_vecs = []
        tail_zero_vecs = []
        for seed in range(3):
            normal_vecs.append(_run_normal(env_id, seed))
            tail_zero_vecs.append(_run_with_tail_zero(env_id, seed))
        normal_mean = np.mean(normal_vecs, axis=0)
        tail_mean = np.mean(tail_zero_vecs, axis=0)
        names = make_env(env_id).reward_spec.names
        print(f"  {'component':<25}{'normal':>10}{'tail_zero':>12}{'delta':>10}")
        for i, name in enumerate(names):
            d = float(tail_mean[i] - normal_mean[i])
            mark = " <-- shift" if abs(d) > 0.01 else ""
            print(f"  {name:<25}{normal_mean[i]:>10.3f}{tail_mean[i]:>12.3f}{d:>10.3f}{mark}")
        print()


if __name__ == "__main__":
    main()
