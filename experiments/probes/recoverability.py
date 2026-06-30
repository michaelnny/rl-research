"""Recoverability probe (acceptance gate 7).

Inject bursts of a "bad" action at early / mid / late fractions of
the horizon, and report how much each terminal-vector component
degrades vs the baseline (no-injection) trajectory.

The signal we want:

  - **graded**: each injection time produces a measurable but
    non-catastrophic terminal-vector shift.
  - **monotonic by recovery time**: an early burst has time to
    recover and should leave less degradation than a late burst.
  - **multi-component**: at least 2 different components should
    shift, not just one.

Run from the repo root::

    PYTHONPATH=src:. .venv/bin/python experiments/probes/recoverability.py
"""

from __future__ import annotations

import numpy as np

from rlh_bench import make_env


def _action(env, intensity: float) -> np.ndarray:
    """Constant scaled action for the env's action shape."""
    return np.clip(
        intensity * np.ones(env.action_space.shape, dtype=np.float32),
        env.action_space.low,
        env.action_space.high,
    )


def _run_with_burst(
    env_id: str,
    *,
    seed: int,
    burst_start: int,
    burst_len: int,
    good_intensity: float = 0.5,
    bad_intensity: float = -1.0,
):
    env = make_env(env_id, reward_mode="vector")
    obs, _ = env.reset(seed=seed)
    horizon = env.config.horizon
    good = _action(env, good_intensity)
    bad = _action(env, bad_intensity)
    last_info = None
    for t in range(horizon):
        a = bad if burst_start <= t < burst_start + burst_len else good
        obs, r, term, trunc, last_info = env.step(a)
        if term:
            break
    return last_info["reward_vector"]


def _probe_env(env_id: str, *, n_seeds: int = 3, burst_len_fraction: float = 0.05):
    env = make_env(env_id)
    horizon = env.config.horizon
    burst_len = max(1, int(burst_len_fraction * horizon))
    fractions = {
        "no_inject": None,
        "early (t=0.05H)": 0.05,
        "mid (t=0.50H)": 0.50,
        "late (t=0.85H)": 0.85,
    }
    means = {label: [] for label in fractions}
    for seed in range(n_seeds):
        for label, frac in fractions.items():
            if frac is None:
                burst_start = horizon + 1  # never inject
            else:
                burst_start = int(frac * horizon)
            rv = _run_with_burst(
                env_id, seed=seed, burst_start=burst_start, burst_len=burst_len
            )
            means[label].append(rv)
    return {label: np.mean(vs, axis=0) for label, vs in means.items()}


def main() -> None:
    targets = [
        "RecoverableCapacityScheduling-Small-v0",
        "RecoverableCapacityScheduling-v0",
        "RecoverableKeyFuelMaze-Small-v0",
    ]
    for env_id in targets:
        print(f"=== {env_id} ===")
        results = _probe_env(env_id)
        names = make_env(env_id).reward_spec.names
        baseline = results["no_inject"]
        labels = ["early (t=0.05H)", "mid (t=0.50H)", "late (t=0.85H)"]
        print(f"  {'component':<25}{'baseline':>10}" + "".join(f"{l:>20}" for l in labels))
        for i, n in enumerate(names):
            row = [f"{baseline[i]:>10.3f}"]
            for label in labels:
                v = results[label][i]
                d = v - baseline[i]
                row.append(f"{v:>10.3f} ({d:+.2f})")
            print(f"  {n:<25}" + "".join(row))
        print()


if __name__ == "__main__":
    main()
