"""Build random-policy baselines for every env in the benchmark suite.

Writes lab/baselines/random.json with mean episodic return (over N episodes)
for each env. Consumed by Stage A's "strictly above random" pass criterion and
documented in docs/benchmarks.md.

Run once on first lab boot (or whenever the benchmark suite changes):

    uv run python scripts/build_baselines.py [--episodes 100] [--quick]

`--quick` runs ~10 episodes per env for fast iteration on the script itself.

Outputs are atomic-replaced into lab/baselines/random.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "lab" / "baselines" / "random.json"


def _gym_random(env_id: str, episodes: int, seed: int) -> dict[str, Any]:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    returns: list[float] = []
    for _ep in range(episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
        returns.append(total)
    env.close()
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": episodes,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if episodes > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _dm_control_random(domain: str, task: str, episodes: int, seed: int) -> dict[str, Any]:
    from dm_control import suite

    env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
    action_spec = env.action_spec()
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    for _ in range(episodes):
        ts = env.reset()
        total = 0.0
        while not ts.last():
            action = rng.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
                size=action_spec.shape,
            ).astype(action_spec.dtype)
            ts = env.step(action)
            total += float(ts.reward)
        returns.append(total)
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": episodes,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if episodes > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _mo_minecart_random(episodes: int, seed: int) -> dict[str, Any]:
    """minecart-v0 returns a 3-vector reward; produce per-channel statistics."""
    import mo_gymnasium as mo_gym

    env = mo_gym.make("minecart-v0")
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    returns: list[np.ndarray] = []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total = np.zeros(3, dtype=np.float64)
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            _obs, reward_vec, terminated, truncated, _info = env.step(action)
            total += np.asarray(reward_vec, dtype=np.float64)
        returns.append(total)
    env.close()
    arr = np.stack(returns, axis=0)
    return {
        "episodes": episodes,
        "channels": arr.shape[1],
        "mean": arr.mean(axis=0).tolist(),
        "std": (arr.std(axis=0, ddof=1) if episodes > 1 else np.zeros(arr.shape[1])).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "scalarized_equal_weight_mean": float(arr.sum(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use ~10 episodes per env for fast script iteration.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional env id (or 'minecart' / 'humanoid') to compute alone.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_eps = 10 if args.quick else args.episodes

    plan = [
        ("CartPole-v1", lambda: _gym_random("CartPole-v1", n_eps, args.seed)),
        ("Pendulum-v1", lambda: _gym_random("Pendulum-v1", n_eps, args.seed)),
        (
            "ALE/MontezumaRevenge-v5",
            lambda: _gym_random("ALE/MontezumaRevenge-v5", n_eps, args.seed),
        ),
        ("humanoid.run", lambda: _dm_control_random("humanoid", "run", n_eps, args.seed)),
        ("minecart-v0", lambda: _mo_minecart_random(n_eps, args.seed)),
    ]
    if args.only is not None:
        plan = [(eid, fn) for eid, fn in plan if args.only in eid]

    results: dict[str, Any] = {}
    for env_id, fn in plan:
        print(f"[build_baselines] {env_id}: starting ({n_eps} episodes)")
        results[env_id] = fn()
        if env_id == "minecart-v0":
            r = results[env_id]
            print(
                f"[build_baselines] {env_id}: per-channel mean={r['mean']} "
                f"scalarized={r['scalarized_equal_weight_mean']:.3f}"
            )
        else:
            r = results[env_id]
            print(f"[build_baselines] {env_id}: mean={r['mean']:.3f} std={r['std']:.3f}")

    payload = {
        "built_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "episodes_per_env": n_eps,
        "seed": args.seed,
        "policy": "uniform-random",
        "envs": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and args.only is not None:
        # Merge into existing baselines instead of clobbering.
        existing = json.loads(OUT_PATH.read_text())
        existing["envs"].update(results)
        existing["built_at"] = payload["built_at"]
        payload = existing
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(OUT_PATH)
    print(f"[build_baselines] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
