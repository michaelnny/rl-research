"""Baseline panel for RecoverableKeyFuelMaze (Small tier).

Used during env calibration to confirm difficulty shape. Reports
mean ± std over N_SEEDS=20 of trivial policies plus a greedy
nearest-landmark proxy baseline.
"""

import numpy as np

from rlh_bench.envs.keyfuel_maze import (
    KeyFuelMazeConfig,
    RecoverableKeyFuelMazeEnv,
)


def _greedy_nearest_landmark(env: RecoverableKeyFuelMazeEnv, obs: np.ndarray) -> np.ndarray:
    """Pick the nearest landmark's direction and activate top-4
    actuators aligned with it. This is a *proxy* greedy, not a
    real heuristic; it just checks that the env admits a
    directional strategy."""
    c = env.config
    K_t = c.n_key_types
    S = c.n_seals
    start = 7 + K_t + S
    dx, dy = obs[start], obs[start + 1]
    direction = np.array([dx, dy], dtype=np.float32)
    A = env.actuator_matrix
    alignment = A.T @ direction
    a = np.zeros(c.action_dim, dtype=np.float32)
    top = np.argsort(-alignment)[:4]
    for i in top:
        if alignment[i] > 0:
            a[i] = 1.0
    return a


def main() -> None:
    cfg = KeyFuelMazeConfig(
        horizon=500, action_dim=16, world_size=24.0,
        n_key_types=2, n_seals=2, n_gates=1, n_fuel_stations=2,
    )
    N_SEEDS = 20

    def _run(policy_fn, seed: int) -> dict:
        env = RecoverableKeyFuelMazeEnv(cfg, reward_mode="vector")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        for t in range(cfg.horizon):
            a = policy_fn(env, obs, t, rng)
            obs, r, term, trunc, info = env.step(a)
            if term:
                break
        return {
            "success": int(info["is_success"]),
            "rv": info["reward_vector"],
            "total_distance": info["total_distance"],
            "total_energy": info["total_energy"],
            "fuel": info["fuel"],
        }

    policies = [
        ("zero", lambda env, obs, t, rng: np.zeros(cfg.action_dim, dtype=np.float32)),
        ("ones", lambda env, obs, t, rng: np.ones(cfg.action_dim, dtype=np.float32)),
        ("rand_pm", lambda env, obs, t, rng: rng.uniform(-1, 1, size=cfg.action_dim).astype(np.float32)),
        ("sin_wave", lambda env, obs, t, rng: np.sin(t * 0.1 + np.arange(cfg.action_dim) * 0.5).astype(np.float32)),
        ("greedy_proxy", lambda env, obs, t, rng: _greedy_nearest_landmark(env, obs)),
    ]
    print(f"{'policy':14s} {'succ':>6} {'seal':>6} {'key':>6} {'fuel':>6} {'dist':>6} {'energy':>8} {'route':>6}")
    for name, fn in policies:
        results = [_run(fn, seed) for seed in range(N_SEEDS)]
        succ = np.mean([r["success"] for r in results])
        seal = np.mean([r["rv"][1] for r in results])
        key = np.mean([r["rv"][2] for r in results])
        fuel = np.mean([r["rv"][3] for r in results])
        dist = np.mean([r["total_distance"] for r in results])
        energy = np.mean([r["total_energy"] for r in results])
        route = np.mean([r["rv"][8] for r in results])
        print(f"{name:14s} {succ:>6.2f} {seal:>6.2f} {key:>6.2f} {fuel:>6.2f} "
              f"{dist:>6.1f} {energy:>8.0f} {route:>6.3f}")


if __name__ == "__main__":
    main()
