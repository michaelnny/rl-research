"""Build `baselines.json` by running each frozen baseline against each
panel env. Run-once script — output committed to git.

Usage:
    uv run scripts/build_baselines.py [--time-budget-s N] [--seed N]
                                       [--envs e1,e2,...] [--workers N]

Output: writes `baselines.json` at repo root with schema:

    {
      "<env_id>": {
        "random": <float>,
        "strong": <float>,
        "per_baseline": {"eps_greedy_q": <float>, "count_bonus": <float>}
      }
    }

`strong` is the max across the strong-baseline algorithms. `random` is the
random baseline's score. `per_baseline` is recorded for diagnostics.

Parallelism:
    Each (env, baseline) pair is a fresh process — they share neither memory
    nor the GPU. With 5 smoke envs x 3 baselines = 15 jobs, default
    --workers=8 finishes in roughly the wallclock of the slowest pair (300 s
    train + eval) instead of the 75-min sequential floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402

BASELINE_NAMES = ["random", "eps_greedy_q", "count_bonus"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_PER_ENV)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=str, default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of (env, baseline) pairs to run concurrently.",
    )
    return p.parse_args()


def _run_one(env: str, name: str, seed: int, time_budget_s: int) -> tuple[str, str, float, float]:
    """Run a single (env, baseline) pair in a fresh process. Returns
    (env, baseline_name, score, wallclock_s)."""
    # Late imports so the worker process owns its own torch/jax state.
    import time as _time

    sys.path.insert(0, str(ROOT))
    import harness as _harness
    from baselines import count_bonus, eps_greedy_q
    from baselines import random as random_baseline

    train_fns = {
        "random": random_baseline.train,
        "eps_greedy_q": eps_greedy_q.train,
        "count_bonus": count_bonus.train,
    }
    t0 = _time.monotonic()
    pol = train_fns[name](env, seed, time_budget_s)
    score = _harness.evaluate(pol, env, seed=seed)
    return env, name, float(score), _time.monotonic() - t0


def main() -> None:
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(",")] if args.envs else list(harness.PANEL_SMOKE)
    jobs = [(env, name) for env in envs for name in BASELINE_NAMES]

    print(
        f"[build_baselines] envs={envs} baselines={BASELINE_NAMES} "
        f"jobs={len(jobs)} workers={args.workers} budget={args.time_budget_s}s",
        flush=True,
    )

    per_env: dict[str, dict[str, float]] = {env: {} for env in envs}
    t_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, env, name, args.seed, args.time_budget_s): (env, name)
            for env, name in jobs
        }
        for fut in as_completed(futures):
            env, name = futures[fut]
            try:
                _env, _name, score, wallclock = fut.result()
                per_env[env][name] = score
                print(
                    f"  [done] {env:36s} {name:15s} score={score:.6f} ({wallclock:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [crash] {env:36s} {name:15s}: {exc!r}", flush=True)

    out: dict[str, dict] = {}
    for env in envs:
        scores = per_env[env]
        rand_score = scores.get("random", 0.0)
        strong_scores = {k: v for k, v in scores.items() if k != "random"}
        strong = max(strong_scores.values()) if strong_scores else float(rand_score)
        out[env] = {
            "random": float(rand_score),
            "strong": float(strong),
            "per_baseline": {k: float(v) for k, v in strong_scores.items()},
        }

    with harness.BASELINES_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(
        f"\nwrote {harness.BASELINES_PATH}  total_wallclock={time.monotonic() - t_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    # ProcessPoolExecutor on Linux defaults to fork — inheriting JAX/torch state
    # is the usual cause of CUDA-after-fork errors. Force spawn.
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    # Stop torch from grabbing every CPU thread per worker.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    main()
