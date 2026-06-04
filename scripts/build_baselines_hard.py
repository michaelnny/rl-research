"""Build the `our_baseline` columns of `baselines_hard.json` by running each
frozen baseline against each hard-tier env. Run-once script — output
committed to git.

Usage:
    uv run scripts/build_baselines_hard.py [--time-budget-s N] [--seed N]
                                            [--envs e1,e2,...]
                                            [--phase2-workers N]

Schedule (Option-B grouped, mirrors `run_panel.py --hard`):
    Phase 1: Craftax-Symbolic-v1 baselines run sequentially (workers=1).
        Each Craftax process spins up its own JAX/XLA GPU context; running
        more than one concurrently on a single 3090 Ti is the cause of the
        2026-05-27 GPU/driver lockup that froze the host. Sequential keeps
        the GPU exclusive to one Craftax baseline at a time.
    Phase 2: MiniHack-Quest-Hard-v0 + mo-halfcheetah-v4 + Humanoid-v5
        baselines (3 envs x 3 baselines = 9 jobs) run in parallel via a
        ProcessPool. These are CPU-bound (MuJoCo, NetHack) and small; 6
        workers default leaves headroom on a 24-core box with OMP=2.

Memory safety:
    Thread caps (OMP_NUM_THREADS=2, MKL_NUM_THREADS=2) are set BEFORE the
    pool is created so spawn-mode workers inherit them. JAX preallocation
    is disabled in `harness.py` (XLA_PYTHON_CLIENT_PREALLOCATE=false), so
    the Craftax process grows on demand instead of grabbing 75% of VRAM at
    import. Phase 2 envs do not use JAX.

The published_sota column in baselines_hard.json is preserved verbatim;
only our_baseline.{random, strong, per_baseline} are rewritten.
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

# Mirror run_panel.py's Option-B grouping. Phase 1 envs run sequentially
# (each saturates the GPU); Phase 2 envs run in parallel (CPU-bound).
PHASE_1_ENVS = ["Craftax-Symbolic-v1"]
PHASE_2_ENVS = ["MiniHack-Quest-Hard-v0", "mo-halfcheetah-v4", "Humanoid-v5"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_HARD)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--envs",
        type=str,
        default=None,
        help="Subset of PANEL_HARD to run; defaults to all 4.",
    )
    p.add_argument(
        "--phase2-workers",
        type=int,
        default=6,
        help="Concurrent CPU-bound jobs in Phase 2. Default 6 (= 9 jobs / ~1.5 waves) "
        "leaves headroom on a 24-core box at OMP_NUM_THREADS=2.",
    )
    p.add_argument(
        "--baselines",
        type=str,
        default=",".join(BASELINE_NAMES),
        help="Comma-separated subset of baselines to run; defaults to all three. "
        "Use 'random' alone to re-eval just the random floor with --random-eval-episodes.",
    )
    p.add_argument(
        "--random-eval-episodes",
        type=int,
        default=None,
        help="Override harness default (20) for the random baseline only. "
        "random.train() is a no-op so its score is a Monte-Carlo estimate of the "
        "uniform-action policy; raising n_episodes tightens that estimate without "
        "burning extra wallclock on training.",
    )
    return p.parse_args()


def _run_one(
    env: str,
    name: str,
    seed: int,
    time_budget_s: int,
    random_eval_episodes: int | None = None,
) -> tuple[str, str, float, float]:
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
    if name == "random" and random_eval_episodes is not None:
        score = _harness.evaluate(pol, env, seed=seed, n_episodes=random_eval_episodes)
    else:
        score = _harness.evaluate(pol, env, seed=seed)
    return env, name, float(score), _time.monotonic() - t0


def _run_phase(
    label: str,
    jobs: list[tuple[str, str]],
    seed: int,
    time_budget_s: int,
    workers: int,
    per_env: dict[str, dict[str, float]],
    random_eval_episodes: int | None = None,
) -> None:
    if not jobs:
        return
    print(
        f"\n[{label}] jobs={len(jobs)} workers={workers} budget={time_budget_s}s",
        flush=True,
    )
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, env, name, seed, time_budget_s, random_eval_episodes): (
                env,
                name,
            )
            for env, name in jobs
        }
        for fut in as_completed(futures):
            env, name = futures[fut]
            try:
                _, _, score, wall = fut.result()
                per_env[env][name] = score
                print(
                    f"  [{label}/done] {env:30s} {name:15s} score={score:.6f} ({wall:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"  [{label}/crash] {env:30s} {name:15s}: {exc!r}",
                    flush=True,
                )
    print(f"[{label}] phase wallclock={time.monotonic() - t0:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(",")] if args.envs else list(harness.PANEL_HARD)
    for e in envs:
        if e not in harness.PANEL_HARD:
            print(f"[build_baselines_hard] unknown hard env: {e}", file=sys.stderr)
            sys.exit(2)

    selected_baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    for b in selected_baselines:
        if b not in BASELINE_NAMES:
            print(f"[build_baselines_hard] unknown baseline: {b}", file=sys.stderr)
            sys.exit(2)

    phase1 = [(env, name) for env in envs if env in PHASE_1_ENVS for name in selected_baselines]
    phase2 = [(env, name) for env in envs if env in PHASE_2_ENVS for name in selected_baselines]

    per_env: dict[str, dict[str, float]] = {env: {} for env in envs}

    print(
        f"[build_baselines_hard] envs={envs} baselines={selected_baselines} "
        f"phase1_jobs={len(phase1)} phase2_jobs={len(phase2)} "
        f"phase2_workers={args.phase2_workers} budget={args.time_budget_s}s "
        f"random_eval_episodes={args.random_eval_episodes}",
        flush=True,
    )
    t_start = time.monotonic()

    # Phase 1: Craftax sequentially (workers=1) — one JAX/GPU context at a time.
    _run_phase(
        "phase1",
        phase1,
        args.seed,
        args.time_budget_s,
        1,
        per_env,
        random_eval_episodes=args.random_eval_episodes,
    )

    # Phase 2: CPU-bound envs in parallel.
    _run_phase(
        "phase2",
        phase2,
        args.seed,
        args.time_budget_s,
        args.phase2_workers,
        per_env,
        random_eval_episodes=args.random_eval_episodes,
    )

    if harness.BASELINES_HARD_PATH.exists():
        with harness.BASELINES_HARD_PATH.open() as f:
            existing = json.load(f)
    else:
        existing = {}

    for env in envs:
        new_scores = per_env[env]
        env_block = existing.get(env, {})
        prev = env_block.get("our_baseline", {})
        prev_per = prev.get("per_baseline", {}) if isinstance(prev, dict) else {}

        # Merge: new scores override prior per-baseline values; baselines we
        # didn't run this pass keep their prior values. Then recompute strong.
        merged_per: dict[str, float] = {k: float(v) for k, v in prev_per.items() if k != "random"}
        for k, v in new_scores.items():
            if k != "random":
                merged_per[k] = float(v)

        if "random" in new_scores:
            rand_score = float(new_scores["random"])
        elif isinstance(prev, dict) and "random" in prev:
            rand_score = float(prev["random"])
        else:
            rand_score = 0.0

        strong = max(merged_per.values()) if merged_per else float(rand_score)
        env_block["our_baseline"] = {
            "random": rand_score,
            "strong": float(strong),
            "per_baseline": merged_per,
        }
        existing[env] = env_block

    with harness.BASELINES_HARD_PATH.open("w") as f:
        json.dump(existing, f, indent=2)
    print(
        f"\nwrote {harness.BASELINES_HARD_PATH}  total_wallclock={time.monotonic() - t_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    main()
