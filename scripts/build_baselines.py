"""Build `baselines.json` by running each frozen baseline against each
panel env. Run-once script — output committed to git.

Usage:
    uv run scripts/build_baselines.py [--time-budget-s N] [--seed N]
                                       [--envs e1,e2,...]

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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
from baselines import count_bonus, eps_greedy_q  # noqa: E402
from baselines import random as random_baseline  # noqa: E402

STRONG_BASELINES = {
    "eps_greedy_q": eps_greedy_q.train,
    "count_bonus": count_bonus.train,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_PER_ENV)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(",")] if args.envs else list(harness.PANEL_SMOKE)

    out: dict[str, dict] = {}
    for env in envs:
        print(f"\n=== {env} ===", flush=True)
        per_b: dict[str, float] = {}

        t0 = time.monotonic()
        rand_pol = random_baseline.train(env, args.seed, args.time_budget_s)
        rand_score = harness.evaluate(rand_pol, env, seed=args.seed)
        print(f"  random         : {rand_score:.6f}  ({time.monotonic() - t0:.1f}s)", flush=True)

        for name, train_fn in STRONG_BASELINES.items():
            t0 = time.monotonic()
            try:
                pol = train_fn(env, args.seed, args.time_budget_s)
                score = harness.evaluate(pol, env, seed=args.seed)
            except Exception as e:
                print(f"  {name:15s}: CRASHED ({e!r})", flush=True)
                continue
            per_b[name] = float(score)
            print(f"  {name:15s}: {score:.6f}  ({time.monotonic() - t0:.1f}s)", flush=True)

        strong = max(per_b.values()) if per_b else float(rand_score)
        out[env] = {
            "random": float(rand_score),
            "strong": float(strong),
            "per_baseline": per_b,
        }

    with harness.BASELINES_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {harness.BASELINES_PATH}", flush=True)


if __name__ == "__main__":
    main()
