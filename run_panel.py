"""Run train.py against the panel and aggregate panel_score.

Frozen file — the agent does not edit this.

Usage:
    uv run run_panel.py                     # smoke tier, 5 envs in parallel (~5 min)
    uv run run_panel.py --hard              # hard tier, Option-B grouped (~2 h)
    uv run run_panel.py --quick             # one-env smoke (DST), <60 s
    uv run run_panel.py --envs e1,e2        # subset of smoke OR hard, parallel
    uv run run_panel.py --seed N            # eval seed (default 0)
    uv run run_panel.py --time-budget-s N   # override per-env wallclock cap

Output: a summary block grep-able by the loop in `program.md`. Key fields:

    panel_n_envs:        total envs run
    panel_n_beat_random: envs where final_score > BASELINES[env]['random']
    panel_n_beat_strong: envs where final_score > BASELINES[env]['strong']
    panel_wallclock_s:   total wallclock across all envs

Per-env detail is also printed; full train.py stdout is captured per env to
`runs/last/<env>.log` (gitignored), readable for debugging.

Exit code is always 0 (a candidate that doesn't beat anything is informative,
not an error). Crashed envs count as `final_score: nan`.

Hard-tier scheduling (Option B):
    Phase 1: Craftax-Symbolic-v1 alone (JAX kernel saturates the GPU).
    Phase 2: MiniHack + mo-halfcheetah + Humanoid in parallel (CPU-bound,
        small policies, fit comfortably in shared CPU + a sliver of VRAM).
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import harness

ROOT = Path(__file__).parent.resolve()
RUN_LOG_DIR = ROOT / "runs" / "last"
TIMEOUT_GRACE_S = 30  # SIGTERM buffer past the train-side budget

# Hard-tier scheduling: which envs share GPU and which run alone.
HARD_PHASE_1 = ["Craftax-Symbolic-v1"]  # solo (heavy JAX kernel)
HARD_PHASE_2 = ["MiniHack-Quest-Hard-v0", "mo-halfcheetah-v4", "Humanoid-v5"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--time-budget-s",
        type=int,
        default=None,
        help="Per-env wallclock cap. Defaults: smoke=300s, hard=3600s.",
    )
    p.add_argument(
        "--envs",
        type=str,
        default=None,
        help="Comma-separated subset of the chosen tier to run.",
    )
    p.add_argument(
        "--hard",
        action="store_true",
        help="Run the hard tier (Option-B grouped) instead of smoke.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: only deep-sea-treasure-concave-v0.",
    )
    return p.parse_args()


_FINAL_SCORE_RE = re.compile(r"^final_score:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE)


def run_one_env(env_id: str, seed: int, time_budget_s: int, log_path: Path) -> float:
    """Run train.py against env_id; return final_score or nan on failure."""
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--env",
        env_id,
        "--seed",
        str(seed),
        "--time-budget-s",
        str(time_budget_s),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        try:
            subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=time_budget_s + TIMEOUT_GRACE_S,
                check=False,
                cwd=ROOT,
            )
        except subprocess.TimeoutExpired:
            logf.write(f"\n[run_panel] TIMEOUT after {time_budget_s + TIMEOUT_GRACE_S}s\n")
    text = log_path.read_text()
    matches = _FINAL_SCORE_RE.findall(text)
    if not matches:
        return float("nan")
    return float(matches[-1])


def run_envs_parallel(
    envs: list[str],
    seed: int,
    time_budget_s: int,
    max_workers: int,
) -> dict[str, float]:
    """Run a list of envs concurrently via subprocess pool. Returns {env: score}."""
    scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_one_env,
                env,
                seed,
                time_budget_s,
                RUN_LOG_DIR / f"{env}.log",
            ): env
            for env in envs
        }
        for fut in futures:
            env = futures[fut]
            try:
                scores[env] = fut.result()
            except Exception as exc:
                (RUN_LOG_DIR / f"{env}.log").write_text(f"[run_panel] worker raised: {exc!r}\n")
                scores[env] = float("nan")
    return scores


def print_per_env(env: str, score: float, baselines: dict) -> tuple[float, float]:
    rand_b = float(baselines[env]["random"])
    strong_b = float(baselines[env]["strong"])
    beat_rand = 1.0 if (not math.isnan(score) and score > rand_b) else 0.0
    beat_strong = 1.0 if (not math.isnan(score) and score > strong_b) else 0.0
    score_s = "NaN (crash/timeout)" if math.isnan(score) else f"{score:.6f}"
    print(
        f"[panel] {env:36s}  score={score_s:24s}  "
        f"random={rand_b:.4g}  strong={strong_b:.4g}  "
        f"beat_random={int(beat_rand)}  beat_strong={int(beat_strong)}",
        flush=True,
    )
    return beat_rand, beat_strong


def main() -> None:
    args = parse_args()

    # Pick tier + env list.
    if args.quick:
        envs_phase_1: list[str] = []
        envs_phase_2 = ["deep-sea-treasure-concave-v0"]
        tier_label = "quick"
        default_budget = harness.TIME_BUDGET_SMOKE
    elif args.hard:
        # Option-B scheduling
        if args.envs:
            requested = [e.strip() for e in args.envs.split(",") if e.strip()]
            for e in requested:
                if e not in harness.PANEL_HARD:
                    print(f"[run_panel] unknown hard env: {e}", file=sys.stderr)
                    sys.exit(2)
            envs_phase_1 = [e for e in requested if e in HARD_PHASE_1]
            envs_phase_2 = [e for e in requested if e in HARD_PHASE_2]
        else:
            envs_phase_1 = list(HARD_PHASE_1)
            envs_phase_2 = list(HARD_PHASE_2)
        tier_label = "hard"
        default_budget = harness.TIME_BUDGET_HARD
    else:
        # Smoke (default): everything in parallel.
        envs_phase_1 = []
        if args.envs:
            requested = [e.strip() for e in args.envs.split(",") if e.strip()]
            for e in requested:
                if e not in harness.PANEL_SMOKE:
                    print(f"[run_panel] unknown smoke env: {e}", file=sys.stderr)
                    sys.exit(2)
            envs_phase_2 = requested
        else:
            envs_phase_2 = list(harness.PANEL_SMOKE)
        tier_label = "smoke"
        default_budget = harness.TIME_BUDGET_SMOKE

    time_budget_s = args.time_budget_s if args.time_budget_s is not None else default_budget

    baselines = harness.load_baselines()

    # Wipe prior log dir so `runs/last/` always reflects this sweep only.
    if RUN_LOG_DIR.exists():
        shutil.rmtree(RUN_LOG_DIR)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[run_panel] tier={tier_label}  budget={time_budget_s}s/env  "
        f"phase1={envs_phase_1}  phase2={envs_phase_2}",
        flush=True,
    )

    t0 = time.monotonic()
    all_scores: dict[str, float] = {}

    # Phase 1: solo (typically Craftax on hard tier; empty otherwise).
    if envs_phase_1:
        ph1_scores = run_envs_parallel(envs_phase_1, args.seed, time_budget_s, max_workers=1)
        all_scores.update(ph1_scores)

    # Phase 2: parallel pool.
    if envs_phase_2:
        max_workers = len(envs_phase_2)
        ph2_scores = run_envs_parallel(
            envs_phase_2, args.seed, time_budget_s, max_workers=max_workers
        )
        all_scores.update(ph2_scores)

    # Print per-env in a deterministic order: phase1 then phase2.
    n_beat_random = 0
    n_beat_strong = 0
    for env in envs_phase_1 + envs_phase_2:
        beat_r, beat_s = print_per_env(env, all_scores[env], baselines)
        n_beat_random += int(beat_r)
        n_beat_strong += int(beat_s)

    total_wallclock = time.monotonic() - t0
    n_envs = len(all_scores)

    print("---", flush=True)
    print(f"panel_tier:          {tier_label}", flush=True)
    print(f"panel_n_envs:        {n_envs}", flush=True)
    print(f"panel_n_beat_random: {n_beat_random}", flush=True)
    print(f"panel_n_beat_strong: {n_beat_strong}", flush=True)
    print(f"panel_wallclock_s:   {total_wallclock:.1f}", flush=True)


if __name__ == "__main__":
    main()
