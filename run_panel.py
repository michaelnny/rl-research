"""Run train.py on a lean benchmark stage."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import harness

ROOT = Path(__file__).parent.resolve()
RUN_DIR = ROOT / "runs" / "last"
FINAL_RE = re.compile(r"^final_score:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE)
BEAT_EPS = 1e-6
DEFAULT_EVAL_GRACE_S = 30
CRAFTAX_EVAL_GRACE_S = 180


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=sorted(harness.STAGES), default="all")
    p.add_argument("--envs", default=None, help="Comma-separated env override")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "train.py",
        help="Candidate train.py path; defaults to repo-root train.py",
    )
    return p.parse_args()


def select_envs(args: argparse.Namespace) -> list[str]:
    envs = (
        [e.strip() for e in args.envs.split(",") if e.strip()]
        if args.envs
        else harness.STAGES[args.stage]
    )
    unknown = [e for e in envs if e not in harness.ENVS]
    if unknown:
        raise SystemExit(f"unknown env(s): {unknown}; choices={harness.ENVS}")
    return list(envs)


def eval_grace_s(env_id: str) -> int:
    return CRAFTAX_EVAL_GRACE_S if env_id == "Craftax-Symbolic-v1" else DEFAULT_EVAL_GRACE_S


def resolve_train_path(path: Path) -> Path:
    out = path if path.is_absolute() else ROOT / path
    if not out.exists():
        raise SystemExit(f"train path does not exist: {out}")
    return out.resolve()


def run_one(env_id: str, seed: int, budget: int, train_path: Path) -> float:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / f"{env_id}.log"
    cmd = [
        sys.executable,
        str(train_path),
        "--env",
        env_id,
        "--seed",
        str(seed),
        "--time-budget-s",
        str(budget),
    ]
    with log_path.open("w") as f:
        try:
            subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=budget + eval_grace_s(env_id),
                check=False,
            )
        except subprocess.TimeoutExpired:
            f.write(f"\n[run_panel] timeout after {budget + eval_grace_s(env_id)}s\n")
    match = FINAL_RE.findall(log_path.read_text())
    return float(match[-1]) if match else float("nan")


def main() -> None:
    args = parse_args()
    envs = select_envs(args)
    train_path = resolve_train_path(args.train_path)
    workers = args.workers or len(envs)
    baselines = harness.load_baselines()

    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    print(
        f"[run] stage={args.stage} envs={','.join(envs)} budget={args.time_budget_s}s "
        f"workers={workers} train_path={train_path}",
        flush=True,
    )
    t0 = time.monotonic()
    scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, env, args.seed, args.time_budget_s, train_path): env
            for env in envs
        }
        for fut in as_completed(futures):
            env = futures[fut]
            try:
                scores[env] = fut.result()
            except Exception as exc:
                (RUN_DIR / f"{env}.log").write_text(f"worker error: {exc!r}\n")
                scores[env] = float("nan")

    beat_random = 0
    beat_strong = 0
    for env in envs:
        score = scores[env]
        random_b = baselines[env]["random"]
        strong_b = baselines[env]["strong"]
        br = int(not math.isnan(score) and score > random_b + BEAT_EPS)
        bs = int(not math.isnan(score) and score > strong_b + BEAT_EPS)
        beat_random += br
        beat_strong += bs
        score_s = "nan" if math.isnan(score) else f"{score:.6f}"
        mr = "nan" if math.isnan(score) else f"{score - random_b:.6f}"
        ms = "nan" if math.isnan(score) else f"{score - strong_b:.6f}"
        print(
            f"[env] {env:36s} score={score_s:>12s} random={random_b:.6f} strong={strong_b:.6f} "
            f"margin_random={mr:>12s} margin_strong={ms:>12s} beat_random={br} beat_strong={bs}",
            flush=True,
        )

    print("---", flush=True)
    print(f"stage:           {args.stage}", flush=True)
    print(f"n_envs:          {len(envs)}", flush=True)
    print(f"n_beat_random:   {beat_random}", flush=True)
    print(f"n_beat_strong:   {beat_strong}", flush=True)
    print(f"wallclock_s:     {time.monotonic() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
