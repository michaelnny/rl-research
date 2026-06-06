"""Run the probe-v1 empirical ladder for a run directory.

The run directory must contain:
- candidate.json
- train.py
- train_ablate.py

Usage:
    uv run python scripts/run_probe_ladder.py worklogs/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
from scripts.validate_candidate import load_json, validate_candidate  # noqa: E402

ENV_RE = re.compile(
    r"^\[env\]\s+(?P<env>\S+)\s+score=\s*(?P<score>\S+).*"
    r"beat_random=(?P<beat_random>[01])\s+beat_strong=(?P<beat_strong>[01])",
    re.MULTILINE,
)
SUMMARY_RE = re.compile(
    r"^(?P<key>n_beat_random|n_beat_strong|wallclock_s):\s+(?P<value>\S+)", re.MULTILINE
)
BEAT_EPS = 1e-6


@dataclass(frozen=True)
class PanelResult:
    envs: list[str]
    scores: dict[str, float | None]
    beat_random: int
    beat_strong: int
    wallclock_s: float
    status: str


def parse_score(raw: str) -> float | None:
    if raw == "nan":
        return None
    value = float(raw)
    return None if math.isnan(value) else value


def parse_panel_text(text: str) -> PanelResult:
    envs: list[str] = []
    scores: dict[str, float | None] = {}
    beat_random = 0
    beat_strong = 0
    for match in ENV_RE.finditer(text):
        env = match.group("env")
        envs.append(env)
        scores[env] = parse_score(match.group("score"))
        beat_random += int(match.group("beat_random"))
        beat_strong += int(match.group("beat_strong"))

    summary = {m.group("key"): m.group("value") for m in SUMMARY_RE.finditer(text)}
    if "n_beat_random" in summary:
        beat_random = int(summary["n_beat_random"])
    if "n_beat_strong" in summary:
        beat_strong = int(summary["n_beat_strong"])
    wallclock_s = float(summary.get("wallclock_s", 0.0))

    if "[run_panel] timeout" in text:
        status = "killed-budget"
    elif not envs:
        status = "killed-error"
    else:
        status = "completed"
    return PanelResult(envs, scores, beat_random, beat_strong, wallclock_s, status)


def score_delta(
    candidate_scores: dict[str, float | None],
    ablation_scores: dict[str, float | None],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for env, score in candidate_scores.items():
        ablated = ablation_scores.get(env)
        out[env] = None if score is None or ablated is None else score - ablated
    return out


def should_confirm(candidate: PanelResult, ablation: PanelResult) -> bool:
    if candidate.status != "completed" or ablation.status != "completed":
        return False
    if candidate.beat_random <= 0:
        return False
    return any(
        delta is not None and delta > BEAT_EPS
        for delta in score_delta(candidate.scores, ablation.scores).values()
    )


def run_panel(
    *,
    train_path: Path,
    stage: str,
    seed: int,
    budget_s: int,
    output_path: Path,
) -> PanelResult:
    cmd = [
        sys.executable,
        str(ROOT / "run_panel.py"),
        "--train-path",
        str(train_path),
        "--stage",
        stage,
        "--seed",
        str(seed),
        "--time-budget-s",
        str(budget_s),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=budget_s + 240,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        output = stdout + f"\n[run_probe_ladder] timeout after {budget_s + 240}s\n"
        output_path.write_text(output)
        return PanelResult([], {}, 0, 0, 0.0, "killed-budget")
    output_path.write_text(completed.stdout)
    result = parse_panel_text(completed.stdout)
    if completed.returncode != 0 and result.status == "completed":
        return PanelResult(
            result.envs,
            result.scores,
            result.beat_random,
            result.beat_strong,
            result.wallclock_s,
            "killed-error",
        )
    return result


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage", choices=sorted(harness.STAGES), default=None)
    parser.add_argument("--smoke-budget-s", type=int, default=30)
    parser.add_argument("--claim-budget-s", type=int, default=harness.TIME_BUDGET_S)
    parser.add_argument("--confirm-seeds", default="1,2")
    return parser.parse_args()


def load_candidate(run_dir: Path) -> dict[str, object]:
    path = run_dir / "candidate.json"
    data = load_json(path)
    errors = validate_candidate(data)
    if errors:
        for error in errors:
            print(f"candidate.json: {error}", file=sys.stderr)
        raise SystemExit(1)
    assert isinstance(data, dict)
    return data


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate = load_candidate(run_dir)
    stage = args.stage or str(candidate["claimed_stage"])
    train_path = run_dir / "train.py"
    ablation_path = run_dir / "train_ablate.py"
    if not train_path.exists():
        raise SystemExit(f"missing candidate train file: {train_path}")
    if not ablation_path.exists():
        raise SystemExit(f"missing ablation train file: {ablation_path}")

    t0 = time.monotonic()
    ladder: dict[str, str] = {}
    confirmation: list[dict[str, object]] = []

    smoke = run_panel(
        train_path=train_path,
        stage=stage,
        seed=0,
        budget_s=args.smoke_budget_s,
        output_path=run_dir / "panel-smoke.txt",
    )
    ladder["smoke"] = smoke.status
    claim = PanelResult([], {}, 0, 0, 0.0, "skipped")
    ablation = PanelResult([], {}, 0, 0, 0.0, "skipped")

    if smoke.status == "completed":
        claim = run_panel(
            train_path=train_path,
            stage=stage,
            seed=0,
            budget_s=args.claim_budget_s,
            output_path=run_dir / "panel-claim.txt",
        )
    ladder["claim"] = claim.status

    if claim.status == "completed":
        ablation = run_panel(
            train_path=ablation_path,
            stage=stage,
            seed=0,
            budget_s=args.claim_budget_s,
            output_path=run_dir / "panel-ablation.txt",
        )
    ladder["ablation"] = ablation.status

    if should_confirm(claim, ablation):
        seeds = [int(s.strip()) for s in args.confirm_seeds.split(",") if s.strip()]
        confirm_status = "completed"
        for seed in seeds:
            cand = run_panel(
                train_path=train_path,
                stage=stage,
                seed=seed,
                budget_s=args.claim_budget_s,
                output_path=run_dir / f"panel-confirm-candidate-seed{seed}.txt",
            )
            abl = run_panel(
                train_path=ablation_path,
                stage=stage,
                seed=seed,
                budget_s=args.claim_budget_s,
                output_path=run_dir / f"panel-confirm-ablation-seed{seed}.txt",
            )
            if cand.status != "completed" or abl.status != "completed":
                confirm_status = (
                    "killed-budget"
                    if "killed-budget" in {cand.status, abl.status}
                    else "killed-error"
                )
            confirmation.append(
                {
                    "seed": seed,
                    "candidate_scores": cand.scores,
                    "ablation_scores": abl.scores,
                    "candidate_beat_random": cand.beat_random,
                    "candidate_beat_strong": cand.beat_strong,
                    "ablation_beat_random": abl.beat_random,
                    "ablation_beat_strong": abl.beat_strong,
                }
            )
        ladder["confirmation"] = confirm_status
    else:
        ladder["confirmation"] = "skipped"

    status = "completed"
    for rung in ("smoke", "claim", "ablation"):
        if ladder[rung] not in {"completed", "skipped"}:
            status = ladder[rung]
            break
    if status == "completed" and ladder["confirmation"] not in {"completed", "skipped"}:
        status = ladder["confirmation"]

    result = {
        "run_id": candidate["run_id"],
        "mode": "probe-v1",
        "stage": stage,
        "envs": claim.envs,
        "scores": claim.scores,
        "beat_random": claim.beat_random,
        "beat_strong": claim.beat_strong,
        "ablation_scores": ablation.scores,
        "ablation_beat_random": ablation.beat_random,
        "ablation_beat_strong": ablation.beat_strong,
        "ablation_delta": score_delta(claim.scores, ablation.scores),
        "confirmation": confirmation,
        "ladder": ladder,
        "wallclock_s": time.monotonic() - t0,
        "n_retries": 0,
        "status": status,
        "commit": git_commit(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
