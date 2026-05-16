"""Runtime helpers that enforce the run-artifact contract from `docs/contract.md`.

The Operator MUST call these helpers; ad-hoc result.json writing is a contract
violation. The Researcher MAY use `next_run_id` when allocating a directory at
iteration start.

Public API:
    next_run_id(thread_slug)     -> str
    write_result(...)            -> Path           # builds + writes a result.json
    validate_result_json(path)   -> None           # raises on any deviation
    append_to_ledger(result_path) -> None          # atomic append to ledger.jsonl
"""

from __future__ import annotations

import fcntl
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAB_DIR = REPO_ROOT / "lab"
RUNS_DIR = LAB_DIR / "runs"
LEDGER_PATH = LAB_DIR / "ledger.jsonl"
SCHEMA_PATH = LAB_DIR / "result.schema.json"

RUN_ID_PATTERN = re.compile(r"^([0-9]{4})-([a-z0-9-]+)$")
THREAD_SLUG_PATTERN = re.compile(r"^[a-z0-9-]+$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")

VALID_STAGES = {"A-only", "A+B"}
VALID_STATUSES = {
    "sanity-failed",
    "benchmark-failed",
    "killed-budget",
    "killed-error",
    "completed",
}
VALID_PILLARS = {"sparse-long-horizon", "long-horizon-dense", "multi-signal"}


class ContractViolation(Exception):
    """Raised when a result.json fails contract validation."""


def next_run_id(thread_slug: str) -> str:
    """Allocate the next NNNN-thread-slug under lab/runs/.

    Scans existing run directories for the highest 4-digit prefix and increments.
    Sequence is shared across all threads — `0007-foo` and `0008-bar` are
    consecutive even if their threads differ.
    """
    if not THREAD_SLUG_PATTERN.fullmatch(thread_slug):
        raise ValueError(
            f"thread_slug must match {THREAD_SLUG_PATTERN.pattern!r}, got {thread_slug!r}"
        )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    highest = 0
    for child in RUNS_DIR.iterdir():
        if not child.is_dir():
            continue
        m = RUN_ID_PATTERN.fullmatch(child.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return f"{highest + 1:04d}-{thread_slug}"


def write_result(
    *,
    run_id: str,
    stage: str,
    status: str,
    primary_benchmark: str,
    pillar: str,
    thread: str,
    seeds: list[int],
    env_steps: int,
    wallclock_s: float,
    best_return: float | list[float],
    final_return: float | list[float],
    by_seed: dict[str, dict[str, Any]],
    sanity: dict[str, Any],
    git_sha: str,
    started_at: datetime,
    ended_at: datetime,
    deps_lock: str = "",
    parent_run_id: str | None = None,
    notes: str = "",
) -> Path:
    """Build and atomically write a `result.json` for the given run.

    Returns the path written. Caller is responsible for ensuring the run
    directory exists and contains hypothesis.md / train.py.
    """
    payload: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "primary_benchmark": primary_benchmark,
        "pillar": pillar,
        "thread": thread,
        "seeds": seeds,
        "env_steps": env_steps,
        "wallclock_s": wallclock_s,
        "best_return": best_return,
        "final_return": final_return,
        "by_seed": by_seed,
        "sanity": sanity,
        "git_sha": git_sha,
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "deps_lock": deps_lock,
        "parent_run_id": parent_run_id,
        "notes": notes,
    }
    out = RUNS_DIR / run_id / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out)
    validate_result_json(out)
    return out


def validate_result_json(path: Path | str) -> None:
    """Enforce the contract from `docs/contract.md` / `lab/result.schema.json`.

    Raises ContractViolation on any deviation. Does not depend on `jsonschema`
    so it stays runnable in the sub-process train.py environment.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    required = {
        "run_id",
        "stage",
        "status",
        "primary_benchmark",
        "pillar",
        "thread",
        "seeds",
        "env_steps",
        "wallclock_s",
        "best_return",
        "final_return",
        "by_seed",
        "sanity",
        "git_sha",
        "started_at",
        "ended_at",
    }
    missing = required - data.keys()
    if missing:
        raise ContractViolation(f"{path}: missing required keys {sorted(missing)}")

    _check(
        RUN_ID_PATTERN.fullmatch(data["run_id"]) is not None,
        f"run_id {data['run_id']!r} does not match {RUN_ID_PATTERN.pattern}",
    )
    _check(
        data["stage"] in VALID_STAGES, f"stage must be one of {VALID_STAGES}, got {data['stage']!r}"
    )
    _check(
        data["status"] in VALID_STATUSES,
        f"status must be one of {VALID_STATUSES}, got {data['status']!r}",
    )
    _check(
        data["pillar"] in VALID_PILLARS,
        f"pillar must be one of {VALID_PILLARS}, got {data['pillar']!r}",
    )
    _check(
        THREAD_SLUG_PATTERN.fullmatch(data["thread"]) is not None,
        f"thread {data['thread']!r} not kebab-case",
    )
    _check(
        GIT_SHA_PATTERN.fullmatch(data["git_sha"]) is not None,
        f"git_sha {data['git_sha']!r} not a sha",
    )
    _check(
        isinstance(data["seeds"], list)
        and len(data["seeds"]) >= 1
        and all(isinstance(s, int) for s in data["seeds"]),
        "seeds must be a non-empty list of ints",
    )
    _check(
        isinstance(data["env_steps"], int) and data["env_steps"] >= 0,
        "env_steps must be a non-negative int",
    )
    _check(
        isinstance(data["wallclock_s"], int | float) and data["wallclock_s"] >= 0,
        "wallclock_s must be non-negative",
    )
    _check_return_value(data["best_return"], "best_return")
    _check_return_value(data["final_return"], "final_return")

    sanity = data["sanity"]
    _check(
        set(sanity.keys()) >= {"passed", "by_env", "retries"},
        "sanity must have passed/by_env/retries",
    )
    _check(isinstance(sanity["passed"], bool), "sanity.passed must be bool")
    _check(
        isinstance(sanity["retries"], int) and sanity["retries"] >= 0,
        "sanity.retries must be non-negative int",
    )
    for env, entry in sanity["by_env"].items():
        _check(
            set(entry.keys()) >= {"return_random", "return_final", "passed"},
            f"sanity.by_env[{env}] missing required keys",
        )

    for parsed_at in ("started_at", "ended_at"):
        try:
            datetime.fromisoformat(data[parsed_at].replace("Z", "+00:00"))
        except (ValueError, AttributeError) as exc:
            raise ContractViolation(f"{parsed_at}={data[parsed_at]!r} not iso8601") from exc

    # Cross-field consistency
    _check(
        set(data["by_seed"].keys()) == {str(s) for s in data["seeds"]},
        "by_seed keys must match seeds list (as strings)",
    )


def append_to_ledger(result_path: Path | str) -> None:
    """Append a one-line ledger entry derived from result.json. Atomic via flock."""
    result_path = Path(result_path)
    validate_result_json(result_path)
    with result_path.open() as f:
        r = json.load(f)
    line = {
        "run_id": r["run_id"],
        "thread": r["thread"],
        "pillar": r["pillar"],
        "primary_benchmark": r["primary_benchmark"],
        "status": r["status"],
        "best_return": r["best_return"],
        "wallclock_s": r["wallclock_s"],
        "verdict_curator": None,
        "hypothesis_path": f"lab/runs/{r['run_id']}/hypothesis.md",
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(line) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractViolation(msg)


def _check_return_value(v: Any, name: str) -> None:
    if isinstance(v, int | float):
        return
    if isinstance(v, list) and all(isinstance(x, int | float) for x in v):
        return
    raise ContractViolation(f"{name} must be number or list of numbers, got {v!r}")
