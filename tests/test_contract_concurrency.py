"""Tests for concurrent-safety + Curator-edit primitives in
``rl_research.contract``.

These tests stress the parts of the contract module that are load-bearing for
the autonomous loop:

* ``next_run_id`` must never hand out the same id to two concurrent callers.
* ``append_to_ledger`` lines must never interleave with each other.
* ``append_to_ledger`` must never interleave with ``update_ledger_verdict``.
* ``update_ledger_verdict`` is atomic (replaces only the matching line) and
  validates inputs.

Concurrency is exercised via ``ProcessPoolExecutor`` so each worker has its
own ``fcntl`` file table — interpreter-level monkeypatching can't paper over
a missing flock the way thread-level testing might.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rl_research import contract


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Redirect the contract module's lab-relative paths into tmp_path so the
    real corpus is never touched."""
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    (lab_dir / "runs").mkdir()
    monkeypatch.setattr(contract, "LAB_DIR", lab_dir)
    monkeypatch.setattr(contract, "RUNS_DIR", lab_dir / "runs")
    monkeypatch.setattr(contract, "LEDGER_PATH", lab_dir / "ledger.jsonl")
    monkeypatch.setattr(contract, "LEDGER_LOCK_PATH", lab_dir / ".ledger.lock")
    monkeypatch.setattr(contract, "NEXT_RUN_ID_LOCK", lab_dir / ".next_run_id.lock")
    return lab_dir


def _make_result(run_id: str, status: str = "completed") -> dict:
    """Build a contract-valid result.json payload."""
    return {
        "run_id": run_id,
        "stage": "A+B",
        "status": status,
        "primary_benchmark": "CartPole-v1",
        "pillar": "sparse-long-horizon",
        "thread": "test-thread",
        "seeds": [0],
        "env_steps": 1000,
        "wallclock_s": 1.5,
        "best_return": 100.0,
        "final_return": 100.0,
        "by_seed": {"0": {"best_return": 100.0, "final_return": 100.0, "env_steps": 1000}},
        "sanity": {
            "passed": True,
            "by_env": {
                "CartPole-v1": {
                    "return_random": 20.0,
                    "return_final": 100.0,
                    "passed": True,
                }
            },
            "retries": 0,
        },
        "git_sha": "0" * 40,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "ended_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "deps_lock": "",
        "parent_run_id": None,
        "notes": "",
    }


def _write_result_file(lab: Path, run_id: str, status: str = "completed") -> Path:
    run_dir = lab / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "result.json"
    p.write_text(json.dumps(_make_result(run_id, status), indent=2, sort_keys=True))
    return p


def test_update_ledger_verdict_round_trip(lab):
    p = _write_result_file(lab, "0001-foo")
    contract.append_to_ledger(p)
    p2 = _write_result_file(lab, "0002-foo")
    contract.append_to_ledger(p2)

    updated = contract.update_ledger_verdict("0001-foo", "promising", notes="strong novelty signal")
    assert updated is True

    lines = contract.LEDGER_PATH.read_text().splitlines()
    assert len(lines) == 2
    e0, e1 = json.loads(lines[0]), json.loads(lines[1])
    assert e0["run_id"] == "0001-foo"
    assert e0["verdict_curator"] == "promising"
    assert e0["verdict_notes"] == "strong novelty signal"
    assert e1["run_id"] == "0002-foo"
    assert e1["verdict_curator"] is None  # untouched


def test_update_ledger_verdict_unknown_run_id(lab):
    p = _write_result_file(lab, "0001-foo")
    contract.append_to_ledger(p)
    assert contract.update_ledger_verdict("9999-nope", "promising") is False
    # ledger unchanged
    [entry] = [json.loads(line) for line in contract.LEDGER_PATH.read_text().splitlines()]
    assert entry["run_id"] == "0001-foo"
    assert entry["verdict_curator"] is None


def test_update_ledger_verdict_rejects_invalid(lab):
    p = _write_result_file(lab, "0001-foo")
    contract.append_to_ledger(p)
    with pytest.raises(contract.ContractViolation):
        contract.update_ledger_verdict("0001-foo", "amazing")


def test_update_ledger_verdict_no_ledger_yet(lab):
    """If no ledger exists, the helper returns False rather than crashing —
    the Curator can be invoked before the Engineer's first run completes."""
    assert contract.update_ledger_verdict("0001-foo", "promising") is False


def test_update_ledger_verdict_preserves_malformed_lines(lab):
    """A truncated last line in the ledger (preflight will catch it on the
    next iteration) should be preserved verbatim — not silently discarded."""
    contract.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    p = _write_result_file(lab, "0001-foo")
    contract.append_to_ledger(p)
    # Tack on a malformed trailing line.
    with contract.LEDGER_PATH.open("a") as f:
        f.write("this is not json\n")

    contract.update_ledger_verdict("0001-foo", "dead-end")
    text = contract.LEDGER_PATH.read_text()
    assert "this is not json" in text
    valid = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    assert valid[0]["verdict_curator"] == "dead-end"


# --- concurrency stress -------------------------------------------------------


def _worker_append(args):
    """Top-level worker (must be picklable for ProcessPoolExecutor)."""
    lab_dir, run_id = args
    # Re-bind module-level paths inside this child process.
    from rl_research import contract as c

    c.LAB_DIR = Path(lab_dir)
    c.RUNS_DIR = c.LAB_DIR / "runs"
    c.LEDGER_PATH = c.LAB_DIR / "ledger.jsonl"
    c.LEDGER_LOCK_PATH = c.LAB_DIR / ".ledger.lock"
    c.NEXT_RUN_ID_LOCK = c.LAB_DIR / ".next_run_id.lock"
    p = _write_result_file(c.LAB_DIR, run_id)
    c.append_to_ledger(p)
    return run_id


def test_append_to_ledger_concurrent(tmp_path):
    """16 workers each append a distinct ledger line. No flock = ledger
    corruption (interleaved bytes, missing newlines, JSON parse failures).
    With flock, every line is intact and all 16 are present."""
    lab_dir = tmp_path / "lab"
    (lab_dir / "runs").mkdir(parents=True)
    n = 16
    args = [(str(lab_dir), f"{i:04d}-conc") for i in range(1, n + 1)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_worker_append, args))
    assert sorted(results) == sorted(r for _, r in args)
    lines = (lab_dir / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == n
    parsed = [json.loads(line) for line in lines]  # would raise on corruption
    assert {p["run_id"] for p in parsed} == {f"{i:04d}-conc" for i in range(1, n + 1)}


def _worker_next_run_id(args):
    lab_dir, _ = args
    from rl_research import contract as c

    c.LAB_DIR = Path(lab_dir)
    c.RUNS_DIR = c.LAB_DIR / "runs"
    c.NEXT_RUN_ID_LOCK = c.LAB_DIR / ".next_run_id.lock"
    return c.next_run_id("conc")


def test_next_run_id_concurrent(tmp_path):
    """8 workers each call next_run_id concurrently. Without the flock they
    can all see the same highest dir and allocate the same id; with it, each
    gets a distinct id 0001..0008."""
    lab_dir = tmp_path / "lab"
    (lab_dir / "runs").mkdir(parents=True)
    args = [(str(lab_dir), i) for i in range(8)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        ids = list(ex.map(_worker_next_run_id, args))
    assert len(set(ids)) == 8, f"duplicate run_ids: {ids}"
    nums = sorted(int(r.split("-", 1)[0]) for r in ids)
    assert nums == list(range(1, 9))


def _worker_mixed(args):
    """Half the workers append, half update — exercises append/update
    interleaving under flock."""
    lab_dir, kind, run_id = args
    from rl_research import contract as c

    c.LAB_DIR = Path(lab_dir)
    c.RUNS_DIR = c.LAB_DIR / "runs"
    c.LEDGER_PATH = c.LAB_DIR / "ledger.jsonl"
    c.LEDGER_LOCK_PATH = c.LAB_DIR / ".ledger.lock"
    c.NEXT_RUN_ID_LOCK = c.LAB_DIR / ".next_run_id.lock"
    if kind == "append":
        p = _write_result_file(c.LAB_DIR, run_id)
        c.append_to_ledger(p)
        return ("append", run_id)
    else:
        # Best-effort update: if the line isn't there yet, returns False.
        ok = c.update_ledger_verdict(run_id, "promising")
        return ("update", run_id, ok)


def test_append_and_update_no_interleave(tmp_path):
    """Pre-seed the ledger with 4 entries, then concurrently fire 4 more
    appends + 4 verdict-updates targeting the seeded entries. The ledger
    must remain valid JSONL and contain exactly 8 entries; updates against
    pre-existing run_ids must succeed."""
    lab_dir = tmp_path / "lab"
    (lab_dir / "runs").mkdir(parents=True)
    # Seed
    from rl_research import contract as c

    c.LAB_DIR = lab_dir
    c.RUNS_DIR = lab_dir / "runs"
    c.LEDGER_PATH = lab_dir / "ledger.jsonl"
    c.LEDGER_LOCK_PATH = lab_dir / ".ledger.lock"
    for i in range(1, 5):
        rid = f"{i:04d}-pre"
        p = _write_result_file(lab_dir, rid)
        c.append_to_ledger(p)

    args = []
    for i in range(5, 9):
        args.append((str(lab_dir), "append", f"{i:04d}-new"))
    for i in range(1, 5):
        args.append((str(lab_dir), "update", f"{i:04d}-pre"))

    with ProcessPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(_worker_mixed, a) for a in args]):
            f.result()  # surface any worker exception

    text = (lab_dir / "ledger.jsonl").read_text()
    parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(parsed) == 8

    by_id = {p["run_id"]: p for p in parsed}
    for i in range(1, 5):
        assert by_id[f"{i:04d}-pre"]["verdict_curator"] == "promising"
    for i in range(5, 9):
        assert by_id[f"{i:04d}-new"]["verdict_curator"] is None
