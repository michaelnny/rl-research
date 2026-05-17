"""Unit tests for ``rl_research.contract``.

The contract is the corpus' load-bearing invariant: every run validates here
or it never enters the ledger. These tests exercise:

  * ``next_run_id`` allocation under empty / populated / mixed-noise dirs
  * ``write_result`` round-trip + atomic write
  * ``validate_result_json`` against every documented invariant
  * ``append_to_ledger`` produces a well-formed line and refuses invalid input

The contract module reaches for ``lab/runs`` and ``lab/ledger.jsonl`` at the
repo root. Tests rebind those module-level paths into ``tmp_path`` via the
``patched_corpus`` fixture so they never touch the real corpus.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rl_research import contract as contract_mod
from rl_research.contract import (
    ContractViolation,
    append_to_ledger,
    next_run_id,
    validate_result_json,
    write_result,
)


@pytest.fixture
def patched_corpus(tmp_path, monkeypatch):
    """Redirect contract module's RUNS_DIR and LEDGER_PATH into tmp_path."""
    runs_dir = tmp_path / "lab" / "runs"
    ledger = tmp_path / "lab" / "ledger.jsonl"
    runs_dir.mkdir(parents=True)
    monkeypatch.setattr(contract_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(contract_mod, "LEDGER_PATH", ledger)
    return runs_dir, ledger


def _good_payload(run_id: str = "0001-foo", seeds=(42,)) -> dict:
    """A minimal payload that passes validate_result_json — tests mutate it."""
    by_seed = {str(s): {"best_return": 1.0, "final_return": 0.5, "env_steps": 100} for s in seeds}
    return {
        "run_id": run_id,
        "stage": "A+B",
        "status": "completed",
        "primary_benchmark": "CartPole-v1",
        "pillar": "sparse-long-horizon",
        "thread": "foo",
        "seeds": list(seeds),
        "env_steps": 100,
        "wallclock_s": 12.5,
        "best_return": 1.0,
        "final_return": 0.5,
        "by_seed": by_seed,
        "sanity": {
            "passed": True,
            "by_env": {
                "CartPole-v1": {"return_random": 22.0, "return_final": 99.0, "passed": True}
            },
            "retries": 0,
        },
        "git_sha": "abcdef1",
        "started_at": "2026-05-16T11:00:00Z",
        "ended_at": "2026-05-16T12:00:00Z",
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "result.json"
    p.write_text(json.dumps(payload))
    return p


# -- next_run_id ------------------------------------------------------------


def test_next_run_id_empty_dir_starts_at_0001(patched_corpus):
    assert next_run_id("foo-bar") == "0001-foo-bar"


def test_next_run_id_increments_past_highest(patched_corpus):
    runs_dir, _ = patched_corpus
    (runs_dir / "0001-alpha").mkdir()
    (runs_dir / "0007-beta").mkdir()
    (runs_dir / "0003-gamma").mkdir()
    assert next_run_id("delta") == "0008-delta"


def test_next_run_id_ignores_non_conforming_dirs(patched_corpus):
    runs_dir, _ = patched_corpus
    (runs_dir / "0002-ok").mkdir()
    (runs_dir / "garbage").mkdir()
    (runs_dir / "0099_underscore").mkdir()  # not kebab-case
    (runs_dir / "01-too-short").mkdir()
    assert next_run_id("ok") == "0003-ok"


def test_next_run_id_ignores_files(patched_corpus):
    runs_dir, _ = patched_corpus
    (runs_dir / "0050-stray.txt").write_text("noise")
    assert next_run_id("ok") == "0001-ok"


def test_next_run_id_rejects_bad_slug(patched_corpus):
    with pytest.raises(ValueError, match="thread_slug"):
        next_run_id("Has_Caps")
    with pytest.raises(ValueError, match="thread_slug"):
        next_run_id("with spaces")
    with pytest.raises(ValueError, match="thread_slug"):
        next_run_id("")


# -- write_result -----------------------------------------------------------


def test_write_result_round_trips(patched_corpus):
    runs_dir, _ = patched_corpus
    (runs_dir / "0001-foo").mkdir()
    p = write_result(
        run_id="0001-foo",
        stage="A+B",
        status="completed",
        primary_benchmark="CartPole-v1",
        pillar="sparse-long-horizon",
        thread="foo",
        seeds=[42],
        env_steps=100,
        wallclock_s=12.5,
        best_return=1.0,
        final_return=0.5,
        by_seed={"42": {"best_return": 1.0, "final_return": 0.5, "env_steps": 100}},
        sanity={
            "passed": True,
            "by_env": {
                "CartPole-v1": {"return_random": 22.0, "return_final": 99.0, "passed": True}
            },
            "retries": 0,
        },
        git_sha="abcdef1234",
        started_at=datetime(2026, 5, 16, 11, 0, tzinfo=UTC),
        ended_at=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
    )
    assert p == runs_dir / "0001-foo" / "result.json"
    data = json.loads(p.read_text())
    assert data["run_id"] == "0001-foo"
    assert data["seeds"] == [42]
    assert data["started_at"].endswith("Z")


def test_write_result_validates_after_write(patched_corpus):
    runs_dir, _ = patched_corpus
    (runs_dir / "0001-foo").mkdir()
    # Mismatched seeds vs by_seed should be caught by post-write validation.
    with pytest.raises(ContractViolation, match="by_seed"):
        write_result(
            run_id="0001-foo",
            stage="A+B",
            status="completed",
            primary_benchmark="CartPole-v1",
            pillar="sparse-long-horizon",
            thread="foo",
            seeds=[42, 43],
            env_steps=0,
            wallclock_s=0.0,
            best_return=0.0,
            final_return=0.0,
            by_seed={"42": {"best_return": 0.0, "final_return": 0.0, "env_steps": 0}},
            sanity={"passed": True, "by_env": {}, "retries": 0},
            git_sha="abcdef1",
            started_at=datetime(2026, 5, 16, tzinfo=UTC),
            ended_at=datetime(2026, 5, 16, tzinfo=UTC),
        )


def test_write_result_handles_naive_datetime(patched_corpus):
    runs_dir, _ = patched_corpus
    (runs_dir / "0001-foo").mkdir()
    p = write_result(
        run_id="0001-foo",
        stage="A+B",
        status="completed",
        primary_benchmark="CartPole-v1",
        pillar="sparse-long-horizon",
        thread="foo",
        seeds=[42],
        env_steps=100,
        wallclock_s=12.5,
        best_return=1.0,
        final_return=0.5,
        by_seed={"42": {"best_return": 1.0, "final_return": 0.5, "env_steps": 100}},
        sanity={
            "passed": True,
            "by_env": {
                "CartPole-v1": {"return_random": 22.0, "return_final": 99.0, "passed": True}
            },
            "retries": 0,
        },
        git_sha="abcdef1",
        started_at=datetime(2026, 5, 16, 11, 0),  # naive
        ended_at=datetime(2026, 5, 16, 12, 0),  # naive
    )
    data = json.loads(p.read_text())
    assert data["started_at"].endswith("Z")
    assert "+00:00" not in data["started_at"]


# -- validate_result_json: success path -------------------------------------


def test_validate_accepts_good_payload(tmp_path):
    p = _write(tmp_path, _good_payload())
    validate_result_json(p)  # no raise


def test_validate_accepts_vector_returns(tmp_path):
    payload = _good_payload()
    payload["best_return"] = [1.0, 2.0, 3.0]
    payload["final_return"] = [0.5, 1.0, 1.5]
    payload["pillar"] = "multi-signal"
    payload["primary_benchmark"] = "minecart-v0"
    p = _write(tmp_path, payload)
    validate_result_json(p)


def test_validate_accepts_a_only_stage(tmp_path):
    payload = _good_payload()
    payload["stage"] = "A-only"
    payload["status"] = "sanity-failed"
    p = _write(tmp_path, payload)
    validate_result_json(p)


# -- validate_result_json: each failure mode --------------------------------


@pytest.mark.parametrize(
    "drop_key",
    [
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
    ],
)
def test_validate_rejects_missing_key(tmp_path, drop_key):
    payload = _good_payload()
    payload.pop(drop_key)
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="missing required keys"):
        validate_result_json(p)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("run_id", "1-foo", "run_id"),
        ("run_id", "0001-Foo", "run_id"),  # capital
        ("run_id", "0001foo", "run_id"),  # missing dash
        ("stage", "C", "stage"),
        ("status", "wat", "status"),
        ("pillar", "made-up", "pillar"),
        ("thread", "Has_Caps", "thread"),
        ("git_sha", "xyz", "git_sha"),
        ("git_sha", "ZZ", "git_sha"),
    ],
)
def test_validate_rejects_bad_enum_or_pattern(tmp_path, field, bad_value, match):
    payload = _good_payload()
    payload[field] = bad_value
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match=match):
        validate_result_json(p)


def test_validate_rejects_empty_seeds(tmp_path):
    payload = _good_payload()
    payload["seeds"] = []
    payload["by_seed"] = {}
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="seeds"):
        validate_result_json(p)


def test_validate_rejects_non_int_seeds(tmp_path):
    payload = _good_payload()
    payload["seeds"] = [42, "43"]
    payload["by_seed"] = {
        "42": {"best_return": 0.0, "final_return": 0.0, "env_steps": 0},
        "43": {"best_return": 0.0, "final_return": 0.0, "env_steps": 0},
    }
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="seeds"):
        validate_result_json(p)


def test_validate_rejects_negative_env_steps(tmp_path):
    payload = _good_payload()
    payload["env_steps"] = -1
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="env_steps"):
        validate_result_json(p)


def test_validate_rejects_negative_wallclock(tmp_path):
    payload = _good_payload()
    payload["wallclock_s"] = -0.1
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="wallclock_s"):
        validate_result_json(p)


def test_validate_rejects_string_return(tmp_path):
    payload = _good_payload()
    payload["best_return"] = "high"
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="best_return"):
        validate_result_json(p)


def test_validate_rejects_mixed_list_return(tmp_path):
    payload = _good_payload()
    payload["final_return"] = [1.0, "two", 3.0]
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="final_return"):
        validate_result_json(p)


def test_validate_rejects_sanity_missing_keys(tmp_path):
    payload = _good_payload()
    payload["sanity"] = {"passed": True}  # missing by_env + retries
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="sanity"):
        validate_result_json(p)


def test_validate_rejects_sanity_passed_not_bool(tmp_path):
    payload = _good_payload()
    payload["sanity"]["passed"] = "yes"
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match=r"sanity\.passed"):
        validate_result_json(p)


def test_validate_rejects_sanity_by_env_missing_keys(tmp_path):
    payload = _good_payload()
    payload["sanity"]["by_env"] = {"CartPole-v1": {"passed": True}}
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="by_env"):
        validate_result_json(p)


def test_validate_rejects_bad_iso8601(tmp_path):
    payload = _good_payload()
    payload["started_at"] = "yesterday"
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="started_at"):
        validate_result_json(p)


def test_validate_rejects_seeds_not_matching_by_seed(tmp_path):
    payload = _good_payload(seeds=(42, 43))
    payload["by_seed"] = {"42": {"best_return": 0.0, "final_return": 0.0, "env_steps": 0}}
    p = _write(tmp_path, payload)
    with pytest.raises(ContractViolation, match="by_seed"):
        validate_result_json(p)


# -- append_to_ledger -------------------------------------------------------


def test_append_to_ledger_writes_one_line(patched_corpus):
    _, ledger = patched_corpus
    p = _write(ledger.parent, _good_payload())
    append_to_ledger(p)
    assert ledger.exists()
    lines = ledger.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["run_id"] == "0001-foo"
    assert rec["thread"] == "foo"
    assert rec["pillar"] == "sparse-long-horizon"
    assert rec["primary_benchmark"] == "CartPole-v1"
    assert rec["status"] == "completed"
    assert rec["best_return"] == 1.0
    assert rec["wallclock_s"] == 12.5
    assert rec["verdict_curator"] is None
    assert rec["hypothesis_path"] == "lab/runs/0001-foo/hypothesis.md"


def test_append_to_ledger_appends_multiple(patched_corpus):
    _, ledger = patched_corpus
    p1 = ledger.parent / "r1.json"
    p2 = ledger.parent / "r2.json"
    p1.write_text(json.dumps(_good_payload(run_id="0001-foo")))
    p2.write_text(json.dumps(_good_payload(run_id="0002-foo")))
    append_to_ledger(p1)
    append_to_ledger(p2)
    lines = ledger.read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["0001-foo", "0002-foo"]


def test_append_to_ledger_refuses_invalid(patched_corpus):
    _, ledger = patched_corpus
    bad = _good_payload()
    bad["pillar"] = "made-up"
    p = ledger.parent / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ContractViolation):
        append_to_ledger(p)
    assert not ledger.exists()  # nothing written
