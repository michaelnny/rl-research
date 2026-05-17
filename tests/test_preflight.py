"""Smoke tests for ``scripts/preflight.sh``.

These verify that the gate refuses to run an iteration when the corpus is in
a known-bad state. We don't exercise every code path (GPU / disk / orphan
process detection require root or hardware that the test env may not have);
we focus on the bits that are pure shell + filesystem.

The script reads ``lab/HALT_REQUESTED.md`` and ``lab/ledger.jsonl`` relative
to ``cd "$(dirname "$0")/.."``, so we copy the script into a tmp working tree
and invoke it from there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT = REPO_ROOT / "scripts" / "preflight.sh"


@pytest.fixture
def workspace(tmp_path):
    """Build a minimal repo skeleton in tmp_path so preflight.sh has the
    layout it expects to read (lab/, scripts/, lab/baselines/random.json)."""
    (tmp_path / "lab" / "runs").mkdir(parents=True)
    (tmp_path / "lab" / "threads").mkdir()
    (tmp_path / "lab" / "baselines").mkdir()
    (tmp_path / "lab" / "baselines" / "random.json").write_text("{}")
    (tmp_path / "scripts").mkdir()
    shutil.copy(PREFLIGHT, tmp_path / "scripts" / "preflight.sh")
    (tmp_path / "scripts" / "preflight.sh").chmod(0o755)
    return tmp_path


def _run(workspace, env_overrides=None):
    env = os.environ.copy()
    # Skip GPU + disk checks by default in tests — they depend on host state.
    env.setdefault("PREFLIGHT_SKIP_GPU", "1")
    env.setdefault("PREFLIGHT_MIN_GB", "0")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", "scripts/preflight.sh"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
    )


def test_preflight_passes_on_clean_workspace(workspace):
    result = _run(workspace)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "preflight: OK" in result.stdout


def test_preflight_halts_on_halt_file(workspace):
    (workspace / "lab" / "HALT_REQUESTED.md").write_text("manual halt\n")
    result = _run(workspace)
    assert result.returncode == 2
    assert "HALT" in result.stderr


def test_preflight_fails_on_corrupt_ledger(workspace):
    (workspace / "lab" / "ledger.jsonl").write_text(
        json.dumps({"run_id": "0001-x"}) + "\nnot-valid-json\n"
    )
    result = _run(workspace)
    assert result.returncode == 1
    assert "malformed" in result.stderr


def test_preflight_passes_with_valid_ledger(workspace):
    (workspace / "lab" / "ledger.jsonl").write_text(
        json.dumps({"run_id": "0001-x"}) + "\n" + json.dumps({"run_id": "0002-x"}) + "\n"
    )
    result = _run(workspace)
    assert result.returncode == 0


def test_preflight_fails_when_baselines_missing(workspace):
    (workspace / "lab" / "baselines" / "random.json").unlink()
    result = _run(workspace, env_overrides={"PREFLIGHT_SKIP_BASELINES": "0"})
    assert result.returncode == 1
    assert "random.json missing" in result.stderr


def test_preflight_fails_on_low_disk(workspace):
    # Force a min that the actual partition almost certainly can't satisfy.
    result = _run(workspace, env_overrides={"PREFLIGHT_MIN_GB": "999999999"})
    assert result.returncode == 1
    assert "free on lab/ partition" in result.stderr


def test_preflight_fails_on_low_memory(workspace):
    # Force a memory floor that nothing can satisfy.
    result = _run(workspace, env_overrides={"PREFLIGHT_MIN_MEM_MB": "999999999"})
    assert result.returncode == 1
    assert "MB RAM available" in result.stderr


def test_preflight_fails_on_empty_baseline(workspace):
    """A 0-byte random.json passes a naive existence test but breaks every
    Stage A run; preflight must catch it."""
    (workspace / "lab" / "baselines" / "random.json").write_text("")
    result = _run(workspace)
    assert result.returncode == 1
    assert "empty" in result.stderr


def test_preflight_fails_on_invalid_json_baseline(workspace):
    """A non-JSON random.json fails the same as missing — Engineer would
    crash trying to read it."""
    (workspace / "lab" / "baselines" / "random.json").write_text("not json {")
    result = _run(workspace)
    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


def test_preflight_skip_baselines_bypasses(workspace):
    """SKIP_BASELINES=1 lets preflight pass on a partial workspace — useful
    early in setup before random.json exists."""
    (workspace / "lab" / "baselines" / "random.json").unlink()
    result = _run(workspace, env_overrides={"PREFLIGHT_SKIP_BASELINES": "1"})
    assert result.returncode == 0


def test_preflight_does_not_use_uv(workspace):
    """Sanity: the hardened preflight must validate the ledger without
    invoking uv (which adds ~1s per call and is fragile to lockfile drift).
    We assert by giving the script a PATH that lacks `uv` but has python3."""
    env = os.environ.copy()
    # Strip uv from PATH while keeping system python3.
    paths = [p for p in env.get("PATH", "").split(":") if p]
    keep = []
    for p in paths:
        if not (Path(p) / "uv").exists():
            keep.append(p)
    env["PATH"] = ":".join(keep)
    env.setdefault("PREFLIGHT_SKIP_GPU", "1")
    env.setdefault("PREFLIGHT_SKIP_AUTH", "1")
    env.setdefault("PREFLIGHT_MIN_GB", "0")

    (workspace / "lab" / "ledger.jsonl").write_text(json.dumps({"run_id": "0001-x"}) + "\n")
    result = subprocess.run(
        ["bash", "scripts/preflight.sh"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
