"""Tests for ``scripts.corpus_stats``.

Exercises the renderer on a synthetic ledger to verify:
  * Empty-ledger handling (writes a valid empty-stats file).
  * Status / pillar / verdict histograms reflect the ledger.
  * Mode-collapse warning fires when one thread dominates the recent window.
  * Bad lines in the ledger are skipped, not raised on (so the dashboard
    still renders when the ledger is partially corrupt).

We import the script as a module via importlib because ``scripts/`` is not a
package. The script's module-level paths (``LEDGER``, ``OUT``, etc.) are
monkeypatched into ``tmp_path`` so the test never touches the real corpus.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "corpus_stats.py"


@pytest.fixture
def stats_module(tmp_path, monkeypatch):
    """Load scripts/corpus_stats.py as a module with its paths redirected."""
    spec = importlib.util.spec_from_file_location("corpus_stats_test", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["corpus_stats_test"] = mod
    spec.loader.exec_module(mod)

    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "threads").mkdir()
    monkeypatch.setattr(mod, "LAB", lab)
    monkeypatch.setattr(mod, "LEDGER", lab / "ledger.jsonl")
    monkeypatch.setattr(mod, "THREADS_DIR", lab / "threads")
    monkeypatch.setattr(mod, "OUT", lab / "CORPUS_STATS.md")
    yield mod
    sys.modules.pop("corpus_stats_test", None)


def _write_ledger(mod, entries):
    mod.LEDGER.write_text("".join(json.dumps(e) + "\n" for e in entries))


def _entry(
    run_id,
    thread="foo",
    status="completed",
    pillar="sparse-long-horizon",
    verdict=None,
    wallclock=10.0,
):
    return {
        "run_id": run_id,
        "thread": thread,
        "status": status,
        "pillar": pillar,
        "primary_benchmark": "CartPole-v1",
        "best_return": 1.0,
        "wallclock_s": wallclock,
        "verdict_curator": verdict,
        "hypothesis_path": f"lab/runs/{run_id}/hypothesis.md",
    }


def test_render_empty_ledger(stats_module):
    text = stats_module.render()
    assert "Ledger is empty" in text
    assert "no runs yet" in text


def test_render_with_entries(stats_module):
    _write_ledger(
        stats_module,
        [
            _entry("0001-foo", thread="alpha", status="completed", verdict="promising"),
            _entry("0002-bar", thread="beta", status="sanity-failed", verdict="dead-end"),
            _entry("0003-baz", thread="alpha", status="completed", verdict=None),
        ],
    )
    text = stats_module.render()
    assert "3 runs total" in text
    assert "completed" in text
    assert "sanity-failed" in text
    assert "alpha" in text
    assert "beta" in text
    assert "uncurated" in text  # null verdict gets bucketed as uncurated


def test_mode_collapse_warning_fires(stats_module):
    entries = [_entry(f"{i:04d}-foo", thread="dominant") for i in range(1, 16)]
    entries += [_entry("0016-other", thread="other") for _ in range(5)]
    _write_ledger(stats_module, entries)
    text = stats_module.render()
    assert "MODE-COLLAPSE WARNING" in text
    assert "dominant" in text


def test_no_mode_collapse_when_diversified(stats_module):
    threads = ["alpha", "beta", "gamma", "delta"]
    entries = [_entry(f"{i:04d}-x", thread=threads[i % 4]) for i in range(1, 21)]
    _write_ledger(stats_module, entries)
    text = stats_module.render()
    assert "MODE-COLLAPSE WARNING" not in text


def test_malformed_lines_skipped(stats_module):
    """A truncated last line shouldn't break the renderer — preflight catches
    it separately and refuses to start a new iteration, but stats must still
    render so the operator can read what's there."""
    stats_module.LEDGER.write_text(
        json.dumps(_entry("0001-foo")) + "\n"
        "this is not json\n" + json.dumps(_entry("0002-bar")) + "\n"
    )
    text = stats_module.render()
    assert "2 runs total" in text


def test_thread_status_picked_up(stats_module):
    _write_ledger(stats_module, [_entry("0001-foo", thread="alpha")])
    (stats_module.THREADS_DIR / "alpha.md").write_text("---\nstatus: paused\n---\n# Thread\n")
    text = stats_module.render()
    assert "thread_status=paused" in text


def test_recent_failures_grouped(stats_module):
    _write_ledger(
        stats_module,
        [
            _entry("0001-x", status="sanity-failed"),
            _entry("0002-x", status="killed-error"),
            _entry("0003-x", status="completed"),
            _entry("0004-x", status="sanity-failed"),
        ],
    )
    text = stats_module.render()
    assert "Recent failures" in text
    assert "sanity-failed" in text
    assert "killed-error" in text


def test_main_writes_atomically(stats_module, tmp_path):
    _write_ledger(stats_module, [_entry("0001-foo")])
    stats_module.main()
    assert stats_module.OUT.exists()
    assert not stats_module.OUT.with_suffix(".md.tmp").exists()
    text = stats_module.OUT.read_text()
    assert "1 runs total" in text
