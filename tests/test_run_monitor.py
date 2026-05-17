"""Tests for ``rl_research.run_monitor``.

The pure decision function ``evaluate`` is exercised with synthetic inputs
(no I/O, no subprocesses) so each kill condition can be triggered
deterministically. Then a few end-to-end checks spawn a sleep process and
verify the monitor's kill path.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from rl_research import run_monitor

# --- evaluate() --------------------------------------------------------------


def test_evaluate_returns_continue_in_grace_with_no_events():
    state = run_monitor.MonitorState()
    action, reason = run_monitor.evaluate(
        state,
        now=state.started_at + 60,  # 60s elapsed, grace=300
        newest_event_mtime=None,
        scalars={},
        grace_period_s=300,
        tb_stale_kill_s=1200,
        max_frozen_checks=3,
        max_nan_checks=2,
    )
    assert action == "continue"
    assert reason is None


def test_evaluate_kills_on_stale_tb_writer():
    state = run_monitor.MonitorState()
    now = state.started_at + 1500  # past grace
    action, reason = run_monitor.evaluate(
        state,
        now=now,
        newest_event_mtime=now - 1300,  # 1300s of silence > 1200 threshold
        scalars={"loss": [0.5]},
        grace_period_s=300,
        tb_stale_kill_s=1200,
        max_frozen_checks=3,
        max_nan_checks=2,
    )
    assert action == "kill"
    assert reason.startswith("stalled:")


def test_evaluate_kills_on_no_events_after_grace():
    state = run_monitor.MonitorState()
    action, reason = run_monitor.evaluate(
        state,
        now=state.started_at + 400,  # past grace=300
        newest_event_mtime=None,
        scalars={},
        grace_period_s=300,
        tb_stale_kill_s=1200,
        max_frozen_checks=3,
        max_nan_checks=2,
    )
    assert action == "kill"
    assert "stalled" in reason


def test_evaluate_kills_on_frozen_param_checksum():
    state = run_monitor.MonitorState()
    now = state.started_at + 600
    # Three consecutive identical checksums → kill
    for i in range(3):
        action, reason = run_monitor.evaluate(
            state,
            now=now + i * 60,
            newest_event_mtime=now + i * 60 - 10,  # writer is alive
            scalars={"progress/param_checksum": [42.0]},
            grace_period_s=300,
            tb_stale_kill_s=1200,
            max_frozen_checks=3,
            max_nan_checks=2,
        )
    assert action == "kill"
    assert "param_checksum" in reason
    assert reason.startswith("stalled:")


def test_evaluate_resets_frozen_count_when_checksum_changes():
    state = run_monitor.MonitorState()
    now = state.started_at + 600
    for cs in [1.0, 1.0, 2.0, 2.0, 2.0]:
        # 1.0 first → frozen=1; 1.0 second → frozen=2.
        # 2.0 first → resets frozen=1; 2.0 second → 2; 2.0 third would be 3.
        action, _ = run_monitor.evaluate(
            state,
            now=now,
            newest_event_mtime=now - 5,
            scalars={"progress/param_checksum": [cs]},
            grace_period_s=300,
            tb_stale_kill_s=1200,
            max_frozen_checks=4,  # 4 so the 5-step sequence does NOT kill
            max_nan_checks=2,
        )
        now += 60
        assert action != "kill"
    assert state.frozen_checks == 3
    assert state.last_param_checksum == 2.0


def test_evaluate_kills_on_nan_loss_even_in_grace():
    state = run_monitor.MonitorState()
    now = state.started_at + 30  # IN grace period
    for _ in range(2):
        action, reason = run_monitor.evaluate(
            state,
            now=now,
            newest_event_mtime=now - 1,
            scalars={"loss/policy": [math.nan]},
            grace_period_s=300,
            tb_stale_kill_s=1200,
            max_frozen_checks=3,
            max_nan_checks=2,
        )
    assert action == "kill"
    assert reason.startswith("diverged:")
    assert "NaN" in reason


def test_evaluate_kills_on_inf_loss():
    state = run_monitor.MonitorState()
    now = state.started_at + 600
    for _ in range(2):
        action, reason = run_monitor.evaluate(
            state,
            now=now,
            newest_event_mtime=now - 1,
            scalars={"loss/policy": [math.inf]},
            grace_period_s=300,
            tb_stale_kill_s=1200,
            max_frozen_checks=3,
            max_nan_checks=2,
        )
    assert action == "kill"
    assert reason.startswith("diverged:")


def test_evaluate_resets_nan_when_value_recovers():
    state = run_monitor.MonitorState()
    now = state.started_at + 600
    # NaN once
    run_monitor.evaluate(
        state,
        now=now,
        newest_event_mtime=now - 1,
        scalars={"loss/policy": [math.nan]},
        grace_period_s=300,
        tb_stale_kill_s=1200,
        max_frozen_checks=3,
        max_nan_checks=2,
    )
    assert state.nan_checks == 1
    # Then a finite value — counter resets, no kill
    action, _ = run_monitor.evaluate(
        state,
        now=now + 60,
        newest_event_mtime=now + 50,
        scalars={"loss/policy": [0.5]},
        grace_period_s=300,
        tb_stale_kill_s=1200,
        max_frozen_checks=3,
        max_nan_checks=2,
    )
    assert state.nan_checks == 0
    assert action == "continue"


def test_evaluate_continue_when_param_checksum_progressing_normally():
    """A run that is updating its weights every check must NEVER be killed."""
    state = run_monitor.MonitorState()
    now = state.started_at + 600
    for cs in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        action, _ = run_monitor.evaluate(
            state,
            now=now,
            newest_event_mtime=now - 5,
            scalars={
                "progress/param_checksum": [cs],
                "loss/policy": [0.5],
                "eval/return_mean": [10.0 + cs],
            },
            grace_period_s=300,
            tb_stale_kill_s=1200,
            max_frozen_checks=3,
            max_nan_checks=2,
        )
        now += 60
        assert action == "continue"


# --- _newest_event_mtime + _read_scalars ------------------------------------


def _write_minimal_tfevent(path: Path) -> None:
    """Write a single empty TF events header so SummaryReader sees the file
    even if it can't parse anything inside. Sufficient for the mtime-based
    test (which only cares about file presence + mtime)."""
    # The smallest valid TFRecord is just a 12-byte header for an empty
    # payload; tbparse skips entries it can't decode rather than raising.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        # Minimal: a length-0 record header. tbparse will skip this gracefully.
        f.write(struct.pack("<Q", 0))  # length
        f.write(struct.pack("<I", 0))  # length crc (bogus but tolerated)


def test_newest_event_mtime_returns_none_when_no_events(tmp_path):
    assert run_monitor._newest_event_mtime(tmp_path / "tb") is None


def test_newest_event_mtime_finds_latest(tmp_path):
    logdir = tmp_path / "tb" / "0"
    e1 = logdir / "events.out.tfevents.1.host.0.0"
    e2 = logdir / "events.out.tfevents.2.host.0.0"
    _write_minimal_tfevent(e1)
    _write_minimal_tfevent(e2)
    # Force e2 to be newer
    os.utime(e1, (1000, 1000))
    os.utime(e2, (2000, 2000))
    assert run_monitor._newest_event_mtime(tmp_path / "tb") == 2000.0


# --- run() with a real subprocess -------------------------------------------


@pytest.fixture
def sleeper():
    """Spawn a long-sleeping child in its own process group so the monitor
    can SIGTERM/SIGKILL it without taking down pytest."""
    procs: list[subprocess.Popen] = []

    def _spawn() -> subprocess.Popen:
        # New session = new process group; isolates kills.
        p = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"], start_new_session=True
        )
        procs.append(p)
        return p

    yield _spawn

    for p in procs:
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait(timeout=5)


def test_run_returns_completed_when_pid_exits(tmp_path, sleeper):
    """If the watched process exits cleanly, the monitor returns 'completed'
    and writes no verdict file."""
    proc = sleeper()
    # Schedule the process to exit immediately so the first poll sees it gone.
    proc.terminate()
    proc.wait(timeout=5)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    verdict = run_monitor.run(
        pid=proc.pid,
        logdir=tmp_path / "tb",
        run_dir=run_dir,
        check_interval_s=0.1,
        grace_period_s=0,
        tb_stale_kill_s=10,
        max_frozen_checks=3,
        max_nan_checks=2,
        term_grace_s=1.0,
    )
    assert verdict == "completed"
    assert not (run_dir / ".monitor_verdict").exists()


def test_run_kills_stalled_process(tmp_path, sleeper):
    """No TB events at all + grace=0 + tiny kill threshold → monitor kills
    the sleeper and writes 'stalled' verdict."""
    proc = sleeper()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    verdict = run_monitor.run(
        pid=proc.pid,
        logdir=tmp_path / "tb",
        run_dir=run_dir,
        check_interval_s=0.1,
        grace_period_s=0,
        tb_stale_kill_s=0,  # any silence kills
        max_frozen_checks=3,
        max_nan_checks=2,
        term_grace_s=2.0,
    )
    assert verdict == "stalled"
    # Verdict file written exactly once with single word
    assert (run_dir / ".monitor_verdict").read_text().strip() == "stalled"
    # Kill actually delivered
    assert proc.wait(timeout=5) is not None
    # Monitor log records the start + kill
    log_text = (run_dir / "monitor.log").read_text()
    assert "start" in log_text
    assert "kill" in log_text


def test_run_writes_log_in_jsonl(tmp_path, sleeper):
    proc = sleeper()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_monitor.run(
        pid=proc.pid,
        logdir=tmp_path / "tb",
        run_dir=run_dir,
        check_interval_s=0.1,
        grace_period_s=0,
        tb_stale_kill_s=0,
        max_frozen_checks=3,
        max_nan_checks=2,
        term_grace_s=2.0,
    )
    import json as _json

    lines = (run_dir / "monitor.log").read_text().splitlines()
    assert len(lines) >= 2
    for line in lines:
        _json.loads(line)  # raises if any line isn't JSON


def test_kill_process_handles_already_dead(tmp_path, sleeper):
    """Exiting before the kill is sent must not raise."""
    proc = sleeper()
    proc.terminate()
    proc.wait(timeout=5)
    # Should be a no-op
    run_monitor.kill_process(proc.pid, term_grace_s=0.1)


def test_max_iterations_returns_unknown(tmp_path, sleeper):
    """Internal escape hatch for tests: the monitor exits with verdict
    'unknown' if we cap its iteration count, even with a live PID."""
    proc = sleeper()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Plenty of grace so no kill condition fires.
    verdict = run_monitor.run(
        pid=proc.pid,
        logdir=tmp_path / "tb",
        run_dir=run_dir,
        check_interval_s=0.05,
        grace_period_s=3600,
        tb_stale_kill_s=3600,
        max_frozen_checks=999,
        max_nan_checks=999,
        max_iterations=2,
    )
    assert verdict == "unknown"
    assert not (run_dir / ".monitor_verdict").exists()
