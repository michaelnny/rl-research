"""External sidecar that watches a Stage B ``train.py`` and kills it if the
run has clearly stalled or diverged — without overhauling slow-but-real
progress.

Threat model (the gaps left open by the in-process ``WallclockBudget`` and the
shell-level ``timeout`` cap):

* The 2h hard timeout fires on every run regardless of state, so a run that
  hung at minute 5 still burns 1h55m of wallclock.
* ``WallclockBudget`` only checks at training-loop boundaries; if the loop
  itself has deadlocked (dataloader, env reset, NCCL, GPU OOM that left the
  process alive), the budget never fires.
* A run that is silently NaN-ing for 30 minutes is wasting the same budget
  as a healthy one and will surface as ``best_return=NaN`` only at the end.

The monitor watches three signals via the run's TensorBoard event files:

1. **TB writer mtime.** If no event has been written for
   ``--tb-stale-kill-s`` (default 1200s = 20 min), the process is frozen.
   Send SIGTERM, wait, then SIGKILL. Verdict: ``stalled``.
2. **``progress/param_checksum``.** A scalar that the framework writes every
   eval pass. If it does not change across ``--max-frozen-checks`` (default 3)
   consecutive checks, the network parameters are not being updated even if
   the writer is alive. Verdict: ``stalled``.
3. **NaN in any logged loss/return.** If any scalar value is NaN/Inf across
   ``--max-nan-checks`` (default 2) consecutive checks, the run has diverged
   beyond recovery. Verdict: ``diverged``.

The first ``--grace-period-s`` (default 300s = 5min) seconds after launch are
exempt from the stall-kill rules — train.py needs time to compile, build the
vec env, and emit its first event.

Output:

* ``lab/runs/<run_id>/monitor.log`` — JSON-line per check with timestamps and
  signal values, for postmortem.
* ``lab/runs/<run_id>/.monitor_verdict`` — single-word file written exactly
  once when the monitor decides to kill. The Engineer reads this file after
  ``train.py`` exits to disambiguate the kill cause and pick the correct
  ``status`` for ``result.json`` (``killed-stalled`` vs ``killed-diverged``).

The monitor itself never writes ``result.json`` or touches the ledger — the
Engineer owns those. It also exits cleanly if the watched PID disappears
(train.py finished on its own), without writing a verdict file.

CLI:

    uv run python -m rl_research.run_monitor \\
        --pid <train_pid> \\
        --logdir lab/runs/<run_id>/tb \\
        --run-dir lab/runs/<run_id>

Designed to be launched by the Engineer as a background subprocess
alongside ``train.py`` and reaped after train exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Tunable defaults. Chosen to match the user-facing requirement: every ~20 min
# we make a kill/no-kill decision, and a slow-but-real run is never killed
# before its 2h budget. A run is allowed to be silent for 20 min once but not
# hang indefinitely.
# ---------------------------------------------------------------------------

DEFAULT_CHECK_INTERVAL_S = 60
DEFAULT_GRACE_PERIOD_S = 300
DEFAULT_TB_STALE_WARN_S = 600
DEFAULT_TB_STALE_KILL_S = 1200
DEFAULT_MAX_FROZEN_CHECKS = 3
DEFAULT_MAX_NAN_CHECKS = 2

PARAM_CHECKSUM_TAG = "progress/param_checksum"


@dataclass
class MonitorState:
    """All decision-relevant state. Kept out of globals so unit tests can
    construct one and feed it deterministic inputs without spawning a real
    subprocess."""

    last_param_checksum: float | None = None
    frozen_checks: int = 0
    nan_checks: int = 0
    started_at: float = field(default_factory=time.time)
    kill_sent: bool = False
    verdict: str | None = None  # "stalled" | "diverged" — None until we kill


def _newest_event_mtime(logdir: Path) -> float | None:
    """Latest mtime across all TB event files under ``logdir``. Returns None
    if no event file exists yet (e.g., train.py is still spinning up)."""
    if not logdir.exists():
        return None
    best: float | None = None
    for p in logdir.rglob("events.out.tfevents.*"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if best is None or m > best:
            best = m
    return best


def _read_scalars(logdir: Path) -> dict[str, list[float]]:
    """Parse TB events under ``logdir`` and return {tag: [values...]}.

    Uses tbparse, which is permissive about partially-written event files
    (it skips truncated trailing records). Returns an empty dict on any
    parse error — we treat parse failure as "no information" rather than
    a kill signal, since the file may simply be mid-write.
    """
    try:
        from tbparse import SummaryReader
    except ImportError:
        return {}
    if not logdir.exists():
        return {}
    try:
        reader = SummaryReader(str(logdir), pivot=False, extra_columns=set())
        df = reader.scalars
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    out: dict[str, list[float]] = {}
    for tag, sub in df.groupby("tag"):
        out[str(tag)] = [float(v) for v in sub["value"].tolist()]
    return out


def _has_nan(scalars: dict[str, list[float]]) -> str | None:
    """Return the first tag containing NaN/Inf, or None."""
    for tag, vals in scalars.items():
        for v in vals[-50:]:  # only check recent values; older NaNs are stale
            if math.isnan(v) or math.isinf(v):
                return tag
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal — still alive
    return True


def _send_signal(pid: int, sig: int) -> None:
    try:
        # Best-effort: kill the whole process group so dataloaders and
        # subprocess workers die with the parent. Falls back to bare PID
        # if no process group exists.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def kill_process(pid: int, *, term_grace_s: float = 30.0) -> None:
    """Send SIGTERM, wait up to ``term_grace_s``, then SIGKILL if still alive."""
    if not _pid_alive(pid):
        return
    _send_signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + term_grace_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    _send_signal(pid, signal.SIGKILL)


def evaluate(
    state: MonitorState,
    *,
    now: float,
    newest_event_mtime: float | None,
    scalars: dict[str, list[float]],
    grace_period_s: float,
    tb_stale_kill_s: float,
    max_frozen_checks: int,
    max_nan_checks: int,
) -> tuple[str, str | None]:
    """Pure decision function. Returns (action, verdict_or_reason).

    action ∈ {"continue", "warn", "kill"}.
    If action=="kill", the second element is the verdict ("stalled" or
    "diverged") plus a colon-prefixed reason; if action=="warn", it's the
    reason; if action=="continue", it's None.
    """
    elapsed = now - state.started_at
    in_grace = elapsed < grace_period_s

    # 1. NaN — kill regardless of grace. A model that emits NaN in the first
    #    5 minutes is not going to recover.
    nan_tag = _has_nan(scalars)
    if nan_tag is not None:
        state.nan_checks += 1
        if state.nan_checks >= max_nan_checks:
            return "kill", f"diverged:NaN in {nan_tag} for {state.nan_checks} checks"
        return "warn", f"NaN in {nan_tag} (check {state.nan_checks}/{max_nan_checks})"
    else:
        state.nan_checks = 0

    if in_grace:
        return "continue", None

    # 2. TB writer stalled. Treat "no event file at all after grace" the same
    #    as a stall — a healthy train.py emits its first event within the
    #    first 60s (RunLogger.add_scalar at startup).
    if newest_event_mtime is None:
        return "kill", f"stalled:no TB events after {elapsed:.0f}s grace"
    age = now - newest_event_mtime
    if age >= tb_stale_kill_s:
        return "kill", f"stalled:TB writer silent for {age:.0f}s (>{tb_stale_kill_s:.0f})"

    # 3. param_checksum frozen. If the tag isn't logged at all we can't infer
    #    anything from it — bail out of this check, don't kill.
    checksums = scalars.get(PARAM_CHECKSUM_TAG, [])
    if checksums:
        latest = checksums[-1]
        if state.last_param_checksum is None or latest != state.last_param_checksum:
            state.last_param_checksum = latest
            state.frozen_checks = 1
        else:
            state.frozen_checks += 1
            if state.frozen_checks >= max_frozen_checks:
                return (
                    "kill",
                    f"stalled:param_checksum frozen at {latest} for {state.frozen_checks} checks",
                )

    return "continue", None


def _log(monitor_log: Path, record: dict) -> None:
    monitor_log.parent.mkdir(parents=True, exist_ok=True)
    with monitor_log.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _write_verdict(run_dir: Path, verdict: str) -> None:
    """Single-word verdict file. The Engineer reads this after train.py exits
    to pick killed-stalled vs killed-diverged for result.json."""
    p = run_dir / ".monitor_verdict"
    tmp = p.with_suffix(".verdict.tmp")
    tmp.write_text(verdict + "\n")
    tmp.replace(p)


def run(
    *,
    pid: int,
    logdir: Path,
    run_dir: Path,
    check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
    grace_period_s: float = DEFAULT_GRACE_PERIOD_S,
    tb_stale_warn_s: float = DEFAULT_TB_STALE_WARN_S,
    tb_stale_kill_s: float = DEFAULT_TB_STALE_KILL_S,
    max_frozen_checks: int = DEFAULT_MAX_FROZEN_CHECKS,
    max_nan_checks: int = DEFAULT_MAX_NAN_CHECKS,
    term_grace_s: float = 30.0,
    max_iterations: int | None = None,
) -> str:
    """Main monitor loop. Returns the verdict string, or 'completed' if the
    watched PID exited on its own, or 'unknown' if max_iterations was hit
    (test-only)."""
    state = MonitorState()
    monitor_log = run_dir / "monitor.log"
    iters = 0
    _log(
        monitor_log,
        {
            "ts": time.time(),
            "event": "start",
            "pid": pid,
            "logdir": str(logdir),
            "thresholds": {
                "check_interval_s": check_interval_s,
                "grace_period_s": grace_period_s,
                "tb_stale_warn_s": tb_stale_warn_s,
                "tb_stale_kill_s": tb_stale_kill_s,
                "max_frozen_checks": max_frozen_checks,
                "max_nan_checks": max_nan_checks,
            },
        },
    )
    while True:
        if max_iterations is not None and iters >= max_iterations:
            _log(monitor_log, {"ts": time.time(), "event": "max_iterations"})
            return "unknown"
        iters += 1
        if not _pid_alive(pid):
            _log(monitor_log, {"ts": time.time(), "event": "pid_gone"})
            return "completed"
        time.sleep(check_interval_s)
        if not _pid_alive(pid):
            _log(monitor_log, {"ts": time.time(), "event": "pid_gone"})
            return "completed"

        now = time.time()
        newest = _newest_event_mtime(logdir)
        scalars = _read_scalars(logdir)
        wall_age = (now - newest) if newest else None
        action, reason = evaluate(
            state,
            now=now,
            newest_event_mtime=newest,
            scalars=scalars,
            grace_period_s=grace_period_s,
            tb_stale_kill_s=tb_stale_kill_s,
            max_frozen_checks=max_frozen_checks,
            max_nan_checks=max_nan_checks,
        )
        _log(
            monitor_log,
            {
                "ts": time.time(),
                "event": "check",
                "action": action,
                "reason": reason,
                "tb_age_s": wall_age,
                "frozen_checks": state.frozen_checks,
                "nan_checks": state.nan_checks,
                "last_param_checksum": state.last_param_checksum,
            },
        )
        if action == "kill":
            verdict = (reason or "").split(":", 1)[0] or "stalled"
            state.verdict = verdict
            state.kill_sent = True
            _write_verdict(run_dir, verdict)
            _log(monitor_log, {"ts": time.time(), "event": "kill", "verdict": verdict})
            kill_process(pid, term_grace_s=term_grace_s)
            return verdict


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rl_research.run_monitor")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--check-interval-s", type=float, default=DEFAULT_CHECK_INTERVAL_S)
    p.add_argument("--grace-period-s", type=float, default=DEFAULT_GRACE_PERIOD_S)
    p.add_argument("--tb-stale-warn-s", type=float, default=DEFAULT_TB_STALE_WARN_S)
    p.add_argument("--tb-stale-kill-s", type=float, default=DEFAULT_TB_STALE_KILL_S)
    p.add_argument("--max-frozen-checks", type=int, default=DEFAULT_MAX_FROZEN_CHECKS)
    p.add_argument("--max-nan-checks", type=int, default=DEFAULT_MAX_NAN_CHECKS)
    p.add_argument("--term-grace-s", type=float, default=30.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    verdict = run(
        pid=args.pid,
        logdir=args.logdir,
        run_dir=args.run_dir,
        check_interval_s=args.check_interval_s,
        grace_period_s=args.grace_period_s,
        tb_stale_warn_s=args.tb_stale_warn_s,
        tb_stale_kill_s=args.tb_stale_kill_s,
        max_frozen_checks=args.max_frozen_checks,
        max_nan_checks=args.max_nan_checks,
        term_grace_s=args.term_grace_s,
    )
    print(verdict)
    return 0 if verdict in {"completed", "unknown"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
