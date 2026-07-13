from __future__ import annotations

import sys
import threading
import time

from rlx_lab.executor import ExecutionLimits, ExecutionSpec, LocalExecutor


def test_executor_captures_output_and_exit_code(tmp_path):
    result = LocalExecutor().run(
        ExecutionSpec(
            argv=(sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(3)"),
            cwd=tmp_path,
            limits=ExecutionLimits(timeout_seconds=5),
        )
    )
    assert result.exit_code == 3
    assert result.stdout == b"out\n"
    assert result.stderr == b"err\n"
    assert not result.timed_out


def test_executor_terminates_process_group_on_timeout(tmp_path):
    result = LocalExecutor().run(
        ExecutionSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            limits=ExecutionLimits(timeout_seconds=0.1),
        )
    )
    assert result.timed_out
    assert result.exit_code != 0
    assert result.duration_seconds < 5


def test_executor_truncates_large_outputs(tmp_path):
    result = LocalExecutor().run(
        ExecutionSpec(
            argv=(sys.executable, "-c", "print('x' * 10000)"),
            cwd=tmp_path,
            limits=ExecutionLimits(timeout_seconds=5, output_bytes=100),
        )
    )
    assert len(result.stdout) == 100
    assert result.stdout_truncated


def test_executor_can_inherit_auth_but_drop_all_rlx_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-auth")
    monkeypatch.setenv("RLX_FACTORLAB_SUITE_KEY", "evaluator-secret")
    result = LocalExecutor().run(
        ExecutionSpec(
            argv=(
                sys.executable,
                "-c",
                "import os; print(os.getenv('OPENAI_API_KEY')); "
                "print(os.getenv('RLX_FACTORLAB_SUITE_KEY'))",
            ),
            cwd=tmp_path,
            limits=ExecutionLimits(timeout_seconds=5),
            inherit_env=True,
            drop_env_prefixes=("RLX_",),
        )
    )
    assert result.stdout == b"provider-auth\nNone\n"


def test_executor_external_cancellation_terminates_active_process_group(tmp_path):
    executor = LocalExecutor()
    results = []

    def run() -> None:
        results.append(
            executor.run(
                ExecutionSpec(
                    argv=(sys.executable, "-c", "import time; time.sleep(30)"),
                    cwd=tmp_path,
                    limits=ExecutionLimits(timeout_seconds=60),
                )
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2.0
    while executor.active_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert executor.terminate_active() == 1
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert results[0].exit_code != 0
    assert results[0].timed_out is False
