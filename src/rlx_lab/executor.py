"""Bounded, shell-free local process execution."""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_seconds: float
    memory_bytes: int | None = None
    cpu_seconds: int | None = None
    output_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.output_bytes <= 0:
            raise ValueError("output_bytes must be positive")


@dataclass(frozen=True)
class ExecutionSpec:
    argv: tuple[str, ...]
    cwd: Path
    limits: ExecutionLimits
    stdin: bytes = b""
    env: Mapping[str, str] = field(default_factory=dict)
    inherit_env: bool = False
    drop_env_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.argv or any(not isinstance(part, str) or not part for part in self.argv):
            raise ValueError("argv must contain non-empty strings")


@dataclass(frozen=True)
class ExecutionResult:
    argv: tuple[str, ...]
    cwd: Path
    exit_code: int
    timed_out: bool
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float


class LocalExecutor:
    """Runs an argv directly and terminates its process group on timeout."""

    _SAFE_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")

    def __init__(self) -> None:
        self._active: set[subprocess.Popen[bytes]] = set()
        self._active_lock = threading.Lock()

    def run(self, spec: ExecutionSpec) -> ExecutionResult:
        cwd = spec.cwd.resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(cwd)
        environment = self._environment(spec)
        started = time.monotonic()
        process = subprocess.Popen(
            spec.argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
            preexec_fn=self._limit_fn(spec.limits),
        )
        with self._active_lock:
            self._active.add(process)
        timed_out = False
        try:
            try:
                stdout, stderr = process.communicate(
                    input=spec.stdin, timeout=spec.limits.timeout_seconds
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_group(process)
                stdout, stderr = process.communicate()
        finally:
            with self._active_lock:
                self._active.discard(process)
        duration = time.monotonic() - started
        stdout, stdout_truncated = _truncate(stdout, spec.limits.output_bytes)
        stderr, stderr_truncated = _truncate(stderr, spec.limits.output_bytes)
        return ExecutionResult(
            argv=spec.argv,
            cwd=cwd,
            exit_code=int(process.returncode),
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_seconds=duration,
        )

    def _environment(self, spec: ExecutionSpec) -> dict[str, str]:
        if spec.inherit_env:
            environment = dict(os.environ)
        else:
            environment = {key: os.environ[key] for key in self._SAFE_ENV_KEYS if key in os.environ}
            environment.setdefault("PATH", "/usr/bin:/bin")
        environment.update({str(key): str(value) for key, value in spec.env.items()})
        for key in tuple(environment):
            if any(key.startswith(prefix) for prefix in spec.drop_env_prefixes):
                environment.pop(key)
        return environment

    @staticmethod
    def _limit_fn(limits: ExecutionLimits):
        def apply_limits() -> None:
            if limits.memory_bytes is not None:
                resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
            if limits.cpu_seconds is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))

        return apply_limits

    @staticmethod
    def _terminate_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def terminate_active(self) -> int:
        """Signal every active process group; safe to call from a worker handler."""

        with self._active_lock:
            processes = tuple(self._active)
        signalled = 0
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
                signalled += 1
            except ProcessLookupError:
                pass
        return signalled

    def active_count(self) -> int:
        with self._active_lock:
            return sum(process.poll() is None for process in self._active)


def _truncate(data: bytes, limit: int) -> tuple[bytes, bool]:
    if len(data) <= limit:
        return data, False
    marker = b"\n...[truncated by rlx_lab]...\n"
    keep = max(0, limit - len(marker))
    return data[:keep] + marker, True


def macos_deny_read_command(
    argv: tuple[str, ...], unreadable_roots: tuple[Path, ...]
) -> tuple[str, ...]:
    """Wrap an argv in a narrow macOS profile that only adds read denials."""

    if not unreadable_roots:
        return argv
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox_exec.exists():
        raise RuntimeError("provider read isolation requires macOS sandbox-exec")
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny process-info*)",
        "(allow process-info* (target self))",
    ]
    for root in unreadable_roots:
        quoted = json.dumps(str(root.resolve()))
        rules.append(f"(deny file-read* (subpath {quoted}))")
        rules.append(f"(deny file-write* (subpath {quoted}))")
    return (str(sandbox_exec), "-p", " ".join(rules), *argv)
