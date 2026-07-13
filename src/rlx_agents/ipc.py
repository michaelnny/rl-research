"""Process-isolated JSON Lines protocol for candidate RL algorithms."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CandidateProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateProcessLimits:
    response_seconds: float = 5.0
    max_message_bytes: int = 10_000_000

    def __post_init__(self) -> None:
        if self.response_seconds <= 0.0 or self.max_message_bytes < 1024:
            raise ValueError("candidate process limits must be positive")


class CandidateClient:
    """Own a candidate subprocess without inheriting evaluator secrets."""

    _SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")

    def __init__(
        self,
        argv: tuple[str, ...] | list[str],
        *,
        cwd: Path,
        limits: CandidateProcessLimits = CandidateProcessLimits(),
        unreadable_roots: tuple[Path, ...] = (),
        unwritable_roots: tuple[Path, ...] = (),
        seed_artifacts: Mapping[str, bytes] | None = None,
        sandbox: bool | None = None,
    ):
        if not argv or any(not isinstance(part, str) or not part for part in argv):
            raise ValueError("candidate argv must contain non-empty strings")
        self.cwd = cwd.resolve()
        self.limits = limits
        self._temporary = tempfile.TemporaryDirectory(prefix="rlx-candidate-")
        self._scratch = Path(self._temporary.name).resolve()
        for name, content in (seed_artifacts or {}).items():
            path = self._artifact_path(name)
            if not isinstance(content, bytes):
                raise TypeError("seed artifact content must be bytes")
            path.write_bytes(content)
            path.chmod(0o400)
        environment = {
            key: os.environ[key] for key in self._SAFE_ENV_KEYS if key in os.environ
        }
        environment.setdefault("PATH", "/usr/bin:/bin")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TMPDIR"] = self._temporary.name
        environment["RLX_CANDIDATE_SCRATCH"] = self._temporary.name
        use_sandbox = sys.platform == "darwin" if sandbox is None else sandbox
        command = tuple(argv)
        if use_sandbox:
            command = _sandbox_command(
                command,
                unreadable_roots=unreadable_roots,
                unwritable_roots=unwritable_roots,
            )
        self._stderr_file = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            env=environment,
            start_new_session=False,
            text=False,
            bufsize=0,
        )
        assert self.process.stdout is not None
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_lines, daemon=True)
        self._reader.start()
        self._closed = False

    def _artifact_path(self, name: str) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise CandidateProtocolError("candidate artifact name must be one plain filename")
        path = (self._scratch / name).resolve()
        if path.parent != self._scratch:
            raise CandidateProtocolError("candidate artifact escaped scratch directory")
        return path

    def read_artifact(self, name: str, *, max_bytes: int) -> bytes:
        """Read a regular, non-symlink artifact from evaluator-owned scratch."""

        path = self._artifact_path(name)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CandidateProtocolError("candidate checkpoint artifact is missing") from exc
        if path.is_symlink() or not path.is_file():
            raise CandidateProtocolError("candidate checkpoint must be a regular file")
        if metadata.st_size > max_bytes:
            raise CandidateProtocolError("candidate checkpoint exceeds artifact limit")
        return path.read_bytes()

    def _read_lines(self) -> None:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline(self.limits.max_message_bytes + 1)
            if not line:
                self._lines.put(None)
                return
            self._lines.put(line)

    def notify(self, message: Mapping[str, Any]) -> None:
        self._write(message)

    def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        self._write(message)
        try:
            line = self._lines.get(timeout=self.limits.response_seconds)
        except queue.Empty as exc:
            self.kill()
            raise CandidateProtocolError(
                f"candidate did not respond within {self.limits.response_seconds} seconds"
            ) from exc
        if line is None:
            code = self.process.poll()
            raise CandidateProtocolError(f"candidate exited before responding (exit {code})")
        if len(line) > self.limits.max_message_bytes:
            self.kill()
            raise CandidateProtocolError("candidate response exceeded message limit")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateProtocolError("candidate response is not JSON") from exc
        if not isinstance(value, dict):
            raise CandidateProtocolError("candidate response must be a JSON object")
        return value

    def _write(self, message: Mapping[str, Any]) -> None:
        if self._closed or self.process.poll() is not None:
            raise CandidateProtocolError("candidate process is not running")
        try:
            encoded = (json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n").encode()
        except (TypeError, ValueError) as exc:
            raise CandidateProtocolError("evaluator attempted to send invalid JSON") from exc
        if len(encoded) > self.limits.max_message_bytes:
            raise CandidateProtocolError("evaluator message exceeded protocol limit")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CandidateProtocolError("candidate closed its input") from exc

    def close(self) -> None:
        if self._closed:
            return
        try:
            response = self.request({"type": "close"})
            if response.get("type") != "closed":
                raise CandidateProtocolError("candidate did not acknowledge close")
            self.process.wait(timeout=self.limits.response_seconds)
        except (CandidateProtocolError, subprocess.TimeoutExpired):
            self.kill()
        finally:
            self._closed = True

    def kill(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
        self._closed = True

    def stderr_digest(self) -> tuple[str, int]:
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        digest = hashlib.sha256()
        size = 0
        while chunk := self._stderr_file.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size

    def __enter__(self) -> CandidateClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close() if exc is None else self.kill()
        self._stderr_file.close()
        self._temporary.cleanup()


def _sandbox_command(
    argv: tuple[str, ...],
    *,
    unreadable_roots: tuple[Path, ...],
    unwritable_roots: tuple[Path, ...],
) -> tuple[str, ...]:
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if not sandbox_exec.exists():
        raise RuntimeError("candidate sandbox requested but sandbox-exec is unavailable")
    home = Path.home()
    sensitive = (
        home / ".ssh",
        home / ".aws",
        home / ".config" / "gh",
        home / ".codex",
        home / ".claude",
    )
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny process-fork)",
        "(deny process-info*)",
        "(allow process-info* (target self))",
    ]
    for path in sensitive:
        quoted = json.dumps(str(path))
        rules.append(f"(deny file-read* (subpath {quoted}))")
        rules.append(f"(deny file-write* (subpath {quoted}))")
    for path in unreadable_roots:
        quoted = json.dumps(str(path.resolve()))
        rules.append(f"(deny file-read* (subpath {quoted}))")
        rules.append(f"(deny file-write* (subpath {quoted}))")
    for path in unwritable_roots:
        quoted = json.dumps(str(path.resolve()))
        rules.append(f"(deny file-write* (subpath {quoted}))")
    return (str(sandbox_exec), "-p", " ".join(rules), *argv)
