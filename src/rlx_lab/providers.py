"""Provider-neutral model worker adapters for Codex, Claude, and tests."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from rlx_lab.executor import (
    ExecutionLimits,
    ExecutionSpec,
    LocalExecutor,
    macos_deny_read_command,
)
from rlx_lab.models import JobMode


@dataclass(frozen=True)
class ProviderRequest:
    role: str
    prompt: str
    cwd: Path
    mode: JobMode
    schema: Mapping[str, Any] | None
    timeout_seconds: float


@dataclass(frozen=True)
class ProviderResult:
    output: Mapping[str, Any]
    stdout: bytes = b""
    stderr: bytes = b""
    duration_seconds: float = 0.0


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, error_class: str) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_class = error_class


class Provider(Protocol):
    def run(self, request: ProviderRequest) -> ProviderResult:
        ...


class FakeProvider:
    """Deterministic provider used to exercise the full orchestration path."""

    def __init__(self, responder: Callable[[ProviderRequest], Mapping[str, Any]]) -> None:
        self.responder = responder

    def run(self, request: ProviderRequest) -> ProviderResult:
        output = dict(self.responder(request))
        return ProviderResult(output=output, stdout=json.dumps(output).encode("utf-8"))


class CodexProvider:
    def __init__(
        self,
        *,
        command: str = "codex",
        model: str | None = None,
        unreadable_roots: tuple[Path, ...] = (),
    ) -> None:
        self.command = command
        self.model = model
        self.unreadable_roots = unreadable_roots
        self.executor = LocalExecutor()

    def build_command(self, request: ProviderRequest, schema_path: Path | None, output_path: Path) -> tuple[str, ...]:
        sandbox = "read-only" if request.mode == JobMode.READ else "workspace-write"
        command = [
            self.command,
            "-a",
            "never",
            "--search",
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            sandbox,
            "--output-last-message",
            str(output_path),
            "--cd",
            str(request.cwd),
        ]
        if self.model:
            command.extend(("--model", self.model))
        if schema_path is not None:
            command.extend(("--output-schema", str(schema_path)))
        command.append("-")
        return tuple(command)

    def run(self, request: ProviderRequest) -> ProviderResult:
        with tempfile.TemporaryDirectory(prefix="rlx-codex-") as temporary:
            root = Path(temporary)
            output_path = root / "final.json"
            schema_path = None
            if request.schema is not None:
                schema_path = root / "schema.json"
                schema_path.write_text(json.dumps(request.schema), encoding="utf-8")
            command = macos_deny_read_command(
                self.build_command(request, schema_path, output_path),
                self.unreadable_roots,
            )
            result = self.executor.run(
                ExecutionSpec(
                    argv=command,
                    cwd=request.cwd,
                    stdin=request.prompt.encode("utf-8"),
                    limits=ExecutionLimits(timeout_seconds=request.timeout_seconds, output_bytes=10_000_000),
                    inherit_env=True,
                    drop_env_prefixes=("RLX_",),
                )
            )
            if result.timed_out:
                raise ProviderError("Codex timed out", retryable=True, error_class="provider_timeout")
            if result.exit_code != 0:
                raise ProviderError(
                    _exit_detail("Codex", result.exit_code, result.stderr, result.stdout),
                    retryable=True,
                    error_class="provider_exit",
                )
            if not output_path.exists():
                raise ProviderError("Codex produced no final output", retryable=True, error_class="missing_output")
            output = _parse_mapping(output_path.read_text(encoding="utf-8"))
            return ProviderResult(output, result.stdout, result.stderr, result.duration_seconds)


class ClaudeProvider:
    def __init__(
        self,
        *,
        command: str = "claude",
        model: str | None = None,
        unreadable_roots: tuple[Path, ...] = (),
    ) -> None:
        self.command = command
        self.model = model
        self.unreadable_roots = unreadable_roots
        self.executor = LocalExecutor()

    def build_command(self, request: ProviderRequest) -> tuple[str, ...]:
        command = [
            self.command,
            "--print",
            "--safe-mode",
            "--no-session-persistence",
            "--output-format",
            "json",
        ]
        if request.mode == JobMode.READ:
            command.extend(
                (
                    "--permission-mode",
                    "dontAsk",
                    "--tools",
                    "Read,Glob,Grep,WebSearch,WebFetch",
                )
            )
        else:
            command.extend(("--permission-mode", "acceptEdits", "--tools", "default"))
        if self.model:
            command.extend(("--model", self.model))
        if request.schema is not None:
            command.extend(("--json-schema", json.dumps(request.schema, separators=(",", ":"))))
        command.append(request.prompt)
        return tuple(command)

    def run(self, request: ProviderRequest) -> ProviderResult:
        result = self.executor.run(
            ExecutionSpec(
                argv=macos_deny_read_command(
                    self.build_command(request), self.unreadable_roots
                ),
                cwd=request.cwd,
                limits=ExecutionLimits(timeout_seconds=request.timeout_seconds, output_bytes=10_000_000),
                inherit_env=True,
                drop_env_prefixes=("RLX_",),
            )
        )
        if result.timed_out:
            raise ProviderError("Claude timed out", retryable=True, error_class="provider_timeout")
        if result.exit_code != 0:
            raise ProviderError(
                _exit_detail("Claude", result.exit_code, result.stderr, result.stdout),
                retryable=True,
                error_class="provider_exit",
            )
        envelope = _parse_mapping(result.stdout.decode("utf-8"))
        structured = envelope.get("structured_output")
        if isinstance(structured, dict):
            output = structured
        elif isinstance(envelope.get("result"), str):
            output = _parse_mapping(envelope["result"])
        else:
            output = envelope
        return ProviderResult(output, result.stdout, result.stderr, result.duration_seconds)


def _parse_mapping(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError("provider output is not JSON", retryable=False, error_class="invalid_output") from exc
    if not isinstance(value, dict):
        raise ProviderError("provider output is not an object", retryable=False, error_class="invalid_output")
    return value


def _exit_detail(label: str, exit_code: int, stderr: bytes, stdout: bytes) -> str:
    diagnostic = b"\n".join(part for part in (stderr[-2000:], stdout[-4000:]) if part)
    excerpt = diagnostic.decode("utf-8", errors="replace").strip()
    return f"{label} exited with {exit_code}" + (f": {excerpt}" if excerpt else "")
