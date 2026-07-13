from __future__ import annotations

from pathlib import Path

from rlx_lab.models import JobMode
from rlx_lab.providers import ClaudeProvider, CodexProvider, ProviderRequest


def request(tmp_path, mode=JobMode.READ):
    return ProviderRequest(
        role="theorist",
        prompt="think",
        cwd=tmp_path,
        mode=mode,
        schema={"type": "object"},
        timeout_seconds=30,
    )


def test_codex_command_uses_ephemeral_structured_sandbox(tmp_path):
    provider = CodexProvider(model="test-model")
    command = provider.build_command(
        request(tmp_path),
        Path("/tmp/schema.json"),
        Path("/tmp/final.json"),
    )
    assert command[:6] == ("codex", "-a", "never", "--search", "exec", "--json")
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "test-model"
    assert command[-1] == "-"


def test_codex_write_mode_is_workspace_scoped(tmp_path):
    provider = CodexProvider()
    command = provider.build_command(
        request(tmp_path, JobMode.WRITE),
        None,
        Path("/tmp/final.json"),
    )
    assert command[command.index("--sandbox") + 1] == "workspace-write"


def test_claude_read_mode_has_no_shell_or_edit_tools(tmp_path):
    command = ClaudeProvider(model="test-model").build_command(request(tmp_path))
    assert "--safe-mode" in command
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    tools = command[command.index("--tools") + 1]
    assert tools == "Read,Glob,Grep,WebSearch,WebFetch"
    assert "Bash" not in tools and "Edit" not in tools
