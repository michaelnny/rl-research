from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import rlx_lab.preflight as preflight
from rlx_lab.campaign import create_controlled_campaign
from rlx_lab.secrets import CampaignSecretStore
from rlx_lab.store import ResearchStore


def _git(repository: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repository, check=True, capture_output=True)


def test_preflight_requires_a_committed_rebuild_and_valid_secret(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    for relative in (
        "src/rlx_lab/cli.py",
        "src/rlx_agents/evaluate.py",
        "src/rlx_bench/suite.py",
        "campaigns/schemas/example.json",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# fixture\n")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")

    runtime = tmp_path / "runtime"
    store = ResearchStore(runtime / "state.db")
    secret_store = CampaignSecretStore(runtime / "secrets")
    campaign = create_controlled_campaign(store, "test", "question")
    secret_store.ensure(campaign)

    real_command = preflight._command

    def fake_command(argv, cwd, timeout=15.0):
        if argv[0] == "codex":
            return subprocess.CompletedProcess(
                argv, 0, "--ask-for-approval --sandbox --search exec", ""
            )
        if argv[0] == "claude":
            return subprocess.CompletedProcess(
                argv, 0, "--safe-mode --json-schema --permission-mode", ""
            )
        return real_command(argv, cwd, timeout)

    monkeypatch.setattr(preflight, "_command", fake_command)
    monkeypatch.setattr(preflight.shutil, "which", lambda command: f"/mock/{command}")

    report = preflight.run_preflight(
        repository=repository,
        runtime=runtime,
        store=store,
        secrets=secret_store,
        campaign_id=campaign,
    )
    checks = {check.name: check for check in report.checks}
    assert checks["committed_snapshot"].passed
    assert checks["head_contains_rebuild"].passed
    assert checks["evaluator_key"].passed
    assert report.ready is (
        sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()
    )

    (repository / "dirty.txt").write_text("not committed\n")
    dirty = preflight.run_preflight(
        repository=repository,
        runtime=runtime,
        store=store,
        secrets=secret_store,
        campaign_id=campaign,
    )
    assert not {check.name: check for check in dirty.checks}["committed_snapshot"].passed
