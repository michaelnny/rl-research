from __future__ import annotations

import json
import os
from pathlib import Path

from rlx_lab.cli import main


REPOSITORY = Path(__file__).resolve().parents[2]


def test_cli_initializes_campaign_and_reports_status(tmp_path, capsys):
    assert main(["--repo", str(tmp_path), "--runtime", "runtime", "init", "--name", "test", "--question", "why?"]) == 0
    campaign = capsys.readouterr().out.strip()
    assert campaign.startswith("campaign-")
    secret = tmp_path / "runtime" / "secrets" / f"{campaign}.key"
    assert len(secret.read_bytes()) == 32
    assert os.stat(secret).st_mode & 0o777 == 0o600
    assert main(["--repo", str(tmp_path), "--runtime", "runtime", "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_cli_renders_brief_from_graph_state(tmp_path, capsys):
    output = tmp_path / "brief.md"
    assert main(["--repo", str(tmp_path), "--runtime", "runtime", "brief", "--output", str(output)]) == 0
    capsys.readouterr()
    assert "Generated from the immutable research graph" in output.read_text()


def test_cli_controls_campaign_lifecycle_and_detailed_status(tmp_path, capsys):
    assert (
        main(
            [
                "--repo",
                str(tmp_path),
                "--runtime",
                "runtime",
                "init",
                "--name",
                "controlled",
                "--question",
                "which mechanism survives?",
                "--max-branches",
                "2",
                "--concurrent-branches",
                "1",
            ]
        )
        == 0
    )
    campaign = capsys.readouterr().out.strip()

    assert main(["--repo", str(tmp_path), "--runtime", "runtime", "status", "--campaign", campaign]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["campaign"]["status"] == "active"
    assert status["usage"]["provider_attempts"] == 0

    for command, expected in (("pause", "paused"), ("resume", "active"), ("stop", "stopped")):
        assert (
            main(
                [
                    "--repo",
                    str(tmp_path),
                    "--runtime",
                    "runtime",
                    command,
                    "--campaign",
                    campaign,
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["status"] == expected


def test_cli_execution_entry_points_refuse_an_unadmitted_tier(tmp_path, capsys):
    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "--repo",
                str(REPOSITORY),
                "--runtime",
                str(runtime),
                "init",
                "--name",
                "blocked",
                "--question",
                "should not execute",
            ]
        )
        == 0
    )
    campaign = capsys.readouterr().out.strip()

    assert (
        main(
            [
                "--repo",
                str(REPOSITORY),
                "--runtime",
                str(runtime),
                "controller",
                "--once",
                "--campaign",
                campaign,
            ]
        )
        == 2
    )
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["ready"] is False
    assert any(check["name"] == "qualified_benchmark_tier" for check in refusal["checks"])

    assert (
        main(
            [
                "--repo",
                str(REPOSITORY),
                "--runtime",
                str(runtime),
                "worker",
                "--once",
                "--campaign",
                campaign,
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["ready"] is False
