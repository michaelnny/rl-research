from __future__ import annotations

import json

from rlx_agents.calibration import SmokeCalibrationSettings, run_smoke_calibration
from rlx_agents.cli import main
from rlx_bench.qualification import REQUIRED_QUALIFICATION_CHECKS, CheckStatus


def test_smoke_calibration_measures_all_gates_without_claiming_qualification() -> None:
    report = run_smoke_calibration(
        SmokeCalibrationSettings(learner_episodes=60, headroom_episodes=3, master_seed=4)
    )
    checks = {check.name: check for check in report.checks}

    assert set(checks) == set(REQUIRED_QUALIFICATION_CHECKS)
    assert report.qualified is False
    assert checks["mechanics"].status is CheckStatus.OBSERVED
    assert checks["mechanics"].measurements["terminal_only"] is True
    assert checks["causal_audit"].measurements["unexpected_edges"] == 0
    assert checks["learnability"].measurements["after_utility"] > checks[
        "learnability"
    ].measurements["before_utility"]
    assert checks["statistics"].status is CheckStatus.NOT_RUN
    assert checks["independent_audit"].status is CheckStatus.NOT_RUN


def test_calibration_cli_writes_machine_readable_report(tmp_path) -> None:
    output = tmp_path / "smoke.json"

    result = main(
        [
            "smoke",
            "--learner-episodes",
            "20",
            "--headroom-episodes",
            "1",
            "--master-seed",
            "9",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert result == 0
    assert payload["qualified"] is False
    assert payload["benchmark_revision"] == "factorlab-v0-under-calibration"
