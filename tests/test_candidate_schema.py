from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from scripts.validate_candidate import validate_candidate  # noqa: E402


def valid_candidate() -> dict[str, object]:
    return {
        "run_id": "20260606-99-auto",
        "algorithm_name": "Test Probe",
        "principle": "Update actions by a typed non-baseline operator.",
        "primitive_name": "Signed transition witness",
        "primitive_type": "map: observation_hash -> action -> signed measure",
        "update_family": "state_action_operator",
        "memory": "table",
        "feedback_signal": 'native vector reward from info["vector"]',
        "uses_reward": True,
        "uses_vector_reward": True,
        "claimed_stage": "vector",
        "nearest_disqualifier": "scalarization",
        "novelty_boundary": "Does not scalarize vector reward into a fixed scalar weight.",
        "empirical_claim": "Should improve hypervolume on vector envs.",
        "falsifier": "No lift over random on vector stage.",
        "ablation_plan": "Randomize the signed witness before action selection.",
        "proof_debt": "Show whether the induced operator has a fixed point.",
    }


def test_valid_candidate_passes() -> None:
    assert validate_candidate(valid_candidate()) == []


def test_missing_required_field_fails() -> None:
    data = valid_candidate()
    data.pop("ablation_plan")

    assert "ablation_plan must be a non-empty string" in validate_candidate(data)


def test_unknown_stage_fails() -> None:
    data = valid_candidate()
    data["claimed_stage"] = "moon"

    errors = validate_candidate(data)

    assert any("claimed_stage must be one of" in error for error in errors)


def test_vector_claim_requires_vector_stage() -> None:
    data = valid_candidate()
    data["claimed_stage"] = "sparse"

    assert "uses_vector_reward=true requires a stage containing vector envs" in validate_candidate(
        data
    )


def test_scalarization_risk_requires_boundary_explanation() -> None:
    data = valid_candidate()
    data["novelty_boundary"] = "Different from the nearest method."

    assert (
        "scalarization-risk probes must explain the scalarization boundary"
        in validate_candidate(data)
    )


def test_validate_candidate_cli(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(valid_candidate()))

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_candidate.py"), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "candidate.json: ok" in completed.stdout
