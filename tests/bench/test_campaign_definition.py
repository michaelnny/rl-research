from __future__ import annotations

import json
from pathlib import Path


DEFINITION = (
    Path(__file__).resolve().parents[2] / "campaigns" / "factorlab_v0" / "definition.json"
)


def test_factorlab_campaign_is_explicitly_unqualified_and_keyless() -> None:
    definition = json.loads(DEFINITION.read_text())

    assert definition["status"] == "under_calibration"
    assert definition["world_suite"]["master_key_source"].startswith("runtime/secrets/")
    assert definition["world_suite"]["derivation"] == "HMAC-SHA256"
    assert definition["world_suite"]["shared_hidden_cue_transform"] is True
    assert "master_key" not in definition["world_suite"]
    assert "master_seed" not in definition["world_suite"]
    assert definition["world_suite"]["heldout_worlds"] > definition["world_suite"]["public_worlds"]


def test_factorlab_campaign_covers_required_extremes_without_cartesian_explosion() -> None:
    definition = json.loads(DEFINITION.read_text())
    sweeps = definition["diagnostic_sweeps"]

    assert max(sweeps["horizon"]) >= 1024
    assert max(item["joint_choices"] for item in sweeps["joint_discrete_designs"]) >= 10**12
    assert max(sweeps["continuous_dimensions"]) >= 64
    assert 1 in sweeps["reward_events"]
    assert max(sweeps["n_objectives"]) >= 8
    assert definition["fractional_design"]["forbidden"] == "unreviewed full Cartesian sweep"
