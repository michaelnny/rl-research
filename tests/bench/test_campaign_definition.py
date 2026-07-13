from __future__ import annotations

import json
from pathlib import Path


DEFINITION = (
    Path(__file__).resolve().parents[2] / "campaigns" / "factorlab_v1" / "definition.json"
)


def test_factorlab_campaign_requires_compact_neural_learners_and_hidden_kernel() -> None:
    definition = json.loads(DEFINITION.read_text())

    assert definition["status"] == "qualification_pending"
    assert definition["admitted_tiers"] == []
    assert definition["learner_class"]["required"] == "compact_neural_policy"
    assert "tabular_policy" in definition["learner_class"]["excluded"]
    assert definition["world_suite"]["shared_hidden_neural_kernel"] is True
    assert definition["world_suite"]["continuous_procedural_observations"] is True
    assert definition["world_suite"]["master_key_source"].startswith("runtime/secrets/")
    assert "master_key" not in definition["world_suite"]


def test_campaign_covers_extremes_with_consumer_gpu_budgets() -> None:
    definition = json.loads(DEFINITION.read_text())
    sweeps = definition["diagnostic_sweeps"]

    assert max(sweeps["horizon"]) >= 1024
    assert max(item["joint_choices"] for item in sweeps["joint_discrete_designs"]) >= 10**12
    assert max(sweeps["continuous_dimensions"]) >= 64
    assert definition["budgets"]["probe"]["accelerator_seconds"] > 0
    assert definition["budgets"]["confirmation"]["max_trainable_parameters"] == 2_000_000
    assert definition["fractional_design"]["forbidden"] == "unreviewed full Cartesian sweep"
