from __future__ import annotations

import json
from pathlib import Path

from rlx_agents.evaluate import CandidateEvaluationConfig


ROOT = Path(__file__).resolve().parents[2]
DEFINITION = ROOT / "campaigns" / "factorlab_v1" / "definition.json"


def test_factorlab_campaign_requires_compact_neural_learners_and_hidden_kernel() -> None:
    definition = json.loads(DEFINITION.read_text())

    assert definition["status"] == "qualified_tier_available"
    assert definition["admitted_tiers"] == ["factorlab-small-v1"]
    report = definition["qualification_reports"]["factorlab-small-v1"]
    assert len(report["evidence_sha256"]) == 64
    assert report["admitted_scope"]["memory_lag"] == 0
    assert "memory_lag_greater_than_zero" in report["not_admitted"]
    assert definition["learner_class"]["required"] == "compact_neural_policy"
    assert "tabular_policy" in definition["learner_class"]["excluded"]
    assert definition["world_suite"]["shared_hidden_neural_kernel"] is True
    assert definition["world_suite"]["continuous_procedural_observations"] is True
    assert definition["world_suite"]["master_key_source"].startswith("runtime/secrets/")
    assert "master_key" not in definition["world_suite"]

    qualification = json.loads((ROOT / report["report_path"]).read_text())
    evaluator = CandidateEvaluationConfig()
    assert evaluator.factor_config().task_id == qualification["task_id"]
    assert evaluator.training_episodes == report["admitted_scope"]["training_episodes"]
    assert evaluator.public_worlds == report["admitted_scope"]["public_worlds"]
    assert evaluator.heldout_worlds == report["admitted_scope"]["heldout_worlds"]


def test_campaign_covers_extremes_with_consumer_gpu_budgets() -> None:
    definition = json.loads(DEFINITION.read_text())
    sweeps = definition["diagnostic_sweeps"]

    assert max(sweeps["horizon"]) >= 1024
    assert max(item["joint_choices"] for item in sweeps["joint_discrete_designs"]) >= 10**12
    assert max(sweeps["continuous_dimensions"]) >= 64
    assert definition["budgets"]["probe"]["accelerator_seconds"] > 0
    assert definition["budgets"]["confirmation"]["max_trainable_parameters"] == 2_000_000
    assert definition["fractional_design"]["forbidden"] == "unreviewed full Cartesian sweep"
