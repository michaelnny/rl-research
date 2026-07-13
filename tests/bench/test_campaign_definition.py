from __future__ import annotations

import json
from pathlib import Path

from rlx_agents.evaluate import CandidateEvaluationConfig
from rlx_bench.factorlab import FactorLabConfig


ROOT = Path(__file__).resolve().parents[2]
DEFINITION = ROOT / "campaigns" / "factorlab_long_v1" / "definition.json"
PROTOCOL = ROOT / "campaigns" / "factorlab_long_v1" / "qualification_protocol.json"


def test_factorlab_campaign_requires_compact_neural_learners_and_hidden_kernel() -> None:
    definition = json.loads(DEFINITION.read_text())

    assert definition["status"] == "qualification_pending"
    assert definition["admitted_tiers"] == []
    assert definition["learner_class"]["required"] == "compact_neural_policy"
    assert "tabular_policy" in definition["learner_class"]["excluded"]
    assert definition["anchor_configuration"]["horizon"] == 5000
    assert definition["anchor_configuration"]["joint_choices"] == 10**12
    assert definition["anchor_configuration"]["reward_events"] == 1
    assert max(definition["target_horizons"]) == 20_000

    protocol = json.loads(PROTOCOL.read_text())
    anchor = dict(protocol["anchor_configuration"])
    anchor["levels_per_factor"] = (anchor["levels_per_factor"],)
    evaluator = CandidateEvaluationConfig()
    assert evaluator.factor_config().task_id == FactorLabConfig(**anchor).task_id
    assert evaluator.training_episodes == protocol["reference"]["episodes"]
    assert evaluator.public_worlds == protocol["suite"]["public_worlds"]
    assert evaluator.heldout_worlds == protocol["suite"]["heldout_worlds"]
    assert definition["evaluation_protocol"]["wall_seconds_total"] == 14_400
    assert protocol["reference"]["device"] == "auto"


def test_campaign_covers_extremes_with_consumer_gpu_budgets() -> None:
    definition = json.loads(DEFINITION.read_text())
    assert min(definition["target_horizons"]) >= 5000
    assert max(definition["target_horizons"]) >= 20_000
    assert definition["learner_class"]["max_accelerator_seconds_per_seed"] == 14_400
    assert definition["learner_class"]["max_trainable_parameters"] == 2_000_000
    assert definition["learner_class"]["max_transitions_per_seed"] == 5_000_000
