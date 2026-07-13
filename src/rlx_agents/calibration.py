"""Executable provisional calibration for the FactorLab scientific kernel."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rlx_bench.actions import ActionEnumerationError
from rlx_bench.audit import audit_causal_contract
from rlx_bench.factorlab import FactorLabConfig, FactorLabEnv, FactorLabInspector, generate_world
from rlx_bench.oracle import exact_weighted_solution
from rlx_bench.qualification import (
    CheckStatus,
    QualificationCheck,
    QualificationReport,
    make_qualification_report,
)
from rlx_bench.suite import EvaluatorWorldSuite, WorldBand, WorldSuiteSpec

from .probes import cue_oracle_probe
from .tabular import FactorizedReinforce, TabularReinforce


@dataclass(frozen=True)
class SmokeCalibrationSettings:
    learner_episodes: int = 200
    headroom_episodes: int = 20
    master_seed: int = 20260713

    def __post_init__(self) -> None:
        if self.learner_episodes < 1 or self.headroom_episodes < 1:
            raise ValueError("calibration episode counts must be positive")


def _observed(name: str, **measurements: object) -> QualificationCheck:
    return QualificationCheck(name, CheckStatus.OBSERVED, dict(measurements))


def run_smoke_calibration(
    settings: SmokeCalibrationSettings = SmokeCalibrationSettings(),
) -> QualificationReport:
    """Run low-cost checks while deliberately leaving the tier unqualified."""

    suite_config = FactorLabConfig(horizon=8, n_factors=1, max_causal_lag=8)
    suite = EvaluatorWorldSuite(
        suite_config,
        WorldSuiteSpec(
            namespace="factorlab-v0-smoke",
            version=0,
            master_key=hashlib.sha256(
                f"factorlab-smoke|{settings.master_seed}".encode()
            ).digest(),
            public_worlds=2,
            tune_worlds=2,
            heldout_worlds=3,
            audit_worlds=2,
        ),
    )
    world = generate_world(suite_config, 12)
    inspector = FactorLabInspector(world)
    terminal_actions = [
        tuple(1 if value > 0 else 0 for value in world.cues[time])
        for time in range(world.config.horizon)
    ]
    terminal = inspector.simulate(terminal_actions, preference=(1.0, 0.0))
    checks: list[QualificationCheck] = [
        _observed(
            "mechanics",
            horizon=world.config.horizon,
            terminal_only=all(not any(reward) for reward in terminal.rewards[:-1]),
            return_vector=terminal.return_vector,
            action_modes_tested=5,
        )
    ]

    audit = audit_causal_contract(world)
    checks.append(
        _observed(
            "causal_audit",
            passed=audit.passed,
            detection_rate=audit.detection_rate,
            unexpected_edges=len(audit.unexpected_edges),
            interventions=audit.interventions,
        )
    )

    tiny = generate_world(
        FactorLabConfig(horizon=2, n_factors=1, max_causal_lag=2), seed=1
    )
    ceiling = exact_weighted_solution(tiny, (1.0, 0.0), max_sequences=10)
    checks.append(
        _observed(
            "feasibility",
            exact_sequences=ceiling.sequences_evaluated,
            exact_return=ceiling.return_vector,
        )
    )

    learner = TabularReinforce(world.action_spec, seed=1, learning_rate=0.5)
    before = learner.run_episode(FactorLabEnv(world), (1.0, 0.0), train=False, greedy=True)
    for _ in range(settings.learner_episodes):
        learner.run_episode(FactorLabEnv(world), (1.0, 0.0), train=True)
    after = learner.run_episode(FactorLabEnv(world), (1.0, 0.0), train=False, greedy=True)
    checks.append(
        _observed(
            "learnability",
            learner="tabular_reinforce",
            episodes=settings.learner_episodes,
            before_utility=before.scalar_utility,
            after_utility=after.scalar_utility,
            exact_ceiling=world.config.decision_count,
        )
    )

    long_config = FactorLabConfig(
        horizon=128, n_factors=1, max_causal_lag=128, cue_cardinality=4
    )
    headroom_values: list[float] = []
    for seed in range(3):
        long_world = generate_world(long_config, 100 + seed)
        limited = TabularReinforce(long_world.action_spec, seed=10 + seed, learning_rate=0.3)
        for _ in range(settings.headroom_episodes):
            limited.run_episode(FactorLabEnv(long_world), (1.0, 0.0), train=True)
        result = limited.run_episode(
            FactorLabEnv(long_world), (1.0, 0.0), train=False, greedy=True
        )
        headroom_values.append(result.scalar_utility / long_config.decision_count)
    checks.append(
        _observed(
            "headroom",
            horizon=128,
            fixed_training_episodes=settings.headroom_episodes,
            normalized_utilities=headroom_values,
            mean_normalized_utility=sum(headroom_values) / len(headroom_values),
            ceiling=1.0,
        )
    )

    no_lag = generate_world(
        FactorLabConfig(horizon=16, n_factors=2, max_causal_lag=8, memory_lag=0), 2
    )
    lagged = generate_world(
        FactorLabConfig(horizon=16, n_factors=2, max_causal_lag=8, memory_lag=4), 2
    )
    no_lag_forgetful = cue_oracle_probe(no_lag, (1.0, 0.0), use_memory=False)
    lagged_forgetful = cue_oracle_probe(lagged, (1.0, 0.0), use_memory=False)
    lagged_memory = cue_oracle_probe(lagged, (1.0, 0.0), use_memory=True)
    huge = generate_world(
        FactorLabConfig(horizon=4, n_factors=40, max_causal_lag=4), 7
    )
    flat_applicable = True
    try:
        TabularReinforce(huge.action_spec, enumeration_limit=10_000)
    except ActionEnumerationError:
        flat_applicable = False
    structured = FactorizedReinforce(huge.action_spec, seed=8)
    structured_result = structured.run_episode(
        FactorLabEnv(huge), (0.5, 0.5), train=True
    )
    checks.append(
        _observed(
            "factor_sensitivity",
            no_lag_forgetful_normalized=(
                no_lag_forgetful.scalar_utility / no_lag.config.decision_count
            ),
            lagged_forgetful_normalized=(
                lagged_forgetful.scalar_utility / lagged.config.decision_count
            ),
            lagged_memory_normalized=(
                lagged_memory.scalar_utility / lagged.config.decision_count
            ),
            joint_action_choices=huge.config.joint_discrete_choices,
            flat_tabular_applicable=flat_applicable,
            structured_policy_transitions=structured_result.transitions,
        )
    )
    checks.append(
        _observed(
            "specificity",
            capable_probe_no_lag_normalized=cue_oracle_probe(
                no_lag, (1.0, 0.0), use_memory=True
            ).scalar_utility
            / no_lag.config.decision_count,
            capable_probe_lagged_normalized=(
                lagged_memory.scalar_utility / lagged.config.decision_count
            ),
            note="provisional one-factor counterfactual only",
        )
    )

    heldout_utilities = []
    for index in range(suite.spec.heldout_worlds):
        heldout = learner.run_episode(
            FactorLabEnv(suite.world(WorldBand.HELDOUT, index)),
            (1.0, 0.0),
            train=False,
            greedy=True,
        )
        heldout_utilities.append(heldout.scalar_utility)
    checks.append(
        _observed(
            "generalization",
            heldout_world_count=len(heldout_utilities),
            heldout_utilities=heldout_utilities,
            interaction_budget_shared=True,
        )
    )
    checks.extend(
        [
            QualificationCheck(
                "statistics",
                CheckStatus.NOT_RUN,
                {"reason": "smoke run has no preregistered multi-seed confidence analysis"},
            ),
            QualificationCheck(
                "independent_audit",
                CheckStatus.NOT_RUN,
                {"reason": "requires a separate implementation path and evidence artifact"},
            ),
        ]
    )
    return make_qualification_report(
        task_id=suite_config.task_id,
        suite_id=suite.suite_id,
        benchmark_revision="factorlab-v0-under-calibration",
        checks=checks,
    )
