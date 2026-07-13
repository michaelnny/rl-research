from __future__ import annotations

import itertools

import numpy as np
import pytest

from rlx_bench.actions import ActionEnumerationError
from rlx_bench.audit import audit_causal_contract
from rlx_bench.factorlab import FactorLabConfig, FactorLabInspector, ObjectiveProtocol, generate_world
from rlx_bench.independent_audit import run_independent_audit
from rlx_bench.oracle import exact_pareto_front, exact_weighted_solution


def _tiny(protocol: ObjectiveProtocol = ObjectiveProtocol.PREFERENCE_CONDITIONED):
    config = FactorLabConfig(
        horizon=3,
        n_factors=1,
        levels_per_factor=(2,),
        signal_dim=2,
        context_dim=2,
        state_dim=2,
        teacher_hidden_dim=4,
        max_causal_lag=3,
        terminal_state_weight=1.0,
        conflict_strength=1.0,
        protocol=protocol,
    )
    return generate_world(config, 1, kernel_key=b"i" * 32)


def test_exact_weighted_solver_matches_direct_tiny_enumeration() -> None:
    world = _tiny()
    solution = exact_weighted_solution(world, (1.0, 0.0), max_sequences=100)
    actions = tuple(item.action for item in world.action_spec.enumerate())
    direct = [
        FactorLabInspector(world).simulate(sequence, preference=(1.0, 0.0)).return_vector[0]
        for sequence in itertools.product(actions, repeat=world.config.decision_count)
    ]
    assert solution.sequences_evaluated == 8
    assert solution.scalar_value == pytest.approx(max(direct))


def test_exact_solver_refuses_to_hide_intractable_enumeration() -> None:
    config = FactorLabConfig(horizon=8, n_factors=3, levels_per_factor=(3,), max_causal_lag=4)
    with pytest.raises(ActionEnumerationError, match="exact planning needs"):
        exact_weighted_solution(generate_world(config, 2), (0.5, 0.5), max_sequences=1000)


def test_exact_pareto_front_is_unique_and_nondominated() -> None:
    front = exact_pareto_front(_tiny(ObjectiveProtocol.POLICY_COVERAGE), max_sequences=100)
    returns = [item.return_vector for item in front]
    assert returns and len(returns) == len(set(returns))
    for left in returns:
        assert not any(
            all(a >= b for a, b in zip(right, left, strict=True))
            and any(a > b for a, b in zip(right, left, strict=True))
            for right in returns
            if right != left
        )


def test_interventions_recover_neural_dynamics_influence_graph() -> None:
    config = FactorLabConfig(
        horizon=8,
        n_factors=2,
        levels_per_factor=(3,),
        signal_dim=3,
        context_dim=2,
        state_dim=2,
        teacher_hidden_dim=4,
        max_causal_lag=8,
        reward_events=4,
        terminal_state_weight=1.0,
    )
    result = audit_causal_contract(generate_world(config, seed=5))
    assert result.passed is True
    assert result.detection_rate == pytest.approx(1.0)
    assert not result.unexpected_edges


def test_independent_path_checks_scores_trajectories_oracle_and_metrics() -> None:
    report = run_independent_audit(_tiny())
    assert report.passed is True
    assert report.nonlinear_midpoint_residual > 1e-4
    assert np.max(
        [
            report.score_max_abs_error,
            report.trajectory_max_abs_error,
            report.oracle_max_abs_error,
            report.normalization_max_abs_error,
            report.random_fingerprint_max_abs_error,
        ]
    ) <= 1e-10
