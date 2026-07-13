from __future__ import annotations

import pytest

from rlx_bench.actions import ActionEnumerationError
from rlx_bench.audit import audit_causal_contract
from rlx_bench.factorlab import FactorLabConfig, ObjectiveProtocol, generate_world
from rlx_bench.oracle import exact_pareto_front, exact_weighted_solution


def test_exact_weighted_solver_establishes_tiny_instance_ceiling() -> None:
    config = FactorLabConfig(
        horizon=2,
        n_factors=1,
        max_causal_lag=2,
        conflict_strength=1.0,
    )
    world = generate_world(config, seed=1)

    solution = exact_weighted_solution(world, (1.0, 0.0), max_sequences=10)

    assert solution.sequences_evaluated == 4
    assert solution.return_vector == pytest.approx((2.0, 0.0))
    assert solution.scalar_value == pytest.approx(2.0)


def test_exact_solver_refuses_to_hide_intractable_enumeration() -> None:
    config = FactorLabConfig(horizon=8, n_factors=3, max_causal_lag=4)
    world = generate_world(config, seed=2)

    with pytest.raises(ActionEnumerationError, match="exact planning needs"):
        exact_weighted_solution(world, (0.5, 0.5), max_sequences=1000)


def test_exact_pareto_front_preserves_policy_tradeoffs() -> None:
    config = FactorLabConfig(
        horizon=2,
        n_factors=1,
        max_causal_lag=2,
        conflict_strength=1.0,
        protocol=ObjectiveProtocol.POLICY_COVERAGE,
    )
    world = generate_world(config, seed=3)

    front = exact_pareto_front(world, max_sequences=10)

    returns = {item.return_vector for item in front}
    assert returns == {(0.0, 2.0), (1.0, 1.0), (2.0, 0.0)}


def test_interventions_recover_additive_terminal_influence_graph() -> None:
    config = FactorLabConfig(
        horizon=6,
        n_factors=2,
        max_causal_lag=6,
        reward_events=1,
    )
    result = audit_causal_contract(generate_world(config, seed=5))

    assert result.passed is True
    assert result.detection_rate == pytest.approx(1.0)
    assert result.recovered_edges == result.declared_edges
    assert not result.unexpected_edges


def test_composed_mechanisms_never_create_undeclared_observed_edges() -> None:
    config = FactorLabConfig(
        horizon=8,
        n_factors=2,
        max_causal_lag=3,
        reward_events=4,
        effects=("additive", "pairwise", "prerequisite", "threshold"),
    )
    result = audit_causal_contract(
        generate_world(config, seed=6), min_detection_rate=0.0
    )

    assert not result.unexpected_edges
    assert result.passed is True
