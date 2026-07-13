from __future__ import annotations

import numpy as np
import pytest

from rlx_bench.budget import BudgetExceeded, BudgetLedger, BudgetLimits, BudgetedEnv
from rlx_bench.factorlab import FactorLabConfig, FactorLabEnv, generate_world
from rlx_bench.metrics import (
    constraint_metrics,
    coverage_metrics,
    hypervolume_2d,
    normalize_returns,
    pareto_mask,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_budget_wrapper_enforces_transitions_and_episodes() -> None:
    clock = FakeClock()
    limits = BudgetLimits(transitions=3, episodes=1, wall_seconds=10.0)
    ledger = BudgetLedger(limits, clock=clock)
    config = FactorLabConfig(horizon=4, n_factors=1, max_causal_lag=2)
    env = BudgetedEnv(FactorLabEnv(generate_world(config, 1)), ledger)

    env.reset(preference=(1.0, 0.0))
    for _ in range(3):
        env.step((0,))
    with pytest.raises(BudgetExceeded) as transition_error:
        env.step((0,))
    assert transition_error.value.resource == "transitions"
    with pytest.raises(BudgetExceeded) as episode_error:
        env.reset(preference=(1.0, 0.0))
    assert episode_error.value.resource == "episodes"
    assert ledger.snapshot().transitions == 3


def test_budget_ledger_enforces_wall_policy_and_preference_limits() -> None:
    clock = FakeClock()
    ledger = BudgetLedger(
        BudgetLimits(
            transitions=10,
            episodes=2,
            wall_seconds=5.0,
            policies=2,
            preference_queries=1,
        ),
        clock=clock,
    )

    ledger.register_policy(2)
    ledger.record_preference_query()
    with pytest.raises(BudgetExceeded, match="policies"):
        ledger.register_policy()
    with pytest.raises(BudgetExceeded, match="preference_queries"):
        ledger.record_preference_query()
    clock.now = 5.0
    with pytest.raises(BudgetExceeded, match="wall_seconds"):
        ledger.check()


def test_normalization_uses_declared_bounds_not_candidate_extrema() -> None:
    returns = [[0.0, 5.0], [5.0, 10.0]]

    normalized = normalize_returns(returns, lower=(0.0, 0.0), upper=(10.0, 20.0))

    np.testing.assert_allclose(normalized, [[0.0, 0.25], [0.5, 0.5]])


def test_pareto_mask_removes_dominated_points_without_scalarization() -> None:
    points = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4], [0.3, 0.3]])

    assert pareto_mask(points).tolist() == [True, True, True, False]


def test_coverage_metrics_report_panel_regret_and_policy_count() -> None:
    candidates = [[1.0, 0.0], [0.0, 0.8]]
    reference = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]]
    panel = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    report = coverage_metrics(candidates, reference, panel)

    assert report.utility_regrets == pytest.approx((0.0, 0.2, 0.1))
    assert report.expected_utility_regret == pytest.approx(0.1)
    assert report.worst_preference_regret == pytest.approx(0.2)
    assert report.policy_count == 2


def test_constraint_metrics_put_feasibility_before_primary_value() -> None:
    values = [[0.9, 0.8, 0.8], [1.0, 0.4, 0.9], [0.7, 0.7, 0.6]]

    report = constraint_metrics(values, floors=(0.6, 0.7))

    assert report.feasibility_rate == pytest.approx(1 / 3)
    assert report.feasible_primary_mean == pytest.approx(0.9)
    assert report.mean_total_violation == pytest.approx((0.0 + 0.2 + 0.1) / 3)


def test_two_objective_hypervolume_uses_nondominated_union() -> None:
    points = [[1.0, 0.5], [0.5, 1.0], [0.25, 0.25]]

    assert hypervolume_2d(points, reference_point=(0.0, 0.0)) == pytest.approx(0.75)
