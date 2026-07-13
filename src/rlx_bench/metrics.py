"""Protocol-specific vector metrics with fixed semantic normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _matrix(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def normalize_returns(
    returns: Any,
    lower: tuple[float, ...] | list[float],
    upper: tuple[float, ...] | list[float],
) -> np.ndarray:
    values = _matrix(returns, name="returns")
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != (values.shape[1],) or high.shape != low.shape:
        raise ValueError("normalization bounds must match objective width")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(high <= low):
        raise ValueError("normalization bounds must be finite with upper > lower")
    return (values - low) / (high - low)


def pareto_mask(points: Any, *, atol: float = 1e-12) -> np.ndarray:
    values = _matrix(points, name="points")
    keep = np.ones(values.shape[0], dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values >= point - atol, axis=1) & np.any(
            values > point + atol, axis=1
        )
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return keep


@dataclass(frozen=True)
class CoverageMetrics:
    utility_regrets: tuple[float, ...]
    expected_utility_regret: float
    worst_preference_regret: float
    tail_preference_regret: float
    preference_count: int
    policy_count: int


def coverage_metrics(
    candidate_returns: Any,
    reference_returns: Any,
    preference_panel: Any,
    *,
    tail_fraction: float = 0.1,
) -> CoverageMetrics:
    candidates = _matrix(candidate_returns, name="candidate_returns")
    references = _matrix(reference_returns, name="reference_returns")
    panel = _matrix(preference_panel, name="preference_panel")
    width = candidates.shape[1]
    if references.shape[1] != width or panel.shape[1] != width:
        raise ValueError("returns and preference panel must share objective width")
    if np.any(panel < 0.0) or np.any(panel.sum(axis=1) <= 0.0):
        raise ValueError("preference rows must be non-negative with positive mass")
    panel = panel / panel.sum(axis=1, keepdims=True)
    reference_utility = np.max(references @ panel.T, axis=0)
    candidate_utility = np.max(candidates @ panel.T, axis=0)
    regrets = np.maximum(0.0, reference_utility - candidate_utility)
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    tail_count = max(1, math.ceil(len(regrets) * tail_fraction))
    tail = np.sort(regrets)[-tail_count:]
    return CoverageMetrics(
        utility_regrets=tuple(float(value) for value in regrets),
        expected_utility_regret=float(np.mean(regrets)),
        worst_preference_regret=float(np.max(regrets)),
        tail_preference_regret=float(np.mean(tail)),
        preference_count=len(panel),
        policy_count=len(candidates),
    )


@dataclass(frozen=True)
class ConstraintMetrics:
    feasibility_rate: float
    mean_total_violation: float
    worst_total_violation: float
    feasible_primary_mean: float | None
    feasible_count: int
    sample_count: int


def constraint_metrics(normalized_returns: Any, floors: Any) -> ConstraintMetrics:
    values = _matrix(normalized_returns, name="normalized_returns")
    thresholds = np.asarray(floors, dtype=np.float64)
    if thresholds.shape != (values.shape[1] - 1,):
        raise ValueError("floors must cover every non-primary objective")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("constraint floors must be finite")
    violations = np.maximum(0.0, thresholds - values[:, 1:])
    totals = violations.sum(axis=1)
    feasible = totals <= 1e-12
    feasible_primary = float(np.mean(values[feasible, 0])) if np.any(feasible) else None
    return ConstraintMetrics(
        feasibility_rate=float(np.mean(feasible)),
        mean_total_violation=float(np.mean(totals)),
        worst_total_violation=float(np.max(totals)),
        feasible_primary_mean=feasible_primary,
        feasible_count=int(np.sum(feasible)),
        sample_count=len(values),
    )


def hypervolume_2d(points: Any, reference_point: tuple[float, float]) -> float:
    values = _matrix(points, name="points")
    if values.shape[1] != 2:
        raise ValueError("hypervolume_2d requires exactly two objectives")
    reference = np.asarray(reference_point, dtype=np.float64)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_point must contain two finite values")
    eligible = values[np.all(values >= reference, axis=1)]
    if len(eligible) == 0:
        return 0.0
    frontier = eligible[pareto_mask(eligible)]
    frontier = frontier[np.argsort(frontier[:, 0])]
    area = 0.0
    previous_x = reference[0]
    for x, y in frontier:
        area += max(0.0, x - previous_x) * max(0.0, y - reference[1])
        previous_x = max(previous_x, x)
    return float(area)
