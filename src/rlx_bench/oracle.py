"""Exact tiny-instance planners used only as FactorLab feasibility ceilings."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .actions import ActionEnumerationError
from .factorlab import FactorLabInspector, FactorLabWorld, ObjectiveProtocol


@dataclass(frozen=True)
class ExactSolution:
    decision_actions: tuple[Any, ...]
    return_vector: tuple[float, ...]
    scalar_value: float
    sequences_evaluated: int


@dataclass(frozen=True)
class ParetoSolution:
    decision_actions: tuple[Any, ...]
    return_vector: tuple[float, ...]


def _normalized_weights(weights: tuple[float, ...] | list[float], width: int) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("weights must be a finite non-negative objective vector")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    return array / total


def _finite_actions(world: FactorLabWorld, limit: int) -> tuple[Any, ...]:
    return tuple(item.action for item in world.action_spec.enumerate(limit=limit))


def _sequence_count(n_actions: int, decisions: int, max_sequences: int) -> int:
    count = n_actions**decisions
    if count > max_sequences:
        raise ActionEnumerationError(
            f"exact planning needs {count} sequences, above limit {max_sequences}"
        )
    return count


def exact_weighted_solution(
    world: FactorLabWorld,
    weights: tuple[float, ...] | list[float],
    *,
    max_sequences: int = 1_000_000,
) -> ExactSolution:
    """Exhaustively solve a finite tiny world under one linear utility."""

    normalized = _normalized_weights(weights, world.config.n_objectives)
    actions = _finite_actions(world, limit=max_sequences)
    count = _sequence_count(len(actions), world.config.decision_count, max_sequences)
    inspector = FactorLabInspector(world)
    preference = (
        tuple(float(value) for value in normalized)
        if world.config.protocol is ObjectiveProtocol.PREFERENCE_CONDITIONED
        else None
    )
    best_actions: tuple[Any, ...] | None = None
    best_return: tuple[float, ...] | None = None
    best_value = -math.inf
    for sequence in itertools.product(actions, repeat=world.config.decision_count):
        result = inspector.simulate(sequence, preference=preference)
        value = float(np.dot(normalized, result.return_vector))
        if value > best_value:
            best_value = value
            best_actions = sequence
            best_return = result.return_vector
    assert best_actions is not None and best_return is not None
    return ExactSolution(best_actions, best_return, best_value, count)


def _dominates(left: tuple[float, ...], right: tuple[float, ...], atol: float) -> bool:
    return all(a >= b - atol for a, b in zip(left, right, strict=True)) and any(
        a > b + atol for a, b in zip(left, right, strict=True)
    )


def exact_pareto_front(
    world: FactorLabWorld,
    *,
    max_sequences: int = 1_000_000,
    atol: float = 1e-12,
) -> tuple[ParetoSolution, ...]:
    """Return unique nondominated policies for a finite tiny world."""

    actions = _finite_actions(world, limit=max_sequences)
    _sequence_count(len(actions), world.config.decision_count, max_sequences)
    inspector = FactorLabInspector(world)
    preference = (
        (1.0 / world.config.n_objectives,) * world.config.n_objectives
        if world.config.protocol is ObjectiveProtocol.PREFERENCE_CONDITIONED
        else None
    )
    candidates: dict[tuple[float, ...], tuple[Any, ...]] = {}
    for sequence in itertools.product(actions, repeat=world.config.decision_count):
        result = inspector.simulate(sequence, preference=preference)
        rounded = tuple(round(value, 12) for value in result.return_vector)
        candidates.setdefault(rounded, sequence)
    returns = tuple(candidates)
    front = [
        ParetoSolution(candidates[point], point)
        for point in returns
        if not any(_dominates(other, point, atol) for other in returns if other != point)
    ]
    return tuple(sorted(front, key=lambda item: item.return_vector))
