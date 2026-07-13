"""Intervention-based checks against FactorLab's declared influence graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actions import ActionEnumerationError
from .factorlab import FactorLabInspector, FactorLabWorld, ObjectiveProtocol


ObservedEdge = tuple[int, int, int]


@dataclass(frozen=True)
class CausalAuditResult:
    interventions: int
    declared_edges: frozenset[ObservedEdge]
    recovered_edges: frozenset[ObservedEdge]
    unexpected_edges: frozenset[ObservedEdge]
    detection_rate: float
    passed: bool


def _candidate_actions(world: FactorLabWorld, limit: int) -> tuple[Any, ...]:
    try:
        return tuple(item.action for item in world.action_spec.enumerate(limit=limit))
    except ActionEnumerationError:
        rng = np.random.default_rng(0)
        return tuple(world.action_spec.sample(rng) for _ in range(min(limit, 32)))


def audit_causal_contract(
    world: FactorLabWorld,
    *,
    max_action_candidates: int = 64,
    max_intervention_times: int = 64,
    atol: float = 1e-10,
    min_detection_rate: float = 0.8,
) -> CausalAuditResult:
    """Change one decision at a time and compare observed and declared edges.

    Detection rate is intentionally separate from soundness. Conditional
    threshold/prerequisite edges may be dormant under a particular reference
    trajectory, but an observed edge outside the declared graph always fails.
    """

    inspector = FactorLabInspector(world)
    config = world.config
    candidates = _candidate_actions(world, max_action_candidates)
    if len(candidates) < 2:
        raise ValueError("causal audit needs at least two distinguishable actions")
    if max_intervention_times < 1:
        raise ValueError("max_intervention_times must be positive")
    baseline_decisions = [candidates[0]] * config.decision_count
    preference = (
        (1.0,) + (0.0,) * (config.n_objectives - 1)
        if config.protocol is ObjectiveProtocol.PREFERENCE_CONDITIONED
        else None
    )
    baseline = np.asarray(
        inspector.simulate(baseline_decisions, preference=preference).rewards,
        dtype=np.float64,
    )
    recovered: set[ObservedEdge] = set()
    interventions = 0
    intervention_count = min(config.decision_count, max_intervention_times)
    decision_indices = tuple(
        dict.fromkeys(
            int(value)
            for value in np.linspace(
                0,
                config.decision_count - 1,
                num=intervention_count,
                dtype=np.int64,
            )
        )
    )
    action_times = {
        config.memory_lag + decision_index for decision_index in decision_indices
    }
    for decision_index in decision_indices:
        action_time = config.memory_lag + decision_index
        best_edges: set[ObservedEdge] = set()
        for alternative in candidates[1:]:
            sequence = list(baseline_decisions)
            sequence[decision_index] = alternative
            changed = np.asarray(
                inspector.simulate(sequence, preference=preference).rewards,
                dtype=np.float64,
            )
            differences = np.argwhere(np.abs(changed - baseline) > atol)
            observed = {
                (action_time, int(after_step) + 1, int(objective))
                for after_step, objective in differences
            }
            interventions += 1
            if len(observed) > len(best_edges):
                best_edges = observed
            if len(best_edges) >= config.n_objectives:
                break
        recovered.update(best_edges)

    declared = frozenset(
        (edge.action_time, edge.reward_time, objective)
        for edge in inspector.influence_edges()
        if edge.action_time in action_times
        for objective in edge.objectives
    )
    unexpected = frozenset(recovered - declared)
    recovered_declared = frozenset(recovered & declared)
    detection_rate = len(recovered_declared) / len(declared) if declared else 1.0
    return CausalAuditResult(
        interventions=interventions,
        declared_edges=declared,
        recovered_edges=recovered_declared,
        unexpected_edges=unexpected,
        detection_rate=detection_rate,
        passed=not unexpected and detection_rate >= min_detection_rate,
    )
