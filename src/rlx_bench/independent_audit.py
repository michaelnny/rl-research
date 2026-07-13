"""Independent numerical path for Neural FactorLab qualification.

This module intentionally does not call ``FactorLabEnv``, ``FactorLabInspector``,
the main action-score function, or the benchmark normalization implementation.
It consumes an immutable world record and reimplements the additive+dynamics
reference equations for audit-sized configurations.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .factorlab import EffectKind, FactorLabWorld


@dataclass(frozen=True)
class IndependentAuditReport:
    score_max_abs_error: float
    trajectory_max_abs_error: float
    oracle_max_abs_error: float
    normalization_max_abs_error: float
    nonlinear_midpoint_residual: float
    random_fingerprint_max_abs_error: float
    passed: bool


def _targets(world: FactorLabWorld, signal: np.ndarray, state: np.ndarray) -> np.ndarray:
    kernel = world.task_kernel
    config = world.config
    dense = np.concatenate(
        (
            config.signal_target_scale * signal,
            config.context_target_scale * np.asarray(world.context),
            config.state_target_scale * state,
        )
    )
    hidden = np.tanh(np.asarray(kernel.encoder) @ dense + np.asarray(kernel.encoder_bias))
    return np.tanh(
        np.einsum("ofh,h->of", np.asarray(kernel.objective_heads), hidden)
        + np.asarray(kernel.objective_bias)
    )


def independent_scores(
    world: FactorLabWorld,
    signal: Sequence[float],
    state: Sequence[float],
    canonical_action: Sequence[float],
) -> np.ndarray:
    targets = _targets(world, np.asarray(signal), np.asarray(state))
    action = np.asarray(canonical_action)
    return np.mean(np.exp(-5.0 * np.square((action[None, :] - targets) / 2.0)), axis=1)


def _schedule(horizon: int, events: int) -> tuple[int, ...]:
    return tuple(math.ceil(index * horizon / events) for index in range(1, events + 1))


def independent_trajectory(
    world: FactorLabWorld,
    actions: Sequence[Any],
) -> np.ndarray:
    """Reimplement additive+dynamics returns without the environment path."""

    config = world.config
    if set(config.effects) - {EffectKind.ADDITIVE, EffectKind.DYNAMICS}:
        raise ValueError("independent trajectory supports additive+dynamics audit worlds")
    if len(actions) != config.decision_count:
        raise ValueError("action count must equal decision_count")
    schedule = _schedule(config.horizon, config.reward_events)
    state = np.asarray(world.initial_state, dtype=np.float64)
    pending: dict[int, np.ndarray] = {}
    total = np.zeros(config.n_objectives, dtype=np.float64)
    decision_index = 0
    for time in range(config.horizon):
        if time < config.memory_lag:
            canonical = np.zeros(config.n_factors, dtype=np.float64)
        else:
            canonical = np.asarray(world.action_spec.decode(actions[decision_index]))
            decision_index += 1
            scores = independent_scores(world, world.signals[time], state, canonical)
            maturity = min(config.horizon, time + int(world.intrinsic_lags[time]))
            release = next(value for value in schedule if value >= maturity)
            pending.setdefault(release, np.zeros(config.n_objectives))
            pending[release] += scores
        signal = np.asarray(world.signals[time])
        action_term = config.action_influence * (
            np.asarray(world.task_kernel.action_to_state) @ canonical
        )
        state = np.tanh(
            config.state_decay * (np.asarray(world.task_kernel.transition) @ state)
            + action_term
            + config.exogenous_influence
            * (np.asarray(world.task_kernel.signal_to_state) @ signal)
        )
        after_step = time + 1
        total += pending.pop(after_step, np.zeros(config.n_objectives))
    targets = np.tanh(
        np.einsum(
            "osc,c->os",
            np.asarray(world.task_kernel.terminal_heads),
            np.asarray(world.context),
        )
    )
    terminal = np.mean(
        np.exp(-5.0 * np.square((state[None, :] - targets) / 2.0)), axis=1
    )
    total += config.terminal_state_weight * terminal
    return total


def independent_normalize(
    returns: Sequence[Sequence[float]],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def nonlinear_midpoint_residual(world: FactorLabWorld) -> float:
    state = np.asarray(world.initial_state)
    left = np.asarray(world.signals[0])
    right = np.asarray(world.signals[min(1, world.config.horizon - 1)])
    midpoint = 0.5 * (left + right)
    left_targets = _targets(world, left, state)
    right_targets = _targets(world, right, state)
    middle_targets = _targets(world, midpoint, state)
    return float(np.max(np.abs(middle_targets - 0.5 * (left_targets + right_targets))))


def run_independent_audit(
    world: FactorLabWorld,
    *,
    atol: float = 1e-10,
) -> IndependentAuditReport:
    from .factorlab import FactorLabInspector, score_canonical_action
    from .metrics import normalize_returns
    from .oracle import exact_weighted_solution

    actions = tuple(item.action for item in world.action_spec.enumerate(limit=10_000))
    sample_actions = actions[: min(8, len(actions))]
    score_errors = []
    for action in sample_actions:
        canonical = world.action_spec.decode(action)
        main = np.asarray(
            score_canonical_action(
                world, world.signals[0], world.initial_state, canonical
            )
        )
        alternate = independent_scores(
            world, world.signals[0], world.initial_state, canonical
        )
        score_errors.append(float(np.max(np.abs(main - alternate))))

    sequence_values: list[np.ndarray] = []
    trajectory_errors: list[float] = []
    main_values: list[np.ndarray] = []
    inspector = FactorLabInspector(world)
    preference = (1.0,) + (0.0,) * (world.config.n_objectives - 1)
    for sequence in itertools.product(actions, repeat=world.config.decision_count):
        independent = independent_trajectory(world, sequence)
        main = np.asarray(inspector.simulate(sequence, preference=preference).return_vector)
        sequence_values.append(independent)
        main_values.append(main)
        trajectory_errors.append(float(np.max(np.abs(main - independent))))
    weights = np.asarray(preference)
    independent_best = max(float(np.dot(weights, value)) for value in sequence_values)
    oracle = exact_weighted_solution(world, preference, max_sequences=100_000)
    oracle_error = abs(independent_best - oracle.scalar_value)

    values = np.asarray(main_values)
    alternate_normalized = independent_normalize(
        values, (0.0,) * world.config.n_objectives, world.return_upper_bound
    )
    main_normalized = normalize_returns(
        values, (0.0,) * world.config.n_objectives, world.return_upper_bound
    )
    normalization_error = float(np.max(np.abs(alternate_normalized - main_normalized)))

    rng = np.random.default_rng(71)
    random_sequences = [
        tuple(world.action_spec.sample(rng) for _ in range(world.config.decision_count))
        for _ in range(32)
    ]
    independent_random = np.mean(
        [independent_trajectory(world, sequence) for sequence in random_sequences], axis=0
    )
    main_random = np.mean(
        [inspector.simulate(sequence, preference=preference).return_vector for sequence in random_sequences],
        axis=0,
    )
    random_error = float(np.max(np.abs(independent_random - main_random)))
    residual = nonlinear_midpoint_residual(world)
    values_to_check = (
        max(score_errors, default=0.0),
        max(trajectory_errors, default=0.0),
        oracle_error,
        normalization_error,
        random_error,
    )
    return IndependentAuditReport(
        score_max_abs_error=values_to_check[0],
        trajectory_max_abs_error=values_to_check[1],
        oracle_max_abs_error=values_to_check[2],
        normalization_max_abs_error=values_to_check[3],
        nonlinear_midpoint_residual=residual,
        random_fingerprint_max_abs_error=values_to_check[4],
        passed=all(value <= atol for value in values_to_check) and residual > 1e-4,
    )
