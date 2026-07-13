from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rlx_bench.actions import InvalidAction
from rlx_bench.factorlab import (
    EffectKind,
    FactorLabConfig,
    FactorLabEnv,
    FactorLabInspector,
    ObjectiveProtocol,
    generate_world,
    score_canonical_action,
)


def _config(**changes: object) -> FactorLabConfig:
    values = {
        "horizon": 8,
        "n_factors": 2,
        "levels_per_factor": (3,),
        "signal_dim": 4,
        "context_dim": 3,
        "state_dim": 3,
        "teacher_hidden_dim": 8,
        "max_causal_lag": 8,
        "terminal_state_weight": 1.0,
    }
    values.update(changes)
    return FactorLabConfig(**values)


def test_world_generation_is_deterministic_continuous_and_content_addressed() -> None:
    config = _config()
    first = generate_world(config, seed=123, kernel_key=b"k" * 32)
    second = generate_world(config, seed=123, kernel_key=b"k" * 32)
    other = generate_world(config, seed=124, kernel_key=b"k" * 32)

    assert first == second
    assert first.world_id == second.world_id
    assert first.world_id != other.world_id
    assert first.task_kernel.kernel_id == other.task_kernel.kernel_id
    assert len(set(first.signals)) == config.horizon
    assert first.intrinsic_lags[0] == config.max_causal_lag


def test_task_id_changes_with_declared_factor_but_not_world_seed() -> None:
    base = _config()
    changed = dataclasses.replace(base, signal_target_scale=2.0)

    assert generate_world(base, 1).config.task_id == generate_world(base, 2).config.task_id
    assert base.task_id != changed.task_id
    assert base.task_id.startswith("factorlab-v1-")


def test_hidden_task_mapping_is_nonlinear_and_objectives_conflict() -> None:
    world = generate_world(_config(conflict_strength=1.0), 5, kernel_key=b"c" * 32)
    choices = tuple(world.action_spec.enumerate())
    scores = [
        score_canonical_action(
            world, world.signals[0], world.initial_state, choice.canonical
        )
        for choice in choices
    ]
    best_first = int(np.argmax([value[0] for value in scores]))
    best_second = int(np.argmax([value[1] for value in scores]))

    assert best_first != best_second
    left = np.asarray(world.signals[0])
    right = np.asarray(world.signals[1])
    midpoint = 0.5 * (left + right)
    action = choices[0].canonical
    midpoint_score = np.asarray(
        score_canonical_action(world, midpoint, world.initial_state, action)
    )
    endpoint_mean = 0.5 * (
        np.asarray(score_canonical_action(world, left, world.initial_state, action))
        + np.asarray(score_canonical_action(world, right, world.initial_state, action))
    )
    assert not np.allclose(midpoint_score, endpoint_mean)


def test_terminal_only_vector_reward_has_no_intermediate_feedback() -> None:
    config = _config(reward_events=1)
    world = generate_world(config, seed=5)
    actions = [(1, 1)] * config.decision_count
    trajectory = FactorLabInspector(world).simulate(actions, preference=(1.0, 0.0))

    assert all(reward == (0.0, 0.0) for reward in trajectory.rewards[:-1])
    assert all(value > 0.0 for value in trajectory.return_vector)


def test_sparse_rewards_only_appear_on_declared_schedule_and_sum_to_return() -> None:
    config = _config(horizon=12, max_causal_lag=3, reward_events=4)
    world = generate_world(config, seed=7)
    trajectory = FactorLabInspector(world).simulate(
        [(1, 1)] * config.decision_count, preference=(0.7, 0.3)
    )

    nonzero = {index + 1 for index, reward in enumerate(trajectory.rewards) if any(reward)}
    assert nonzero <= set(world.reward_schedule)
    assert tuple(np.sum(np.asarray(trajectory.rewards), axis=0)) == pytest.approx(
        trajectory.return_vector
    )


def test_matched_action_renderings_hold_neural_world_and_returns_fixed() -> None:
    common = dict(
        horizon=6,
        n_factors=2,
        levels_per_factor=(2,),
        signal_dim=3,
        context_dim=2,
        state_dim=2,
        teacher_hidden_dim=4,
        max_causal_lag=4,
        reward_events=3,
        terminal_state_weight=1.0,
    )
    factored = generate_world(FactorLabConfig(**common, action_mode="factored_discrete"), 11)
    flat = generate_world(FactorLabConfig(**common, action_mode="flat_discrete"), 11)
    continuous = generate_world(FactorLabConfig(**common, action_mode="continuous"), 11)

    assert factored.signals == flat.signals == continuous.signals
    assert factored.task_kernel == flat.task_kernel == continuous.task_kernel
    factored_actions = [(0, 1)] * factored.config.decision_count
    canonical = [factored.action_spec.decode(action) for action in factored_actions]
    flat_lookup = {item.canonical: item.action for item in flat.action_spec.enumerate()}
    flat_actions = [flat_lookup[action] for action in canonical]

    first = FactorLabInspector(factored).simulate(factored_actions, preference=(0.6, 0.4))
    second = FactorLabInspector(flat).simulate(flat_actions, preference=(0.6, 0.4))
    third = FactorLabInspector(continuous).simulate(canonical, preference=(0.6, 0.4))
    assert first.return_vector == pytest.approx(second.return_vector)
    assert first.return_vector == pytest.approx(third.return_vector)


def test_memory_lag_reveals_future_signal_and_requires_noop_warmup() -> None:
    config = _config(horizon=9, max_causal_lag=3, memory_lag=3)
    world = generate_world(config, seed=4)
    env = FactorLabEnv(world)
    observation, public = env.reset(preference=(0.5, 0.5))

    assert observation["signal_for_time"] == 3
    assert observation["signal"] == list(world.signals[3])
    assert observation["action_required"] is False
    assert len(observation["features"]) == config.observation_width
    assert "world_id" not in public and "seed" not in public
    with pytest.raises(InvalidAction):
        env.step((0, 0))
    for _ in range(3):
        observation, _, _, _, _ = env.step(None)
    assert observation["action_required"] is True
    with pytest.raises(InvalidAction):
        env.step(None)


def test_protocols_are_distinct_and_preferences_are_normalized() -> None:
    preferred = FactorLabEnv(generate_world(_config(horizon=4, max_causal_lag=4), 1))
    with pytest.raises(ValueError):
        preferred.reset()
    _, info = preferred.reset(preference=(2.0, 1.0))
    assert info["preference"] == pytest.approx([2 / 3, 1 / 3])

    coverage = FactorLabEnv(
        generate_world(_config(horizon=4, max_causal_lag=4, protocol=ObjectiveProtocol.POLICY_COVERAGE), 1)
    )
    coverage.reset()
    with pytest.raises(ValueError):
        coverage.reset(preference=(0.5, 0.5))

    constrained = FactorLabConfig(
        horizon=4,
        n_objectives=3,
        protocol=ObjectiveProtocol.CONSTRAINED,
    )
    assert constrained.constraint_floors == (0.5, 0.5)


def test_effect_mechanisms_declare_dynamics_and_long_range_edges() -> None:
    config = _config(
        horizon=10,
        max_causal_lag=3,
        reward_events=5,
        effects=("additive", "dynamics", "pairwise", "prerequisite", "threshold"),
        pairwise_gap=2,
        prerequisite_span=3,
    )
    edges = FactorLabInspector(generate_world(config, seed=2)).influence_edges()

    assert {edge.mechanism for edge in edges} == set(EffectKind)
    assert any(edge.reward_time == config.horizon for edge in edges)
    assert max(edge.reward_time - edge.action_time for edge in edges) >= 8


def test_returns_stay_inside_declared_bounds_and_lifecycle_is_strict() -> None:
    config = _config(horizon=6, max_causal_lag=5)
    world = generate_world(config, seed=3)
    trajectory = FactorLabInspector(world).simulate(
        [(1, 1)] * config.decision_count, preference=(0.5, 0.5)
    )
    assert all(
        0.0 <= value <= upper
        for value, upper in zip(trajectory.return_vector, world.return_upper_bound, strict=True)
    )

    env = FactorLabEnv(world)
    with pytest.raises(RuntimeError):
        env.step((0, 0))
    env.reset(preference=(0.5, 0.5))
    for _ in range(config.horizon):
        _, _, terminated, truncated, _ = env.step((0, 0))
        assert truncated is False
    assert terminated is True
    with pytest.raises(RuntimeError):
        env.step((0, 0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon", 1),
        ("n_objectives", 1),
        ("memory_lag", 128),
        ("reward_events", 129),
        ("conflict_strength", 1.1),
        ("signal_autocorrelation", 1.0),
    ],
)
def test_invalid_factor_configurations_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        FactorLabConfig(**{field: value})
