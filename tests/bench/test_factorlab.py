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
)


def _action_for_cue(cue: tuple[float, ...], *, opposite: bool = False) -> tuple[int, ...]:
    if opposite:
        cue = tuple(-value for value in cue)
    return tuple(1 if value > 0 else 0 for value in cue)


def _objective_zero_actions(world) -> list[tuple[int, ...]]:
    config = world.config
    return [
        _action_for_cue(world.cues[time])
        for time in range(config.memory_lag, config.horizon)
    ]


def test_world_generation_is_deterministic_and_content_addressed() -> None:
    config = FactorLabConfig(horizon=8, n_factors=3, max_causal_lag=6)
    first = generate_world(config, seed=123)
    second = generate_world(config, seed=123)
    other = generate_world(config, seed=124)

    assert first == second
    assert first.world_id == second.world_id
    assert first.world_id != other.world_id
    assert first.intrinsic_lags[0] == 6


def test_task_id_changes_with_declared_factor_but_not_world_seed() -> None:
    base = FactorLabConfig(horizon=8, n_factors=2, max_causal_lag=4)
    changed = dataclasses.replace(base, conflict_strength=0.5)

    assert generate_world(base, 1).config.task_id == generate_world(base, 2).config.task_id
    assert base.task_id != changed.task_id


def test_terminal_only_vector_reward_preserves_real_objective_conflict() -> None:
    config = FactorLabConfig(
        horizon=8,
        n_factors=3,
        max_causal_lag=8,
        reward_events=1,
        conflict_strength=1.0,
    )
    world = generate_world(config, seed=5)
    trajectory = FactorLabInspector(world).simulate(
        _objective_zero_actions(world), preference=(1.0, 0.0)
    )

    assert all(reward == (0.0, 0.0) for reward in trajectory.rewards[:-1])
    assert trajectory.return_vector == pytest.approx((8.0, 0.0))


def test_sparse_rewards_only_appear_on_declared_schedule_and_sum_to_return() -> None:
    config = FactorLabConfig(
        horizon=12,
        n_factors=2,
        max_causal_lag=3,
        reward_events=4,
        conflict_strength=0.8,
    )
    world = generate_world(config, seed=7)
    trajectory = FactorLabInspector(world).simulate(
        _objective_zero_actions(world), preference=(0.7, 0.3)
    )

    schedule = set(world.reward_schedule)
    nonzero_after_steps = {
        index + 1 for index, reward in enumerate(trajectory.rewards) if any(reward)
    }
    assert nonzero_after_steps <= schedule
    assert tuple(np.sum(np.asarray(trajectory.rewards), axis=0)) == pytest.approx(
        trajectory.return_vector
    )


def test_matched_action_renderings_hold_latent_world_and_returns_fixed() -> None:
    common = dict(
        horizon=6,
        n_factors=2,
        levels_per_factor=(2,),
        max_causal_lag=4,
        reward_events=3,
    )
    factored = generate_world(FactorLabConfig(**common, action_mode="factored_discrete"), 11)
    flat = generate_world(FactorLabConfig(**common, action_mode="flat_discrete"), 11)
    continuous = generate_world(FactorLabConfig(**common, action_mode="continuous"), 11)

    assert factored.cues == flat.cues == continuous.cues
    assert factored.intrinsic_lags == flat.intrinsic_lags == continuous.intrinsic_lags
    canonical = [factored.action_spec.decode(action) for action in _objective_zero_actions(factored)]
    flat_lookup = {
        encoded.canonical: encoded.action for encoded in flat.action_spec.enumerate()
    }
    flat_actions = [flat_lookup[action] for action in canonical]

    factored_return = FactorLabInspector(factored).simulate(
        _objective_zero_actions(factored), preference=(0.6, 0.4)
    )
    flat_return = FactorLabInspector(flat).simulate(
        flat_actions, preference=(0.6, 0.4)
    )
    continuous_return = FactorLabInspector(continuous).simulate(
        canonical, preference=(0.6, 0.4)
    )

    assert factored_return.return_vector == pytest.approx(flat_return.return_vector)
    assert factored_return.return_vector == pytest.approx(continuous_return.return_vector)


def test_memory_lag_reveals_future_cue_and_requires_noop_warmup() -> None:
    config = FactorLabConfig(horizon=9, n_factors=2, max_causal_lag=3, memory_lag=3)
    world = generate_world(config, seed=4)
    env = FactorLabEnv(world)
    observation, public = env.reset(preference=(0.5, 0.5))

    assert observation["cue_for_time"] == 3
    assert observation["revealed_cue"] == world.cues[3]
    assert observation["action_required"] is False
    assert "world_id" not in public
    assert "seed" not in public
    with pytest.raises(InvalidAction):
        env.step((0, 0))
    for _ in range(3):
        observation, _, _, _, _ = env.step(None)
    assert observation["action_required"] is True
    with pytest.raises(InvalidAction):
        env.step(None)


def test_protocols_are_distinct_and_preferences_are_normalized() -> None:
    preferred = FactorLabEnv(generate_world(FactorLabConfig(horizon=4), 1))
    with pytest.raises(ValueError):
        preferred.reset()
    _, info = preferred.reset(preference=(2.0, 1.0))
    assert info["preference"] == pytest.approx([2 / 3, 1 / 3])

    coverage_config = FactorLabConfig(
        horizon=4, protocol=ObjectiveProtocol.POLICY_COVERAGE
    )
    coverage = FactorLabEnv(generate_world(coverage_config, 1))
    coverage.reset()
    with pytest.raises(ValueError):
        coverage.reset(preference=(0.5, 0.5))

    constrained_config = FactorLabConfig(
        horizon=4, n_objectives=3, protocol=ObjectiveProtocol.CONSTRAINED
    )
    assert constrained_config.constraint_floors == (0.5, 0.5)


def test_effect_mechanisms_create_declared_longer_range_edges() -> None:
    config = FactorLabConfig(
        horizon=10,
        n_factors=2,
        max_causal_lag=3,
        reward_events=5,
        effects=("additive", "pairwise", "prerequisite", "threshold"),
        pairwise_gap=2,
        prerequisite_span=3,
    )
    inspector = FactorLabInspector(generate_world(config, seed=2))
    edges = inspector.influence_edges()

    mechanisms = {edge.mechanism for edge in edges}
    assert mechanisms == set(EffectKind)
    assert any(
        edge.reward_time == config.horizon and edge.mechanism is EffectKind.THRESHOLD
        for edge in edges
    )
    assert max(edge.reward_time - edge.action_time for edge in edges) >= 8


def test_all_composed_effects_stay_inside_fixed_return_bounds() -> None:
    config = FactorLabConfig(
        horizon=12,
        n_factors=2,
        max_causal_lag=5,
        effects=("additive", "pairwise", "prerequisite", "threshold"),
    )
    world = generate_world(config, seed=3)
    trajectory = FactorLabInspector(world).simulate(
        _objective_zero_actions(world), preference=(0.5, 0.5)
    )

    assert all(
        0.0 <= value <= upper
        for value, upper in zip(trajectory.return_vector, world.return_upper_bound, strict=True)
    )


def test_episode_lifecycle_is_strict() -> None:
    config = FactorLabConfig(horizon=3, n_factors=2, max_causal_lag=2)
    env = FactorLabEnv(generate_world(config, seed=1))
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
        ("memory_lag", 64),
        ("reward_events", 65),
        ("conflict_strength", 1.1),
    ],
)
def test_invalid_factor_configurations_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        FactorLabConfig(**{field: value})
