from __future__ import annotations

import pytest

from rlx_agents import FactorizedReinforce, TabularReinforce, cue_oracle_probe
from rlx_bench.actions import ActionEnumerationError
from rlx_bench.factorlab import FactorLabConfig, FactorLabEnv, generate_world


def test_tabular_reinforce_learns_from_terminal_only_vector_return() -> None:
    config = FactorLabConfig(
        horizon=8,
        n_factors=1,
        max_causal_lag=8,
        reward_events=1,
        cue_cardinality=4,
        conflict_strength=1.0,
    )
    world = generate_world(config, seed=12)
    learner = TabularReinforce(world.action_spec, seed=1, learning_rate=0.5)
    before = learner.run_episode(
        FactorLabEnv(world), (1.0, 0.0), train=False, greedy=True
    )

    for _ in range(400):
        learner.run_episode(FactorLabEnv(world), (1.0, 0.0), train=True)
    after = learner.run_episode(
        FactorLabEnv(world), (1.0, 0.0), train=False, greedy=True
    )

    assert before.scalar_utility < 4.0
    assert after.scalar_utility == pytest.approx(8.0)
    assert learner.episodes == 400


def test_learned_cue_policy_transfers_to_unseen_world_sequences() -> None:
    config = FactorLabConfig(
        horizon=8, n_factors=1, max_causal_lag=8, cue_cardinality=4
    )
    training_world = generate_world(config, seed=12)
    learner = TabularReinforce(training_world.action_spec, seed=2, learning_rate=0.5)
    for _ in range(400):
        learner.run_episode(FactorLabEnv(training_world), (1.0, 0.0), train=True)

    held_out = learner.run_episode(
        FactorLabEnv(generate_world(config, seed=15)),
        (1.0, 0.0),
        train=False,
        greedy=True,
    )

    assert held_out.scalar_utility == pytest.approx(8.0)


def test_flat_tabular_baseline_fails_explicitly_at_large_joint_action_count() -> None:
    config = FactorLabConfig(
        horizon=4, n_factors=40, max_causal_lag=4, levels_per_factor=(2,)
    )
    world = generate_world(config, seed=1)

    with pytest.raises(ActionEnumerationError):
        TabularReinforce(world.action_spec, enumeration_limit=10_000)


def test_factorized_reinforce_learns_without_enumerating_joint_actions() -> None:
    config = FactorLabConfig(
        horizon=8,
        n_factors=8,
        max_causal_lag=8,
        cue_cardinality=8,
        levels_per_factor=(2,),
    )
    world = generate_world(config, seed=3)
    learner = FactorizedReinforce(world.action_spec, seed=4, learning_rate=0.4)

    for _ in range(1000):
        learner.run_episode(FactorLabEnv(world), (1.0, 0.0), train=True)
    result = learner.run_episode(
        FactorLabEnv(world), (1.0, 0.0), train=False, greedy=True
    )

    assert world.config.joint_discrete_choices == 256
    assert result.scalar_utility == pytest.approx(8.0)


def test_factorized_policy_constructs_actions_in_trillion_choice_space() -> None:
    config = FactorLabConfig(
        horizon=4,
        n_factors=40,
        max_causal_lag=4,
        levels_per_factor=(2,),
    )
    world = generate_world(config, seed=7)
    learner = FactorizedReinforce(world.action_spec, seed=8)

    result = learner.run_episode(FactorLabEnv(world), (0.5, 0.5), train=True)

    assert world.config.joint_discrete_choices >= 10**12
    assert result.transitions == 4


def test_memory_probe_is_sensitive_to_memory_lag_only_when_memory_is_removed() -> None:
    no_lag = generate_world(
        FactorLabConfig(horizon=16, n_factors=2, max_causal_lag=8, memory_lag=0),
        seed=2,
    )
    lagged = generate_world(
        FactorLabConfig(horizon=16, n_factors=2, max_causal_lag=8, memory_lag=4),
        seed=2,
    )

    no_lag_memory = cue_oracle_probe(no_lag, (1.0, 0.0), use_memory=True)
    no_lag_forgetful = cue_oracle_probe(no_lag, (1.0, 0.0), use_memory=False)
    lagged_memory = cue_oracle_probe(lagged, (1.0, 0.0), use_memory=True)
    lagged_forgetful = cue_oracle_probe(lagged, (1.0, 0.0), use_memory=False)

    assert no_lag_memory.scalar_utility == no_lag_forgetful.scalar_utility
    assert lagged_memory.scalar_utility == pytest.approx(lagged.config.decision_count)
    assert lagged_forgetful.scalar_utility < lagged_memory.scalar_utility * 0.75
