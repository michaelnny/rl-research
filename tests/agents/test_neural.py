from __future__ import annotations

import numpy as np
import pytest
import torch

from rlx_agents import (
    BranchingActorCritic,
    NeuralReferenceConfig,
    evaluate_actor_critic,
    train_actor_critic,
)
from rlx_bench.factorlab import FactorLabConfig
from rlx_bench.suite import EvaluatorWorldSuite, WorldBand, WorldSuiteSpec


def _suite(memory_lag: int = 0) -> EvaluatorWorldSuite:
    config = FactorLabConfig(
        horizon=12,
        n_factors=3,
        levels_per_factor=(3,),
        signal_dim=4,
        context_dim=3,
        state_dim=3,
        teacher_hidden_dim=8,
        max_causal_lag=12,
        memory_lag=memory_lag,
        terminal_state_weight=1.0,
    )
    return EvaluatorWorldSuite(
        config,
        WorldSuiteSpec(
            namespace=f"neural-test-{memory_lag}",
            version=1,
            master_key=b"n" * 32,
            public_worlds=4,
            tune_worlds=2,
            heldout_worlds=3,
            audit_worlds=2,
        ),
    )


@pytest.mark.parametrize(
    ("architecture", "recurrent"),
    [
        ("mlp", False),
        ("residual_mlp", False),
        ("residual_gru", True),
        ("transformer", True),
    ],
)
def test_compact_neural_architectures_have_branching_heads_and_bounded_parameters(
    architecture: str, recurrent: bool
) -> None:
    model = BranchingActorCritic(
        24,
        (3, 4, 5),
        hidden_size=32,
        architecture=architecture,
        transformer_heads=4,
        context_window=8,
    )
    memory = model.initial_state(2, torch.device("cpu"))
    logits, values, next_memory = model(torch.zeros(2, 24), memory)

    assert [tuple(value.shape) for value in logits] == [(2, 3), (2, 4), (2, 5)]
    assert tuple(values.shape) == (2,)
    assert model.recurrent is recurrent
    assert (next_memory is not None) is recurrent
    assert sum(parameter.numel() for parameter in model.parameters()) < 500_000


def test_neural_training_uses_shared_parameters_and_evaluates_unseen_worlds() -> None:
    suite = _suite()
    public = [suite.world(WorldBand.PUBLIC, index) for index in range(4)]
    heldout = [suite.world(WorldBand.HELDOUT, index) for index in range(3)]
    result = train_actor_critic(
        public,
        (1.0, 0.0),
        config=NeuralReferenceConfig(
            hidden_size=32,
            architecture="residual_mlp",
            episodes=16,
            batch_episodes=4,
            device="cpu",
            max_parameters=100_000,
        ),
        seed=3,
    )
    evaluated = evaluate_actor_critic(
        result.model, heldout, (1.0, 0.0), device="cpu"
    )

    assert result.manifest.architecture == "branching_residual_mlp_actor_critic"
    assert result.manifest.trainable_parameters < 100_000
    assert result.transitions == 16 * suite.config.horizon
    assert len(evaluated) == 3
    assert all(0.0 <= item.normalized_utility <= 1.0 for item in evaluated)
    assert np.isfinite(result.losses).all()


def test_parameter_cap_fails_closed_before_training() -> None:
    suite = _suite()
    with pytest.raises(ValueError, match="above cap"):
        train_actor_critic(
            [suite.world(WorldBand.PUBLIC, 0)],
            (1.0, 0.0),
            config=NeuralReferenceConfig(
                hidden_size=64,
                architecture="transformer",
                episodes=1,
                batch_episodes=1,
                max_parameters=10,
                device="cpu",
            ),
        )
