"""Compact neural reference learners and process-isolated candidate tools."""

from .neural import (
    BranchingActorCritic,
    EpisodeResult,
    NeuralModelManifest,
    NeuralReferenceConfig,
    NeuralTrainingResult,
    evaluate_actor_critic,
    train_actor_critic,
)

__all__ = [
    "BranchingActorCritic",
    "EpisodeResult",
    "NeuralModelManifest",
    "NeuralReferenceConfig",
    "NeuralTrainingResult",
    "evaluate_actor_critic",
    "train_actor_critic",
]
