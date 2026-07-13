"""Compute-light reference learners for FactorLab calibration.

These are intentionally small scientific baselines, not candidate algorithms.
They consume only learner-facing observations, vector rewards, and action specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rlx_bench.actions import ActionEnumerationError, FactoredDiscreteActionSpec


@dataclass(frozen=True)
class EpisodeResult:
    return_vector: tuple[float, ...]
    scalar_utility: float
    transitions: int


def _weights(preference: tuple[float, ...] | list[float]) -> np.ndarray:
    values = np.asarray(preference, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0.0):
        raise ValueError("preference must be a non-negative objective vector")
    total = float(values.sum())
    if not np.all(np.isfinite(values)) or total <= 0.0:
        raise ValueError("preference must be finite with positive mass")
    return values / total


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


class TabularReinforce:
    """A softmax Monte-Carlo policy gradient over finite structured actions."""

    def __init__(
        self,
        action_spec: Any,
        *,
        seed: int = 0,
        learning_rate: float = 0.25,
        baseline_rate: float = 0.05,
        enumeration_limit: int = 10_000,
        use_memory: bool = True,
    ):
        if learning_rate <= 0.0 or not 0.0 < baseline_rate <= 1.0:
            raise ValueError("learning rate must be positive and baseline rate in (0, 1]")
        encoded = tuple(action_spec.enumerate(limit=enumeration_limit))
        if len(encoded) < 2:
            raise ValueError("TabularReinforce needs at least two actions")
        self.actions = tuple(item.action for item in encoded)
        self.learning_rate = learning_rate
        self.baseline_rate = baseline_rate
        self.use_memory = use_memory
        self.rng = np.random.default_rng(seed)
        self.logits: dict[tuple[float, ...], np.ndarray] = {}
        self.baseline = 0.0
        self.episodes = 0

    def _policy(self, cue: tuple[float, ...]) -> np.ndarray:
        logits = self.logits.setdefault(cue, np.zeros(len(self.actions), dtype=np.float64))
        return _softmax(logits)

    def run_episode(
        self,
        env: Any,
        preference: tuple[float, ...] | list[float],
        *,
        train: bool,
        greedy: bool = False,
    ) -> EpisodeResult:
        weights = _weights(preference)
        observation, _ = env.reset(preference=tuple(float(value) for value in weights))
        cue_memory: dict[int, tuple[float, ...]] = {}
        visits: list[tuple[tuple[float, ...], int, np.ndarray]] = []
        rewards: list[tuple[float, ...]] = []
        transitions = 0
        terminated = False
        while not terminated:
            cue_for_time = observation.get("cue_for_time")
            revealed = observation.get("revealed_cue")
            if cue_for_time is not None and revealed is not None:
                cue_memory[int(cue_for_time)] = tuple(float(value) for value in revealed)
            if observation["action_required"]:
                if self.use_memory:
                    cue = cue_memory.get(int(observation["time"]))
                else:
                    cue = tuple(revealed) if revealed is not None else None
                if cue is None:
                    cue = (0.0,)
                probabilities = self._policy(cue)
                index = (
                    int(np.argmax(probabilities))
                    if greedy
                    else int(self.rng.choice(len(self.actions), p=probabilities))
                )
                action = self.actions[index]
                visits.append((cue, index, probabilities.copy()))
            else:
                action = None
            observation, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
            transitions += 1

        return_vector = tuple(float(value) for value in np.sum(np.asarray(rewards), axis=0))
        utility = float(np.dot(weights, return_vector))
        if train:
            scale = max(1, len(visits))
            advantage = (utility - self.baseline) / scale
            for cue, selected, probabilities in visits:
                gradient = -probabilities
                gradient[selected] += 1.0
                self.logits[cue] += self.learning_rate * advantage * gradient
                np.clip(self.logits[cue], -20.0, 20.0, out=self.logits[cue])
            self.baseline += self.baseline_rate * (utility - self.baseline)
            self.episodes += 1
        return EpisodeResult(return_vector, utility, transitions)


class FactorizedReinforce:
    """Policy gradient whose parameter count grows with factors, not joint choices."""

    def __init__(
        self,
        action_spec: FactoredDiscreteActionSpec,
        *,
        seed: int = 0,
        learning_rate: float = 0.25,
        baseline_rate: float = 0.05,
        use_memory: bool = True,
    ):
        if not isinstance(action_spec, FactoredDiscreteActionSpec):
            raise TypeError("FactorizedReinforce requires a factored discrete action spec")
        self.action_spec = action_spec
        self.learning_rate = learning_rate
        self.baseline_rate = baseline_rate
        self.use_memory = use_memory
        self.rng = np.random.default_rng(seed)
        self.logits: dict[tuple[tuple[float, ...], int], np.ndarray] = {}
        self.baseline = 0.0
        self.episodes = 0

    def _policy(self, cue: tuple[float, ...], factor: int) -> np.ndarray:
        key = (cue, factor)
        logits = self.logits.setdefault(
            key, np.zeros(len(self.action_spec.levels[factor]), dtype=np.float64)
        )
        return _softmax(logits)

    def run_episode(
        self,
        env: Any,
        preference: tuple[float, ...] | list[float],
        *,
        train: bool,
        greedy: bool = False,
    ) -> EpisodeResult:
        weights = _weights(preference)
        observation, _ = env.reset(preference=tuple(float(value) for value in weights))
        cue_memory: dict[int, tuple[float, ...]] = {}
        visits: list[tuple[tuple[float, ...], int, int, np.ndarray]] = []
        rewards: list[tuple[float, ...]] = []
        terminated = False
        transitions = 0
        while not terminated:
            cue_for_time = observation.get("cue_for_time")
            revealed = observation.get("revealed_cue")
            if cue_for_time is not None and revealed is not None:
                cue_memory[int(cue_for_time)] = tuple(float(value) for value in revealed)
            if observation["action_required"]:
                cue = (
                    cue_memory.get(int(observation["time"]))
                    if self.use_memory
                    else tuple(revealed) if revealed is not None else None
                )
                if cue is None:
                    cue = (0.0,) * self.action_spec.canonical_dim
                indices: list[int] = []
                for factor in range(self.action_spec.canonical_dim):
                    probabilities = self._policy(cue, factor)
                    selected = (
                        int(np.argmax(probabilities))
                        if greedy
                        else int(self.rng.choice(len(probabilities), p=probabilities))
                    )
                    indices.append(selected)
                    visits.append((cue, factor, selected, probabilities.copy()))
                action = tuple(indices)
            else:
                action = None
            observation, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
            transitions += 1

        return_vector = tuple(float(value) for value in np.sum(np.asarray(rewards), axis=0))
        utility = float(np.dot(weights, return_vector))
        if train:
            decisions = max(1, len(visits) // self.action_spec.canonical_dim)
            advantage = (utility - self.baseline) / decisions
            for cue, factor, selected, probabilities in visits:
                gradient = -probabilities
                gradient[selected] += 1.0
                self.logits[(cue, factor)] += self.learning_rate * advantage * gradient
                np.clip(self.logits[(cue, factor)], -20.0, 20.0, out=self.logits[(cue, factor)])
            self.baseline += self.baseline_rate * (utility - self.baseline)
            self.episodes += 1
        return EpisodeResult(return_vector, utility, transitions)


def require_small_enumerable(action_spec: Any, limit: int) -> None:
    """Useful qualification preflight with an explicit failure mode."""

    try:
        tuple(action_spec.enumerate(limit=limit))
    except ActionEnumerationError as exc:
        raise ActionEnumerationError(
            "flat tabular baseline is inapplicable at this action scale; use a structured baseline"
        ) from exc
