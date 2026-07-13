"""Privileged mechanism probes; these are not learning baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from rlx_bench.factorlab import (
    FactorLabEnv,
    FactorLabWorld,
    ObjectiveProtocol,
    score_canonical_action,
)

from .tabular import EpisodeResult, _weights


def cue_oracle_probe(
    world: FactorLabWorld,
    preference: tuple[float, ...] | list[float],
    *,
    use_memory: bool,
    enumeration_limit: int = 100_000,
) -> EpisodeResult:
    """Probe whether the declared memory factor changes a known-capable policy.

    This function uses privileged world semantics to isolate memory. Its output
    must be labeled ``mechanism_probe``, never ``learned`` or ``oracle ceiling``.
    """

    weights = _weights(preference)
    choices = tuple(world.action_spec.enumerate(limit=enumeration_limit))
    env = FactorLabEnv(world)
    reset_preference: Any = (
        tuple(float(value) for value in weights)
        if world.config.protocol is ObjectiveProtocol.PREFERENCE_CONDITIONED
        else None
    )
    observation, _ = env.reset(preference=reset_preference)
    cue_memory: dict[int, tuple[float, ...]] = {}
    rewards: list[tuple[float, ...]] = []
    terminated = False
    transitions = 0
    while not terminated:
        revealed = observation["revealed_cue"]
        cue_for = observation["cue_for_time"]
        if revealed is not None and cue_for is not None:
            cue_memory[int(cue_for)] = tuple(revealed)
        if observation["action_required"]:
            cue = (
                cue_memory.get(int(observation["time"]))
                if use_memory
                else tuple(revealed) if revealed is not None else None
            )
            if cue is None:
                action = choices[0].action
            else:
                action = max(
                    choices,
                    key=lambda item: float(
                        np.dot(
                            weights,
                            score_canonical_action(
                                world.config,
                                world.objective_signs,
                                cue,
                                item.canonical,
                            ),
                        )
                    ),
                ).action
        else:
            action = None
        observation, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
        transitions += 1
    return_vector = tuple(float(value) for value in np.sum(np.asarray(rewards), axis=0))
    return EpisodeResult(return_vector, float(np.dot(weights, return_vector)), transitions)
