"""FactorLab: controlled causal diagnostics for long-horizon vector RL.

This module intentionally contains no imports from ``rlh_bench``. It defines a
new procedural task family with separate learner and privileged-inspector APIs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from .actions import ActionSpec, InvalidAction, make_action_spec


class ObjectiveProtocol(str, Enum):
    PREFERENCE_CONDITIONED = "preference_conditioned"
    POLICY_COVERAGE = "policy_coverage"
    CONSTRAINED = "constrained"


class EffectKind(str, Enum):
    ADDITIVE = "additive"
    PAIRWISE = "pairwise"
    THRESHOLD = "threshold"
    PREREQUISITE = "prerequisite"


ACTION_MODES = frozenset(
    {
        "flat_discrete",
        "embedded_catalog",
        "factored_discrete",
        "continuous",
        "conditional_hybrid",
    }
)


@dataclass(frozen=True)
class FactorLabConfig:
    horizon: int = 64
    n_objectives: int = 2
    n_factors: int = 4
    action_mode: str = "factored_discrete"
    levels_per_factor: tuple[int, ...] = (2,)
    catalog_size: int = 128
    max_causal_lag: int | None = None
    memory_lag: int = 0
    reward_events: int = 1
    conflict_strength: float = 1.0
    effects: tuple[EffectKind | str, ...] = (EffectKind.ADDITIVE,)
    pairwise_gap: int = 1
    interaction_strength: float = 0.25
    prerequisite_span: int = 4
    threshold: float = 0.7
    threshold_bonus: float = 0.5
    cue_cardinality: int = 8
    protocol: ObjectiveProtocol | str = ObjectiveProtocol.PREFERENCE_CONDITIONED
    constraint_floors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if self.n_objectives < 2:
            raise ValueError("FactorLab requires at least two objectives")
        if self.n_factors < 1:
            raise ValueError("n_factors must be positive")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"unknown action mode: {self.action_mode}")
        counts = tuple(int(value) for value in self.levels_per_factor)
        if len(counts) == 1:
            counts = counts * self.n_factors
        if len(counts) != self.n_factors or any(value < 2 for value in counts):
            raise ValueError("levels_per_factor must give >=2 levels for every factor")
        object.__setattr__(self, "levels_per_factor", counts)
        if self.catalog_size < 2:
            raise ValueError("catalog_size must be at least 2")
        max_causal_lag = min(32, self.horizon) if self.max_causal_lag is None else self.max_causal_lag
        if not 1 <= max_causal_lag <= self.horizon:
            raise ValueError("max_causal_lag must lie in [1, horizon]")
        object.__setattr__(self, "max_causal_lag", int(max_causal_lag))
        if not 0 <= self.memory_lag < self.horizon:
            raise ValueError("memory_lag must lie in [0, horizon)")
        if not 1 <= self.reward_events <= self.horizon:
            raise ValueError("reward_events must lie in [1, horizon]")
        if not 0.0 <= self.conflict_strength <= 1.0:
            raise ValueError("conflict_strength must lie in [0, 1]")
        normalized_effects = tuple(EffectKind(effect) for effect in self.effects)
        if not normalized_effects or len(set(normalized_effects)) != len(normalized_effects):
            raise ValueError("effects must be a non-empty set of mechanisms")
        if EffectKind.ADDITIVE not in normalized_effects:
            raise ValueError("additive must be present so every decision has a direct effect")
        object.__setattr__(self, "effects", normalized_effects)
        if self.pairwise_gap < 1:
            raise ValueError("pairwise_gap must be positive")
        if self.interaction_strength < 0.0:
            raise ValueError("interaction_strength cannot be negative")
        if self.prerequisite_span < 2:
            raise ValueError("prerequisite_span must be at least 2")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if self.threshold_bonus < 0.0:
            raise ValueError("threshold_bonus cannot be negative")
        if self.cue_cardinality < 2:
            raise ValueError("cue_cardinality must be at least 2")
        protocol = ObjectiveProtocol(self.protocol)
        object.__setattr__(self, "protocol", protocol)
        if protocol is ObjectiveProtocol.CONSTRAINED:
            floors = self.constraint_floors or (0.5,) * (self.n_objectives - 1)
            if len(floors) != self.n_objectives - 1:
                raise ValueError("constrained protocol needs one floor after the primary objective")
            if any(not 0.0 <= floor <= 1.0 for floor in floors):
                raise ValueError("constraint floors are normalized and must lie in [0, 1]")
            object.__setattr__(self, "constraint_floors", tuple(float(x) for x in floors))
        elif self.constraint_floors:
            raise ValueError("constraint_floors only apply to the constrained protocol")

    @property
    def decision_count(self) -> int:
        return self.horizon - self.memory_lag

    @property
    def joint_discrete_choices(self) -> int:
        return math.prod(self.levels_per_factor)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["protocol"] = self.protocol.value
        data["effects"] = [effect.value for effect in self.effects]
        data["levels_per_factor"] = list(self.levels_per_factor)
        data["constraint_floors"] = list(self.constraint_floors)
        return data

    @property
    def task_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return f"factorlab-v0-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class CueTransform:
    """Evaluator-owned signed permutation shared by a task suite.

    Learners observe the untransformed cue and must infer this mapping from
    training feedback instead of exploiting a public cue/action identity.
    """

    permutation: tuple[int, ...]
    signs: tuple[float, ...]

    def __post_init__(self) -> None:
        width = len(self.permutation)
        if width < 1 or tuple(sorted(self.permutation)) != tuple(range(width)):
            raise ValueError("cue transform permutation must contain each dimension once")
        if len(self.signs) != width or any(sign not in (-1.0, 1.0) for sign in self.signs):
            raise ValueError("cue transform signs must be +/-1 for every dimension")

    @classmethod
    def identity(cls, width: int) -> CueTransform:
        if width < 1:
            raise ValueError("cue transform width must be positive")
        return cls(tuple(range(width)), (1.0,) * width)

    def apply(self, cue: tuple[float, ...]) -> tuple[float, ...]:
        if len(cue) != len(self.permutation):
            raise ValueError("cue width does not match cue transform")
        return tuple(
            self.signs[index] * float(cue[source])
            for index, source in enumerate(self.permutation)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"permutation": list(self.permutation), "signs": list(self.signs)}


@dataclass(frozen=True)
class FactorLabWorld:
    config: FactorLabConfig
    seed: int
    cues: tuple[tuple[float, ...], ...]
    intrinsic_lags: tuple[int, ...]
    objective_signs: tuple[tuple[float, ...], ...]
    cue_transform: CueTransform
    action_spec: ActionSpec
    world_id: str

    @property
    def reward_schedule(self) -> tuple[int, ...]:
        return reward_schedule(self.config.horizon, self.config.reward_events)

    @property
    def return_upper_bound(self) -> tuple[float, ...]:
        config = self.config
        pair_count = max(0, config.decision_count - config.pairwise_gap)
        upper = float(config.decision_count)
        if EffectKind.PAIRWISE in config.effects:
            upper += pair_count * config.interaction_strength
        if EffectKind.THRESHOLD in config.effects:
            upper += config.threshold_bonus
        return (upper,) * config.n_objectives


@dataclass(frozen=True)
class InfluenceEdge:
    action_time: int
    latent_maturity: int
    reward_time: int
    objectives: tuple[int, ...]
    mechanism: EffectKind


@dataclass(frozen=True)
class Trajectory:
    actions: tuple[Any, ...]
    rewards: tuple[tuple[float, ...], ...]
    return_vector: tuple[float, ...]


def _objective_signs(n_objectives: int, n_factors: int) -> tuple[tuple[float, ...], ...]:
    signs: list[tuple[float, ...]] = []
    for objective in range(n_objectives):
        if objective == 0:
            signs.append((1.0,) * n_factors)
        elif objective == 1:
            signs.append((-1.0,) * n_factors)
        else:
            signs.append(
                tuple(
                    1.0 if ((factor + 1) * (objective + 1)) % (objective + 2) else -1.0
                    for factor in range(n_factors)
                )
            )
    return tuple(signs)


def score_canonical_action(
    config: FactorLabConfig,
    objective_signs: tuple[tuple[float, ...], ...],
    cue: tuple[float, ...],
    canonical_action: tuple[float, ...],
) -> tuple[float, ...]:
    """Return fixed-semantics objective scores in [0, 1]."""

    if len(cue) != config.n_factors or len(canonical_action) != config.n_factors:
        raise ValueError("cue and canonical action dimensions must match the configuration")
    cue_array = np.asarray(cue, dtype=np.float64)
    action_array = np.asarray(canonical_action, dtype=np.float64)
    scores: list[float] = []
    for signs in objective_signs:
        alternate = cue_array * np.asarray(signs, dtype=np.float64)
        ideal = (1.0 - config.conflict_strength) * cue_array + (
            config.conflict_strength * alternate
        )
        squared_distance = np.square((action_array - ideal) / 2.0)
        scores.append(float(np.clip(1.0 - np.mean(squared_distance), 0.0, 1.0)))
    return tuple(scores)


def reward_schedule(horizon: int, reward_events: int) -> tuple[int, ...]:
    """Return after-step indices; one event means terminal-only feedback."""

    return tuple(math.ceil(index * horizon / reward_events) for index in range(1, reward_events + 1))


def _release_time(world: FactorLabWorld, action_time: int) -> tuple[int, int]:
    maturity = min(world.config.horizon, action_time + world.intrinsic_lags[action_time])
    release = next(time for time in world.reward_schedule if time >= maturity)
    return maturity, release


def _world_payload(
    config: FactorLabConfig,
    seed: int,
    cues: tuple[tuple[float, ...], ...],
    lags: tuple[int, ...],
    signs: tuple[tuple[float, ...], ...],
    cue_transform: CueTransform,
    action_spec: ActionSpec,
) -> dict[str, Any]:
    return {
        "family": "factorlab",
        "version": 0,
        "config": config.to_dict(),
        "seed": seed,
        "cues": cues,
        "intrinsic_lags": lags,
        "objective_signs": signs,
        "cue_transform": cue_transform.to_dict(),
        "action_schema": action_spec.public_schema(),
    }


def generate_world(
    config: FactorLabConfig,
    seed: int,
    *,
    cue_transform: CueTransform | None = None,
) -> FactorLabWorld:
    """Generate an immutable world; its seed is evaluator-only metadata."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("world seed must be an integer")
    seed = int(seed)
    cue_transform = cue_transform or CueTransform.identity(config.n_factors)
    if len(cue_transform.permutation) != config.n_factors:
        raise ValueError("cue transform width must match n_factors")
    rng = np.random.default_rng(seed)
    prototypes = rng.choice(
        (-1.0, 1.0), size=(config.cue_cardinality, config.n_factors), replace=True
    )
    prototype_ids = rng.integers(config.cue_cardinality, size=config.horizon)
    cues = tuple(
        tuple(float(value) for value in prototypes[int(index)]) for index in prototype_ids
    )
    lags = [0] * config.horizon
    for action_time in range(config.memory_lag, config.horizon):
        maximum = max(1, min(config.max_causal_lag, config.horizon - action_time))
        lags[action_time] = int(rng.integers(1, maximum + 1))
    first_decision = config.memory_lag
    lags[first_decision] = min(config.max_causal_lag, config.horizon - first_decision)
    lags_tuple = tuple(lags)
    signs = _objective_signs(config.n_objectives, config.n_factors)
    action_spec = make_action_spec(
        config.action_mode,
        n_factors=config.n_factors,
        levels_per_factor=config.levels_per_factor,
        catalog_size=config.catalog_size,
        seed=seed ^ 0xA5A5A5A5,
    )
    payload = _world_payload(
        config, seed, cues, lags_tuple, signs, cue_transform, action_spec
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    world_id = f"flw-{hashlib.sha256(canonical.encode()).hexdigest()}"
    return FactorLabWorld(
        config=config,
        seed=seed,
        cues=cues,
        intrinsic_lags=lags_tuple,
        objective_signs=signs,
        cue_transform=cue_transform,
        action_spec=action_spec,
        world_id=world_id,
    )


def _validate_preference(config: FactorLabConfig, preference: Any) -> tuple[float, ...] | None:
    if config.protocol is ObjectiveProtocol.PREFERENCE_CONDITIONED:
        if preference is None:
            raise ValueError("preference-conditioned tasks require a preference at reset")
        try:
            values = tuple(float(value) for value in preference)
        except (TypeError, ValueError) as exc:
            raise ValueError("preference must be a numeric vector") from exc
        if len(values) != config.n_objectives:
            raise ValueError("preference width must match n_objectives")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("preference entries must be finite and non-negative")
        total = sum(values)
        if total <= 0.0:
            raise ValueError("preference must have positive mass")
        return tuple(value / total for value in values)
    if preference is not None:
        raise ValueError(f"{config.protocol.value} tasks do not reveal a utility preference")
    return None


class FactorLabEnv:
    """Learner-facing deterministic environment with no privileged metadata."""

    def __init__(self, world: FactorLabWorld):
        self._world = world
        self.action_spec = world.action_spec
        self._time = 0
        self._done = False
        self._started = False
        self._preference: tuple[float, ...] | None = None
        self._canonical_actions: dict[int, tuple[float, ...]] = {}
        self._base_scores: dict[int, tuple[float, ...]] = {}
        self._pending: dict[int, np.ndarray] = {}
        self._return = np.zeros(world.config.n_objectives, dtype=np.float64)

    def _observation(self) -> dict[str, Any]:
        config = self._world.config
        reveal_for = self._time + config.memory_lag
        cue = self._world.cues[reveal_for] if reveal_for < config.horizon else None
        return {
            "time": self._time,
            "time_fraction": self._time / config.horizon,
            "action_required": config.memory_lag <= self._time < config.horizon,
            "revealed_cue": cue,
            "cue_for_time": reveal_for if cue is not None else None,
        }

    def reset(self, *, preference: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._preference = _validate_preference(self._world.config, preference)
        self._time = 0
        self._done = False
        self._started = True
        self._canonical_actions.clear()
        self._base_scores.clear()
        self._pending.clear()
        self._return = np.zeros(self._world.config.n_objectives, dtype=np.float64)
        config = self._world.config
        public_info = {
            "task_id": config.task_id,
            "family": "factorlab",
            "version": 0,
            "horizon": config.horizon,
            "objective_names": [f"objective_{index}" for index in range(config.n_objectives)],
            "objective_orientation": ["maximize"] * config.n_objectives,
            "objective_protocol": config.protocol.value,
            "constraint_floors": list(config.constraint_floors),
            "preference": list(self._preference) if self._preference is not None else None,
            "action_spec": self.action_spec.public_schema(),
            "return_lower_bound": [0.0] * config.n_objectives,
            "return_upper_bound": list(self._world.return_upper_bound),
        }
        return self._observation(), public_info

    def _add_pending(self, release: int, values: np.ndarray) -> None:
        bucket = self._pending.setdefault(
            release, np.zeros(self._world.config.n_objectives, dtype=np.float64)
        )
        bucket += values

    def _apply_action(self, action: Any) -> None:
        world = self._world
        config = world.config
        action_time = self._time
        canonical = self.action_spec.decode(action)
        scores = score_canonical_action(
            config,
            world.objective_signs,
            world.cue_transform.apply(world.cues[action_time]),
            canonical,
        )
        self._canonical_actions[action_time] = canonical
        self._base_scores[action_time] = scores
        contribution = np.asarray(scores, dtype=np.float64)

        if EffectKind.PREREQUISITE in config.effects:
            relative = action_time - config.memory_lag
            anchor = action_time - (relative % config.prerequisite_span)
            if anchor != action_time:
                contribution *= np.asarray(self._base_scores[anchor], dtype=np.float64)

        if EffectKind.PAIRWISE in config.effects:
            prior = action_time - config.pairwise_gap
            if prior >= config.memory_lag:
                prior_scores = np.asarray(self._base_scores[prior], dtype=np.float64)
                pairwise = config.interaction_strength * np.sqrt(
                    np.asarray(scores, dtype=np.float64) * prior_scores
                )
                contribution += pairwise

        _, release = _release_time(world, action_time)
        self._add_pending(release, contribution)

    def _threshold_reward(self) -> np.ndarray:
        config = self._world.config
        if EffectKind.THRESHOLD not in config.effects:
            return np.zeros(config.n_objectives, dtype=np.float64)
        scores = np.asarray(list(self._base_scores.values()), dtype=np.float64)
        means = scores.mean(axis=0)
        return config.threshold_bonus * (means >= config.threshold)

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], tuple[float, ...], bool, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("cannot step a terminated episode")
        config = self._world.config
        if self._time < config.memory_lag:
            if action is not None:
                raise InvalidAction("warm-up steps require action=None")
        else:
            if action is None:
                raise InvalidAction("decision steps require an action")
            self._apply_action(action)

        self._time += 1
        reward = self._pending.pop(
            self._time, np.zeros(config.n_objectives, dtype=np.float64)
        )
        if self._time == config.horizon:
            reward += self._threshold_reward()
        self._return += reward
        self._done = self._time == config.horizon
        info = {
            "task_id": config.task_id,
            "time": self._time,
            "reward_event": bool(np.any(reward)),
        }
        return (
            self._observation(),
            tuple(float(value) for value in reward),
            self._done,
            False,
            info,
        )


class FactorLabInspector:
    """Privileged evaluator-only access to a generated world's causal truth."""

    def __init__(self, world: FactorLabWorld):
        self.world = world

    def manifest(self) -> dict[str, Any]:
        world = self.world
        return {
            **_world_payload(
                world.config,
                world.seed,
                world.cues,
                world.intrinsic_lags,
                world.objective_signs,
                world.cue_transform,
                world.action_spec,
            ),
            "world_id": world.world_id,
            "reward_schedule": world.reward_schedule,
            "return_upper_bound": world.return_upper_bound,
        }

    def influence_edges(self) -> tuple[InfluenceEdge, ...]:
        world = self.world
        config = world.config
        objectives = tuple(range(config.n_objectives))
        edges: set[InfluenceEdge] = set()
        for action_time in range(config.memory_lag, config.horizon):
            maturity, release = _release_time(world, action_time)
            edges.add(
                InfluenceEdge(
                    action_time, maturity, release, objectives, EffectKind.ADDITIVE
                )
            )
            if EffectKind.PAIRWISE in config.effects:
                prior = action_time - config.pairwise_gap
                if prior >= config.memory_lag:
                    edges.add(
                        InfluenceEdge(prior, maturity, release, objectives, EffectKind.PAIRWISE)
                    )
            if EffectKind.PREREQUISITE in config.effects:
                relative = action_time - config.memory_lag
                anchor = action_time - (relative % config.prerequisite_span)
                if anchor != action_time:
                    edges.add(
                        InfluenceEdge(
                            anchor, maturity, release, objectives, EffectKind.PREREQUISITE
                        )
                    )
        if EffectKind.THRESHOLD in config.effects:
            for action_time in range(config.memory_lag, config.horizon):
                edges.add(
                    InfluenceEdge(
                        action_time,
                        config.horizon,
                        config.horizon,
                        objectives,
                        EffectKind.THRESHOLD,
                    )
                )
        return tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.action_time,
                    edge.reward_time,
                    edge.mechanism.value,
                ),
            )
        )

    def expand_decision_actions(self, actions: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
        config = self.world.config
        if len(actions) == config.horizon:
            return tuple(actions)
        if len(actions) != config.decision_count:
            raise ValueError("actions must cover either the horizon or every decision step")
        return (None,) * config.memory_lag + tuple(actions)

    def simulate(
        self,
        actions: tuple[Any, ...] | list[Any],
        *,
        preference: Any = None,
    ) -> Trajectory:
        expanded = self.expand_decision_actions(actions)
        env = FactorLabEnv(self.world)
        env.reset(preference=preference)
        rewards: list[tuple[float, ...]] = []
        for action in expanded:
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        returns = tuple(float(value) for value in np.sum(np.asarray(rewards), axis=0))
        return Trajectory(actions=expanded, rewards=tuple(rewards), return_vector=returns)
