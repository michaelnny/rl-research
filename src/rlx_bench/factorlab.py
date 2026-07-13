"""Neural FactorLab: controlled diagnostics for compact neural RL.

FactorLab is deliberately not a finite cue/action lookup problem.  Every world
contains continuous procedural observations and shares an evaluator-owned,
nonlinear task kernel with the other worlds in its suite.  A useful policy must
learn a representation that transfers to unseen trajectories and worlds.
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
    DYNAMICS = "dynamics"
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
    """Public task dimensions; no hidden task-kernel parameters live here."""

    horizon: int = 128
    n_objectives: int = 2
    n_factors: int = 8
    action_mode: str = "factored_discrete"
    levels_per_factor: tuple[int, ...] = (4,)
    catalog_size: int = 256
    signal_dim: int = 16
    context_dim: int = 8
    state_dim: int = 8
    teacher_hidden_dim: int = 32
    signal_autocorrelation: float = 0.2
    signal_target_scale: float = 1.5
    context_target_scale: float = 0.75
    state_target_scale: float = 0.75
    max_causal_lag: int | None = None
    memory_lag: int = 0
    reward_events: int = 1
    conflict_strength: float = 0.75
    effects: tuple[EffectKind | str, ...] = (
        EffectKind.ADDITIVE,
        EffectKind.DYNAMICS,
    )
    pairwise_gap: int = 1
    interaction_strength: float = 0.15
    prerequisite_span: int = 4
    threshold: float = 0.72
    threshold_bonus: float = 0.5
    terminal_state_weight: float = 2.0
    state_decay: float = 0.85
    action_influence: float = 0.25
    exogenous_influence: float = 0.15
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
        for name in ("signal_dim", "context_dim", "state_dim", "teacher_hidden_dim"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if not 0.0 <= self.signal_autocorrelation < 1.0:
            raise ValueError("signal_autocorrelation must lie in [0, 1)")
        if min(
            self.signal_target_scale,
            self.context_target_scale,
            self.state_target_scale,
        ) <= 0.0:
            raise ValueError("target channel scales must be positive")
        lag = min(64, self.horizon) if self.max_causal_lag is None else self.max_causal_lag
        if not 1 <= lag <= self.horizon:
            raise ValueError("max_causal_lag must lie in [1, horizon]")
        object.__setattr__(self, "max_causal_lag", int(lag))
        if not 0 <= self.memory_lag < self.horizon:
            raise ValueError("memory_lag must lie in [0, horizon)")
        if not 1 <= self.reward_events <= self.horizon:
            raise ValueError("reward_events must lie in [1, horizon]")
        if not 0.0 <= self.conflict_strength <= 1.0:
            raise ValueError("conflict_strength must lie in [0, 1]")
        effects = tuple(EffectKind(effect) for effect in self.effects)
        if not effects or len(set(effects)) != len(effects):
            raise ValueError("effects must be a non-empty set of mechanisms")
        if EffectKind.ADDITIVE not in effects:
            raise ValueError("additive must be present so every action has a direct effect")
        object.__setattr__(self, "effects", effects)
        if self.pairwise_gap < 1 or self.prerequisite_span < 2:
            raise ValueError("interaction spans must be positive")
        if self.interaction_strength < 0.0 or self.threshold_bonus < 0.0:
            raise ValueError("reward bonuses cannot be negative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if self.terminal_state_weight < 0.0:
            raise ValueError("terminal_state_weight cannot be negative")
        if EffectKind.DYNAMICS not in effects and self.terminal_state_weight:
            raise ValueError("terminal_state_weight requires the dynamics mechanism")
        if not 0.0 <= self.state_decay < 1.0:
            raise ValueError("state_decay must lie in [0, 1)")
        if self.action_influence < 0.0 or self.exogenous_influence < 0.0:
            raise ValueError("state influences cannot be negative")
        protocol = ObjectiveProtocol(self.protocol)
        object.__setattr__(self, "protocol", protocol)
        if protocol is ObjectiveProtocol.CONSTRAINED:
            floors = self.constraint_floors or (0.5,) * (self.n_objectives - 1)
            if len(floors) != self.n_objectives - 1:
                raise ValueError("constrained protocol needs one floor after the primary objective")
            if any(not 0.0 <= floor <= 1.0 for floor in floors):
                raise ValueError("constraint floors must lie in [0, 1]")
            object.__setattr__(self, "constraint_floors", tuple(float(x) for x in floors))
        elif self.constraint_floors:
            raise ValueError("constraint_floors only apply to the constrained protocol")

    @property
    def decision_count(self) -> int:
        return self.horizon - self.memory_lag

    @property
    def joint_discrete_choices(self) -> int:
        return math.prod(self.levels_per_factor)

    @property
    def observation_width(self) -> int:
        # time fraction, sin/cos time, action mask, signal mask, then dense channels.
        return (
            5
            + self.signal_dim
            + self.context_dim
            + self.state_dim
            + self.n_factors
            + self.n_objectives
        )

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
        return f"factorlab-v1-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


FloatVector = tuple[float, ...]
FloatMatrix = tuple[FloatVector, ...]


def _nested_tuple(array: np.ndarray) -> Any:
    if array.ndim == 1:
        return tuple(float(value) for value in array)
    return tuple(_nested_tuple(value) for value in array)


@dataclass(frozen=True)
class NeuralTaskKernel:
    """Evaluator-owned nonlinear task and dynamics parameters."""

    encoder: tuple[FloatVector, ...]
    encoder_bias: FloatVector
    objective_heads: tuple[tuple[FloatVector, ...], ...]
    objective_bias: tuple[FloatVector, ...]
    transition: FloatMatrix
    action_to_state: FloatMatrix
    signal_to_state: FloatMatrix
    terminal_heads: tuple[FloatMatrix, ...]
    kernel_id: str


def derive_task_kernel(config: FactorLabConfig, kernel_key: bytes) -> NeuralTaskKernel:
    """Derive one suite-shared hidden kernel without exposing its key."""

    if not isinstance(kernel_key, bytes) or len(kernel_key) < 32:
        raise ValueError("kernel_key must contain at least 256 bits")
    dimensions = (
        config.signal_dim,
        config.context_dim,
        config.state_dim,
        config.teacher_hidden_dim,
        config.n_factors,
        config.n_objectives,
    )
    digest = hashlib.sha256(kernel_key + repr(dimensions).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    input_width = config.signal_dim + config.context_dim + config.state_dim
    encoder = rng.normal(0.0, 1.25 / math.sqrt(input_width), (config.teacher_hidden_dim, input_width))
    encoder_bias = rng.normal(0.0, 0.3, config.teacher_hidden_dim)
    base_head = rng.normal(
        0.0,
        1.1 / math.sqrt(config.teacher_hidden_dim),
        (config.n_factors, config.teacher_hidden_dim),
    )
    heads: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for objective in range(config.n_objectives):
        independent = rng.normal(
            0.0,
            1.1 / math.sqrt(config.teacher_hidden_dim),
            (config.n_factors, config.teacher_hidden_dim),
        )
        if objective == 0:
            head = base_head
        elif objective == 1:
            angle = math.pi * config.conflict_strength
            head = math.cos(angle) * base_head + math.sin(angle) * independent
        else:
            angle = math.pi * config.conflict_strength * objective / (config.n_objectives - 1)
            head = math.cos(angle) * base_head + math.sin(angle) * independent
        heads.append(head)
        biases.append(rng.normal(0.0, 0.15, config.n_factors))
    transition_raw = rng.normal(size=(config.state_dim, config.state_dim))
    transition, _ = np.linalg.qr(transition_raw)
    action_to_state = rng.normal(
        0.0, 1.0 / math.sqrt(config.n_factors), (config.state_dim, config.n_factors)
    )
    signal_to_state = rng.normal(
        0.0, 1.0 / math.sqrt(config.signal_dim), (config.state_dim, config.signal_dim)
    )
    terminal_heads = rng.normal(
        0.0,
        1.0 / math.sqrt(config.context_dim),
        (config.n_objectives, config.state_dim, config.context_dim),
    )
    identity = hashlib.sha256(
        kernel_key + b"|factorlab-neural-kernel-v1|" + repr(dimensions).encode()
    ).hexdigest()
    return NeuralTaskKernel(
        encoder=_nested_tuple(encoder),
        encoder_bias=_nested_tuple(encoder_bias),
        objective_heads=_nested_tuple(np.asarray(heads)),
        objective_bias=_nested_tuple(np.asarray(biases)),
        transition=_nested_tuple(transition),
        action_to_state=_nested_tuple(action_to_state),
        signal_to_state=_nested_tuple(signal_to_state),
        terminal_heads=_nested_tuple(terminal_heads),
        kernel_id=f"flk-{identity}",
    )


@dataclass(frozen=True)
class FactorLabWorld:
    config: FactorLabConfig
    seed: int
    signals: tuple[FloatVector, ...]
    context: FloatVector
    initial_state: FloatVector
    intrinsic_lags: tuple[int, ...]
    task_kernel: NeuralTaskKernel
    action_spec: ActionSpec
    world_id: str

    @property
    def reward_schedule(self) -> tuple[int, ...]:
        return reward_schedule(self.config.horizon, self.config.reward_events)

    @property
    def return_upper_bound(self) -> tuple[float, ...]:
        config = self.config
        upper = float(config.decision_count)
        if EffectKind.PAIRWISE in config.effects:
            upper += max(0, config.decision_count - config.pairwise_gap) * (
                config.interaction_strength
            )
        if EffectKind.THRESHOLD in config.effects:
            upper += config.threshold_bonus
        if EffectKind.DYNAMICS in config.effects:
            upper += config.terminal_state_weight
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


def reward_schedule(horizon: int, reward_events: int) -> tuple[int, ...]:
    """Return after-step indices; one event means terminal-only feedback."""

    return tuple(math.ceil(index * horizon / reward_events) for index in range(1, reward_events + 1))


def _release_time(world: FactorLabWorld, action_time: int) -> tuple[int, int]:
    maturity = min(world.config.horizon, action_time + world.intrinsic_lags[action_time])
    release = next(time for time in world.reward_schedule if time >= maturity)
    return maturity, release


def generate_world(
    config: FactorLabConfig,
    seed: int,
    *,
    kernel_key: bytes | None = None,
) -> FactorLabWorld:
    """Generate an immutable neural task world; its seed remains evaluator metadata."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("world seed must be an integer")
    seed = int(seed)
    if kernel_key is None:
        kernel_key = hashlib.sha256(b"factorlab-v1-standalone-test-kernel").digest()
    kernel = derive_task_kernel(config, kernel_key)
    rng = np.random.default_rng(seed)
    context = np.tanh(rng.normal(size=config.context_dim))
    innovations = rng.normal(size=(config.horizon, config.signal_dim))
    signals = np.zeros_like(innovations)
    signals[0] = innovations[0]
    correlation = config.signal_autocorrelation
    for time in range(1, config.horizon):
        signals[time] = correlation * signals[time - 1] + math.sqrt(
            1.0 - correlation**2
        ) * innovations[time]
    signals = np.tanh(signals)
    initial_state = np.tanh(rng.normal(0.0, 0.35, config.state_dim))
    lags = [0] * config.horizon
    for action_time in range(config.memory_lag, config.horizon):
        maximum = max(1, min(config.max_causal_lag, config.horizon - action_time))
        lags[action_time] = int(rng.integers(1, maximum + 1))
    lags[config.memory_lag] = min(config.max_causal_lag, config.horizon - config.memory_lag)
    action_spec = make_action_spec(
        config.action_mode,
        n_factors=config.n_factors,
        levels_per_factor=config.levels_per_factor,
        catalog_size=config.catalog_size,
        seed=seed ^ 0xA5A5A5A5,
    )
    payload = {
        "family": "factorlab",
        "version": 1,
        "config": config.to_dict(),
        "seed": seed,
        "signals": _nested_tuple(signals),
        "context": _nested_tuple(context),
        "initial_state": _nested_tuple(initial_state),
        "intrinsic_lags": lags,
        "kernel_id": kernel.kernel_id,
        "action_schema": action_spec.public_schema(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    world_id = f"flw-{hashlib.sha256(canonical.encode()).hexdigest()}"
    return FactorLabWorld(
        config=config,
        seed=seed,
        signals=_nested_tuple(signals),
        context=_nested_tuple(context),
        initial_state=_nested_tuple(initial_state),
        intrinsic_lags=tuple(lags),
        task_kernel=kernel,
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


def _target_values(
    world: FactorLabWorld,
    signal: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
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
    return np.tanh(np.einsum("ofh,h->of", np.asarray(kernel.objective_heads), hidden) + np.asarray(kernel.objective_bias))


def score_canonical_action(
    world: FactorLabWorld,
    signal: tuple[float, ...] | np.ndarray,
    state: tuple[float, ...] | np.ndarray,
    canonical_action: tuple[float, ...] | np.ndarray,
) -> tuple[float, ...]:
    """Evaluator-side nonlinear score for one state, cue, and structured action."""

    action = np.asarray(canonical_action, dtype=np.float64)
    if action.shape != (world.config.n_factors,):
        raise ValueError("canonical action width does not match n_factors")
    targets = _target_values(world, np.asarray(signal), np.asarray(state))
    squared = np.square((action[None, :] - targets) / 2.0)
    # A bounded radial score gives useful separation between random and
    # representation-aware actions without introducing discontinuous labels.
    scores = np.mean(np.exp(-5.0 * squared), axis=1)
    return tuple(float(value) for value in np.clip(scores, 0.0, 1.0))


class FactorLabEnv:
    """Learner-facing neural environment with no privileged task parameters."""

    def __init__(self, world: FactorLabWorld):
        self._world = world
        self.action_spec = world.action_spec
        self._time = 0
        self._done = False
        self._started = False
        self._preference: tuple[float, ...] | None = None
        self._state = np.asarray(world.initial_state, dtype=np.float64)
        self._previous_action = np.zeros(world.config.n_factors, dtype=np.float64)
        self._canonical_actions: dict[int, tuple[float, ...]] = {}
        self._base_scores: dict[int, tuple[float, ...]] = {}
        self._pending: dict[int, np.ndarray] = {}
        self._return = np.zeros(world.config.n_objectives, dtype=np.float64)

    def _observation(self) -> dict[str, Any]:
        config = self._world.config
        reveal_for = self._time + config.memory_lag
        signal = self._world.signals[reveal_for] if reveal_for < config.horizon else None
        signal_values = signal or (0.0,) * config.signal_dim
        preference = self._preference or (0.0,) * config.n_objectives
        phase = 2.0 * math.pi * self._time / config.horizon
        action_required = config.memory_lag <= self._time < config.horizon
        features = (
            self._time / config.horizon,
            math.sin(phase),
            math.cos(phase),
            float(action_required),
            float(signal is not None),
            *signal_values,
            *self._world.context,
            *(float(value) for value in self._state),
            *(float(value) for value in self._previous_action),
            *preference,
        )
        return {
            "time": self._time,
            "action_required": action_required,
            "signal": list(signal) if signal is not None else None,
            "signal_for_time": reveal_for if signal is not None else None,
            "state": [float(value) for value in self._state],
            "context": list(self._world.context),
            "previous_action": [float(value) for value in self._previous_action],
            "preference": list(preference),
            "features": [float(value) for value in features],
        }

    def reset(self, *, preference: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._preference = _validate_preference(self._world.config, preference)
        self._time = 0
        self._done = False
        self._started = True
        self._state = np.asarray(self._world.initial_state, dtype=np.float64)
        self._previous_action = np.zeros(self._world.config.n_factors, dtype=np.float64)
        self._canonical_actions.clear()
        self._base_scores.clear()
        self._pending.clear()
        self._return = np.zeros(self._world.config.n_objectives, dtype=np.float64)
        config = self._world.config
        public_info = {
            "task_id": config.task_id,
            "family": "factorlab",
            "version": 1,
            "horizon": config.horizon,
            "observation_spec": {"kind": "dense_float", "width": config.observation_width},
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

    def _apply_action(self, action: Any) -> np.ndarray:
        world = self._world
        config = world.config
        action_time = self._time
        canonical = self.action_spec.decode(action)
        scores = score_canonical_action(
            world, world.signals[action_time], self._state, canonical
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
                pair = config.interaction_strength * np.sqrt(
                    np.asarray(scores) * np.asarray(self._base_scores[prior])
                )
                contribution += pair
        _, release = _release_time(world, action_time)
        self._add_pending(release, contribution)
        return np.asarray(canonical, dtype=np.float64)

    def _advance_state(self, canonical: np.ndarray) -> None:
        config = self._world.config
        kernel = self._world.task_kernel
        signal = np.asarray(self._world.signals[self._time])
        action_term = np.zeros(config.state_dim)
        if EffectKind.DYNAMICS in config.effects:
            action_term = config.action_influence * (np.asarray(kernel.action_to_state) @ canonical)
        self._state = np.tanh(
            config.state_decay * (np.asarray(kernel.transition) @ self._state)
            + action_term
            + config.exogenous_influence * (np.asarray(kernel.signal_to_state) @ signal)
        )
        self._previous_action = canonical

    def _terminal_reward(self) -> np.ndarray:
        config = self._world.config
        reward = np.zeros(config.n_objectives, dtype=np.float64)
        if EffectKind.THRESHOLD in config.effects:
            means = np.asarray(list(self._base_scores.values()), dtype=np.float64).mean(axis=0)
            reward += config.threshold_bonus * (means >= config.threshold)
        if EffectKind.DYNAMICS in config.effects and config.terminal_state_weight:
            targets = np.tanh(
                np.einsum(
                    "osc,c->os",
                    np.asarray(self._world.task_kernel.terminal_heads),
                    np.asarray(self._world.context),
                )
            )
            scores = np.mean(
                np.exp(-5.0 * np.square((self._state[None, :] - targets) / 2.0)),
                axis=1,
            )
            reward += config.terminal_state_weight * np.clip(scores, 0.0, 1.0)
        return reward

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
            canonical = np.zeros(config.n_factors, dtype=np.float64)
        else:
            if action is None:
                raise InvalidAction("decision steps require an action")
            canonical = self._apply_action(action)
        self._advance_state(canonical)
        self._time += 1
        reward = self._pending.pop(
            self._time, np.zeros(config.n_objectives, dtype=np.float64)
        )
        if self._time == config.horizon:
            reward += self._terminal_reward()
        self._return += reward
        self._done = self._time == config.horizon
        info = {
            "task_id": config.task_id,
            "time": self._time,
            "reward_event": bool(np.any(reward)),
        }
        return self._observation(), tuple(float(value) for value in reward), self._done, False, info


class FactorLabInspector:
    """Privileged evaluator-only access to causal structure and simulation."""

    def __init__(self, world: FactorLabWorld):
        self.world = world

    def manifest(self) -> dict[str, Any]:
        world = self.world
        return {
            "family": "factorlab",
            "version": 1,
            "config": world.config.to_dict(),
            "world_id": world.world_id,
            "world_seed": world.seed,
            "kernel_id": world.task_kernel.kernel_id,
            "intrinsic_lags": list(world.intrinsic_lags),
            "reward_schedule": list(world.reward_schedule),
            "return_upper_bound": list(world.return_upper_bound),
            "action_schema": world.action_spec.public_schema(),
        }

    def influence_edges(self) -> tuple[InfluenceEdge, ...]:
        world = self.world
        config = world.config
        objectives = tuple(range(config.n_objectives))
        edges: set[InfluenceEdge] = set()
        for action_time in range(config.memory_lag, config.horizon):
            maturity, release = _release_time(world, action_time)
            edges.add(InfluenceEdge(action_time, maturity, release, objectives, EffectKind.ADDITIVE))
            if EffectKind.PAIRWISE in config.effects:
                prior = action_time - config.pairwise_gap
                if prior >= config.memory_lag:
                    edges.add(InfluenceEdge(prior, maturity, release, objectives, EffectKind.PAIRWISE))
            if EffectKind.PREREQUISITE in config.effects:
                relative = action_time - config.memory_lag
                anchor = action_time - (relative % config.prerequisite_span)
                if anchor != action_time:
                    edges.add(InfluenceEdge(anchor, maturity, release, objectives, EffectKind.PREREQUISITE))
            if EffectKind.DYNAMICS in config.effects:
                for affected_time in range(action_time + 1, config.horizon):
                    affected_maturity, affected_release = _release_time(
                        world, affected_time
                    )
                    edges.add(
                        InfluenceEdge(
                            action_time,
                            affected_maturity,
                            affected_release,
                            objectives,
                            EffectKind.DYNAMICS,
                        )
                    )
                edges.add(
                    InfluenceEdge(
                        action_time,
                        config.horizon,
                        config.horizon,
                        objectives,
                        EffectKind.DYNAMICS,
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
                key=lambda edge: (edge.action_time, edge.reward_time, edge.mechanism.value),
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
