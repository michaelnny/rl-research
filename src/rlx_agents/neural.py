"""Compact neural reference agents for Neural FactorLab.

These are intentionally ordinary neural RL baselines, not proposed research
algorithms.  They establish learnability, headroom, memory sensitivity, and
resource fingerprints under evaluator-measured budgets.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rlx_bench.actions import FactoredDiscreteActionSpec
from rlx_bench.factorlab import FactorLabEnv, FactorLabWorld


@dataclass(frozen=True)
class NeuralReferenceConfig:
    hidden_size: int = 64
    architecture: str = "residual_mlp"
    residual_blocks: int = 2
    transformer_layers: int = 2
    transformer_heads: int = 4
    context_window: int = 32
    learning_rate: float = 3e-4
    gamma: float = 1.0
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    episodes: int = 512
    batch_episodes: int = 16
    optimization_batch_size: int = 8192
    optimization_epochs: int = 4
    ppo_clip: float = 0.2
    truncated_bptt_steps: int = 256
    gradient_clip: float = 1.0
    max_parameters: int = 1_000_000
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.hidden_size < 8:
            raise ValueError("hidden_size must be at least 8")
        if self.architecture not in {"mlp", "residual_mlp", "residual_gru", "transformer"}:
            raise ValueError("unknown compact neural architecture")
        if self.residual_blocks < 1 or self.transformer_layers < 1:
            raise ValueError("neural depth settings must be positive")
        if self.transformer_heads < 1 or self.hidden_size % self.transformer_heads:
            raise ValueError("transformer_heads must divide hidden_size")
        if self.context_window < 2:
            raise ValueError("context_window must be at least two")
        if self.learning_rate <= 0.0 or not 0.0 < self.gamma <= 1.0:
            raise ValueError("optimizer settings must be positive")
        if self.entropy_coefficient < 0.0 or self.value_coefficient < 0.0:
            raise ValueError("loss coefficients cannot be negative")
        if min(
            self.episodes,
            self.batch_episodes,
            self.optimization_batch_size,
            self.optimization_epochs,
            self.truncated_bptt_steps,
        ) < 1:
            raise ValueError("episode counts must be positive")
        if not 0.0 < self.ppo_clip < 1.0:
            raise ValueError("ppo_clip must lie in (0, 1)")
        if self.gradient_clip <= 0.0 or self.max_parameters < 1:
            raise ValueError("gradient and parameter limits must be positive")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")


@dataclass(frozen=True)
class NeuralModelManifest:
    architecture: str
    observation_width: int
    hidden_size: int
    factor_levels: tuple[int, ...]
    trainable_parameters: int
    recurrent: bool
    framework: str
    device: str


@dataclass(frozen=True)
class EpisodeResult:
    return_vector: tuple[float, ...]
    normalized_utility: float
    transitions: int


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: BranchingActorCritic
    manifest: NeuralModelManifest
    episode_utilities: tuple[float, ...]
    losses: tuple[float, ...]
    transitions: int


class ResidualBlock(nn.Module):
    """Pre-normalized compact residual MLP block."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(self.norm(inputs)) / math.sqrt(2.0)


class CausalAttentionBlock(nn.Module):
    """One online multi-head attention block over bounded past tokens."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.feedforward = ResidualBlock(width)

    def forward(self, current: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, length, width = memory.shape
        query = self.query(self.query_norm(current)).view(
            batch, self.heads, self.head_width
        )
        normalized_memory = self.memory_norm(memory)
        keys = self.key(normalized_memory).view(
            batch, length, self.heads, self.head_width
        ).transpose(1, 2)
        values = self.value(normalized_memory).view(
            batch, length, self.heads, self.head_width
        ).transpose(1, 2)
        scores = torch.einsum("bhd,bhld->bhl", query, keys) / math.sqrt(
            self.head_width
        )
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhl,bhld->bhd", attention, values).reshape(
            batch, width
        )
        attended = current + self.output(context) / math.sqrt(2.0)
        return self.feedforward(attended)


class BranchingActorCritic(nn.Module):
    """Shared compact encoder with one categorical head per action factor."""

    def __init__(
        self,
        observation_width: int,
        factor_levels: Sequence[int],
        *,
        hidden_size: int = 64,
        architecture: str = "residual_mlp",
        residual_blocks: int = 2,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        context_window: int = 32,
    ):
        super().__init__()
        if observation_width < 1 or not factor_levels:
            raise ValueError("model dimensions must be positive")
        self.observation_width = int(observation_width)
        self.factor_levels = tuple(int(value) for value in factor_levels)
        self.hidden_size = int(hidden_size)
        self.architecture = architecture
        self.recurrent = architecture in {"residual_gru", "transformer"}
        self.context_window = int(context_window)
        self.input_projection = nn.Linear(self.observation_width, hidden_size)
        if architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh()
            )
        else:
            self.encoder = nn.Sequential(
                *(ResidualBlock(hidden_size) for _ in range(residual_blocks))
            )
        self.gru = (
            nn.GRUCell(hidden_size, hidden_size)
            if architecture == "residual_gru"
            else None
        )
        if architecture == "transformer":
            self.transformer = nn.ModuleList(
                CausalAttentionBlock(hidden_size, transformer_heads)
                for _ in range(transformer_layers)
            )
            self.relative_position = nn.Parameter(
                torch.zeros(context_window, hidden_size)
            )
            nn.init.normal_(self.relative_position, std=0.02)
        else:
            self.transformer = None
            self.register_parameter("relative_position", None)
        self.actor_heads = nn.ModuleList(
            nn.Linear(hidden_size, levels) for levels in self.factor_levels
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.gru is not None:
            return torch.zeros(batch_size, self.hidden_size, device=device)
        if self.transformer is not None:
            return torch.zeros(batch_size, 0, self.hidden_size, device=device)
        return None

    def forward(
        self,
        observation: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor | None]:
        features = self.encoder(self.input_projection(observation))
        next_memory = None
        if self.gru is not None:
            if memory is None:
                memory = torch.zeros_like(features)
            features = self.gru(features, memory)
            next_memory = features
        elif self.transformer is not None:
            if memory is None:
                memory = torch.zeros(
                    features.shape[0], 0, self.hidden_size, device=features.device
                )
            tokens = torch.cat((memory, features.unsqueeze(1)), dim=1)
            tokens = tokens[:, -self.context_window :]
            positioned = tokens + self.relative_position[-tokens.shape[1] :].unsqueeze(0)
            for block in self.transformer:
                features = block(features, positioned)
            next_memory = tokens
        logits = tuple(head(features) for head in self.actor_heads)
        value = self.value_head(features).squeeze(-1)
        return logits, value, next_memory


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _make_model(
    observation_width: int,
    factor_levels: Sequence[int],
    config: NeuralReferenceConfig,
) -> BranchingActorCritic:
    return BranchingActorCritic(
        observation_width,
        factor_levels,
        hidden_size=config.hidden_size,
        architecture=config.architecture,
        residual_blocks=config.residual_blocks,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        context_window=config.context_window,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weights(preference: Sequence[float], width: int) -> np.ndarray:
    values = np.asarray(preference, dtype=np.float64)
    if values.shape != (width,) or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("preference must be a finite non-negative objective vector")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("preference must have positive mass")
    return values / total


def _check_worlds(worlds: Sequence[FactorLabWorld]) -> tuple[FactorLabWorld, ...]:
    values = tuple(worlds)
    if not values:
        raise ValueError("at least one training world is required")
    first = values[0]
    if not isinstance(first.action_spec, FactoredDiscreteActionSpec):
        raise TypeError("branching actor-critic requires factored-discrete actions")
    if any(world.config.task_id != first.config.task_id for world in values):
        raise ValueError("all worlds must share one task configuration")
    if any(world.task_kernel.kernel_id != first.task_kernel.kernel_id for world in values):
        raise ValueError("all worlds must share one suite task kernel")
    return values


def _manifest(
    model: BranchingActorCritic,
    device: torch.device,
) -> NeuralModelManifest:
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return NeuralModelManifest(
        architecture=f"branching_{model.architecture}_actor_critic",
        observation_width=model.observation_width,
        hidden_size=model.hidden_size,
        factor_levels=model.factor_levels,
        trainable_parameters=parameters,
        recurrent=model.recurrent,
        framework=f"torch-{torch.__version__}",
        device=str(device),
    )


def train_actor_critic(
    worlds: Sequence[FactorLabWorld],
    preference: Sequence[float],
    *,
    config: NeuralReferenceConfig = NeuralReferenceConfig(),
    seed: int = 0,
) -> NeuralTrainingResult:
    """Train a compact Monte-Carlo actor-critic on procedural public worlds.

    Rollouts are collected without autograd, then replayed through the current
    policy for one bounded optimization pass. This keeps memory linear and
    bounded at 5k--20k steps instead of retaining a full-horizon computation
    graph while the environment is running.
    """

    worlds = _check_worlds(worlds)
    task = worlds[0].config
    weights = _weights(preference, task.n_objectives)
    device = resolve_device(config.device)
    _seed_everything(seed)
    model = _make_model(task.observation_width, task.levels_per_factor, config).to(device)
    manifest = _manifest(model, device)
    if manifest.trainable_parameters > config.max_parameters:
        raise ValueError(
            f"reference model has {manifest.trainable_parameters} parameters, "
            f"above cap {config.max_parameters}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    upper = np.asarray(worlds[0].return_upper_bound, dtype=np.float64)
    preference_tuple = tuple(float(value) for value in weights)
    episode_utilities: list[float] = []
    losses: list[float] = []
    transitions = 0

    for batch_start in range(0, config.episodes, config.batch_episodes):
        batch_size = min(config.batch_episodes, config.episodes - batch_start)
        batch_worlds = [
            worlds[(batch_start + offset) % len(worlds)] for offset in range(batch_size)
        ]
        environments = [FactorLabEnv(world) for world in batch_worlds]
        observations = [env.reset(preference=preference_tuple)[0] for env in environments]
        memory = model.initial_state(batch_size, device)
        feature_steps = np.empty(
            (task.horizon, batch_size, task.observation_width), dtype=np.float32
        )
        action_steps = np.zeros(
            (task.horizon, batch_size, task.n_factors), dtype=np.int64
        )
        behavior_log_prob_steps = np.zeros(
            (task.horizon, batch_size), dtype=np.float32
        )
        action_masks = np.empty((task.horizon, batch_size), dtype=bool)
        batch_returns = np.zeros((batch_size, task.n_objectives), dtype=np.float64)

        with torch.no_grad():
            for time_index in range(task.horizon):
                features = np.asarray(
                    [item["features"] for item in observations], dtype=np.float32
                )
                feature_steps[time_index] = features
                observation_tensor = torch.as_tensor(features, device=device)
                logits, _, memory = model(observation_tensor, memory)
                sampled = torch.stack(
                    tuple(
                        torch.multinomial(torch.softmax(value, dim=-1), 1).squeeze(1)
                        for value in logits
                    ),
                    dim=1,
                )
                sampled_cpu = sampled.cpu().numpy()
                action_steps[time_index] = sampled_cpu
                behavior_log_prob_steps[time_index] = (
                    torch.stack(
                        tuple(
                            torch.log_softmax(value, dim=-1)
                            .gather(1, sampled[:, factor : factor + 1])
                            .squeeze(1)
                            for factor, value in enumerate(logits)
                        ),
                        dim=1,
                    )
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                )
                required = np.asarray(
                    [item["action_required"] for item in observations], dtype=bool
                )
                action_masks[time_index] = required
                rewards = np.zeros((batch_size, task.n_objectives), dtype=np.float64)
                next_observations: list[dict[str, Any]] = []
                for index, env in enumerate(environments):
                    action: Any = tuple(int(value) for value in sampled_cpu[index])
                    if not required[index]:
                        action = None
                    next_observation, reward, _, _, _ = env.step(action)
                    next_observations.append(next_observation)
                    rewards[index] = reward
                batch_returns += rewards
                observations = next_observations
                transitions += batch_size

        normalized_utilities = (batch_returns / upper) @ weights
        centered = np.empty_like(normalized_utilities)
        for world_id in {world.world_id for world in batch_worlds}:
            indices = np.asarray(
                [world.world_id == world_id for world in batch_worlds], dtype=bool
            )
            centered[indices] = (
                normalized_utilities[indices] - normalized_utilities[indices].mean()
            )
        if float(centered.std()) <= 1e-8:
            centered = normalized_utilities - normalized_utilities.mean()
        scale = float(normalized_utilities.std())
        if scale > 1e-8:
            centered /= scale
        policy_targets = np.broadcast_to(
            centered[None, :], (task.horizon, batch_size)
        )
        value_targets = np.broadcast_to(
            normalized_utilities[None, :], (task.horizon, batch_size)
        )
        valid_actions = int(action_masks.sum())
        if valid_actions < 1:
            raise ValueError("training rollout did not contain any decision steps")

        total_loss = 0.0
        if not model.recurrent:
            flat_features = feature_steps.reshape(-1, task.observation_width)
            flat_actions = action_steps.reshape(-1, task.n_factors)
            flat_masks = action_masks.reshape(-1)
            flat_behavior_log_probs = behavior_log_prob_steps.reshape(-1)
            flat_policy_targets = policy_targets.reshape(-1)
            flat_value_targets = value_targets.reshape(-1)
            for _ in range(config.optimization_epochs):
                optimizer.zero_grad(set_to_none=True)
                epoch_loss = 0.0
                for start in range(
                    0, flat_features.shape[0], config.optimization_batch_size
                ):
                    stop = min(
                        start + config.optimization_batch_size, flat_features.shape[0]
                    )
                    mask = torch.as_tensor(
                        flat_masks[start:stop], dtype=torch.bool, device=device
                    )
                    if not bool(mask.any()):
                        continue
                    features = torch.as_tensor(flat_features[start:stop], device=device)
                    actions = torch.as_tensor(flat_actions[start:stop], device=device)
                    old_log_prob = torch.as_tensor(
                        flat_behavior_log_probs[start:stop], device=device
                    )
                    policy_target = torch.as_tensor(
                        flat_policy_targets[start:stop], dtype=torch.float32, device=device
                    )
                    value_target = torch.as_tensor(
                        flat_value_targets[start:stop], dtype=torch.float32, device=device
                    )
                    logits, values, _ = model(features)
                    distributions = tuple(
                        Categorical(logits=value, validate_args=False) for value in logits
                    )
                    log_prob = torch.stack(
                        tuple(
                            distribution.log_prob(actions[:, factor])
                            for factor, distribution in enumerate(distributions)
                        ),
                        dim=1,
                    ).sum(dim=1)
                    entropy = torch.stack(
                        tuple(distribution.entropy() for distribution in distributions),
                        dim=1,
                    ).mean(dim=1)
                    ratio = torch.exp(log_prob - old_log_prob)
                    unclipped = ratio * policy_target
                    clipped = torch.clamp(
                        ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip
                    ) * policy_target
                    policy_loss = -torch.minimum(unclipped[mask], clipped[mask]).mean()
                    value_loss = 0.5 * (values - value_target).square().mean()
                    entropy_bonus = entropy[mask].mean()
                    chunk_weight = float(mask.sum().item()) / valid_actions
                    loss = chunk_weight * (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy_bonus
                    )
                    loss.backward()
                    epoch_loss += float(loss.detach().cpu())
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                total_loss += epoch_loss / config.optimization_epochs
        else:
            for _ in range(config.optimization_epochs):
                optimizer.zero_grad(set_to_none=True)
                replay_memory = model.initial_state(batch_size, device)
                epoch_loss = 0.0
                for start in range(0, task.horizon, config.truncated_bptt_steps):
                    stop = min(start + config.truncated_bptt_steps, task.horizon)
                    step_log_probs: list[torch.Tensor] = []
                    step_entropies: list[torch.Tensor] = []
                    step_values: list[torch.Tensor] = []
                    for time_index in range(start, stop):
                        features = torch.as_tensor(feature_steps[time_index], device=device)
                        actions = torch.as_tensor(action_steps[time_index], device=device)
                        logits, values, replay_memory = model(features, replay_memory)
                        distributions = tuple(
                            Categorical(logits=value, validate_args=False) for value in logits
                        )
                        step_log_probs.append(
                            torch.stack(
                                tuple(
                                    distribution.log_prob(actions[:, factor])
                                    for factor, distribution in enumerate(distributions)
                                ),
                                dim=1,
                            ).sum(dim=1)
                        )
                        step_entropies.append(
                            torch.stack(
                                tuple(
                                    distribution.entropy()
                                    for distribution in distributions
                                ),
                                dim=1,
                            ).mean(dim=1)
                        )
                        step_values.append(values)
                    mask = torch.as_tensor(
                        action_masks[start:stop], dtype=torch.bool, device=device
                    )
                    old_log_prob = torch.as_tensor(
                        behavior_log_prob_steps[start:stop], device=device
                    )
                    policy_target = torch.as_tensor(
                        policy_targets[start:stop], dtype=torch.float32, device=device
                    )
                    value_target = torch.as_tensor(
                        value_targets[start:stop], dtype=torch.float32, device=device
                    )
                    log_prob = torch.stack(step_log_probs)
                    entropy = torch.stack(step_entropies)
                    values = torch.stack(step_values)
                    ratio = torch.exp(log_prob - old_log_prob)
                    unclipped = ratio * policy_target
                    clipped = torch.clamp(
                        ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip
                    ) * policy_target
                    policy_loss = -torch.minimum(
                        unclipped[mask], clipped[mask]
                    ).mean()
                    value_loss = 0.5 * (values - value_target).square().mean()
                    entropy_bonus = entropy[mask].mean()
                    chunk_weight = float(mask.sum().item()) / valid_actions
                    loss = chunk_weight * (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy_bonus
                    )
                    loss.backward()
                    epoch_loss += float(loss.detach().cpu())
                    if replay_memory is not None:
                        replay_memory = replay_memory.detach()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                total_loss += epoch_loss / config.optimization_epochs

        losses.append(total_loss)
        episode_utilities.extend(normalized_utilities.tolist())

    return NeuralTrainingResult(
        model=model,
        manifest=manifest,
        episode_utilities=tuple(float(value) for value in episode_utilities),
        losses=tuple(losses),
        transitions=transitions,
    )


@torch.no_grad()
def evaluate_actor_critic(
    model: BranchingActorCritic,
    worlds: Sequence[FactorLabWorld],
    preference: Sequence[float],
    *,
    device: str | torch.device = "cpu",
    greedy: bool = True,
    seed: int = 0,
) -> tuple[EpisodeResult, ...]:
    """Evaluate without updates; each world receives fresh recurrent state."""

    worlds = _check_worlds(worlds)
    task = worlds[0].config
    weights = _weights(preference, task.n_objectives)
    target_device = resolve_device(device) if isinstance(device, str) else device
    model = model.to(target_device)
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    del generator  # torch distributions use the process RNG; seed it explicitly below.
    torch.manual_seed(seed)
    preference_tuple = tuple(float(value) for value in weights)
    environments = [FactorLabEnv(world) for world in worlds]
    observations = [
        environment.reset(preference=preference_tuple)[0]
        for environment in environments
    ]
    memory = model.initial_state(len(environments), target_device)
    returns = np.zeros((len(environments), task.n_objectives), dtype=np.float64)
    for _ in range(task.horizon):
        inputs = torch.as_tensor(
            np.asarray([observation["features"] for observation in observations]),
            dtype=torch.float32,
            device=target_device,
        )
        logits, _, memory = model(inputs, memory)
        if greedy:
            sampled = torch.stack(tuple(torch.argmax(head, dim=1) for head in logits), dim=1)
        else:
            sampled = torch.stack(
                tuple(
                    torch.multinomial(torch.softmax(head, dim=-1), 1).squeeze(1)
                    for head in logits
                ),
                dim=1,
            )
        sampled_cpu = sampled.cpu().numpy()
        next_observations = []
        for index, environment in enumerate(environments):
            action = (
                tuple(int(value) for value in sampled_cpu[index])
                if observations[index]["action_required"]
                else None
            )
            observation, reward, _, _, _ = environment.step(action)
            next_observations.append(observation)
            returns[index] += np.asarray(reward)
        observations = next_observations

    results: list[EpisodeResult] = []
    for world, return_values in zip(worlds, returns, strict=True):
        return_vector = tuple(float(value) for value in return_values)
        normalized_utility = float(
            np.dot(weights, np.asarray(return_vector) / np.asarray(world.return_upper_bound))
        )
        results.append(EpisodeResult(return_vector, normalized_utility, task.horizon))
    return tuple(results)


def model_manifest_dict(result: NeuralTrainingResult) -> dict[str, Any]:
    return asdict(result.manifest)
