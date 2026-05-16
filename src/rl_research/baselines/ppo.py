"""PPO baseline (frozen yardstick).

This is the **only** allowed RL algorithm implementation in the project. It is
NOT a candidate; it is the reference the Curator weighs candidates against.

References (all hyperparameter choices below trace back to one of these):

  [PPO]      Schulman, Wolski, Dhariwal, Radford, Klimov.
             "Proximal Policy Optimization Algorithms."
             arXiv:1707.06347 (2017). https://arxiv.org/abs/1707.06347
             — the clipped-surrogate objective, value-loss coefficient, and
             original Atari (Table 5) and MuJoCo (Table 3) hyperparameters.
  [GAE]      Schulman, Moritz, Levine, Jordan, Abbeel.
             "High-Dimensional Continuous Control Using Generalized Advantage
             Estimation." arXiv:1506.02438 (2015).
             https://arxiv.org/abs/1506.02438
             — the GAE(λ) advantage estimator used in `_compute_gae`.
  [37D]      Huang, Dossa, Raffin, Kanervisto, Wang.
             "The 37 Implementation Details of PPO." ICLR Blog Track (2022).
             https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
             — orthogonal init (gain √2 trunk, 0.01 policy head, 1 value head),
             advantage normalization, value-function loss clipping, learning-rate
             annealing, Adam ε=1e-5, gradient clipping at 0.5. Implemented here.
  [CleanRL]  Huang et al. "CleanRL: High-quality Single-file Implementations of
             Deep Reinforcement Learning Algorithms." JMLR 23(274) (2022).
             https://github.com/vwxyzjn/cleanrl
             — single-file reference matching [37D]. The discrete-control
             defaults (`ppo.py`) and continuous-control defaults
             (`ppo_continuous_action.py`) are the source of truth for the
             classic-control hyperparameters below.

Implementation notes:

- Implemented from primitives — no `stable_baselines3` / `cleanrl` / `tianshou`
  imports. Uses only `torch`, `numpy`, `gymnasium`, `dm_control`, `mo_gymnasium`,
  `ale_py`, and `tensorboard`. References above are honored algorithmically; no
  source code is copied.
- Single entrypoint `train(config)` that obeys `docs/contract.md` §train.py
  contract: it accepts the same CLI flags, produces the same TensorBoard
  scalars, and writes `result.json`.
- Frozen hyperparameter sets per domain. The hyperparameters are documented
  inline (see `_HPARAMS`) and never re-tuned per benchmark. Each set carries
  an inline citation to the paper / reference repo it came from.
- Multi-signal handling: PPO is a scalar-credit method, so on `minecart-v0` the
  baseline runs with a fixed equal-weight scalarization of the reward vector.
  This is documented as a *known limitation* of the baseline — handling vector
  rewards natively is exactly what a third-family candidate is supposed to do
  better. We log per-channel evaluation returns for evidence.

PPO is also the reference implementation for the contract: any candidate's
`train.py` must produce the same TensorBoard scalars and a contract-valid
`result.json`. See `docs/roles/engineer.md`.

Usage (CLI):

    uv run python -m rl_research.baselines.ppo \\
        --env CartPole-v1 --seed 0 \\
        --total-env-steps 50000 --max-wallclock-s 300 \\
        --logdir lab/baselines/ppo/CartPole-v1/0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Frozen hyperparameter sets per domain.
#
# Four domain buckets, each citing the reference whose values it follows:
#
#   discrete-mlp     — CartPole-v1 (sanity)              [CleanRL ppo.py]
#   continuous-mlp   — Pendulum-v1, humanoid.run         [PPO §Sec 6.1, Table 3] / [CleanRL ppo_continuous_action.py]
#   atari-cnn        — ALE/MontezumaRevenge-v5           [PPO §Sec 6.2, Table 5] / [CleanRL ppo_atari.py]
#   minecart-mlp     — minecart-v0 (MORL)                [CleanRL ppo.py]
#
# Implementation-detail toggles common to all buckets (anneal_lr, advantage
# normalization, value-loss clipping, orthogonal init, Adam ε=1e-5, grad-norm
# 0.5) follow [37D]. We do NOT retune per benchmark.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Hyperparams:
    n_envs: int
    n_steps: int
    n_epochs: int
    minibatch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    eval_episodes: int = 10


_HPARAMS: dict[str, _Hyperparams] = {
    # CartPole-v1 (and other small discrete classic-control envs).
    # Source: [CleanRL ppo.py] default config — the well-tested reference for
    # discrete classic-control PPO. See also [37D] §"PPO on classic control".
    # batch_size = n_envs * n_steps = 4 * 128 = 512; minibatch_size = 128 → 4 minibatches.
    "discrete-mlp": _Hyperparams(
        n_envs=4,
        n_steps=128,
        n_epochs=4,
        minibatch_size=128,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    ),
    # Atari (Montezuma's Revenge and other ALE envs).
    # Source: [PPO] Table 5 (Sec 6.2 Atari experiments). [CleanRL ppo_atari.py]
    # mirrors these. clip=0.1 and ent_coef=0.01 are the Atari-specific values
    # from [PPO]; n_envs=8 * n_steps=128 = batch 1024, minibatch 256 -> 4 mb.
    "atari-cnn": _Hyperparams(
        n_envs=8,
        n_steps=128,
        n_epochs=4,
        minibatch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
    ),
    # Continuous control: Pendulum-v1 (sanity) and dm_control humanoid.run.
    # Source: [PPO] Table 3 (Sec 6.1 MuJoCo), n_envs=1, n_steps=2048,
    # n_epochs=10, minibatch=64, lr=3e-4, ent_coef=0. [CleanRL
    # ppo_continuous_action.py L46-72] uses the exact same set as its
    # continuous default. State-independent diagonal Gaussian policy follows
    # [PPO] (`Normal(mean(s), std)` with std a learned bias) per [37D §2-4
    # continuous]. n_envs=1 matches both the original paper and CleanRL —
    # batch_size = 1 * 2048 = 2048; minibatch_size = 64 → 32 minibatches.
    "continuous-mlp": _Hyperparams(
        n_envs=1,
        n_steps=2048,
        n_epochs=10,
        minibatch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
    ),
    # MORL minecart-v0 with equal-weight scalarization.
    # Source: [CleanRL ppo.py] discrete defaults verbatim (ent_coef=0.01 is
    # the canonical discrete value, not "bumped"). No canonical PPO
    # benchmark exists for minecart — this is a documented baseline
    # limitation (see module docstring "Multi-signal").
    "minecart-mlp": _Hyperparams(
        n_envs=4,
        n_steps=128,
        n_epochs=4,
        minibatch_size=128,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    ),
}


# ---------------------------------------------------------------------------
# Public config.
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    env_id: str
    seed: int
    total_env_steps: int
    max_wallclock_s: int
    logdir: str
    record_result_json: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Env classification.
# ---------------------------------------------------------------------------


_DISCRETE_CLASSIC = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"}
_CONTINUOUS_CLASSIC = {"Pendulum-v1", "MountainCarContinuous-v0"}


def _classify_env(env_id: str) -> str:
    """Map env_id → hyperparameter bucket (key into `_HPARAMS`)."""
    if env_id in _DISCRETE_CLASSIC:
        return "discrete-mlp"
    if env_id in _CONTINUOUS_CLASSIC:
        return "continuous-mlp"
    if env_id.startswith("ALE/"):
        return "atari-cnn"
    if env_id == "minecart-v0":
        return "minecart-mlp"
    if env_id == "humanoid.run":
        return "continuous-mlp"
    raise ValueError(f"PPO baseline does not have frozen hyperparameters for env {env_id!r}")


def _classify_adapter(env_id: str) -> str:
    """Map env_id → adapter family. Decoupled from `_classify_env` because, e.g.,
    Pendulum and humanoid.run share the `continuous-mlp` hparam bucket but use
    different env adapters (gymnasium vs dm_control)."""
    if env_id in _DISCRETE_CLASSIC or env_id in _CONTINUOUS_CLASSIC:
        return "gym-classic"
    if env_id.startswith("ALE/"):
        return "gym-atari"
    if env_id == "minecart-v0":
        return "mo-minecart"
    if env_id == "humanoid.run":
        return "dm-control"
    raise ValueError(f"PPO baseline does not have an adapter for env {env_id!r}")


# ---------------------------------------------------------------------------
# Observation + reward normalization.
#
# Implements [37D] §6-9 of "9 details for continuous action domains":
#   §6 normalization of observation, §7 observation clipping to [-10, 10],
#   §8 reward scaling by running discounted-return std, §9 reward clipping
#   to [-10, 10]. Critical for high-dimensional continuous control like
#   dm_control humanoid — without it PPO never escapes the random baseline.
#
# Implementations are line-for-line ports (no copy) of the gymnasium
# canonical wrappers, used by CleanRL's ppo_continuous_action.py:
#   - RunningMeanStd: gymnasium/wrappers/utils.py L31-67
#     (Chan/Welford parallel algorithm; same defaults: count=eps=1e-4,
#      mean=zeros, var=ones).
#   - NormalizeObservation: gymnasium/wrappers/stateful_observation.py L497-537
#     ((obs - mean) / sqrt(var + 1e-8); the [-10, 10] clip comes from
#      CleanRL's separate TransformObservation wrapper).
#   - NormalizeReward: gymnasium/wrappers/stateful_reward.py L107-122
#     (discounted_reward = prev * gamma * (1 - terminated) + reward; rms
#      updated on the new discounted_reward; output = reward / sqrt(var + eps);
#      reset is implicit in the (1 - terminated) multiplier — no separate
#      post-zero. Note the wrapper uses `terminated` only, not `done`; this
#      preserves the known one-step leak across truncation/auto-reset
#      boundaries documented in https://github.com/openai/gym/pull/3152).
# ---------------------------------------------------------------------------


class _RunningMeanStd:
    """Chan/Welford parallel running mean/var.

    Port of gymnasium/wrappers/utils.py:RunningMeanStd (L31-67), which itself
    cites Chan, Golub, LeVeque 1979. Used by [37D §6] for observation
    normalization and [37D §8] for reward scaling.
    """

    def __init__(self, shape: tuple[int, ...] = (), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, batch: np.ndarray) -> None:
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot


# ---------------------------------------------------------------------------
# Vector env adapters.
#
# All adapters expose:
#   reset(seed)  -> obs of shape (n_envs, *obs_shape)
#   step(action) -> (next_obs, reward_scalar, terminated, truncated)
#   pop_completed() -> list of episode returns (float, or np.ndarray for
#                       multi-objective)
#
# We do not use gymnasium's VectorEnv because (a) we mix gym, dm_control, and
# mo_gymnasium; (b) we want explicit auto-reset behavior; (c) Atari's
# preprocessing varies and is easier to wire here.
# ---------------------------------------------------------------------------


class _GymVecAdapter:
    def __init__(self, make_env: Callable[[int], Any], n_envs: int, seed: int):
        self.envs = [make_env(seed + i) for i in range(n_envs)]
        self.n_envs = n_envs
        self.completed: list[float] = []
        self._returns = np.zeros(n_envs, dtype=np.float64)

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space

    @property
    def single_action_space(self):
        return self.envs[0].action_space

    def reset(self, seed: int) -> np.ndarray:
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            obs_list.append(np.asarray(obs))
        self._returns[:] = 0.0
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        next_obs_list = []
        final_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, _ = env.step(actions[i])
            self._returns[i] += float(reward)
            rewards[i] = float(reward)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            if term or trunc:
                self.completed.append(float(self._returns[i]))
                self._returns[i] = 0.0
                final_obs_list.append(np.asarray(obs))
                obs, _ = env.reset()
            else:
                final_obs_list.append(np.asarray(obs))
            next_obs_list.append(np.asarray(obs))
        return (
            np.stack(next_obs_list, axis=0),
            rewards,
            terminated,
            truncated,
            np.stack(final_obs_list, axis=0),
        )

    def pop_completed(self) -> list[float]:
        out, self.completed = self.completed, []
        return out

    def close(self) -> None:
        for env in self.envs:
            env.close()


class _MinecartVecAdapter:
    """mo_gymnasium minecart-v0 with equal-weight scalarization for the PPO update."""

    def __init__(self, n_envs: int, seed: int):
        import mo_gymnasium as mo_gym

        self.envs = [mo_gym.make("minecart-v0") for _ in range(n_envs)]
        for i, env in enumerate(self.envs):
            env.action_space.seed(seed + i)
        self.n_envs = n_envs
        self.completed: list[np.ndarray] = []
        self._returns = np.zeros((n_envs, 3), dtype=np.float64)
        self._scalarization = np.array([1.0, 1.0, 1.0]) / 3.0

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space

    @property
    def single_action_space(self):
        return self.envs[0].action_space

    def reset(self, seed: int) -> np.ndarray:
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            obs_list.append(np.asarray(obs, dtype=np.float32))
        self._returns[:] = 0.0
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        next_obs_list = []
        final_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            obs, reward_vec, term, trunc, _ = env.step(int(actions[i]))
            r_vec = np.asarray(reward_vec, dtype=np.float64)
            self._returns[i] += r_vec
            rewards[i] = float(np.dot(r_vec, self._scalarization))
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            if term or trunc:
                self.completed.append(self._returns[i].copy())
                self._returns[i] = 0.0
                final_obs_list.append(np.asarray(obs, dtype=np.float32))
                obs, _ = env.reset()
            else:
                final_obs_list.append(np.asarray(obs, dtype=np.float32))
            next_obs_list.append(np.asarray(obs, dtype=np.float32))
        return (
            np.stack(next_obs_list, axis=0),
            rewards,
            terminated,
            truncated,
            np.stack(final_obs_list, axis=0),
        )

    def pop_completed(self) -> list[np.ndarray]:
        out, self.completed = self.completed, []
        return out

    def close(self) -> None:
        for env in self.envs:
            env.close()


class _DMControlVecAdapter:
    """Vector adapter for dm_control suite (e.g. humanoid.run)."""

    def __init__(self, domain: str, task: str, n_envs: int, seed: int):
        from dm_control import suite

        self.envs = [
            suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed + i})
            for i in range(n_envs)
        ]
        self.n_envs = n_envs
        self.completed: list[float] = []
        self._returns = np.zeros(n_envs, dtype=np.float64)

        ts = self.envs[0].reset()
        self._obs_dim = int(sum(np.prod(v.shape) for v in ts.observation.values()))
        spec = self.envs[0].action_spec()
        self._action_dim = int(np.prod(spec.shape))
        self._action_low = np.broadcast_to(spec.minimum, spec.shape).astype(np.float32)
        self._action_high = np.broadcast_to(spec.maximum, spec.shape).astype(np.float32)
        # Reset all envs and cache their initial obs for the first reset() call.
        self._cached = np.stack(
            [self._flatten(self.envs[i].reset().observation) for i in range(n_envs)], axis=0
        )

    @staticmethod
    def _flatten(observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(v, dtype=np.float32).ravel() for v in observation.values()]
        )

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high

    def reset(self, seed: int) -> np.ndarray:
        return self._cached

    def step(self, actions: np.ndarray):
        next_obs_list = []
        final_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            action = np.clip(actions[i], self._action_low, self._action_high).astype(np.float32)
            ts = env.step(action)
            r = float(ts.reward) if ts.reward is not None else 0.0
            self._returns[i] += r
            rewards[i] = r
            if ts.last():
                # dm_control humanoid.run has no real terminal — `ts.last()` is the
                # 1000-step time-limit. Treat as truncation so GAE bootstraps with
                # V(final_obs) instead of zero. [37D §"time-limits"] / SB3
                # `_handle_timeout_termination`.
                truncated[i] = True
                self.completed.append(float(self._returns[i]))
                self._returns[i] = 0.0
                final_obs_list.append(self._flatten(ts.observation))
                ts = env.reset()
            else:
                final_obs_list.append(self._flatten(ts.observation))
            next_obs_list.append(self._flatten(ts.observation))
        out = np.stack(next_obs_list, axis=0)
        self._cached = out
        return out, rewards, terminated, truncated, np.stack(final_obs_list, axis=0)

    def pop_completed(self) -> list[float]:
        out, self.completed = self.completed, []
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Networks.
# ---------------------------------------------------------------------------


def _ortho_init(layer: nn.Module, gain: float = math.sqrt(2)) -> None:
    if isinstance(layer, nn.Linear | nn.Conv2d):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class _MLPDiscreteActorCritic(nn.Module):
    """Separate actor and critic MLPs ([37D §13]; CleanRL ppo.py L96-112).

    The official `ppo2` implementation uses *separate* networks for the
    policy and value functions — sharing a trunk couples their gradients
    and degrades performance, especially on continuous control. We follow
    the canonical separated-trunk design.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Trunk gain √2, last-layer gains per [37D §11].
        for m in list(self.actor)[:-1]:
            _ortho_init(m)
        _ortho_init(self.actor[-1], gain=0.01)
        for m in list(self.critic)[:-1]:
            _ortho_init(m)
        _ortho_init(self.critic[-1], gain=1.0)
        self._continuous = False

    @property
    def continuous(self) -> bool:
        return self._continuous

    def forward(self, obs: torch.Tensor):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


class _MLPContinuousActorCritic(nn.Module):
    """Separate actor and critic MLPs with state-independent log-std.

    Matches CleanRL ppo_continuous_action.py L108-125 exactly:
      - Two independent 64-Tanh-64-Tanh trunks
      - actor head outputs action mean (gain 0.01)
      - log_std is a state-independent learned bias (init 0)
      - critic head outputs scalar value (gain 1.0)
    [37D §2-4 continuous] documents this as the canonical PPO continuous
    parameterization.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in list(self.actor_mean)[:-1]:
            _ortho_init(m)
        _ortho_init(self.actor_mean[-1], gain=0.01)
        for m in list(self.critic)[:-1]:
            _ortho_init(m)
        _ortho_init(self.critic[-1], gain=1.0)
        self._continuous = True

    @property
    def continuous(self) -> bool:
        return self._continuous

    def forward(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        log_std = self.actor_logstd.expand_as(mean)
        return mean, log_std, self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        mean, log_std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, log_std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value


class _CNNDiscreteActorCritic(nn.Module):
    """Nature-DQN CNN for Atari (84x84 grayscale, 4-frame stack)."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)
        for m in self.conv:
            _ortho_init(m)
        _ortho_init(self.policy, gain=0.01)
        _ortho_init(self.value, gain=1.0)
        self._continuous = False

    @property
    def continuous(self) -> bool:
        return self._continuous

    def forward(self, obs: torch.Tensor):
        x = obs.float() / 255.0
        h = self.conv(x)
        return self.policy(h), self.value(h).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Env factories for non-vectorized eval.
# ---------------------------------------------------------------------------


def _make_classic(env_id: str):
    def _f(seed: int):
        import gymnasium as gym

        env = gym.make(env_id)
        # ClipAction for continuous Box action spaces — matches CleanRL
        # ppo_continuous_action.py L94. Without this, the unbounded Gaussian
        # samples are clipped silently inside the env, creating a mismatch
        # between executed action and the log-prob used in the importance
        # ratio. [37D §5 continuous].
        if env.action_space.__class__.__name__ == "Box":
            env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed)
        return env

    return _f


def _make_atari(env_id: str):
    def _f(seed: int):
        import ale_py
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

        gym.register_envs(ale_py)
        env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        env = FrameStackObservation(env, stack_size=4)
        env.action_space.seed(seed)
        return env

    return _f


# ---------------------------------------------------------------------------
# GAE.
# ---------------------------------------------------------------------------


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ) with proper truncation handling.

    `next_values[t]` is V(s_{t+1}) where s_{t+1} is the state that would have
    *continued* this episode — i.e., the pre-reset obs at truncation, the
    next obs otherwise. `terminated[t]` is 1 only on real terminal transitions
    (zeros the value bootstrap). `dones[t] = terminated[t] | truncated[t]`
    breaks the GAE λ chain.

    Refs: [GAE] Schulman 2015; [SB3 OnPolicyAlgorithm._handle_timeout_termination];
    [37D §"time-limit truncation"]. Distinguishing terminated vs. truncated
    matters for any env where the time-limit is artificial (Pendulum-v1,
    every dm_control task).
    """
    n_steps, _n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(n_steps)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Eval (single-env, deterministic).
# ---------------------------------------------------------------------------


def _evaluate(
    adapter_family: str,
    env_id: str,
    network: nn.Module,
    seed: int,
    n_episodes: int,
    device: torch.device,
    obs_rms: _RunningMeanStd | None = None,
    obs_clip: float = 10.0,
) -> tuple[float, float, np.ndarray | None]:
    """Single-env deterministic evaluation.

    If `obs_rms` is provided, observations are normalized using its frozen stats
    (no updates during eval) — required when the policy was trained on
    normalized observations, per [37D §3].
    """
    network.eval()
    returns: list[float] = []
    per_channel: list[np.ndarray] = []

    def _maybe_norm(x: np.ndarray) -> np.ndarray:
        if obs_rms is None:
            return x
        normed = (x - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        return np.clip(normed, -obs_clip, obs_clip).astype(np.float32)

    if adapter_family == "gym-classic":
        env = _make_classic(env_id)(seed + 10_000)
    elif adapter_family == "gym-atari":
        env = _make_atari(env_id)(seed + 10_000)
    elif adapter_family == "mo-minecart":
        import mo_gymnasium as mo_gym

        env = mo_gym.make("minecart-v0")
    elif adapter_family == "dm-control":
        from dm_control import suite

        env = suite.load(domain_name="humanoid", task_name="run", task_kwargs={"random": seed + 99})
        action_spec = env.action_spec()
        action_low = np.broadcast_to(action_spec.minimum, action_spec.shape).astype(np.float32)
        action_high = np.broadcast_to(action_spec.maximum, action_spec.shape).astype(np.float32)
    else:
        raise ValueError(adapter_family)

    with torch.no_grad():
        for ep in range(n_episodes):
            if adapter_family == "dm-control":
                ts = env.reset()
                obs = _DMControlVecAdapter._flatten(ts.observation)
            elif adapter_family == "mo-minecart":
                obs, _ = env.reset(seed=seed + 5_000 + ep)
                obs = np.asarray(obs, dtype=np.float32)
                ret_vec = np.zeros(3, dtype=np.float64)
            else:
                obs, _ = env.reset(seed=seed + 5_000 + ep)
                obs = np.asarray(obs)
            done = False
            ret_scalar = 0.0
            steps = 0
            max_steps = 10_000
            while not done and steps < max_steps:
                obs_for_net = obs if adapter_family == "gym-atari" else _maybe_norm(obs)
                if adapter_family == "gym-atari":
                    obs_t = torch.as_tensor(
                        obs_for_net, device=device, dtype=torch.uint8
                    ).unsqueeze(0)
                else:
                    obs_t = torch.as_tensor(
                        obs_for_net, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                if network.continuous:
                    mean, _log_std, _ = network(obs_t)
                    action = mean.squeeze(0).cpu().numpy()
                else:
                    logits, _ = network(obs_t)
                    action = int(logits.argmax(-1).item())

                if adapter_family == "dm-control":
                    a = np.clip(action, action_low, action_high).astype(np.float32)
                    ts = env.step(a)
                    obs = _DMControlVecAdapter._flatten(ts.observation)
                    r = float(ts.reward) if ts.reward is not None else 0.0
                    ret_scalar += r
                    done = ts.last()
                elif adapter_family == "mo-minecart":
                    obs, r_vec, term, trunc, _ = env.step(action)
                    obs = np.asarray(obs, dtype=np.float32)
                    r_vec = np.asarray(r_vec, dtype=np.float64)
                    ret_vec += r_vec
                    ret_scalar += float(r_vec.sum())
                    done = term or trunc
                else:
                    obs, r, term, trunc, _ = env.step(action)
                    obs = np.asarray(obs)
                    ret_scalar += float(r)
                    done = term or trunc
                steps += 1
            returns.append(ret_scalar)
            if adapter_family == "mo-minecart":
                per_channel.append(ret_vec.copy())

    if hasattr(env, "close"):
        env.close()
    network.train()

    arr = np.asarray(returns, dtype=np.float64)
    pc = np.stack(per_channel, axis=0).mean(axis=0) if per_channel else None
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, pc


# ---------------------------------------------------------------------------
# Param checksum (for the Stage A "params actually moved" gate).
# ---------------------------------------------------------------------------


def _param_checksum(network: nn.Module) -> str:
    h = hashlib.sha1()
    for p in network.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def train(config: PPOConfig) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    t0 = time.monotonic()
    domain = _classify_env(config.env_id)
    adapter_family = _classify_adapter(config.env_id)
    hp = _HPARAMS[domain]
    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build adapter + network.
    if adapter_family == "gym-classic":
        adapter: Any = _GymVecAdapter(_make_classic(config.env_id), hp.n_envs, config.seed)
        action_space = adapter.single_action_space
        is_discrete = action_space.__class__.__name__ == "Discrete"
        obs_dim = int(np.prod(adapter.single_observation_space.shape))
        if is_discrete:
            network: nn.Module = _MLPDiscreteActorCritic(obs_dim, int(action_space.n)).to(device)
        else:
            network = _MLPContinuousActorCritic(obs_dim, int(np.prod(action_space.shape))).to(
                device
            )
    elif adapter_family == "gym-atari":
        adapter = _GymVecAdapter(_make_atari(config.env_id), hp.n_envs, config.seed)
        is_discrete = True
        network = _CNNDiscreteActorCritic(int(adapter.single_action_space.n)).to(device)
    elif adapter_family == "mo-minecart":
        adapter = _MinecartVecAdapter(hp.n_envs, config.seed)
        is_discrete = True
        obs_dim = int(np.prod(adapter.single_observation_space.shape))
        network = _MLPDiscreteActorCritic(obs_dim, int(adapter.single_action_space.n)).to(device)
    elif adapter_family == "dm-control":
        adapter = _DMControlVecAdapter("humanoid", "run", hp.n_envs, config.seed)
        is_discrete = False
        # Canonical 64-64 trunk per [37D §4 continuous]; humanoid's 67-dim obs
        # and 21-dim action fit within capacity at this width.
        network = _MLPContinuousActorCritic(adapter.obs_dim, adapter.action_dim, hidden=64).to(
            device
        )
    else:
        raise ValueError(adapter_family)

    optimizer = torch.optim.Adam(network.parameters(), lr=hp.learning_rate, eps=1e-5)
    writer = SummaryWriter(log_dir=str(logdir))

    # Observation/reward normalization for high-dim continuous control.
    # [37D] §3-4: critical for dm_control humanoid (raw obs/reward scales
    # are not friendly to vanilla PPO). Pendulum learns without it but we
    # apply it uniformly across the continuous-mlp domain for consistency.
    use_norm = domain == "continuous-mlp"
    obs_rms = (
        _RunningMeanStd(shape=(int(np.prod(adapter.single_observation_space.shape)),))
        if use_norm and adapter_family == "gym-classic"
        else (
            _RunningMeanStd(shape=(adapter.obs_dim,))
            if use_norm and adapter_family == "dm-control"
            else None
        )
    )
    ret_rms = _RunningMeanStd(shape=()) if use_norm else None
    discounted_returns = np.zeros(hp.n_envs, dtype=np.float64) if use_norm else None
    obs_clip = 10.0
    rew_clip = 10.0

    def _norm_obs(obs_np: np.ndarray) -> np.ndarray:
        """Port of gymnasium NormalizeObservation.observation (L531-537) +
        CleanRL's TransformObservation clip to [-10, 10]."""
        if obs_rms is None:
            return obs_np
        obs_rms.update(obs_np)
        normed = (obs_np - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        return np.clip(normed, -obs_clip, obs_clip).astype(np.float32)

    def _norm_obs_frozen(obs_np: np.ndarray) -> np.ndarray:
        """Same as _norm_obs but does NOT update running stats. Used to
        normalize the truncated state's observation when computing its
        bootstrap value V(s_truncated) — the post-reset obs has already
        updated obs_rms, so the truncated state must use the same stats
        the policy/value just saw."""
        if obs_rms is None:
            return obs_np
        normed = (obs_np - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        return np.clip(normed, -obs_clip, obs_clip).astype(np.float32)

    def _norm_reward(rew_np: np.ndarray, terminated_np: np.ndarray) -> np.ndarray:
        """Port of gymnasium NormalizeReward.step (L107-122) + CleanRL's
        TransformReward clip to [-10, 10].

        Canonical formula (note: uses `terminated`, not `done = term|trunc`,
        matching the wrapper exactly — preserves the known one-step leak
        across episode boundaries documented in
        https://github.com/openai/gym/pull/3152). Reset is implicit in the
        (1 - terminated) multiplier."""
        if ret_rms is None or discounted_returns is None:
            return rew_np
        not_term = 1.0 - terminated_np.astype(np.float64)
        discounted_returns[:] = discounted_returns * hp.gamma * not_term + rew_np
        ret_rms.update(discounted_returns.copy())
        normed = rew_np / np.sqrt(ret_rms.var + 1e-8)
        return np.clip(normed, -rew_clip, rew_clip).astype(np.float32)

    initial_checksum = _param_checksum(network)
    final_checksum = initial_checksum

    obs = adapter.reset(config.seed)
    if use_norm:
        obs = _norm_obs(obs)
    obs_dtype = torch.uint8 if adapter_family == "gym-atari" else torch.float32
    obs_t = torch.as_tensor(obs, device=device, dtype=obs_dtype)

    n_steps = hp.n_steps
    n_envs = hp.n_envs
    batch_size = n_steps * n_envs

    obs_buf_dtype = torch.uint8 if adapter_family == "gym-atari" else torch.float32
    obs_buf = torch.zeros((n_steps, n_envs, *obs.shape[1:]), dtype=obs_buf_dtype, device=device)
    if is_discrete:
        actions_buf = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
    else:
        action_dim = (
            adapter.action_dim
            if adapter_family == "dm-control"
            else int(np.prod(adapter.single_action_space.shape))
        )
        actions_buf = torch.zeros((n_steps, n_envs, action_dim), dtype=torch.float32, device=device)
    log_probs_buf = torch.zeros((n_steps, n_envs), device=device)
    values_buf = torch.zeros((n_steps, n_envs), device=device)
    rewards_buf = torch.zeros((n_steps, n_envs), device=device)
    dones_buf = torch.zeros((n_steps, n_envs), device=device)
    terminated_buf = torch.zeros((n_steps, n_envs), device=device)
    # V(s_truncated) bootstrap values, captured per truncation event. Zero
    # everywhere else; only read at truncation positions when assembling
    # next_values for GAE.
    truncation_values_buf = torch.zeros((n_steps, n_envs), device=device)

    global_step = 0
    last_eval_step = -1
    eval_history: list[tuple[int, float, float]] = []
    eval_per_channel_history: list[tuple[int, np.ndarray]] = []
    eval_interval = max(1, config.total_env_steps // 20)

    status = "completed"
    error_traceback: str | None = None
    last_loss = float("nan")
    final_mean = float("nan")
    final_pc: np.ndarray | None = None

    # Initial eval at step 0 (random-init policy).
    try:
        m, s, pc = _evaluate(
            adapter_family,
            config.env_id,
            network,
            config.seed,
            hp.eval_episodes,
            device,
            obs_rms=obs_rms,
        )
        writer.add_scalar("eval/return_mean", m, 0)
        writer.add_scalar("eval/return_std", s, 0)
        if pc is not None:
            for i, v in enumerate(pc):
                writer.add_scalar(f"eval/return_per_channel/{i}", float(v), 0)
            eval_per_channel_history.append((0, pc))
        eval_history.append((0, m, s))
        last_eval_step = 0
    except Exception:
        traceback.print_exc()

    try:
        while global_step < config.total_env_steps:
            if time.monotonic() - t0 > config.max_wallclock_s:
                status = "killed-budget"
                break

            if hp.anneal_lr:
                frac = 1.0 - global_step / max(1, config.total_env_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = hp.learning_rate * frac

            # Rollout collection.
            for step in range(n_steps):
                with torch.no_grad():
                    action, log_prob, _entropy, value = network.act(obs_t)
                obs_buf[step] = obs_t
                actions_buf[step] = action
                log_probs_buf[step] = log_prob
                values_buf[step] = value

                action_np = action.cpu().numpy()
                next_obs, rewards, terminated, truncated, final_obs = adapter.step(action_np)
                done = terminated | truncated
                if use_norm:
                    rewards = _norm_reward(rewards, terminated)
                    next_obs = _norm_obs(next_obs)
                rewards_buf[step] = torch.as_tensor(rewards, device=device, dtype=torch.float32)
                dones_buf[step] = torch.as_tensor(done.astype(np.float32), device=device)
                terminated_buf[step] = torch.as_tensor(terminated.astype(np.float32), device=device)

                # Bootstrap V(final_obs) for any envs that truncated this step.
                # Truncation is artificial time-limit cutoff (no real terminal),
                # so the GAE TD target needs gamma * V(s_truncated), not zero.
                if truncated.any():
                    trunc_idx = np.where(truncated)[0]
                    fo_for_v = final_obs[trunc_idx]
                    if use_norm and adapter_family != "gym-atari":
                        fo_for_v = _norm_obs_frozen(fo_for_v)
                    fo_t = torch.as_tensor(fo_for_v, device=device, dtype=obs_dtype)
                    with torch.no_grad():
                        if network.continuous:
                            _m, _ls, fo_vals = network(fo_t)
                        else:
                            _logits, fo_vals = network(fo_t)
                    truncation_values_buf[step, torch.as_tensor(trunc_idx, device=device)] = (
                        fo_vals.detach()
                    )

                obs_t = torch.as_tensor(next_obs, device=device, dtype=obs_dtype)
                global_step += n_envs

            with torch.no_grad():
                if network.continuous:
                    _m, _ls, last_values = network(obs_t)
                else:
                    _logits, last_values = network(obs_t)

            # Build next_values[t] = V(s_{t+1}) for the GAE TD target.
            # Default chain uses values shifted by 1 + last_values for the tail.
            # Truncation steps override with V(final_obs) captured during rollout.
            next_values_buf = torch.zeros_like(values_buf)
            next_values_buf[:-1] = values_buf[1:]
            next_values_buf[-1] = last_values
            # truncated = done & !terminated. Both are 0/1 floats.
            truncated_mask = (dones_buf - terminated_buf).clamp(min=0).bool()
            next_values_buf = torch.where(truncated_mask, truncation_values_buf, next_values_buf)

            advantages, returns = _compute_gae(
                rewards_buf,
                values_buf,
                next_values_buf,
                terminated_buf,
                dones_buf,
                hp.gamma,
                hp.gae_lambda,
            )

            b_obs = obs_buf.reshape((batch_size, *obs_buf.shape[2:]))
            b_actions = (
                actions_buf.reshape(batch_size)
                if is_discrete
                else actions_buf.reshape(batch_size, -1)
            )
            b_log_probs = log_probs_buf.reshape(batch_size)
            b_advantages = advantages.reshape(batch_size)
            b_returns = returns.reshape(batch_size)
            b_values = values_buf.reshape(batch_size)

            indices = np.arange(batch_size)
            for _epoch in range(hp.n_epochs):
                rng.shuffle(indices)
                for start in range(0, batch_size, hp.minibatch_size):
                    mb = indices[start : start + hp.minibatch_size]
                    mb_t = torch.as_tensor(mb, device=device)
                    new_log_prob, entropy, new_value = network.evaluate(
                        b_obs[mb_t], b_actions[mb_t]
                    )
                    log_ratio = new_log_prob - b_log_probs[mb_t]
                    ratio = log_ratio.exp()

                    # Per-minibatch advantage normalization ([37D §7] / CleanRL
                    # ppo.py L257; ppo_continuous_action.py L272).
                    mb_adv = b_advantages[mb_t]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - hp.clip_coef, 1 + hp.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_clipped = b_values[mb_t] + torch.clamp(
                        new_value - b_values[mb_t], -hp.clip_coef, hp.clip_coef
                    )
                    v_loss_unclipped = (new_value - b_returns[mb_t]) ** 2
                    v_loss_clipped = (v_clipped - b_returns[mb_t]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + hp.vf_coef * v_loss - hp.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), hp.max_grad_norm)
                    optimizer.step()
                    last_loss = float(loss.detach())

            writer.add_scalar("train/loss", last_loss, global_step)
            writer.add_scalar("progress/env_steps", global_step, global_step)
            writer.add_scalar("progress/wallclock_s", time.monotonic() - t0, global_step)
            final_checksum = _param_checksum(network)
            writer.add_scalar(
                "progress/param_checksum", int(final_checksum, 16) % (2**31), global_step
            )

            if global_step - last_eval_step >= eval_interval:
                m, s, pc = _evaluate(
                    adapter_family,
                    config.env_id,
                    network,
                    config.seed,
                    hp.eval_episodes,
                    device,
                    obs_rms=obs_rms,
                )
                writer.add_scalar("eval/return_mean", m, global_step)
                writer.add_scalar("eval/return_std", s, global_step)
                if pc is not None:
                    for i, v in enumerate(pc):
                        writer.add_scalar(f"eval/return_per_channel/{i}", float(v), global_step)
                    eval_per_channel_history.append((global_step, pc))
                eval_history.append((global_step, m, s))
                last_eval_step = global_step
                print(
                    f"[ppo] step={global_step} eval_mean={m:.3f} "
                    f"eval_std={s:.3f} loss={last_loss:.4f} "
                    f"wall={time.monotonic() - t0:.1f}s"
                )

            if math.isnan(last_loss):
                status = "killed-error"
                error_traceback = "NaN loss detected"
                break

    except Exception:
        status = "killed-error"
        error_traceback = traceback.format_exc()
        print(error_traceback)

    # Final eval.
    try:
        final_mean, final_std, final_pc = _evaluate(
            adapter_family,
            config.env_id,
            network,
            config.seed,
            hp.eval_episodes,
            device,
            obs_rms=obs_rms,
        )
        writer.add_scalar("eval/return_mean", final_mean, global_step)
        writer.add_scalar("eval/return_std", final_std, global_step)
        if final_pc is not None:
            for i, v in enumerate(final_pc):
                writer.add_scalar(f"eval/return_per_channel/{i}", float(v), global_step)
            eval_per_channel_history.append((global_step, final_pc))
        eval_history.append((global_step, final_mean, final_std))
    except Exception:
        traceback.print_exc()

    writer.close()
    adapter.close()
    ended_at = datetime.now(UTC)
    wall_total = time.monotonic() - t0

    if eval_per_channel_history:
        scalarized = [pc.sum() for _step, pc in eval_per_channel_history]
        best_idx = int(np.argmax(scalarized))
        best_return: float | list[float] = eval_per_channel_history[best_idx][1].tolist()
        final_return: float | list[float] = (
            final_pc.tolist() if final_pc is not None else eval_per_channel_history[-1][1].tolist()
        )
    else:
        best_return = max((m for _s, m, _ in eval_history), default=float("nan"))
        final_return = (
            final_mean
            if not math.isnan(final_mean)
            else (eval_history[-1][1] if eval_history else float("nan"))
        )

    summary: dict[str, Any] = {
        "env_id": config.env_id,
        "domain": domain,
        "seed": config.seed,
        "env_steps": int(global_step),
        "wallclock_s": float(wall_total),
        "best_return": best_return,
        "final_return": final_return,
        "status": status,
        "initial_param_checksum": initial_checksum,
        "final_param_checksum": final_checksum,
        "params_changed": initial_checksum != final_checksum,
        "n_evals": len(eval_history),
    }
    if error_traceback:
        summary["error"] = error_traceback

    if config.record_result_json:
        try:
            git_sha = _git_sha()
        except Exception:
            git_sha = "0" * 40
        result_payload = {
            "summary": summary,
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
            "git_sha": git_sha,
            "hparams": {
                k: getattr(hp, k)
                for k in (
                    "n_envs",
                    "n_steps",
                    "n_epochs",
                    "minibatch_size",
                    "learning_rate",
                    "gamma",
                    "gae_lambda",
                    "clip_coef",
                    "vf_coef",
                    "ent_coef",
                    "max_grad_norm",
                    "anneal_lr",
                    "eval_episodes",
                )
            },
        }
        (logdir / "result.json").write_text(json.dumps(result_payload, indent=2, sort_keys=True))

    return summary


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO baseline (frozen yardstick).")
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--max-wallclock-s", type=int, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()

    cfg = PPOConfig(
        env_id=args.env,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        max_wallclock_s=args.max_wallclock_s,
        logdir=args.logdir,
    )
    print(json.dumps(train(cfg), indent=2))


__all__ = ["PPOConfig", "train"]


if __name__ == "__main__":
    main()
