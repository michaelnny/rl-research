"""Minimal REINFORCE / Monte-Carlo policy-gradient baseline.

This module is optional and imports PyTorch lazily. It is intentionally compact,
CPU-friendly, and aimed at smoke-test baselines for terminal-reward tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rlh_bench.core import Env
from rlh_bench.spaces import Box


@dataclass
class ReinforceResult:
    """Training output for :func:`train_reinforce`."""

    policy: object
    returns: list[float]
    losses: list[float]


def train_reinforce(
    env_factory: Callable[[], Env],
    episodes: int = 100,
    hidden_size: int = 64,
    lr: float = 3e-3,
    gamma: float = 1.0,
    entropy_coef: float = 1e-3,
    seed: int = 0,
) -> ReinforceResult:
    """Train a small Gaussian MLP policy with REINFORCE.

    Supports environments with ``Box`` observation and action spaces. The action
    distribution samples a Gaussian pre-activation and squashes it through tanh
    into the environment action bounds.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "train_reinforce requires torch. Install with `pip install rlh-bench[torch]`."
        ) from exc

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_factory()
    if not isinstance(env.observation_space, Box) or not isinstance(env.action_space, Box):
        raise TypeError("REINFORCE baseline supports Box observation/action spaces only")

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = torch.as_tensor(env.action_space.low.reshape(-1), dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high.reshape(-1), dtype=torch.float32)

    class GaussianMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, act_dim),
            )
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        def act(self, obs_np: np.ndarray):
            obs_t = torch.as_tensor(obs_np.reshape(1, -1), dtype=torch.float32)
            mean = self.net(obs_t).squeeze(0)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            pre_tanh = dist.rsample()
            squashed = torch.tanh(pre_tanh)
            action = action_low + (squashed + 1.0) * 0.5 * (action_high - action_low)
            log_prob = dist.log_prob(pre_tanh).sum()
            entropy = dist.entropy().sum()
            return action.detach().cpu().numpy().reshape(env.action_space.shape), log_prob, entropy

    policy = GaussianMLP()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    returns: list[float] = []
    losses: list[float] = []
    baseline = 0.0

    for ep in range(episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed + ep)
        log_probs = []
        entropies = []
        rewards = []

        while True:
            action, log_prob, entropy = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if isinstance(reward, np.ndarray):
                if hasattr(env, "reward_spec"):
                    scalar_reward = env.reward_spec.scalarize(reward)
                else:
                    scalar_reward = float(np.sum(reward))
            else:
                scalar_reward = float(reward)
            rewards.append(scalar_reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            if terminated or truncated:
                break

        # Monte-Carlo returns. With terminal-only rewards this is the same final
        # outcome assigned backward with optional discounting.
        discounted = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            discounted.append(running)
        discounted.reverse()
        episode_return = float(discounted[0]) if discounted else 0.0
        returns.append(float(sum(rewards)))
        baseline = 0.9 * baseline + 0.1 * episode_return if ep > 0 else episode_return

        returns_t = torch.as_tensor(discounted, dtype=torch.float32)
        advantages = returns_t - float(baseline)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        loss = -(log_probs_t * advantages.detach()).sum() - entropy_coef * entropies_t.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return ReinforceResult(policy=policy, returns=returns, losses=losses)
