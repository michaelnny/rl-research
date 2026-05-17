"""PPO baseline (frozen yardstick).

This is the **only** allowed RL algorithm implementation in the project. It is
NOT a candidate; it is the reference the Curator weighs candidates against.

References (every hyperparameter and implementation detail traces back to one
of these):

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
  imports. Reusable runtime/env/eval/logging/checkpoint primitives live in
  ``rl_research.{runtime,envs,evaluate,tb,checkpoints}``; this file holds only
  PPO-specific logic (networks, GAE, clipped-surrogate update).
- Single entrypoint ``train(config)`` that obeys ``docs/contract.md`` §train.py
  contract: the same CLI flags, the same TensorBoard scalars, and a per-seed
  ``result.json``.
- Frozen hyperparameter sets per domain. Documented inline (see ``_HPARAMS``)
  and never re-tuned per benchmark.
- Multi-signal handling: PPO is a scalar-credit method, so on ``minecart-v0``
  the baseline runs with a fixed equal-weight scalarization of the reward
  vector. This is documented as a *known limitation* of the baseline — handling
  vector rewards natively is exactly what a third-family candidate is supposed
  to do better. We log per-channel evaluation returns for evidence.

PPO is also the reference implementation for the contract: any candidate's
``train.py`` must produce the same TensorBoard scalars and a contract-valid
``result.json``. See ``docs/roles/engineer.md``.

Usage (CLI)::

    uv run python -m rl_research.baselines.ppo \\
        --env CartPole-v1 --seed 0 \\
        --total-env-steps 50000 --max-wallclock-s 300 \\
        --logdir lab/baselines/ppo/CartPole-v1/0
"""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl_research.envs import (
    DMControlVecAdapter,
    GymVecAdapter,
    MinecartVecAdapter,
    adapter_family,
    make_atari_env,
    make_classic_env,
)
from rl_research.evaluate import evaluate as eval_policy
from rl_research.runtime import (
    RunningMeanStd,
    WallclockBudget,
    param_checksum,
    parse_train_cli,
    seed_everything,
    write_config_json,
)
from rl_research.tb import EvalCadence, RunLogger

# ---------------------------------------------------------------------------
# Frozen hyperparameter sets per domain.
#
# Four domain buckets, each citing the reference whose values it follows:
#
#   discrete-mlp     — CartPole-v1 (sanity)              [CleanRL ppo.py]
#   continuous-mlp   — Pendulum-v1, humanoid.run         [PPO Table 3] (Mujoco 1M) + §6.1/§6.2 prose for clip/ent / [CleanRL ppo_continuous_action.py]
#   atari-cnn        — ALE/MontezumaRevenge-v5           [PPO Table 5] (§6.4 Atari) base values; n_epochs/vf_coef from [CleanRL ppo_atari.py]
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
    # Source: [PPO Table 5] (Schulman 2017, Sec 6.4 Atari experiments) for the
    # base values: n_envs=8, n_steps=128, lr=2.5e-4 (annealed), gamma=0.99,
    # lambda=0.95, clip=0.1, ent_coef=0.01, minibatch=256 (= 32*8 per Table 5).
    # We deviate from Table 5 on TWO values, matching [CleanRL ppo_atari.py]:
    #   - n_epochs = 4 (Table 5 specifies 3; CleanRL uses 4 - L63)
    #   - vf_coef  = 0.5 (Table 5 specifies c1=1; CleanRL uses 0.5 - L73)
    # CleanRL is the de-facto reference whose published Atari learning curves
    # (e.g. Breakout) we calibrate against, so we cite it rather than retune.
    # batch = 8*128 = 1024, minibatch=256 -> 4 minibatches.
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
    # Source: [PPO Table 3] (Schulman 2017 Appendix A, used for the Sec 6.1
    # MuJoCo 1M-step surrogate-objective sweep): T=2048, lr=3e-4, n_epochs=10,
    # minibatch=64, gamma=0.99, lambda=0.95. clip_coef=0.2 from Sec 6.2 prose
    # ("we used the hyperparameters from the previous section, with eps=0.2");
    # ent_coef=0 from Sec 6.1 prose ("we don't use an entropy bonus"). [CleanRL
    # ppo_continuous_action.py] uses the same set as its continuous default.
    # State-independent diagonal Gaussian policy follows [PPO] (`Normal(mean(s),
    # std)` with std a learned bias) per [37D Sec 2-4 continuous]. n_envs=1
    # matches both the original paper's per-env evaluation and CleanRL --
    # batch_size = 1 * 2048 = 2048; minibatch_size = 64 -> 32 minibatches.
    # Note: [PPO Table 3] does NOT cover Pendulum-v1; the paper benchmarks
    # 7 MuJoCo tasks (HalfCheetah, Hopper, Inverted{Double}Pendulum, Reacher,
    # Swimmer, Walker2d). We reuse the same continuous bucket on Pendulum-v1
    # because it is the canonical CleanRL/SpinningUp config for that env.
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


_DISCRETE_CLASSIC = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"}
_CONTINUOUS_CLASSIC = {"Pendulum-v1", "MountainCarContinuous-v0"}


def _classify_env(env_id: str) -> str:
    """Map env_id → hyperparameter bucket (key into ``_HPARAMS``)."""
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
    checkpoint_every_updates: int = 50
    extra: dict[str, Any] = field(default_factory=dict)


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

    The official ``ppo2`` implementation uses *separate* networks for the
    policy and value functions — sharing a trunk couples their gradients and
    degrades performance, especially on continuous control. We follow the
    canonical separated-trunk design.
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

    ``next_values[t]`` is V(s_{t+1}) where s_{t+1} is the state that would
    have *continued* this episode — i.e., the pre-reset obs at truncation,
    the next obs otherwise. ``terminated[t]`` is 1 only on real terminal
    transitions (zeros the value bootstrap). ``dones[t] = terminated[t] |
    truncated[t]`` breaks the GAE λ chain.

    Refs: [GAE] Schulman 2015; [SB3 OnPolicyAlgorithm._handle_timeout_termination];
    [37D §"time-limit truncation"]. Distinguishing terminated vs truncated
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
# Eval-policy adapters.
# ---------------------------------------------------------------------------


def _make_eval_policy(network: nn.Module, fam: str, device: torch.device):
    """Wrap a PPO network as the ``policy_fn(obs) -> action`` callable expected
    by ``rl_research.evaluate.evaluate``. Discrete envs use argmax of logits;
    continuous envs use the action mean. Atari obs are passed as uint8 (the
    network's ``forward`` divides by 255 internally)."""

    @torch.no_grad()
    def _policy(obs_np: np.ndarray) -> np.ndarray:
        if fam == "gym-atari":
            obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.uint8).unsqueeze(0)
        else:
            obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32).unsqueeze(0)
        if network.continuous:
            mean, _log_std, _ = network(obs_t)
            return mean.squeeze(0).cpu().numpy()
        logits, _ = network(obs_t)
        return np.asarray(int(logits.argmax(-1).item()))

    return _policy


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def train(config: PPOConfig) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    budget = WallclockBudget(config.max_wallclock_s)
    domain = _classify_env(config.env_id)
    fam = adapter_family(config.env_id)
    hp = _HPARAMS[domain]
    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build adapter + network.
    if fam == "gym-classic":
        adapter: Any = GymVecAdapter(make_classic_env(config.env_id), hp.n_envs, config.seed)
        action_space = adapter.single_action_space
        is_discrete = action_space.__class__.__name__ == "Discrete"
        obs_dim = int(np.prod(adapter.single_observation_space.shape))
        if is_discrete:
            network: nn.Module = _MLPDiscreteActorCritic(obs_dim, int(action_space.n)).to(device)
        else:
            network = _MLPContinuousActorCritic(obs_dim, int(np.prod(action_space.shape))).to(
                device
            )
    elif fam == "gym-atari":
        adapter = GymVecAdapter(make_atari_env(config.env_id), hp.n_envs, config.seed)
        is_discrete = True
        network = _CNNDiscreteActorCritic(int(adapter.single_action_space.n)).to(device)
    elif fam == "mo-minecart":
        adapter = MinecartVecAdapter(hp.n_envs, config.seed)
        is_discrete = True
        obs_dim = int(np.prod(adapter.single_observation_space.shape))
        network = _MLPDiscreteActorCritic(obs_dim, int(adapter.single_action_space.n)).to(device)
    elif fam == "dm-control":
        adapter = DMControlVecAdapter("humanoid", "run", hp.n_envs, config.seed)
        is_discrete = False
        # Canonical 64-64 trunk per [37D §4 continuous]; humanoid's 67-dim obs
        # and 21-dim action fit within capacity at this width.
        network = _MLPContinuousActorCritic(adapter.obs_dim, adapter.action_dim, hidden=64).to(
            device
        )
    else:
        raise ValueError(fam)

    optimizer = torch.optim.Adam(network.parameters(), lr=hp.learning_rate, eps=1e-5)
    logger = RunLogger(logdir)
    cadence = EvalCadence(total_env_steps=config.total_env_steps, n_evals=20)

    # Observation/reward normalization for high-dim continuous control.
    # [37D continuous §6-9]: Normalization of Observation, Observation
    # Clipping, Reward Scaling, Reward Clipping. Critical for dm_control
    # humanoid (raw obs/reward scales are not friendly to vanilla PPO).
    # Pendulum learns without it but we apply it uniformly across the
    # continuous-mlp domain for consistency.
    use_norm = domain == "continuous-mlp"
    obs_rms = (
        RunningMeanStd(shape=(int(np.prod(adapter.single_observation_space.shape)),))
        if use_norm and fam == "gym-classic"
        else (
            RunningMeanStd(shape=(adapter.obs_dim,)) if use_norm and fam == "dm-control" else None
        )
    )
    ret_rms = RunningMeanStd(shape=()) if use_norm else None
    discounted_returns = np.zeros(hp.n_envs, dtype=np.float64) if use_norm else None
    obs_clip = 10.0
    rew_clip = 10.0

    def _norm_obs_live(obs_np: np.ndarray) -> np.ndarray:
        """Update obs_rms with the new batch, then normalize.
        Port of gymnasium NormalizeObservation.observation (L531-537) +
        CleanRL's TransformObservation clip to [-10, 10]."""
        if obs_rms is None:
            return obs_np
        obs_rms.update(obs_np)
        return obs_rms.normalize(obs_np, clip=obs_clip)

    def _norm_obs_frozen(obs_np: np.ndarray) -> np.ndarray:
        """Same as _norm_obs_live but does NOT update running stats. Used to
        normalize the truncated state's observation when computing its
        bootstrap value V(s_truncated) — the post-reset obs has already
        updated obs_rms, so the truncated state must use the same stats
        the policy/value just saw."""
        if obs_rms is None:
            return obs_np
        return obs_rms.normalize(obs_np, clip=obs_clip)

    def _norm_reward(rew_np: np.ndarray, terminated_np: np.ndarray) -> np.ndarray:
        """Port of gymnasium NormalizeReward.step (L107-122) + CleanRL's
        TransformReward clip to [-10, 10].

        Canonical formula (note: uses ``terminated``, not ``done = term|trunc``,
        matching the wrapper exactly — preserves the known one-step leak
        across episode boundaries documented in
        https://github.com/openai/gym/pull/3152). Reset is implicit in the
        ``(1 - terminated)`` multiplier."""
        if ret_rms is None or discounted_returns is None:
            return rew_np
        not_term = 1.0 - terminated_np.astype(np.float64)
        discounted_returns[:] = discounted_returns * hp.gamma * not_term + rew_np
        ret_rms.update(discounted_returns.copy())
        normed = rew_np / np.sqrt(ret_rms.var + 1e-8)
        return np.clip(normed, -rew_clip, rew_clip).astype(np.float32)

    initial_checksum = param_checksum(network)
    final_checksum = initial_checksum

    obs = adapter.reset(config.seed)
    if use_norm:
        obs = _norm_obs_live(obs)
    obs_dtype = torch.uint8 if fam == "gym-atari" else torch.float32
    obs_t = torch.as_tensor(obs, device=device, dtype=obs_dtype)

    n_steps = hp.n_steps
    n_envs = hp.n_envs
    batch_size = n_steps * n_envs

    obs_buf_dtype = torch.uint8 if fam == "gym-atari" else torch.float32
    obs_buf = torch.zeros((n_steps, n_envs, *obs.shape[1:]), dtype=obs_buf_dtype, device=device)
    if is_discrete:
        actions_buf = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
    else:
        action_dim = (
            adapter.action_dim
            if fam == "dm-control"
            else int(np.prod(adapter.single_action_space.shape))
        )
        actions_buf = torch.zeros((n_steps, n_envs, action_dim), dtype=torch.float32, device=device)
    log_probs_buf = torch.zeros((n_steps, n_envs), device=device)
    values_buf = torch.zeros((n_steps, n_envs), device=device)
    rewards_buf = torch.zeros((n_steps, n_envs), device=device)
    dones_buf = torch.zeros((n_steps, n_envs), device=device)
    terminated_buf = torch.zeros((n_steps, n_envs), device=device)
    truncation_values_buf = torch.zeros((n_steps, n_envs), device=device)

    global_step = 0
    eval_history: list[tuple[int, float, float]] = []
    eval_per_channel_history: list[tuple[int, np.ndarray]] = []

    status = "completed"
    error_traceback: str | None = None
    last_loss = float("nan")
    final_mean = float("nan")
    final_pc: np.ndarray | None = None
    update_idx = 0

    obs_transform_for_eval = (
        (lambda x: obs_rms.normalize(x, clip=obs_clip)) if obs_rms is not None else None
    )

    def _eval_now(step: int) -> tuple[float, float, np.ndarray | None]:
        return eval_policy(
            config.env_id,
            _make_eval_policy(network, fam, device),
            seed=config.seed,
            n_episodes=hp.eval_episodes,
            obs_transform=obs_transform_for_eval,
        )

    # Initial eval at step 0 (random-init policy).
    try:
        m, s, pc = _eval_now(0)
        logger.log_eval(0, m, s, pc)
        eval_history.append((0, m, s))
        if pc is not None:
            eval_per_channel_history.append((0, pc))
        cadence.force(0)
    except Exception:
        traceback.print_exc()

    try:
        while global_step < config.total_env_steps:
            if budget.expired():
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
                    next_obs = _norm_obs_live(next_obs)
                rewards_buf[step] = torch.as_tensor(rewards, device=device, dtype=torch.float32)
                dones_buf[step] = torch.as_tensor(done.astype(np.float32), device=device)
                terminated_buf[step] = torch.as_tensor(terminated.astype(np.float32), device=device)

                # Bootstrap V(final_obs) for any envs that truncated this step.
                if truncated.any():
                    trunc_idx = np.where(truncated)[0]
                    fo_for_v = final_obs[trunc_idx]
                    if use_norm and fam != "gym-atari":
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
            next_values_buf = torch.zeros_like(values_buf)
            next_values_buf[:-1] = values_buf[1:]
            next_values_buf[-1] = last_values
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

            update_idx += 1
            final_checksum = param_checksum(network)
            logger.log_train(global_step, last_loss)
            logger.log_progress(
                global_step,
                env_steps=global_step,
                wallclock_s=budget.elapsed_s(),
                param_checksum=final_checksum,
            )

            if cadence.maybe_eval(global_step):
                m, s, pc = _eval_now(global_step)
                logger.log_eval(global_step, m, s, pc)
                eval_history.append((global_step, m, s))
                if pc is not None:
                    eval_per_channel_history.append((global_step, pc))
                print(
                    f"[ppo] step={global_step} eval_mean={m:.3f} "
                    f"eval_std={s:.3f} loss={last_loss:.4f} "
                    f"wall={budget.elapsed_s():.1f}s"
                )

            if (
                config.checkpoint_every_updates
                and update_idx % config.checkpoint_every_updates == 0
            ):
                from rl_research.checkpoints import save_checkpoint

                save_checkpoint(
                    logdir,
                    {
                        "model": network.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step": global_step,
                        "update": update_idx,
                    },
                    global_step,
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
        final_mean, final_std, final_pc = _eval_now(global_step)
        logger.log_eval(global_step, final_mean, final_std, final_pc)
        eval_history.append((global_step, final_mean, final_std))
        if final_pc is not None:
            eval_per_channel_history.append((global_step, final_pc))
    except Exception:
        traceback.print_exc()

    logger.close()
    adapter.close()
    ended_at = datetime.now(UTC)

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
        "wallclock_s": float(budget.elapsed_s()),
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
        from rl_research.runtime import _git_sha

        result_payload = {
            "summary": summary,
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
            "git_sha": _git_sha(),
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


def main() -> None:
    args = parse_train_cli(description="PPO baseline (frozen yardstick).")

    cfg = PPOConfig(
        env_id=args.env,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        max_wallclock_s=args.max_wallclock_s,
        logdir=args.logdir,
    )

    write_config_json(cfg.logdir, args)
    print(json.dumps(train(cfg), indent=2))


__all__ = ["PPOConfig", "train"]


if __name__ == "__main__":
    main()
