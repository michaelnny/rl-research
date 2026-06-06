"""Ablation of PRISM: replace rolling Pareto-frontier KDE target with a
fixed isotropic Gaussian centered at the origin of vector-return space.

Sigma is estimated from the first 8 random-policy rollouts and frozen.
The per-trajectory weight becomes w_tau = -||y_tau||^2 / (2 sigma^2),
centered by the running mean over a window of size N (same as candidate).

All else (vector-return computation, score-function update, baselining,
hyperparameters) is preserved. The frontier-tracking primitive is what
is removed.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402


# ---------- defaults (must match candidate) ----------
GAMMA = 0.99
WINDOW_N = 64
LR = 3e-3
HIDDEN = 64
WARMUP_EPISODES = 8
EPS = 1e-8


def _flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def action_logprobs(self, obs: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(obs), dim=-1)


def _make_torch_obs(obs_np: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(_flatten_obs(obs_np)).to(device)


def _vector_dim(env_id: str) -> int:
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    return 2


def _read_rvec(info: dict, scalar_reward: float, env_id: str) -> np.ndarray:
    if "vector" in info:
        return np.asarray(info["vector"], dtype=np.float64)
    return np.array([float(scalar_reward), -1.0], dtype=np.float64)


def _rollout(env, policy: PolicyNet, m: int, rng, n_actions: int, env_id: str,
             device, deadline: float, obs_np):
    """One rollout. Returns (log_probs, y_tau, done_flag, next_obs, env_steps)."""
    log_probs: List[torch.Tensor] = []
    y_tau = np.zeros(m, dtype=np.float64)
    gamma_t = 1.0
    steps = 0
    max_steps = harness.MAX_EPISODE_STEPS
    done = False
    while not done and steps < max_steps and time.monotonic() < deadline:
        obs_t = _make_torch_obs(obs_np, device)
        logp = policy.action_logprobs(obs_t.unsqueeze(0)).squeeze(0)
        probs = torch.exp(logp).detach().numpy()
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum()
        action = int(rng.choice(n_actions, p=probs))
        log_probs.append(logp[action])
        try:
            next_obs, reward, term, trunc, info = env.step(action)
        except Exception:
            term, trunc = True, False
            next_obs = obs_np
            reward = 0.0
            info = {}
        r_vec = _read_rvec(info, reward, env_id)
        if r_vec.shape[0] != m:
            if r_vec.shape[0] < m:
                r_vec = np.concatenate([r_vec, np.zeros(m - r_vec.shape[0])])
            else:
                r_vec = r_vec[:m]
        y_tau += gamma_t * r_vec
        gamma_t *= GAMMA
        obs_np = next_obs
        done = bool(term) or bool(trunc)
        steps += 1
    return log_probs, y_tau, done, obs_np, steps


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)

    device = torch.device("cpu")

    env = harness.make_env(env_id, seed)
    act_space = env.action_space
    if not hasattr(act_space, "n"):
        env.close()
        raise RuntimeError("PRISM ablation only supports discrete action spaces")
    n_actions = int(act_space.n)
    obs_sample, _ = env.reset(seed=seed)
    obs_dim = int(_flatten_obs(obs_sample).shape[0])

    policy = PolicyNet(obs_dim, n_actions).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=LR)

    m = _vector_dim(env_id)
    log_w_window: Deque[float] = deque(maxlen=WINDOW_N)

    t0 = time.monotonic()
    deadline = t0 + max(1, int(time_budget_s) - 2)
    episodes = 0
    env_steps = 0
    last_log_t = t0

    obs_np, _ = env.reset(seed=seed)

    # ---- Warmup: collect first 8 rollouts to estimate sigma = std(||y_tau||).
    # During warmup, apply REINFORCE with weight 1.0 (matching the candidate's
    # |F|<2 warmup behavior). After warmup, use the frozen-sigma weight.
    warmup_y: List[np.ndarray] = []

    while len(warmup_y) < WARMUP_EPISODES and time.monotonic() < deadline:
        log_probs, y_tau, done, obs_np, steps = _rollout(
            env, policy, m, rng, n_actions, env_id, device, deadline, obs_np
        )
        env_steps += steps
        if done:
            obs_np, _ = env.reset()
        if not log_probs:
            continue
        warmup_y.append(y_tau.copy())
        episodes += 1

        # Immediate REINFORCE update with weight 1.0 (no backward across optim steps).
        sum_logp = torch.stack(log_probs).sum()
        loss = -1.0 * sum_logp
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optim.step()

    if len(warmup_y) >= 2:
        norms = np.array([np.linalg.norm(y) for y in warmup_y], dtype=np.float64)
        sigma = float(np.std(norms))
        if sigma < 1e-3:
            sigma = float(np.std(np.stack(warmup_y, axis=0))) + 1e-2
    else:
        sigma = 1.0
    sigma = max(sigma, 1e-3)

    print(
        f"[ablate-train] env={env_id} seed={seed} sigma={sigma:.4f} "
        f"warmup_y_norms={[float(np.linalg.norm(y)) for y in warmup_y]}",
        flush=True,
    )

    # ---- Main loop ----
    while time.monotonic() < deadline:
        log_probs, y_tau, done, obs_np, steps = _rollout(
            env, policy, m, rng, n_actions, env_id, device, deadline, obs_np
        )
        env_steps += steps
        if done:
            obs_np, _ = env.reset()
        if not log_probs:
            continue

        # log mu*_ablate(y_tau) up to constant: -||y||^2 / (2 sigma^2)
        log_w = -float(np.dot(y_tau, y_tau)) / (2.0 * sigma * sigma + EPS)
        log_w_window.append(log_w)
        mean_log_w = float(np.mean(list(log_w_window)))
        w_tau = log_w - mean_log_w
        w_clip = float(np.clip(w_tau, -50.0, 50.0))

        sum_logp = torch.stack(log_probs).sum()
        loss = -w_clip * sum_logp
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optim.step()
        episodes += 1

        if time.monotonic() - last_log_t > 10.0:
            last_log_t = time.monotonic()
            print(
                f"[ablate-train] env={env_id} seed={seed} episode={episodes} "
                f"env_steps={env_steps} y_tau={y_tau.tolist()} "
                f"log_w={log_w:.4f} w_tau={w_clip:.4f}",
                flush=True,
            )

    env.close()
    print(
        f"[ablate-train] env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={episodes} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} sigma={sigma:.4f}",
        flush=True,
    )

    policy.eval()

    def policy_fn(obs: np.ndarray):
        with torch.no_grad():
            obs_t = _make_torch_obs(obs, device).unsqueeze(0)
            logits = policy(obs_t).squeeze(0)
            return int(torch.argmax(logits).item())

    return policy_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()
    policy = train(args.env, args.seed, args.time_budget_s)
    score = harness.evaluate(policy, args.env, seed=args.seed)
    print("---", flush=True)
    print(f"env:           {args.env}", flush=True)
    print(f"seed:          {args.seed}", flush=True)
    print(f"env_type:      {harness.ENV_TYPE[args.env]}", flush=True)
    print(f"wallclock_s:   {time.monotonic() - t0:.1f}", flush=True)
    print(f"final_score:   {score:.6f}", flush=True)


if __name__ == "__main__":
    main()
