"""DUAL-IR ablation: m_phi is replaced by the constant-zero function.

Per the candidate's ablation plan:
- Replace the martingale-difference penalty m_phi with the constant-zero
  function (m == 0 identically). Remove the penalty network and its
  optimizer.
- Keep the rest of the algorithm: still compute D(tau) = max_k R_k,
  t* = argmax_k R_k, and apply credit-truncated policy gradient
  weighted by D * 1[k <= t*].

On undiscounted monotone-non-decreasing-reward envs (e.g., CartPole
with reward = 1 per step), this reduces to REINFORCE with terminal
undiscounted return as trajectory weight (no baseline). On the quick
stage (DST-concave), the per-step reward signal is non-monotone in
general, but the m=0 ablation still has a degenerate dual envelope in
the sense that no martingale-difference penalty is being learned --
the discriminating empirical observable is the trajectory fraction
P(t* < T-1) and the gap to the full algorithm's learning curve.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Bootstrap: ensure repo root is importable when this file is invoked as a
# script from within worklogs/runs/<run_id>/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


def flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


def get_action_count(env) -> int:
    space = env.action_space
    if hasattr(space, "n"):
        return int(space.n)
    raise RuntimeError(f"DUAL-IR-ABL requires a discrete action space; got {space}")


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
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


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    obs0, _ = env.reset(seed=seed)
    obs_flat = flatten_obs(obs0)
    obs_dim = int(obs_flat.shape[0])
    n_actions = get_action_count(env)

    device = torch.device("cpu")
    policy = PolicyNet(obs_dim, n_actions).to(device)
    pi_lr = 3e-3
    opt_pi = torch.optim.Adam(policy.parameters(), lr=pi_lr)

    gamma = 0.99
    max_steps_per_ep = 500

    t0 = time.monotonic()
    n_episodes = 0
    n_steps = 0
    n_t_lt_T = 0

    last_returns: list[float] = []

    def policy_logits(np_obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np_obs, dtype=torch.float32, device=device).unsqueeze(0)
        return policy(x).squeeze(0)

    def select_action(np_obs: np.ndarray, sample: bool = True) -> tuple[int, torch.Tensor]:
        logits = policy_logits(np_obs)
        if sample:
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=device))
        else:
            a = int(torch.argmax(logits).item())
            logp = F.log_softmax(logits, dim=-1)[a]
        return a, logp

    while True:
        elapsed = time.monotonic() - t0
        if elapsed > time_budget_s - 5.0:
            break

        obs, _ = env.reset(seed=seed + n_episodes + 1)
        obs_np = flatten_obs(obs)

        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        ep_steps = 0
        while not done and ep_steps < max_steps_per_ep:
            a, logp = select_action(obs_np, sample=True)
            next_obs, scalar_r, term, trunc, info = env.step(a)
            if is_vector and isinstance(info, dict) and "vector" in info:
                vec = np.asarray(info["vector"], dtype=np.float64)
                r_k = float(vec.sum())
            else:
                r_k = float(scalar_r)

            log_probs.append(logp)
            rewards.append(r_k)
            obs_np = flatten_obs(next_obs)
            done = bool(term) or bool(trunc)
            ep_steps += 1
            n_steps += 1

        T = len(rewards)
        if T == 0:
            n_episodes += 1
            continue

        # m == 0 -> M_k == 0 -> D = max_k R_k, t* = argmax_k R_k
        gammas = np.array([gamma**k for k in range(T)], dtype=np.float64)
        disc_r = gammas * np.asarray(rewards, dtype=np.float64)
        R_k = np.cumsum(disc_r)
        env_k = R_k  # M_k = 0 by ablation

        D = float(env_k.max())
        t_star = int(env_k.argmax())
        if t_star < T - 1:
            n_t_lt_T += 1

        ep_return = float(np.sum(rewards))
        last_returns.append(ep_return)
        if len(last_returns) > 50:
            last_returns = last_returns[-50:]

        if len(last_returns) >= 2:
            D_centered = D - float(np.mean(last_returns) / max(1, len(rewards)))
        else:
            D_centered = D

        opt_pi.zero_grad()
        if t_star >= 0:
            credit_logps = torch.stack(log_probs[: t_star + 1])
            policy_loss = -D_centered * credit_logps.sum()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_pi.step()

        n_episodes += 1
        if n_episodes % 25 == 0:
            frac = n_t_lt_T / max(1, n_episodes)
            recent = float(np.mean(last_returns[-20:])) if last_returns else 0.0
            print(
                f"[train-abl] ep={n_episodes} steps={n_steps} t<T_frac={frac:.3f} "
                f"recent_return={recent:.3f} D={D:.3f} t*={t_star} T={T - 1} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    env.close()
    frac_final = n_t_lt_T / max(1, n_episodes)
    print(
        f"[train-abl] env={env_id} seed={seed} env_steps={n_steps} "
        f"episodes={n_episodes} P(t*<T-1)={frac_final:.3f} "
        f"train_s={time.monotonic() - t0:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    policy.eval()

    def policy_fn(np_obs: np.ndarray):
        x = torch.as_tensor(flatten_obs(np_obs), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(x).squeeze(0)
            return int(torch.argmax(logits).item())

    return policy_fn


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
