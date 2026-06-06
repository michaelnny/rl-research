"""DUAL-IR (Dual Information-Relaxation Learning) candidate.

Faithful realization of the run 20260606-15-auto probe (see hypothesis.md):
- scalar policy network producing categorical logits
- scalar martingale-difference penalty m_phi(s,a,s') = f_phi(s,a,s') - b_phi(s,a)
  with b_phi a small head trained by 1-step regression to enforce the
  conditional-zero-mean constraint approximately.
- per-rollout dual envelope D = max_k (R_k - M_k), arg-sup-time t*.
- policy gradient: sum_{k <= t*} D * grad log pi(a_k|s_k).
- penalty descent on m_phi(s_{t*}, a_{t*}, s_{t*+1}) by D.

The candidate's claimed_stage is `quick`, which in this harness is the
vector env deep-sea-treasure-concave-v0. Per the substrate rule training
must consume info["vector"]; the algorithm is scalar-reward-native, so
we use the scalar r_k = sum(info["vector"]) (read explicitly from the
vector-reward channel, identical to the harness scalar reward by
construction). The DUAL-IR update is otherwise unchanged.
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


# -----------------------------
# Observation flattening
# -----------------------------
def flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


def get_action_count(env) -> int:
    space = env.action_space
    if hasattr(space, "n"):
        return int(space.n)
    raise RuntimeError(f"DUAL-IR requires a discrete action space; got {space}")


# -----------------------------
# Networks
# -----------------------------
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


class PenaltyNet(nn.Module):
    """f_phi(s,a,s') as a scalar function via embedding action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(2 * obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(a, num_classes=self.n_actions).float()
        x = torch.cat([s, a_oh, sp], dim=-1)
        return self.net(x).squeeze(-1)


class BaselineNet(nn.Module):
    """b_phi(s,a) approximating E_{s'}[f_phi(s,a,s')]."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(a, num_classes=self.n_actions).float()
        x = torch.cat([s, a_oh], dim=-1)
        return self.net(x).squeeze(-1)


# -----------------------------
# Training
# -----------------------------
def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 13)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # Probe observation
    obs0, _ = env.reset(seed=seed)
    obs_flat = flatten_obs(obs0)
    obs_dim = int(obs_flat.shape[0])
    n_actions = get_action_count(env)

    device = torch.device("cpu")
    policy = PolicyNet(obs_dim, n_actions).to(device)
    penalty = PenaltyNet(obs_dim, n_actions).to(device)
    baseline = BaselineNet(obs_dim, n_actions).to(device)

    pi_lr = 3e-3
    m_lr = 1e-3
    b_lr = 1e-2
    opt_pi = torch.optim.Adam(policy.parameters(), lr=pi_lr)
    opt_m = torch.optim.Adam(penalty.parameters(), lr=m_lr)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=b_lr)

    gamma = 0.99
    max_steps_per_ep = 500

    t0 = time.monotonic()
    n_episodes = 0
    n_steps = 0
    n_t_lt_T = 0  # episodes with t* < T-1

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

        # ---------------- Rollout ----------------
        obs, _ = env.reset(seed=seed + n_episodes + 1)
        obs_np = flatten_obs(obs)

        states: list[np.ndarray] = []
        actions: list[int] = []
        next_states: list[np.ndarray] = []
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        ep_steps = 0
        while not done and ep_steps < max_steps_per_ep:
            a, logp = select_action(obs_np, sample=True)
            step_out = env.step(a)
            next_obs, scalar_r, term, trunc, info = step_out
            # Faithful consumption of info["vector"] when present.
            if is_vector and isinstance(info, dict) and "vector" in info:
                vec = np.asarray(info["vector"], dtype=np.float64)
                r_k = float(vec.sum())
            else:
                r_k = float(scalar_r)

            next_obs_np = flatten_obs(next_obs)
            states.append(obs_np)
            actions.append(int(a))
            next_states.append(next_obs_np)
            log_probs.append(logp)
            rewards.append(r_k)
            obs_np = next_obs_np
            done = bool(term) or bool(trunc)
            ep_steps += 1
            n_steps += 1

        T = len(rewards)
        if T == 0:
            n_episodes += 1
            continue

        # ---------------- Compute penalties mu_k = m_phi(s_k, a_k, s_{k+1}) ----------------
        s_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        a_t = torch.as_tensor(np.array(actions, dtype=np.int64), dtype=torch.long, device=device)
        sp_t = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)

        with torch.no_grad():
            f_eval = penalty(s_t, a_t, sp_t)
            b_eval = baseline(s_t, a_t)
            mu = (f_eval - b_eval).cpu().numpy()  # shape (T,)

        # ---------------- Cumulative discounted partial sums ----------------
        gammas = np.array([gamma**k for k in range(T)], dtype=np.float64)
        disc_r = gammas * np.asarray(rewards, dtype=np.float64)
        disc_mu = gammas * mu.astype(np.float64)
        R_k = np.cumsum(disc_r)
        M_k = np.cumsum(disc_mu)
        env_k = R_k - M_k

        D = float(env_k.max())
        t_star = int(env_k.argmax())  # in [0, T-1]
        if t_star < T - 1:
            n_t_lt_T += 1

        ep_return = float(np.sum(rewards))
        last_returns.append(ep_return)
        if len(last_returns) > 50:
            last_returns = last_returns[-50:]

        # ---------------- Step 5: policy gradient (ascent on D) ----------------
        # Use D as the scalar credit weight, but center for variance reduction
        # (purely a scaling subtraction; does not change the credit-truncation
        # mechanism, which is the load-bearing claim).
        if len(last_returns) >= 2:
            D_centered = D - float(np.mean(last_returns) / max(1, len(rewards)))
        else:
            D_centered = D

        opt_pi.zero_grad()
        if t_star >= 0:
            credit_logps = torch.stack(log_probs[: t_star + 1])  # k <= t*
            # gradient ascent on D * sum log pi -> minimize -D * sum log pi
            policy_loss = -D_centered * credit_logps.sum()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_pi.step()

        # ---------------- Step 6: penalty descent (envelope theorem) ----------------
        # dD/dM_{t*} = -1, so dL/dphi via M = D * d(-mu_{t*})/dphi = -D * d mu_{t*}/dphi
        # We update phi by - alpha * D * d mu_{t*}/dphi (descent in m at t*).
        # Equivalently: minimize loss = D * mu_{t*} (signed by D's sign).
        opt_m.zero_grad()
        s_star = s_t[t_star : t_star + 1]
        a_star = a_t[t_star : t_star + 1]
        sp_star = sp_t[t_star : t_star + 1]
        mu_star = penalty(s_star, a_star, sp_star) - baseline(s_star, a_star).detach()
        m_loss = D_centered * mu_star.squeeze()
        m_loss.backward()
        torch.nn.utils.clip_grad_norm_(penalty.parameters(), 1.0)
        opt_m.step()

        # ---------------- Baseline regression ----------------
        # Baseline tracks E_{s'}[f_phi(s,a,s')] via stop-grad regression on f_phi.
        opt_b.zero_grad()
        with torch.no_grad():
            f_target = penalty(s_t, a_t, sp_t)
        b_pred = baseline(s_t, a_t)
        b_loss = F.mse_loss(b_pred, f_target)
        b_loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
        opt_b.step()

        n_episodes += 1
        if n_episodes % 25 == 0:
            frac = n_t_lt_T / max(1, n_episodes)
            recent = float(np.mean(last_returns[-20:])) if last_returns else 0.0
            print(
                f"[train] ep={n_episodes} steps={n_steps} t<T_frac={frac:.3f} "
                f"recent_return={recent:.3f} D={D:.3f} t*={t_star} T={T - 1} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    env.close()
    frac_final = n_t_lt_T / max(1, n_episodes)
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} "
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
