"""SNELL ablation: random-threshold truncation.

Per the hypothesis's `## Ablation plan`, we replace the learned
continuation value C_phi with a per-rollout uniform random threshold
theta_random ~ Uniform(0, max_t R_t), and set
tau_random = min{ t : R_t >= theta_random } (default tau_random = T-1).

Everything else (prefix gradient, locked-in-reward weighting,
policy network, optimizer) is identical to train.py. This isolates
the load-bearing primitive: the *learned, predictable, state-conditional*
continuation value.

Contract:
    uv run train_ablate.py --env ENV --seed 0 --time-budget-s 120
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Ensure repo root is importable when this file is run as a script from
# a non-root directory by run_panel.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

import harness


# ---------------------------------------------------------------------------
# Observation / action utilities (identical to train.py)
# ---------------------------------------------------------------------------

def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _obs_dim(env) -> int:
    obs_space = env.observation_space
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        n = 1
        for d in obs_space.shape:
            n *= int(d)
        return n
    raise ValueError(f"unknown observation space: {obs_space!r}")


def _n_actions(env) -> int:
    a = env.action_space
    if hasattr(a, "n"):
        return int(a.n)
    raise ValueError(f"unsupported action space: {a!r}")


# ---------------------------------------------------------------------------
# Networks (policy only; no continuation-value net)
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def _scalar_step_reward(reward: float, info: dict, is_vector: bool) -> float:
    """Match train.py: consume info["vector"] for vector envs and
    take its component sum as the per-step scalar reward."""
    if is_vector:
        vec = np.asarray(info["vector"], dtype=np.float64)
        return float(vec.sum())
    return float(reward)


def collect_episode(
    env,
    policy: PolicyNet,
    is_vector: bool,
    *,
    rng: np.random.Generator,
    max_steps: int,
    device: torch.device,
):
    obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []

    for _ in range(max_steps):
        s = _flatten_obs(obs)
        states.append(s)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(s_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a).squeeze(0))
        action = int(a.item())
        actions.append(action)

        next_obs, reward, term, trunc, info = env.step(action)
        rewards.append(_scalar_step_reward(reward, info, is_vector))
        obs = next_obs
        if bool(term) or bool(trunc):
            break

    return states, actions, rewards, log_probs


def cumulative_discounted(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    g = 1.0
    for i, r in enumerate(rewards):
        running += g * r
        out[i] = running
        g *= gamma
    return out


def random_threshold_stop(R: np.ndarray, rng: np.random.Generator) -> int:
    """Per ablation plan:
        theta_random ~ Uniform(0, max_t R_t)
        tau_random = min{ t : R_t >= theta_random }, default = T-1.

    If max_t R_t <= 0 (e.g. all-negative cumulative on DST), we draw
    Uniform(min_t R_t, max_t R_t) so the threshold is still a sample
    from the observed range — this is the natural extension of the
    plan to envs where cumulative reward is non-positive. The
    structural property (random uniform threshold across observed
    range) is preserved.
    """
    n = len(R)
    if n == 0:
        return 0
    R_max = float(R.max())
    R_min = float(R.min())
    if R_max > 0:
        theta = float(rng.uniform(0.0, R_max))
    elif R_max == R_min:
        theta = R_max  # degenerate: threshold = the single value
    else:
        theta = float(rng.uniform(R_min, R_max))
    for t in range(n):
        if R[t] >= theta:
            return t
    return n - 1


# ---------------------------------------------------------------------------
# Train (ablation)
# ---------------------------------------------------------------------------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    in_dim = _obs_dim(env)
    n_actions = _n_actions(env)

    policy = PolicyNet(in_dim, n_actions).to(device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=3e-3)

    gamma = 0.99
    max_steps = harness.MAX_EPISODE_STEPS
    batch_size = 8

    t0 = time.monotonic()
    end_t = t0 + max(1, time_budget_s - 5)
    n_episodes = 0
    n_steps = 0
    tau_frac_below_one = 0
    tau_frac_total = 0
    last_correlation = float("nan")

    while time.monotonic() < end_t:
        batch_log_probs: list[torch.Tensor] = []
        batch_W: list[float] = []
        batch_taus: list[int] = []
        batch_R_taus: list[float] = []

        for _ in range(batch_size):
            if time.monotonic() >= end_t:
                break
            states, actions, rewards, log_probs = collect_episode(
                env, policy, is_vector, rng=rng, max_steps=max_steps, device=device
            )
            T_ep = len(states)
            if T_ep == 0:
                continue
            n_episodes += 1
            n_steps += T_ep

            R = cumulative_discounted(rewards, gamma)
            tau = random_threshold_stop(R, rng)
            R_tau = float(R[tau])

            batch_taus.append(tau)
            batch_R_taus.append(R_tau)
            tau_frac_total += 1
            if tau < T_ep - 1:
                tau_frac_below_one += 1

            for t in range(tau + 1):
                batch_log_probs.append(log_probs[t])
                batch_W.append(R_tau)

        if not batch_log_probs:
            continue

        W = torch.as_tensor(batch_W, dtype=torch.float32, device=device)
        if W.numel() > 1 and W.std().item() > 1e-8:
            W_norm = (W - W.mean()) / (W.std() + 1e-8)
        else:
            W_norm = W - W.mean() if W.numel() > 0 else W
        log_probs_t = torch.stack(batch_log_probs)
        pi_loss = -(W_norm * log_probs_t).mean()

        opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_pi.step()

        if len(batch_taus) >= 3:
            taus_arr = np.asarray(batch_taus, dtype=np.float64)
            rtau_arr = np.asarray(batch_R_taus, dtype=np.float64)
            if taus_arr.std() > 1e-8 and rtau_arr.std() > 1e-8:
                last_correlation = float(np.corrcoef(taus_arr, rtau_arr)[0, 1])

    env.close()

    fired_frac = (tau_frac_below_one / tau_frac_total) if tau_frac_total > 0 else 0.0
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} "
        f"episodes={n_episodes} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} tau_fired_frac={fired_frac:.3f} "
        f"corr_tau_Rtau={last_correlation:.3f}",
        flush=True,
    )

    policy.eval()

    def policy_fn(obs: np.ndarray):
        s = _flatten_obs(obs)
        with torch.no_grad():
            logits = policy(torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(torch.argmax(logits, dim=-1).item())
        return a

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
    if math.isnan(score):
        score = 0.0
    print("---", flush=True)
    print(f"env:           {args.env}", flush=True)
    print(f"seed:          {args.seed}", flush=True)
    print(f"env_type:      {harness.ENV_TYPE[args.env]}", flush=True)
    print(f"wallclock_s:   {time.monotonic() - t0:.1f}", flush=True)
    print(f"final_score:   {score:.6f}", flush=True)


if __name__ == "__main__":
    main()
