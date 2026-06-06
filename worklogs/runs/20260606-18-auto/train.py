"""PRISM: Pareto-Rolling-Imitation Score-function Match.

Per-trajectory weight on REINFORCE-style score-function policy gradient is
the log-density of the realized terminal vector return under a rolling
KDE supported on the Pareto-non-dominated subset of recent terminal
vector returns.

Vector-stage probe. Reads info["vector"] directly from the env.
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


# ---------- defaults ----------
GAMMA = 0.99
WINDOW_N = 64
LR = 3e-3
HIDDEN = 64
WARMUP_EPISODES = 8  # before |F| >= 2 weight = 1.0 anyway, but also for sigma in ablation
EPS = 1e-8


def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr


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
        logits = self.forward(obs)
        return F.log_softmax(logits, dim=-1)


def _pareto_nondominated(points: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-non-dominated points (componentwise strict).

    A point p is dominated iff exists q with q >= p elementwise and q > p
    in at least one coord.
    """
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        p = points[i]
        # q dominates p if all(q >= p) and any(q > p)
        ge = np.all(points >= p, axis=1)
        gt = np.any(points > p, axis=1)
        dominates = ge & gt
        dominates[i] = False
        if dominates.any():
            keep[i] = False
    return np.where(keep)[0]


def _median_heuristic(points: np.ndarray) -> float:
    n = points.shape[0]
    if n < 2:
        return 1.0
    diffs = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(d2[iu])
    dists = dists[dists > 0]
    if dists.size == 0:
        return 1.0
    return float(np.median(dists))


def _log_kde(y: np.ndarray, frontier: np.ndarray, h: float) -> float:
    """Log of mu*(y) = (1/|F|) sum_p N(y; p, h^2 I)."""
    m = y.shape[0]
    diffs = frontier - y[None, :]
    sq = np.sum(diffs * diffs, axis=-1)  # (|F|,)
    # log N(y; p, h^2 I) = -m/2 log(2pi h^2) - ||y-p||^2 / (2h^2)
    log_norm = -0.5 * m * math.log(2.0 * math.pi * h * h + EPS)
    log_kernels = log_norm - sq / (2.0 * h * h + EPS)
    # log (1/|F|) * sum exp(log_kernels) = logsumexp - log|F|
    M = np.max(log_kernels)
    log_sum = M + math.log(np.sum(np.exp(log_kernels - M)) + EPS)
    return float(log_sum - math.log(frontier.shape[0]))


def _make_torch_obs(obs_np: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(_flatten_obs(obs_np)).to(device)


def _vector_dim(env_id: str) -> int:
    # DST 2-d; RG 3-d. Probe target is vector stage.
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    # Sparse fallback (synthetic 2-vector): scalar reward + step marker.
    # NOTE: per Reviewer guidance, this is *not* used in vector-stage runs.
    return 2


def _read_rvec(info: dict, scalar_reward: float, env_id: str) -> np.ndarray:
    if "vector" in info:
        return np.asarray(info["vector"], dtype=np.float64)
    # Sparse fallback only: synthetic vector (scalar reward, -1 step penalty).
    # Vector envs always provide info["vector"], so this branch is unused on
    # the vector stage.
    return np.array([float(scalar_reward), -1.0], dtype=np.float64)


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)

    device = torch.device("cpu")  # small nets, vector envs trivially fit on CPU

    env = harness.make_env(env_id, seed)
    obs_space = env.observation_space
    act_space = env.action_space
    if not hasattr(act_space, "n"):
        env.close()
        raise RuntimeError("PRISM only supports discrete action spaces")
    n_actions = int(act_space.n)
    obs_sample, _ = env.reset(seed=seed)
    obs_dim = int(_flatten_obs(obs_sample).shape[0])

    policy = PolicyNet(obs_dim, n_actions).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=LR)

    m = _vector_dim(env_id)

    window: Deque[np.ndarray] = deque(maxlen=WINDOW_N)
    log_mu_window: Deque[float] = deque(maxlen=WINDOW_N)

    t0 = time.monotonic()
    deadline = t0 + max(1, int(time_budget_s) - 2)
    episodes = 0
    env_steps = 0
    coverage_log: List[int] = []
    last_log_t = t0

    obs_np, _ = env.reset(seed=seed)
    while time.monotonic() < deadline:
        # ----- rollout -----
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
                # Defensive: treat env errors as terminal
                term, trunc = True, False
                next_obs = obs_np
                reward = 0.0
                info = {}
            r_vec = _read_rvec(info, reward, env_id)
            if r_vec.shape[0] != m:
                # If env disagrees, pad/truncate to m to avoid crashes
                if r_vec.shape[0] < m:
                    r_vec = np.concatenate([r_vec, np.zeros(m - r_vec.shape[0])])
                else:
                    r_vec = r_vec[:m]
            y_tau += gamma_t * r_vec
            gamma_t *= GAMMA
            obs_np = next_obs
            done = bool(term) or bool(trunc)
            steps += 1
            env_steps += 1

        if done:
            obs_np, _ = env.reset()

        if not log_probs:
            continue

        # ----- update window + frontier -----
        window.append(y_tau.copy())
        W_arr = np.stack(list(window), axis=0)  # (n_w, m)
        idx = _pareto_nondominated(W_arr)
        F_arr = W_arr[idx]
        coverage_n = int(F_arr.shape[0])

        # ----- compute weight -----
        if F_arr.shape[0] < 2:
            w_tau = 1.0
            log_mu_y = 0.0
        else:
            h = _median_heuristic(F_arr) + EPS
            log_mu_y = _log_kde(y_tau, F_arr, h)
            log_mu_window.append(log_mu_y)
            # baseline: running mean of recent log_mu over a window comparable to N
            mean_log_mu = float(np.mean(list(log_mu_window))) if log_mu_window else 0.0
            w_tau = log_mu_y - mean_log_mu
        if F_arr.shape[0] >= 2 and len(log_mu_window) == 0:
            log_mu_window.append(log_mu_y)

        # Clip the weight to prevent rare blow-ups (variance reduction, not
        # algorithm-changing): equivalent to score-function gradient with a
        # finite scale.
        w_clip = float(np.clip(w_tau, -50.0, 50.0))

        # ----- score-function policy gradient -----
        sum_logp = torch.stack(log_probs).sum()
        loss = -w_clip * sum_logp
        optim.zero_grad()
        loss.backward()
        # Gradient clip to keep training stable; does not change update form.
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optim.step()

        episodes += 1
        coverage_log.append(coverage_n)

        # Progress logging (sparse, to avoid spamming).
        if time.monotonic() - last_log_t > 10.0:
            last_log_t = time.monotonic()
            print(
                f"[train] env={env_id} seed={seed} episode={episodes} "
                f"env_steps={env_steps} coverage={coverage_n} "
                f"y_tau={y_tau.tolist()} w_tau={w_clip:.4f}",
                flush=True,
            )

    env.close()

    final_coverage = coverage_log[-1] if coverage_log else 0
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={episodes} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} final_coverage={final_coverage}",
        flush=True,
    )

    # Deterministic greedy policy at evaluation time.
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
