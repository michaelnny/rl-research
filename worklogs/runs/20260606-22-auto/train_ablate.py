"""PARGRAD ablation: random-uniform per-step weight.

Replaces the per-step Pareto-rank weight p_t with a fresh per-step
Uniform(0, 1) sample. Preserves: rollout, policy net, optimizer,
hyperparameters, score-function update structure, gradient-norm
logging. Removes: the systematic Pareto-direction of p_t (no ring
buffer is consulted to weight the gradient).

For diagnostic comparability with the candidate, a *shadow* per-t ring
buffer is maintained and a "mean_pt_trend" is logged as the dominance-
count fraction would have been (so the comparison is apples-to-apples
on the discriminating observable). The shadow buffer is NEVER read by
the gradient update, only by the logger.

NOTE (per Reviewer): the unicriterial channel-1-only sanity ablation is
deliberately NOT the canonical ablation here; the random-uniform
ablation is the primary load-bearing test, per Researcher's
ablation_plan.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402


RUN_DIR = Path(__file__).parent.resolve()
LOG_GRADNORM = RUN_DIR / "gradnorm_var.txt"
LOG_MEAN_PT = RUN_DIR / "mean_pt_trend.txt"
LOG_EVERY_EPISODES = 200
LOG_EVERY_SECONDS = 1.0
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB cap each
BUFFER_N = 64  # shadow buffer for diagnostic logging only
T_MAX = harness.MAX_EPISODE_STEPS
GAMMA = 0.99
LR = 1e-2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


def _obs_to_vec(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).ravel()


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.head(h)


def _pareto_rank(buffer: deque, y1: float, y2: float) -> float:
    n = len(buffer)
    if n == 0:
        return 0.5
    dom = 0
    for z1, z2 in buffer:
        if z1 <= y1 and z2 <= y2 and not (z1 == y1 and z2 == y2):
            dom += 1
    return dom / n


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    LOG_GRADNORM.write_text("")
    LOG_MEAN_PT.write_text("")

    rng_np = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 0
    if n_actions == 0:
        env.close()
        raise RuntimeError("PARGRAD ablation requires Discrete action space.")

    obs_reset, _ = env.reset(seed=seed)
    obs_dim = int(_obs_to_vec(obs_reset).shape[0])

    policy = PolicyNet(obs_dim=obs_dim, n_actions=n_actions)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    # Shadow per-t ring buffer for diagnostic Pareto-rank logging only.
    # NEVER consulted by the gradient update (which uses Uniform(0,1)).
    shadow_ring: list[deque] = [deque(maxlen=BUFFER_N) for _ in range(T_MAX + 1)]

    t_start = time.monotonic()
    t_deadline = t_start + max(1, time_budget_s - 5)
    episode_idx = 0
    total_env_steps = 0
    last_log_time = 0.0
    last_logged_episode = -10**9

    while time.monotonic() < t_deadline:
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        vec_buf: list[np.ndarray] = []
        steps_in_ep = 0
        done = False

        obs, _ = env.reset(seed=seed + 7919 * (episode_idx + 1))

        while not done and steps_in_ep < T_MAX:
            obs_vec = _obs_to_vec(obs)
            obs_t = torch.from_numpy(obs_vec).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                probs = F.softmax(logits, dim=-1).cpu().numpy().ravel()
            a = int(rng_np.choice(n_actions, p=probs))
            next_obs, reward, term, trunc, info = env.step(a)
            if is_vector:
                vec_r = np.asarray(info["vector"], dtype=np.float64)
                if vec_r.shape[0] != 2:
                    raise RuntimeError(
                        f"PARGRAD assumes 2-channel vector reward; got {vec_r.shape}"
                    )
            else:
                vec_r = np.array([float(reward), 0.0], dtype=np.float64)

            obs_buf.append(obs_vec)
            act_buf.append(a)
            vec_buf.append(vec_r)

            obs = next_obs
            done = bool(term) or bool(trunc)
            steps_in_ep += 1
            total_env_steps += 1
            if time.monotonic() >= t_deadline:
                break

        T = len(act_buf)
        if T == 0:
            episode_idx += 1
            continue

        # Per-step weight: random uniform per step (THE ablation).
        p_used = rng_np.uniform(0.0, 1.0, size=T).astype(np.float64)

        # Shadow Pareto-rank for logging only (compute against shadow
        # buffer and then push current bivariate cumulants).
        vec_arr = np.stack(vec_buf, axis=0)
        gamma_pow = np.power(GAMMA, np.arange(T))
        M = np.cumsum(vec_arr * gamma_pow[:, None], axis=0)
        p_shadow = np.empty(T, dtype=np.float64)
        for t in range(T):
            p_shadow[t] = _pareto_rank(shadow_ring[t], float(M[t, 0]), float(M[t, 1]))
        for t in range(T):
            shadow_ring[t].append((float(M[t, 0]), float(M[t, 1])))

        obs_t_all = torch.from_numpy(np.stack(obs_buf, axis=0).astype(np.float32))
        act_t_all = torch.from_numpy(np.asarray(act_buf, dtype=np.int64))
        logits_all = policy(obs_t_all)
        log_probs = F.log_softmax(logits_all, dim=-1)
        chosen_lp = log_probs.gather(1, act_t_all.unsqueeze(1)).squeeze(1)

        n_norm = min(T, 32)
        if n_norm >= 2:
            idxs = np.linspace(0, T - 1, n_norm).astype(int)
            norms = []
            params = list(policy.parameters())
            for i in idxs:
                grads = torch.autograd.grad(
                    chosen_lp[int(i)], params, retain_graph=True, allow_unused=True
                )
                sqsum = 0.0
                for g in grads:
                    if g is not None:
                        sqsum += float(g.detach().pow(2).sum().item())
                norms.append(math.sqrt(sqsum))
            arr = np.asarray(norms, dtype=np.float64)
            mean_n = float(arr.mean())
            var_n = float(arr.var())
            gradnorm_var = var_n / (mean_n * mean_n) if mean_n > 1e-12 else 0.0
        else:
            gradnorm_var = 0.0

        weights_t = torch.from_numpy(p_used.astype(np.float32))
        loss = -(weights_t * chosen_lp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Log shadow mean_pt (so ablation/candidate are apples-to-apples
        # on the diagnostic observable).
        mean_pt = float(p_shadow.mean())
        now = time.monotonic()
        should_log = (
            (episode_idx - last_logged_episode) >= LOG_EVERY_EPISODES
            and (now - last_log_time) >= LOG_EVERY_SECONDS
        )
        if episode_idx == 0:
            should_log = True
        if should_log:
            try:
                if LOG_GRADNORM.stat().st_size < LOG_MAX_BYTES:
                    with LOG_GRADNORM.open("a") as f:
                        f.write(f"{episode_idx}\t{gradnorm_var:.6e}\n")
            except FileNotFoundError:
                pass
            try:
                if LOG_MEAN_PT.stat().st_size < LOG_MAX_BYTES:
                    with LOG_MEAN_PT.open("a") as f:
                        f.write(f"{episode_idx}\t{mean_pt:.6f}\t{T}\n")
            except FileNotFoundError:
                pass
            last_log_time = now
            last_logged_episode = episode_idx

        episode_idx += 1

    env.close()
    train_s = time.monotonic() - t_start
    print(
        f"[train-ablate] env={env_id} seed={seed} env_steps={total_env_steps} "
        f"episodes={episode_idx} train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    policy.eval()

    def policy_fn(obs_in: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(_obs_to_vec(obs_in)).unsqueeze(0)
            logits = policy(x)
            a = int(torch.argmax(logits, dim=-1).item())
        return a

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
