"""PARGRAD: Per-Step Pareto-Rank Score-Function Gradient.

Update rule (faithful to hypothesis):
- Score-function policy gradient with per-step weight p_t equal to the
  empirical fraction of past bivariate cumulative-return points (M^1_t,
  M^2_t) at the same trajectory time index that are weakly dominated
  with at-least-one-strict by the current rollout's bivariate cumulative
  point. (standard Pareto-dominance, Deb 2002.)
- Per-t ring buffer (deque maxlen=N) of past (M^1_t, M^2_t) bivariate
  pairs. T_max ring buffers total.
- Vector reward consumed per-component from info["vector"]; channels are
  NEVER summed. The cumulative scalar return is not used in the update.

Logged scalars (one line per N=200 episodes or per second wallclock,
whichever is fewer):
- gradnorm_var.txt: Var_t(||g_t||) / Mean_t(||g_t||)^2 per episode.
- mean_pt_trend.txt: per-rollout mean p_t.
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

# Make repo-root importable when this file is invoked directly from a
# nested run directory (run_panel.py sets cwd=ROOT but Python's
# sys.path[0] is the script's own directory).
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
BUFFER_N = 64
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
    arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr


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
    """Weak-dominance with at-least-one-strict count fraction."""
    n = len(buffer)
    if n == 0:
        return 0.5  # symmetric prior
    dom = 0
    for z1, z2 in buffer:
        if z1 <= y1 and z2 <= y2 and not (z1 == y1 and z2 == y2):
            dom += 1
    return dom / n


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    # Reset logs at start of training.
    LOG_GRADNORM.write_text("")
    LOG_MEAN_PT.write_text("")

    rng_np = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 0
    if n_actions == 0:
        env.close()
        raise RuntimeError("PARGRAD requires Discrete action space.")

    obs_reset, _ = env.reset(seed=seed)
    obs_dim = int(_obs_to_vec(obs_reset).shape[0])

    policy = PolicyNet(obs_dim=obs_dim, n_actions=n_actions)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    # Per-t ring buffer of (M^1_t, M^2_t) pairs.
    ring: list[deque] = [deque(maxlen=BUFFER_N) for _ in range(T_MAX + 1)]

    t_start = time.monotonic()
    t_deadline = t_start + max(1, time_budget_s - 5)
    episode_idx = 0
    total_env_steps = 0
    last_log_time = 0.0
    last_logged_episode = -10**9

    obs = obs_reset

    while time.monotonic() < t_deadline:
        # Rollout one episode.
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        vec_buf: list[np.ndarray] = []
        steps_in_ep = 0
        done = False

        # Re-seed reset only at start of each episode (gymnasium auto seed handling).
        obs, _ = env.reset(seed=seed + 7919 * (episode_idx + 1))

        while not done and steps_in_ep < T_MAX:
            obs_vec = _obs_to_vec(obs)
            obs_t = torch.from_numpy(obs_vec).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                probs = F.softmax(logits, dim=-1).cpu().numpy().ravel()
            # Sample
            a = int(rng_np.choice(n_actions, p=probs))
            next_obs, reward, term, trunc, info = env.step(a)
            if is_vector:
                vec_r = np.asarray(info["vector"], dtype=np.float64)
                if vec_r.shape[0] != 2:
                    # PARGRAD assumes m=2.
                    raise RuntimeError(
                        f"PARGRAD assumes 2-channel vector reward; got {vec_r.shape}"
                    )
            else:
                # Fallback: treat scalar reward as single-channel; second
                # channel is zero. This run targets quick-stage (DST) so
                # this branch is not hit, but kept for robustness.
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

        # Cumulative discounted per-channel values.
        vec_arr = np.stack(vec_buf, axis=0)  # (T, 2)
        gamma_pow = np.power(GAMMA, np.arange(T))
        disc = vec_arr * gamma_pow[:, None]  # (T, 2)
        M = np.cumsum(disc, axis=0)  # (T, 2)

        # Per-step Pareto-rank weights p_t against ring buffer; then push.
        p = np.empty(T, dtype=np.float64)
        for t in range(T):
            y1, y2 = float(M[t, 0]), float(M[t, 1])
            p[t] = _pareto_rank(ring[t], y1, y2)
        for t in range(T):
            ring[t].append((float(M[t, 0]), float(M[t, 1])))

        # Score-function gradient with per-step weight p_t.
        # We want per-step gradient norms to log gradnorm_var, so we
        # compute each step's contribution separately.
        obs_t_all = torch.from_numpy(np.stack(obs_buf, axis=0).astype(np.float32))
        act_t_all = torch.from_numpy(np.asarray(act_buf, dtype=np.int64))
        logits_all = policy(obs_t_all)
        log_probs = F.log_softmax(logits_all, dim=-1)
        chosen_lp = log_probs.gather(1, act_t_all.unsqueeze(1)).squeeze(1)  # (T,)

        # Compute per-step gradient norms by autograd in a loop. To keep
        # this cheap we limit to a sample of steps (min(T, 32)) for the
        # gradnorm_var stat. The full update uses all steps.
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

        # Full update: ascent on sum_t p_t * log_pi(a_t|s_t).
        weights_t = torch.from_numpy(p.astype(np.float32))
        loss = -(weights_t * chosen_lp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Logging (rate-limited).
        mean_pt = float(p.mean())
        now = time.monotonic()
        should_log = (
            (episode_idx - last_logged_episode) >= LOG_EVERY_EPISODES
            and (now - last_log_time) >= LOG_EVERY_SECONDS
        )
        # First episode always logged.
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
        f"[train] env={env_id} seed={seed} env_steps={total_env_steps} "
        f"episodes={episode_idx} train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Final policy: deterministic argmax of the trained policy.
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
