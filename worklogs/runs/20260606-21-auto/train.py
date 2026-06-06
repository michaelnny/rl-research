"""COPDEV (Per-Step Copula Deviation Policy Gradient).

Implements per-step score-function policy gradient where each step's
gradient is weighted by the absolute difference between two empirical-CDF
ranks of the cumulative-return process: per-channel-1 cumulative reward
versus a survival-rank surrogate for the (degenerate) channel-2 step
penalty on DST-concave.

Vector reward consumed per-component from info["vector"]; channels are
never summed for the training signal.
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


# ---------------- args / CLI ----------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ---------------- policy ----------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    def logp_and_entropy(self, x: torch.Tensor, a: torch.Tensor):
        logits = self.forward(x)
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, a.unsqueeze(-1)).squeeze(-1), logits


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(obs, dtype=np.float32).ravel(), dtype=torch.float32)


# ---------------- per-(t, c) ring buffer of cumulative channel values ----------------


class CumChannelBuffers:
    """Per-time-index ring buffers of cumulative channel-1 values; plus
    a survival counter used to compute the channel-2 survival-rank
    surrogate on DST-concave (where channel 2 is deterministic).
    """

    def __init__(self, t_max: int, n: int = 256):
        self.t_max = int(t_max)
        self.n = int(n)
        self.buf_c1: list[deque] = [deque(maxlen=self.n) for _ in range(self.t_max + 1)]
        # Survival counts: number of historical episodes that survived >= t
        # steps. Stored as raw counts; rank = survival[t] / total_episodes.
        self.survival = np.zeros(self.t_max + 1, dtype=np.int64)
        self.total_episodes = 0

    def cdf_rank_c1(self, t: int, x: float) -> float:
        """Empirical CDF F-hat_t^1(x) = mean[1[buf <= x]] using current buffer."""
        if t < 0 or t > self.t_max:
            return 0.5
        b = self.buf_c1[t]
        if len(b) == 0:
            return 0.5
        arr = np.asarray(b, dtype=np.float64)
        return float(np.mean(arr <= x))

    def cdf_rank_c2_survival(self, t: int) -> float:
        """Survival-rank: fraction of past episodes that survived >= t steps."""
        if self.total_episodes == 0:
            return 0.5
        if t < 0:
            return 1.0
        if t > self.t_max:
            return float(self.survival[self.t_max]) / max(1, self.total_episodes)
        return float(self.survival[t]) / float(self.total_episodes)

    def push_episode(self, M_c1: list[float]):
        """After episode of length T+1 (indices 0..T), push M^1_t for each
        t and bump survival counter for each t in 0..T.
        """
        T = len(M_c1) - 1
        if T < 0:
            return
        self.total_episodes += 1
        for t, val in enumerate(M_c1):
            if t > self.t_max:
                break
            self.buf_c1[t].append(float(val))
        upper = min(T, self.t_max)
        # survived >= t for t = 0..upper
        self.survival[: upper + 1] += 1


# ---------------- training ----------------


def _is_vector_env(env_id: str) -> bool:
    return harness.ENV_TYPE[env_id] == "vector"


def _flatten_obs_dim(obs_space) -> int:
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        n = 1
        for s in obs_space.shape:
            n *= int(s)
        return int(n)
    return 1


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    """COPDEV training loop.

    Score-function policy gradient with per-step weight d_t equal to the
    L1 distance between empirical-CDF ranks of the per-channel cumulative
    return process. Channel-1 uses a per-(t) ring buffer of historical
    cumulative treasure values. Channel-2 uses a survival-rank surrogate
    (survival CDF over historical episode lengths) because channel 2 is
    deterministic on DST-concave.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    obs_space = env.observation_space
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        env.close()
        raise RuntimeError("COPDEV requires discrete action space")
    n_actions = int(action_space.n)
    obs_dim = _flatten_obs_dim(obs_space)

    is_vector = _is_vector_env(env_id)
    if not is_vector:
        # COPDEV's primitive requires bicriterial vector reward; if a
        # scalar env is used, we synthesize a 2-channel decomposition
        # (reward, -1 step penalty) so the algorithm form is preserved.
        # Quick stage uses DST-concave (vector); scalar path is only a
        # safety net if the runner is invoked elsewhere.
        pass

    policy = PolicyNet(obs_dim, n_actions, hidden=64)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-3)

    # Discount and buffers.
    gamma = 0.99
    t_max = harness.MAX_EPISODE_STEPS  # 2000
    # Cap the buffer arrays at a reasonable trajectory length; DST-concave
    # episodes terminate at the treasure or after MAX_EPISODE_STEPS. We
    # use a smaller cap to bound memory and still cover typical lengths.
    t_cap = 256
    buffers = CumChannelBuffers(t_max=t_cap, n=256)

    # Logging file for the discriminating observable.
    import os

    run_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(run_dir, "gradnorm_var.txt")
    log_f = open(log_path, "a")
    log_f.write(
        f"# variant=COPDEV env={env_id} seed={seed} time_budget_s={time_budget_s} t0={time.time():.3f}\n"
    )
    log_f.flush()

    print(
        f"[train] env={env_id} seed={seed} budget_s={time_budget_s} "
        f"obs_dim={obs_dim} n_actions={n_actions} variant=COPDEV",
        flush=True,
    )

    t0 = time.monotonic()
    deadline = t0 + max(1, time_budget_s - 5)  # leave grace for eval
    episode_idx = 0
    total_env_steps = 0

    # Best-so-far weights (for deterministic policy at eval time we use
    # final policy; we just keep current).
    while time.monotonic() < deadline:
        # ---- rollout one episode ----
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        steps = 0
        ep_logps: list[torch.Tensor] = []
        ep_obs_list: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_vec: list[np.ndarray] = []  # per-step vector reward
        ep_lens: list[int] = []  # noqa: F841 (kept for future logging)

        while not done and steps < t_cap:
            x = obs_to_tensor(obs)
            with torch.no_grad():
                logits = policy.forward(x)
                probs = torch.softmax(logits, dim=-1).numpy()
            a = int(rng.choice(n_actions, p=probs))
            ep_obs_list.append(np.asarray(obs, dtype=np.float32).ravel())
            ep_actions.append(a)
            obs, _r, term, trunc, info = env.step(a)
            if is_vector:
                vec = np.asarray(info.get("vector"), dtype=np.float64)
            else:
                # Synthesize 2-channel decomposition: (reward, -1).
                vec = np.array([float(_r), -1.0], dtype=np.float64)
            ep_vec.append(vec)
            done = bool(term) or bool(trunc)
            steps += 1
            total_env_steps += 1

        T = len(ep_actions)  # episode length in steps
        if T == 0:
            continue

        # ---- compute cumulative per-channel values M^c_t for t = 0..T-1 ----
        # M^c_t = sum_{k <= t} gamma^k * r^c_k
        M_c1 = np.zeros(T, dtype=np.float64)
        # M_c2 unused for the rank (we use survival surrogate), but
        # computed for completeness/logging. Channel index 0 = treasure,
        # 1 = step penalty in DST-concave's vector.
        run1 = 0.0
        for k in range(T):
            run1 += (gamma ** k) * float(ep_vec[k][0])
            M_c1[k] = run1

        # ---- compute per-step copula deviation d_t using current buffers ----
        d = np.zeros(T, dtype=np.float64)
        for t in range(T):
            r1 = buffers.cdf_rank_c1(t, M_c1[t])
            r2 = buffers.cdf_rank_c2_survival(t)
            d[t] = abs(r1 - r2)

        # ---- update buffers AFTER computing d to avoid leakage ----
        buffers.push_episode(M_c1.tolist())

        # ---- score-function policy gradient: g = sum_t d_t * grad log pi ----
        obs_batch = torch.as_tensor(np.stack(ep_obs_list), dtype=torch.float32)
        act_batch = torch.as_tensor(np.asarray(ep_actions, dtype=np.int64), dtype=torch.long)
        d_t = torch.as_tensor(d, dtype=torch.float32)

        # Per-step gradient norms: we need ||g_t|| where g_t = d_t * grad_theta log pi(a_t|s_t).
        # Compute per-step log-probs, then for the discriminator measure
        # ||g_t|| = d_t * ||grad_theta log pi(a_t|s_t)||. We compute
        # ||grad log pi|| per step via autograd (small T, small policy).
        logp_per_step, _logits = policy.logp_and_entropy(obs_batch, act_batch)

        # Per-step gradient norms (for logging discriminator).
        per_step_gn: list[float] = []
        # Sample up to a cap of steps to keep this O(T*params); T is
        # small (<= 256) and policy is tiny, so just iterate.
        params = [p for p in policy.parameters() if p.requires_grad]
        for t in range(T):
            grads = torch.autograd.grad(
                logp_per_step[t], params, retain_graph=True, create_graph=False, allow_unused=True
            )
            sq = 0.0
            for g in grads:
                if g is None:
                    continue
                sq += float((g.detach() ** 2).sum().item())
            score_norm = math.sqrt(sq)
            g_norm = float(d[t]) * score_norm
            per_step_gn.append(g_norm)

        # Compute gradnorm_var = Var_t(||g_t||) / Mean_t(||g_t||)^2.
        gn_arr = np.asarray(per_step_gn, dtype=np.float64)
        mean_gn = float(gn_arr.mean()) if T > 0 else 0.0
        var_gn = float(gn_arr.var()) if T > 0 else 0.0
        if mean_gn > 1e-12:
            gradnorm_var = var_gn / (mean_gn ** 2)
        else:
            gradnorm_var = 0.0

        # Now do the actual update step: loss = - sum_t d_t * logp_t / T.
        # Negative sign because optimizer minimizes; we ascend J.
        weighted = (d_t * logp_per_step).sum()
        # Normalize by episode length to keep step magnitudes comparable.
        loss = -weighted / max(1, T)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optim.step()

        # Log discriminator scalar.
        log_f.write(
            f"episode={episode_idx} T={T} mean_d={d.mean():.6f} var_d={d.var():.6f} "
            f"mean_gn={mean_gn:.6e} var_gn={var_gn:.6e} gradnorm_var={gradnorm_var:.6f} "
            f"env_steps={total_env_steps} t={time.monotonic() - t0:.2f}\n"
        )
        if episode_idx % 50 == 0:
            log_f.flush()
        episode_idx += 1

    log_f.flush()
    log_f.close()
    env.close()

    # Build a deterministic policy_fn that argmaxes the policy.
    policy.eval()

    def policy_fn(obs: np.ndarray):
        x = obs_to_tensor(obs)
        with torch.no_grad():
            logits = policy.forward(x)
            a = int(torch.argmax(logits).item())
        return a

    print(
        f"[train] env={env_id} seed={seed} env_steps={total_env_steps} "
        f"episodes={episode_idx} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} variant=COPDEV",
        flush=True,
    )
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
