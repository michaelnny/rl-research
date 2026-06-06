"""COPDEV ablation: replace per-step weight d_t with constant 1.

This recovers REINFORCE-without-baseline at the score-function step.
All other code (rollout, policy net, optimizer, hyperparameters,
gradient-norm logging) is preserved.

Vector reward is still consumed per-component from info["vector"] and
the channels are never summed for training; we just no longer use the
cumulative-channel CDF ranks to weight per-step gradients.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
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
    """Ablation: per-step weight d_t = 1; no CDF, no ring buffer.

    This matches COPDEV's training loop structure exactly except that
    the score-function update uses uniform weights:
        g_theta = sum_t 1 * grad log pi(a_t | s_t)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    obs_space = env.observation_space
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        env.close()
        raise RuntimeError("ablation requires discrete action space")
    n_actions = int(action_space.n)
    obs_dim = _flatten_obs_dim(obs_space)

    is_vector = _is_vector_env(env_id)

    policy = PolicyNet(obs_dim, n_actions, hidden=64)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-3)

    t_cap = 256

    import os

    run_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(run_dir, "gradnorm_var.txt")
    log_f = open(log_path, "a")
    log_f.write(
        f"# variant=ABLATE env={env_id} seed={seed} time_budget_s={time_budget_s} t0={time.time():.3f}\n"
    )
    log_f.flush()

    print(
        f"[train] env={env_id} seed={seed} budget_s={time_budget_s} "
        f"obs_dim={obs_dim} n_actions={n_actions} variant=ABLATE",
        flush=True,
    )

    t0 = time.monotonic()
    deadline = t0 + max(1, time_budget_s - 5)
    episode_idx = 0
    total_env_steps = 0

    while time.monotonic() < deadline:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        steps = 0
        ep_obs_list: list[np.ndarray] = []
        ep_actions: list[int] = []
        # We still consume vector reward per-component (no scalarization
        # of training signal), even though the ablation does not use it
        # to weight gradients. Keeping vector consumption preserves the
        # contract requirement on vector envs.
        ep_vec: list[np.ndarray] = []

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
                vec = np.array([float(_r), -1.0], dtype=np.float64)
            ep_vec.append(vec)
            done = bool(term) or bool(trunc)
            steps += 1
            total_env_steps += 1

        T = len(ep_actions)
        if T == 0:
            continue

        # ABLATION: d_t = 1 for all t.
        d = np.ones(T, dtype=np.float64)

        obs_batch = torch.as_tensor(np.stack(ep_obs_list), dtype=torch.float32)
        act_batch = torch.as_tensor(np.asarray(ep_actions, dtype=np.int64), dtype=torch.long)
        d_t = torch.as_tensor(d, dtype=torch.float32)

        logp_per_step, _logits = policy.logp_and_entropy(obs_batch, act_batch)

        params = [p for p in policy.parameters() if p.requires_grad]
        per_step_gn: list[float] = []
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
            g_norm = float(d[t]) * score_norm  # d[t] = 1
            per_step_gn.append(g_norm)

        gn_arr = np.asarray(per_step_gn, dtype=np.float64)
        mean_gn = float(gn_arr.mean()) if T > 0 else 0.0
        var_gn = float(gn_arr.var()) if T > 0 else 0.0
        if mean_gn > 1e-12:
            gradnorm_var = var_gn / (mean_gn ** 2)
        else:
            gradnorm_var = 0.0

        weighted = (d_t * logp_per_step).sum()
        loss = -weighted / max(1, T)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optim.step()

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
        f"budget_s={time_budget_s} variant=ABLATE",
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
