"""TEAR ablation: replace the trajectory-empirical backward adjoint with
random i.i.d. Gaussian vectors per step.

Per `## Ablation plan` in hypothesis.md:
  1. Skip the Jacobian step and the backward recursion.
  2. At each step t, sample tilde_lambda_t ~ N(0, I_k) i.i.d.
  3. Compute H_t = <r_vec_t, 1_m> + <tilde_lambda_{t+1}, J_t * onehot(a_t)_proj>
     (we keep J_t in the action-projection inner product, the same form as
     the candidate, so the change is exclusively the lambda primitive).
  4. Apply the same score-function update.

This isolates the load-bearing primitive (the per-trajectory backward
adjoint co-state) from the rest of the algorithm.
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
import torch.nn.functional as F

import harness


# ---------------- Feature map (same as candidate) ---------------- #


class RandomProjection:
    def __init__(self, obs_dim: int, k: int, rng: np.random.Generator) -> None:
        self.obs_dim = obs_dim
        self.k = k
        self.W = rng.standard_normal((obs_dim, k)).astype(np.float32) / np.sqrt(obs_dim)
        self.b = rng.standard_normal(k).astype(np.float32) * 0.01

    def __call__(self, obs_flat: np.ndarray) -> np.ndarray:
        return obs_flat.astype(np.float32) @ self.W + self.b


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


# ---------------- Policy network (same as candidate) ---------------- #


class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def logits(self, obs_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(obs_np, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.forward(x)


def compute_jt_action_proj(
    phi_t: np.ndarray,
    phi_tp1: np.ndarray,
    a_proj: np.ndarray,
    eps: float,
) -> np.ndarray:
    delta = phi_tp1 - phi_t
    denom = float(phi_t @ phi_t) + eps
    return a_proj + delta * (float(phi_t @ a_proj) / denom)


def env_reward_dim(env_id: str) -> int:
    if harness.ENV_TYPE[env_id] == "vector":
        if env_id == "deep-sea-treasure-concave-v0":
            return 2
        if env_id == "resource-gathering-v0":
            return 3
        raise ValueError(f"unknown vector env reward dim: {env_id}")
    return 2


def step_reward_vector(env_id: str, scalar_r: float, info: dict) -> np.ndarray:
    if harness.ENV_TYPE[env_id] == "vector":
        return np.asarray(info["vector"], dtype=np.float32).reshape(-1)
    return np.asarray([float(scalar_r), 1.0], dtype=np.float32)


# ---------------- Training loop (ablation) ---------------- #


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # A separate stream for sampling the iid lambda vectors keeps it
    # well-defined and reproducible.
    lam_rng = np.random.default_rng(seed + 1_000_003)

    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        env.close()
        raise RuntimeError("TEAR currently supports discrete action spaces only.")
    n_actions = int(action_space.n)

    obs0, _ = env.reset(seed=seed)
    obs_flat0 = flatten_obs(obs0)
    obs_dim = int(obs_flat0.shape[0])

    m = env_reward_dim(env_id)
    k = max(8, m, n_actions, obs_dim)

    phi_map = RandomProjection(obs_dim, k, rng)
    P_act = rng.standard_normal((n_actions, k)).astype(np.float32) / np.sqrt(n_actions)

    policy = PolicyNet(obs_dim, n_actions, hidden=64)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    eps_jac = 1e-3
    h_clip = 10.0
    grad_clip = 5.0
    max_episode_steps = harness.MAX_EPISODE_STEPS

    n_episodes = 0
    n_env_steps = 0

    def sample_action(obs_flat: np.ndarray) -> int:
        with torch.no_grad():
            logits = policy.logits(obs_flat).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    while time.monotonic() < deadline:
        obs, _ = env.reset(seed=seed + 7919 * (n_episodes + 1))
        obs_flat = flatten_obs(obs)

        ep_obs: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_phi: list[np.ndarray] = [phi_map(obs_flat)]
        ep_rvec: list[np.ndarray] = []
        done = False
        steps = 0
        while not done and steps < max_episode_steps and time.monotonic() < deadline:
            a = sample_action(obs_flat)
            ep_obs.append(obs_flat)
            ep_actions.append(a)
            obs_next, reward, term, trunc, info = env.step(a)
            r_vec = step_reward_vector(env_id, float(reward), info)
            ep_rvec.append(r_vec)
            obs_flat = flatten_obs(obs_next)
            ep_phi.append(phi_map(obs_flat))
            done = bool(term) or bool(trunc)
            steps += 1
            n_env_steps += 1
        if len(ep_actions) == 0:
            n_episodes += 1
            continue

        T = len(ep_actions)
        phi = np.stack(ep_phi, axis=0)
        rvec = np.zeros((T + 1, m), dtype=np.float32)
        for t in range(T):
            rvec[t] = ep_rvec[t]
        rvec[T] = ep_rvec[-1]

        # ABLATION: lambda is i.i.d. Gaussian per step, no backward recursion.
        lam_random = lam_rng.standard_normal((T + 1, k)).astype(np.float32)

        # Compute H_t the same way, but with lam_random in place of the
        # adjoint co-state.
        H = np.zeros(T, dtype=np.float32)
        for t in range(T):
            jt_aproj = compute_jt_action_proj(phi[t], phi[t + 1], P_act[ep_actions[t]], eps_jac)
            inner = float(lam_random[t + 1] @ jt_aproj)
            r_sum = float(rvec[t].sum())
            H[t] = r_sum + inner

        H_centered = H - float(H.mean())
        H_clipped = np.clip(H_centered, -h_clip, h_clip)

        obs_t = torch.from_numpy(np.stack(ep_obs, axis=0).astype(np.float32))
        a_t = torch.from_numpy(np.asarray(ep_actions, dtype=np.int64))
        H_t = torch.from_numpy(H_clipped.astype(np.float32))

        logits = policy(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)
        log_pi_a = log_probs.gather(1, a_t.unsqueeze(1)).squeeze(1)
        loss = -(H_t * log_pi_a).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step()

        n_episodes += 1

    env.close()

    policy.eval()

    def policy_fn(obs: np.ndarray):
        obs_flat_eval = flatten_obs(obs)
        with torch.no_grad():
            logits = policy.logits(obs_flat_eval).squeeze(0)
        return int(torch.argmax(logits).item())

    train_s = time.monotonic() - t_start
    print(
        f"[train_ablate] env={env_id} seed={seed} env_steps={n_env_steps} "
        f"episodes={n_episodes} train_s={train_s:.1f} budget_s={time_budget_s} k={k}",
        flush=True,
    )
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
