"""TEAR (Trajectory-Empirical Adjoint Reflection) — candidate.

Implements a per-trajectory backward-linear vector adjoint co-state lambda_t
in R^k, propagated by lambda_T = pad(r_vec_T) and
    lambda_t = pad(r_vec_t) + J_t^T lambda_{t+1},
with J_t the rank-1 trajectory-empirical Jacobian
    J_t = I + (phi_{t+1} - phi_t) phi_t^T / (||phi_t||^2 + eps).

The per-step Hamiltonian
    H_t = <r_vec_t, 1_m> + <lambda_{t+1}, J_t * onehot(a_t)_proj>
is used as a SCALAR score-function weight in
    g_theta = sum_t H_t * grad log pi_theta(a_t|s_t),
    theta <- theta + alpha * g_theta.

No learned critic, no Bellman backup, no learned dynamics. The vector
reward channel is consumed via info["vector"] for vector envs; for scalar
envs we augment r_vec_t = (r_t, 1.0) so the constant time-marker channel
keeps lambda_T != 0.
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


# ---------------- Feature map ---------------- #


class RandomProjection:
    """Fixed random projection from raw observation to phi(s) in R^k.

    For DST/RG (small int vectors) we still wrap a normalization +
    projection: it lets phi(s) live in R^k with k decoupled from the obs
    dim, so the action-onehot projection inhabits the same space.
    """

    def __init__(self, obs_dim: int, k: int, rng: np.random.Generator) -> None:
        self.obs_dim = obs_dim
        self.k = k
        # Slight scaling so phi has O(1) norm.
        self.W = rng.standard_normal((obs_dim, k)).astype(np.float32) / np.sqrt(obs_dim)
        self.b = rng.standard_normal(k).astype(np.float32) * 0.01

    def __call__(self, obs_flat: np.ndarray) -> np.ndarray:
        return obs_flat.astype(np.float32) @ self.W + self.b


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


# ---------------- Policy network ---------------- #


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
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

    def logits(self, obs_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(obs_np, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.forward(x)


# ---------------- TEAR core ---------------- #


def compute_adjoints(
    phi: np.ndarray,            # (T+1, k)
    r_vec: np.ndarray,          # (T+1, m)  reward vec at each step (last is terminal)
    eps: float,
    use_jacobian: bool = True,
    use_terminal: bool = True,
) -> np.ndarray:
    """Backward adjoint recursion. Returns lambda of shape (T+1, k).

    For step t, lambda_t = pad(r_vec_t) + J_t^T lambda_{t+1} for t < T,
    and lambda_T = pad(r_vec_T) (or 0 if use_terminal=False).
    """
    T_plus_1, k = phi.shape
    T = T_plus_1 - 1
    m = r_vec.shape[1]
    lam = np.zeros((T_plus_1, k), dtype=np.float32)
    pad = np.zeros((T_plus_1, k), dtype=np.float32)
    pad[:, :m] = r_vec[:, :m]
    if use_terminal:
        lam[T] = pad[T]
    # else lam[T] stays zero
    for t in range(T - 1, -1, -1):
        v = lam[t + 1]
        if use_jacobian:
            phi_t = phi[t]
            delta = phi[t + 1] - phi_t
            denom = float(phi_t @ phi_t) + eps
            # J_t^T v = v + phi_t * (delta . v) / denom
            jt_v = v + phi_t * (float(delta @ v) / denom)
        else:
            jt_v = v
        lam[t] = pad[t] + jt_v
    return lam


def compute_jt_action_proj(
    phi_t: np.ndarray,
    phi_tp1: np.ndarray,
    a_proj: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Compute J_t * a_proj where J_t is the rank-1 trajectory-empirical Jacobian.

    J_t v = v + (phi_{t+1} - phi_t) * (phi_t . v) / (||phi_t||^2 + eps)
    """
    delta = phi_tp1 - phi_t
    denom = float(phi_t @ phi_t) + eps
    return a_proj + delta * (float(phi_t @ a_proj) / denom)


# ---------------- Environment helpers ---------------- #


def env_reward_dim(env_id: str) -> int:
    if harness.ENV_TYPE[env_id] == "vector":
        # 2 for DST, 3 for RG
        if env_id == "deep-sea-treasure-concave-v0":
            return 2
        if env_id == "resource-gathering-v0":
            return 3
        raise ValueError(f"unknown vector env reward dim: {env_id}")
    # Scalar env: augment to (r, 1)
    return 2


def step_reward_vector(env_id: str, scalar_r: float, info: dict) -> np.ndarray:
    if harness.ENV_TYPE[env_id] == "vector":
        return np.asarray(info["vector"], dtype=np.float32).reshape(-1)
    return np.asarray([float(scalar_r), 1.0], dtype=np.float32)


# ---------------- Training loop ---------------- #


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave headroom for eval

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        env.close()
        raise RuntimeError("TEAR currently supports discrete action spaces only.")
    n_actions = int(action_space.n)

    obs0, _ = env.reset(seed=seed)
    obs_flat0 = flatten_obs(obs0)
    obs_dim = int(obs_flat0.shape[0])

    # Feature dim k. Keep k modest; needs k >= reward_dim and k >= n_actions
    # so the action-onehot projection is non-degenerate, but we still use a
    # projection in R^k for action onehots regardless.
    m = env_reward_dim(env_id)
    k = max(8, m, n_actions, obs_dim)

    phi_map = RandomProjection(obs_dim, k, rng)
    # Fixed random action-onehot projection: R^|A| -> R^k
    P_act = rng.standard_normal((n_actions, k)).astype(np.float32) / np.sqrt(n_actions)

    policy = PolicyNet(obs_dim, n_actions, hidden=64)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    eps_jac = 1e-3
    h_clip = 10.0       # clip per-step Hamiltonian to bound variance
    grad_clip = 5.0
    max_episode_steps = harness.MAX_EPISODE_STEPS

    n_episodes = 0
    n_env_steps = 0

    def sample_action(obs_flat: np.ndarray) -> tuple[int, torch.Tensor]:
        with torch.no_grad():
            logits = policy.logits(obs_flat).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        a = int(torch.multinomial(probs, num_samples=1).item())
        return a, logits.detach()

    # main training loop
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
            a, _ = sample_action(obs_flat)
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

        # Append a terminal r_vec entry for time T (the state after final step).
        # For r_vec_T we use the last realized r_vec (so terminal boundary has
        # signal). This honors "lambda_T = r_vec_T" with r_vec_T = realized
        # final-step vector reward.
        T = len(ep_actions)
        phi = np.stack(ep_phi, axis=0)        # shape (T+1, k)
        rvec = np.zeros((T + 1, m), dtype=np.float32)
        for t in range(T):
            rvec[t] = ep_rvec[t]
        rvec[T] = ep_rvec[-1]  # terminal-boundary reward = last realized r_vec

        lam = compute_adjoints(
            phi=phi,
            r_vec=rvec,
            eps=eps_jac,
            use_jacobian=True,
            use_terminal=True,
        )

        # Compute per-step Hamiltonians H_t.
        H = np.zeros(T, dtype=np.float32)
        for t in range(T):
            jt_aproj = compute_jt_action_proj(phi[t], phi[t + 1], P_act[ep_actions[t]], eps_jac)
            inner = float(lam[t + 1] @ jt_aproj)
            r_sum = float(rvec[t].sum())  # <r_vec_t, 1_m>
            H[t] = r_sum + inner

        # Center + clip H to keep variance manageable (still per-trajectory,
        # not a learned baseline; this is a standard score-function variance
        # control and not a critic).
        H_centered = H - float(H.mean())
        H_clipped = np.clip(H_centered, -h_clip, h_clip)

        # Score-function ascent.
        obs_t = torch.from_numpy(np.stack(ep_obs, axis=0).astype(np.float32))
        a_t = torch.from_numpy(np.asarray(ep_actions, dtype=np.int64))
        H_t = torch.from_numpy(H_clipped.astype(np.float32))

        logits = policy(obs_t)                       # (T, n_actions)
        log_probs = F.log_softmax(logits, dim=-1)
        log_pi_a = log_probs.gather(1, a_t.unsqueeze(1)).squeeze(1)  # (T,)
        # We ascend J ~ E[ sum_t H_t * log pi(a_t|s_t) ], i.e. minimize the
        # negation. We average over t to keep step size scale-invariant in T.
        loss = -(H_t * log_pi_a).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step()

        n_episodes += 1

    env.close()

    # Eval policy: argmax of logits (deterministic).
    policy.eval()

    def policy_fn(obs: np.ndarray):
        obs_flat_eval = flatten_obs(obs)
        with torch.no_grad():
            logits = policy.logits(obs_flat_eval).squeeze(0)
        return int(torch.argmax(logits).item())

    train_s = time.monotonic() - t_start
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_env_steps} episodes={n_episodes} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} k={k}",
        flush=True,
    )
    return policy_fn


# ---------------- CLI ---------------- #


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
