"""GRADCOMP ablation: random-unit-vector replacement for v1(tau).

Identical to train.py except that the load-bearing primitive v1(tau)
(top eigenvector of the empirical Fisher via score-Gram) is replaced by
a uniformly random unit vector xi ~ Uniform(S^{n-1}), redrawn each
rollout. The slerp interpolation, eta annealer, gradient floor c, and
all hyperparameters are preserved.

This isolates the *trajectory-informed* nature of v1 against an
identically-scaffolded random-direction baseline.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402


# ---------- defaults (identical to train.py) ----------
GAMMA = 0.99
LR = 5e-2
HIDDEN = 32
GRAD_FLOOR_C = 1e-3
EPS = 1e-8
LOG_INTERVAL_S = 10.0
MAX_STEPS_PER_EPISODE_FOR_GRAM = 256


def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1) / 10.0
    return arr


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def action_logprobs(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        return F.log_softmax(logits, dim=-1)


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n


def _flat_grad(params: List[torch.Tensor]) -> torch.Tensor:
    parts = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel()))
        else:
            parts.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(parts)


def _zero_grad(params: List[torch.Tensor]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _slerp(v1_unit: torch.Tensor, ghat: torch.Tensor, eta: float) -> torch.Tensor:
    n_v = float(torch.linalg.norm(v1_unit))
    n_g = float(torch.linalg.norm(ghat))
    if n_v < EPS and n_g < EPS:
        return torch.zeros_like(v1_unit)
    if n_v < EPS:
        return ghat / max(n_g, EPS)
    if n_g < EPS:
        return v1_unit / max(n_v, EPS)
    dot = float(torch.dot(v1_unit, ghat))
    v1 = v1_unit if dot >= 0 else -v1_unit
    dot = abs(dot)
    dot = min(max(dot, -1.0 + 1e-6), 1.0 - 1e-6)
    omega = math.acos(dot)
    if omega < 1e-4:
        return ghat / max(n_g, EPS)
    s = math.sin(omega)
    a = math.sin((1.0 - eta) * omega) / s
    b = math.sin(eta * omega) / s
    out = a * v1 + b * ghat
    nrm = float(torch.linalg.norm(out))
    if nrm < EPS:
        return ghat / max(n_g, EPS)
    return out / nrm


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _random_unit_vector(n: int, rng: np.random.Generator) -> torch.Tensor:
    """ABLATION: random unit vector replacing v1(tau).

    Uniform on the sphere via normalized standard normal sample.
    """
    xi = rng.standard_normal(n).astype(np.float32)
    nrm = float(np.linalg.norm(xi))
    if nrm < EPS:
        xi = np.zeros(n, dtype=np.float32)
        xi[0] = 1.0
        return torch.from_numpy(xi)
    return torch.from_numpy(xi / nrm)


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)
    # Separate stream for ablation random-unit-vector draws so policy
    # action sampling is bit-identical to train.py given the same seed
    # (though run-to-run trajectories will diverge after the first update
    # because the update direction differs; this is intended).
    rng_xi = np.random.default_rng(seed + 31337)

    device = torch.device("cpu")

    env = harness.make_env(env_id, seed)
    act_space = env.action_space
    if not hasattr(act_space, "n"):
        env.close()
        raise RuntimeError("GRADCOMP ablation only supports discrete action spaces")
    n_actions = int(act_space.n)
    obs_sample, _ = env.reset(seed=seed)
    obs_dim = int(_flatten_obs(obs_sample).shape[0])

    policy = PolicyNet(obs_dim, n_actions).to(device)
    params = list(policy.parameters())
    n_params = sum(p.numel() for p in params)

    t0 = time.monotonic()
    deadline = t0 + max(1.0, float(time_budget_s) - 2.0)
    episodes = 0
    env_steps = 0
    first_reward_ep = -1
    last_log_t = t0
    cum_align = 0.0
    align_count = 0

    obs_np, _ = env.reset(seed=seed)
    while time.monotonic() < deadline:
        log_probs_kept: List[torch.Tensor] = []
        actions_kept: List[int] = []
        obs_kept: List[np.ndarray] = []
        rewards_all: List[float] = []
        kept_indices: List[int] = []

        steps = 0
        max_steps = harness.MAX_EPISODE_STEPS
        done = False
        while not done and steps < max_steps and time.monotonic() < deadline:
            obs_flat = _flatten_obs(obs_np)
            obs_t = torch.from_numpy(obs_flat).to(device)
            with torch.no_grad():
                logits = policy(obs_t.unsqueeze(0)).squeeze(0)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs = np.clip(probs, 1e-8, None)
            probs = probs / probs.sum()
            action = int(rng.choice(n_actions, p=probs))

            try:
                next_obs, reward, term, trunc, info = env.step(action)
            except Exception:
                term, trunc = True, False
                next_obs = obs_np
                reward = 0.0

            rewards_all.append(float(reward))
            if steps < MAX_STEPS_PER_EPISODE_FOR_GRAM:
                obs_kept.append(obs_flat)
                actions_kept.append(action)
                kept_indices.append(steps)

            obs_np = next_obs
            done = bool(term) or bool(trunc)
            steps += 1
            env_steps += 1

        if done:
            obs_np, _ = env.reset()

        if len(actions_kept) == 0:
            continue

        # ----- compute return-to-go for kept steps using full reward seq -----
        T_full = len(rewards_all)
        R = np.zeros(T_full, dtype=np.float64)
        running = 0.0
        for k in range(T_full - 1, -1, -1):
            running = rewards_all[k] + GAMMA * running
            R[k] = running
        ep_return = sum(rewards_all)
        if ep_return != 0.0 and first_reward_ep < 0:
            first_reward_ep = episodes

        # ----- per-step score vectors g_t = grad log pi(a_t|s_t) -----
        T = len(kept_indices)
        scores = torch.zeros(T, n_params, dtype=torch.float32)
        obs_batch = torch.from_numpy(np.stack(obs_kept, axis=0).astype(np.float32))
        for i in range(T):
            _zero_grad(params)
            logp = policy.action_logprobs(obs_batch[i].unsqueeze(0)).squeeze(0)
            logp_a = logp[actions_kept[i]]
            logp_a.backward()
            scores[i] = _flat_grad(params)
        _zero_grad(params)

        # ----- REINFORCE direction g_tau (identical to candidate) -----
        R_kept = torch.from_numpy(
            np.array([R[k] for k in kept_indices], dtype=np.float32)
        )
        g_tau = (scores * R_kept.unsqueeze(1)).sum(dim=0)
        g_norm = float(torch.linalg.norm(g_tau))
        ghat = g_tau / (g_norm + EPS) if g_norm > EPS else torch.zeros_like(g_tau)

        # ----- ABLATION: replace v1(tau) with a random unit vector -----
        v1 = _random_unit_vector(n_params, rng_xi)
        # Principal eigenvalue is undefined here; we keep the slot for parity
        # in logging.
        mu = float("nan")

        # Compass annealer: eta = sigma((|g| - c) / c) [identical schedule]
        eta = _sigmoid((g_norm - GRAD_FLOOR_C) / max(GRAD_FLOOR_C, EPS))

        if g_norm < EPS and float(torch.linalg.norm(v1)) < EPS:
            d_tau = torch.zeros_like(g_tau)
        elif g_norm < EPS:
            d_tau = v1 / (float(torch.linalg.norm(v1)) + EPS)
        elif float(torch.linalg.norm(v1)) < EPS:
            d_tau = ghat
        else:
            d_tau = _slerp(v1, ghat, eta)

        step_mag = max(g_norm, GRAD_FLOOR_C)

        flat = _flat_params(policy).clone()
        flat = flat + LR * step_mag * d_tau.to(flat.dtype)
        if not torch.all(torch.isfinite(flat)):
            flat = _flat_params(policy)
        _set_flat_params(policy, flat)

        if g_norm > EPS and float(torch.linalg.norm(v1)) > EPS:
            align = float(torch.dot(v1, ghat))
            cum_align += align
            align_count += 1

        episodes += 1

        if time.monotonic() - last_log_t > LOG_INTERVAL_S:
            last_log_t = time.monotonic()
            mean_align = cum_align / max(align_count, 1)
            print(
                f"[train-ablate] env={env_id} seed={seed} ep={episodes} "
                f"env_steps={env_steps} T_kept={T} ep_return={ep_return:.4f} "
                f"|g_tau|={g_norm:.5f} eta={eta:.4f} "
                f"mean_align={mean_align:.4f} first_rew_ep={first_reward_ep}",
                flush=True,
            )

    env.close()

    mean_align = cum_align / max(align_count, 1)
    print(
        f"[train-ablate] env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={episodes} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} first_reward_ep={first_reward_ep} "
        f"final_mean_align={mean_align:.4f}",
        flush=True,
    )

    policy.eval()

    def policy_fn(obs: np.ndarray):
        with torch.no_grad():
            obs_t = torch.from_numpy(_flatten_obs(obs)).to(device)
            logits = policy(obs_t.unsqueeze(0)).squeeze(0)
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
