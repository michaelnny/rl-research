"""EFLO ablation: c_t = 1 (trajectory log-likelihood ascent).

Same scaffold as train.py but the per-step weight is the constant 1, removing
the GAE-style entropy-flow weight. Predicted to collapse policy entropy on
the realized actions (gradient ascent on log P_theta(tau)).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


HERE = Path(__file__).parent.resolve()
LOG_FILENAME = "mean_entropy_ablate.txt"
VARIANT_TAG = "ABLATE"
LOG_EVERY_N = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


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


def _flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 99_999)

    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        env.close()

        def random_policy(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} note=continuous-action-fallback",
            flush=True,
        )
        return random_policy

    n_actions = int(action_space.n)
    sample_obs, _ = env.reset(seed=seed)
    obs_dim = int(_flatten_obs(sample_obs).shape[0])

    device = torch.device("cpu")
    net = PolicyNet(obs_dim, n_actions).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=3e-3)

    log_path = HERE / LOG_FILENAME
    log_path.write_text(f"# variant={VARIANT_TAG} env={env_id} seed={seed}\n")
    log_buffer: list[float] = []
    episode_count = 0

    t0 = time.monotonic()
    deadline = t0 + max(1, time_budget_s - 5)
    env_steps = 0

    while time.monotonic() < deadline:
        obs_arr, _ = env.reset(seed=seed + 1 + episode_count)
        states: list[np.ndarray] = []
        actions: list[int] = []

        done = False
        steps = 0
        while not done and steps < harness.MAX_EPISODE_STEPS:
            x = torch.from_numpy(_flatten_obs(obs_arr)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = net(x)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            probs = np.clip(probs, 1e-12, 1.0)
            probs = probs / probs.sum()
            a = int(rng.choice(n_actions, p=probs))
            states.append(_flatten_obs(obs_arr))
            actions.append(a)
            obs_arr, _r, term, trunc, _info = env.step(a)
            done = bool(term) or bool(trunc)
            steps += 1
            env_steps += 1
            if time.monotonic() >= deadline:
                break

        states.append(_flatten_obs(obs_arr))
        T = len(actions)
        if T < 1:
            continue

        states_t = torch.from_numpy(np.stack(states, axis=0)).float().to(device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)

        logits_all = net(states_t)
        logp_all = F.log_softmax(logits_all, dim=-1)
        p_all = logp_all.exp()
        H_all = -(p_all * logp_all).sum(dim=-1)

        with torch.no_grad():
            H_np = H_all.detach().cpu().numpy()

        # ---- ABLATION: c_t = 1 for all t (no entropy residual, no GAE recursion) ----
        c = np.ones(T, dtype=np.float32)

        logp_taken = logp_all[:T].gather(1, actions_t.unsqueeze(1)).squeeze(1)
        c_t = torch.from_numpy(c).to(device)
        loss = -(c_t * logp_taken).sum()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        optim.step()

        mean_H = float(H_np[:T].mean()) if T > 0 else 0.0
        log_buffer.append(mean_H)
        episode_count += 1

        if len(log_buffer) >= LOG_EVERY_N:
            mean_block = float(np.mean(log_buffer))
            std_block = float(np.std(log_buffer))
            with log_path.open("a") as f:
                f.write(
                    f"episode={episode_count} block_size={len(log_buffer)} "
                    f"mean_H_t={mean_block:.6f} std_H_t={std_block:.6f}\n"
                )
            log_buffer = []

    if log_buffer:
        mean_block = float(np.mean(log_buffer))
        std_block = float(np.std(log_buffer))
        with log_path.open("a") as f:
            f.write(
                f"episode={episode_count} block_size={len(log_buffer)} "
                f"mean_H_t={mean_block:.6f} std_H_t={std_block:.6f}\n"
            )

    env.close()
    train_s = time.monotonic() - t0
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={episode_count} train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    net.eval()

    def policy_fn(obs):
        x = torch.from_numpy(_flatten_obs(obs)).unsqueeze(0)
        with torch.no_grad():
            logits = net(x)
        return int(torch.argmax(logits, dim=-1).item())

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
