"""PTW — Prefix-Witness Floor (run 20260605-07-auto).

Realizes the hypothesis in worklogs/runs/20260605-07-auto/hypothesis.md.

Primitive: prefix-witness floor
    w(prefix) = coord-min { v(tau') : tau'_{0:k} = prefix }
over a buffer of completed trajectories with full per-channel terminal
vector outcomes v(tau).

Improvement operator: at each (tau, k), update weight is
    L1 of strictly-positive coords of  Delta(tau, k) = w(tau_{0:k+1}) - w(tau_{0:k}),
applied to grad_theta log pi(a_k | s_k). No baseline subtraction, no critic,
no Bellman target. Per-channel standardization (running median / MAD) is
applied to v before computing w to prevent magnitude-domination collapse.

CLI/contract identical to repo-root train.py.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Policy network
# ----------------------------------------------------------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr


# ----------------------------------------------------------------------------
# Prefix-witness floor over the trajectory buffer
# ----------------------------------------------------------------------------


class PrefixFloor:
    """Maintains, for every observed action prefix, the coord-wise min of
    terminal vector outcomes of all buffered trajectories starting with that
    prefix.

    A dict from prefix-tuple to (count, current coord-min vector). Adding a
    trajectory updates every prefix it touches (length 0 .. T).
    """

    def __init__(self, channels: int):
        self.channels = channels
        self.floor: dict[tuple[int, ...], np.ndarray] = {}
        self.count: dict[tuple[int, ...], int] = defaultdict(int)

    def add(self, action_seq: list[int], v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        assert v.shape[0] == self.channels
        # prefix length 0 (empty) up to length T (full)
        prefix: tuple[int, ...] = ()
        self._update(prefix, v)
        for a in action_seq:
            prefix = prefix + (int(a),)
            self._update(prefix, v)

    def _update(self, prefix: tuple[int, ...], v: np.ndarray) -> None:
        cur = self.floor.get(prefix)
        if cur is None:
            self.floor[prefix] = v.copy()
        else:
            np.minimum(cur, v, out=cur)
        self.count[prefix] += 1

    def get(self, prefix: tuple[int, ...]) -> np.ndarray | None:
        return self.floor.get(prefix)


# ----------------------------------------------------------------------------
# Per-channel running median / MAD standardizer
# ----------------------------------------------------------------------------


class ChannelStandardizer:
    """Per-channel running median / MAD standardization, computed over the
    last `window` trajectory terminal vectors. Uniform across all envs (no
    per-env tuning), as the reviewer's risk note prescribes.
    """

    def __init__(self, channels: int, window: int = 256):
        self.channels = channels
        self.window = window
        self.buf: list[np.ndarray] = []

    def add(self, v: np.ndarray) -> None:
        self.buf.append(np.asarray(v, dtype=np.float64).reshape(-1).copy())
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def transform(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if len(self.buf) < 2:
            return v.copy()
        stack = np.stack(self.buf, axis=0)  # (n, channels)
        med = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - med), axis=0)
        # MAD scaled by 1.4826 ~ sigma for Gaussians; floor to avoid /0.
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-6, 1.0, scale)
        return (v - med) / scale


# ----------------------------------------------------------------------------
# PTW training
# ----------------------------------------------------------------------------


def _terminal_vector(env_id: str, vec_sum: np.ndarray) -> np.ndarray:
    """The hypothesis says the choice (sum vs terminal-state read) is
    uniform across envs; we use the per-trajectory channel-wise sum.
    """
    return vec_sum


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng_np = np.random.default_rng(seed + 13)
    torch.manual_seed(seed + 17)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 4

    # Determine obs dim and channel count by a probe.
    obs0, _ = env.reset(seed=seed)
    obs_dim = _flatten_obs(obs0).shape[0]

    if is_vector:
        # k from HV_REF
        channels = int(harness.HV_REF[env_id].shape[0])
    else:
        channels = 1  # scalar-reward envs collapse to k=1 (degenerate case).

    policy = PolicyNet(obs_dim, n_actions, hidden=64)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-3)

    floor = PrefixFloor(channels=channels)
    stdz = ChannelStandardizer(channels=channels, window=256)

    # Diagnostics
    diag_total_pairs = 0
    diag_nonzero_pairs = 0
    diag_prefix_share_sum = 0.0
    diag_prefix_share_n = 0
    diag_episodes = 0
    diag_first_success_seen = False

    # Sampling temperature; hypothesis says "softmax with optional temperature".
    # We use plain softmax (temperature 1.0) to keep faithful.
    max_steps_cap = harness.MAX_EPISODE_STEPS

    t0 = time.monotonic()
    train_s_budget = max(1.0, float(time_budget_s) - 5.0)  # leave a small margin
    total_env_steps = 0

    BATCH_TRAJS = 8  # collect this many trajectories, then update.

    def collect_trajectory() -> tuple[list[np.ndarray], list[int], np.ndarray, float]:
        """Run one episode under current stochastic policy. Returns
        (states, actions, terminal_vector, scalar_return)."""
        states: list[np.ndarray] = []
        actions: list[int] = []
        vec_sum = np.zeros(channels, dtype=np.float64)
        scalar_ret = 0.0
        obs, _ = env.reset(seed=int(rng_np.integers(0, 2**31 - 1)))
        steps = 0
        while steps < max_steps_cap:
            x = _flatten_obs(obs)
            states.append(x)
            with torch.no_grad():
                logits = policy(torch.from_numpy(x).float().unsqueeze(0))
                probs = F.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
            a = int(rng_np.choice(n_actions, p=probs))
            actions.append(a)
            obs, reward, term, trunc, info = env.step(a)
            scalar_ret += float(reward)
            if is_vector:
                v_step = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
                vec_sum += v_step
            else:
                vec_sum[0] += float(reward)
            steps += 1
            if bool(term) or bool(trunc):
                break
        return states, actions, vec_sum, scalar_ret

    while time.monotonic() - t0 < train_s_budget:
        # 1) Collect a batch of trajectories.
        batch: list[tuple[list[np.ndarray], list[int], np.ndarray]] = []
        for _ in range(BATCH_TRAJS):
            if time.monotonic() - t0 >= train_s_budget:
                break
            states, actions, vec_sum, scalar_ret = collect_trajectory()
            total_env_steps += len(actions)
            diag_episodes += 1
            v_term = _terminal_vector(env_id, vec_sum)
            stdz.add(v_term)
            v_std = stdz.transform(v_term)
            floor.add(actions, v_std)
            batch.append((states, actions, v_std))
            if not diag_first_success_seen and (
                (is_vector and np.any(v_term > 0)) or (not is_vector and scalar_ret > 0)
            ):
                diag_first_success_seen = True

        if not batch:
            break

        # 2) Compute PTW updates from the batch against the buffered floor.
        # For each tau in batch and each k in [0, T-1]:
        #   w_k   = floor[tau_{0:k}]
        #   w_kp1 = floor[tau_{0:k+1}]
        #   Delta = w_kp1 - w_k   (>= 0 coord-wise by construction)
        #   weight = sum of strictly-positive coords of Delta  (L1 on the +cone)
        #   loss contribution: -weight * log pi(a_k | s_k)
        # We do not subtract a baseline; we do not clip the sign.
        loss_terms: list[torch.Tensor] = []
        for states, actions, _v_std in batch:
            T = len(actions)
            if T == 0:
                continue
            # Track average prefix-sharing depth: number of buffered trajs
            # sharing the prefix at depth k, averaged over k.
            share_sum = 0.0
            for k in range(T):
                prefix_k = tuple(actions[:k])
                prefix_kp1 = tuple(actions[: k + 1])
                w_k = floor.get(prefix_k)
                w_kp1 = floor.get(prefix_kp1)
                share_sum += float(floor.count.get(prefix_kp1, 0))
                if w_k is None or w_kp1 is None:
                    continue
                delta = w_kp1 - w_k
                # By construction delta >= 0; numerical-safe positive-cone L1.
                pos = np.maximum(delta, 0.0)
                weight = float(pos.sum())
                diag_total_pairs += 1
                if weight > 0.0:
                    diag_nonzero_pairs += 1
                    x = torch.from_numpy(states[k]).float().unsqueeze(0)
                    logits = policy(x)
                    log_probs = F.log_softmax(logits, dim=-1).reshape(-1)
                    loss_terms.append(-weight * log_probs[actions[k]])
            if T > 0:
                diag_prefix_share_sum += share_sum / T
                diag_prefix_share_n += 1

        if loss_terms:
            loss = torch.stack(loss_terms).sum()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optim.step()

    train_s = time.monotonic() - t0

    # Diagnostics print (audit channels mandated by reviewer).
    nz_frac = (diag_nonzero_pairs / diag_total_pairs) if diag_total_pairs else 0.0
    avg_share = (diag_prefix_share_sum / diag_prefix_share_n) if diag_prefix_share_n else 0.0
    k1_flag = "k=1-degenerate" if channels == 1 else "k>=2"
    print(
        f"[ptw] env={env_id} seed={seed} eps={diag_episodes} steps={total_env_steps} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} channels={channels} "
        f"nonzero_delta_frac={nz_frac:.3f} avg_prefix_share_depth={avg_share:.2f} "
        f"first_success_seen={diag_first_success_seen} mode={k1_flag}",
        flush=True,
    )

    env.close()

    # Build a deterministic policy_fn (argmax of logits) for evaluation.
    policy.eval()

    def policy_fn(obs: np.ndarray):
        x = torch.from_numpy(_flatten_obs(obs)).float().unsqueeze(0)
        with torch.no_grad():
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
