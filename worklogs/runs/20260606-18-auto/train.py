"""PFA — Per-Channel Phase-Flow Asymmetry.

Realizes the hypothesis at worklogs/runs/20260606-18-auto/hypothesis.md:
- Per-channel short/long-horizon firing-probability heads (p_m, q_m), channel-specific.
- Primitive: signed 2-D phase-area on each transition,
  A_m(o_t, a_t) = p_m(o_t)*q_m(o_{t+1}) - p_m(o_{t+1})*q_m(o_t).
- Per-(cluster, action) running-mean k-vector Ā[c, a, :] over channels.
- Pareto-non-dominance logit nudge over the k-vector (no scalar collapse).
- Cluster c(o) = fixed-radius hash of policy-trunk activation.

CLI:
    uv run train.py --env ENV --seed S --time-budget-s T
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# --------- Hyperparameters (algorithmic structure, not tuning knobs) ---------
TRUNK_DIM = 32
HORIZON_H = 16          # long-horizon window
ALPHA = 1.0             # logit-nudge scale
ENTROPY_FLOOR = 0.05    # per hypothesis "single fixed entropy floor"
LR = 3e-4
BATCH = 64
HASH_BITS = 6           # cluster hash bits => 2^6 buckets
BUFFER_CAP = 20000
WARMUP_STEPS = 200      # before applying nudge
UPDATE_EVERY = 16
DEVICE = torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# --------- Networks ---------
class PolicyTrunk(nn.Module):
    """Trunk producing policy logits z(o) and a separate embedding for clustering.

    The clustering embedding is detached so cluster identities depend on the
    trunk activation but do not gradient-couple to the heads.
    """

    def __init__(self, obs_dim: int, n_actions: int, trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, trunk_dim)
        self.fc2 = nn.Linear(trunk_dim, trunk_dim)
        self.head = nn.Linear(trunk_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        logits = self.head(h)
        return logits, h


class ChannelHead(nn.Module):
    """Channel-specific small classifier (no shared trunk across channels).

    Outputs sigmoid probability for the firing indicator at one horizon.
    """

    def __init__(self, obs_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs)).squeeze(-1)


# --------- Cluster hashing ---------
class TrunkHasher:
    """Fixed-radius hash of policy-trunk activation.

    Random projection -> sign bits -> integer cluster id.
    """

    def __init__(self, trunk_dim: int, hash_bits: int, seed: int):
        rng = np.random.default_rng(seed + 7)
        self.W = rng.standard_normal((trunk_dim, hash_bits)).astype(np.float32)

    def hash(self, h_np: np.ndarray) -> int:
        # h_np shape: (trunk_dim,)
        bits = (h_np @ self.W) > 0  # (hash_bits,)
        c = 0
        for b in bits:
            c = (c << 1) | int(b)
        return int(c)


# --------- A-bar running-mean table ---------
class AbarTable:
    """Per-(cluster, action) running-mean k-vector with sample counts."""

    def __init__(self, k: int, n_actions: int):
        self.k = k
        self.n_actions = n_actions
        # cell -> dict[action] -> {"sum": np.ndarray(k), "n": int}
        self.cells: dict[int, dict[int, dict]] = defaultdict(
            lambda: {a: {"sum": np.zeros(k, dtype=np.float64), "n": 0}
                     for a in range(n_actions)}
        )

    def update(self, cluster: int, action: int, area_vec: np.ndarray) -> None:
        cell = self.cells[cluster][action]
        cell["sum"] += area_vec
        cell["n"] += 1

    def row_matrix(self, cluster: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (M, counts) where M is |A| x k of running-mean signed areas."""
        M = np.zeros((self.n_actions, self.k), dtype=np.float64)
        counts = np.zeros(self.n_actions, dtype=np.int64)
        if cluster not in self.cells:
            return M, counts
        cell = self.cells[cluster]
        for a in range(self.n_actions):
            n = cell[a]["n"]
            counts[a] = n
            if n > 0:
                M[a] = cell[a]["sum"] / n
        return M, counts


def pareto_nudge(M: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Compute per-action (n_dom - m_dom) for the Pareto-non-dominance vote.

    n_a^dom = number of actions strictly dominated by a.
    m_a^dom = number of actions strictly dominating a.
    Coordinate-wise partial order on R^k. Actions with zero observed count
    contribute neutrally (no dominance either way) so the nudge defaults to 0.
    """
    A = M.shape[0]
    nudge = np.zeros(A, dtype=np.float64)
    valid = counts > 0
    for a in range(A):
        if not valid[a]:
            continue
        ra = M[a]
        n_dom = 0
        m_dom = 0
        for b in range(A):
            if b == a or not valid[b]:
                continue
            rb = M[b]
            # a dominates b: ra >= rb componentwise and ra > rb in some coord
            if np.all(ra >= rb) and np.any(ra > rb):
                n_dom += 1
            elif np.all(rb >= ra) and np.any(rb > ra):
                m_dom += 1
        nudge[a] = float(n_dom - m_dom)
    return nudge


# --------- Replay buffer ---------
class ReplayBuffer:
    def __init__(self, cap: int, obs_dim: int, k: int):
        self.cap = cap
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.act = np.zeros(cap, dtype=np.int64)
        self.fire_now = np.zeros((cap, k), dtype=np.float32)  # 1{v_{t+1, m} != 0}
        self.fire_h = np.zeros((cap, k), dtype=np.float32)    # 1{any fire in next H steps}
        self.cluster = np.zeros(cap, dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add(self, o, no, a, fnow, c):
        i = self.ptr
        self.obs[i] = o
        self.next_obs[i] = no
        self.act[i] = a
        self.fire_now[i] = fnow
        self.cluster[i] = c
        # fire_h backfilled by fill_horizon; default 0
        self.fire_h[i] = 0.0
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        return i

    def fill_horizon_for_episode(self, ep_indices: list[int], fires: np.ndarray) -> None:
        """For each step in ep_indices, set fire_h to 1 if any channel fires in next H steps.

        fires shape: (T, k), where fires[t] = 1{v_{t+1,m} != 0} (the next-step fire).
        For step at ep_indices[t], horizon window is fires[t : t+H] (i.e., the next H steps).
        """
        T = len(ep_indices)
        H = HORIZON_H
        for t in range(T):
            window = fires[t : min(t + H, T)]
            if len(window) > 0:
                hbits = (window.sum(axis=0) > 0).astype(np.float32)
            else:
                hbits = np.zeros(fires.shape[1], dtype=np.float32)
            self.fire_h[ep_indices[t]] = hbits

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.fire_now[idx]),
            torch.from_numpy(self.fire_h[idx]),
        )


# --------- Training loop ---------
def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n)

    # Sample a reset to determine obs shape and probe info["vector"] dimensionality
    obs0, _ = env.reset(seed=seed)
    obs0 = np.asarray(obs0, dtype=np.float32).reshape(-1)
    obs_dim = int(obs0.shape[0])

    # Probe k by stepping once
    a_probe = env.action_space.sample()
    _o, _r, _term, _trunc, info_probe = env.step(a_probe)
    if "vector" not in info_probe:
        # Non-vector envs: PFA defined only on the vector axis; fall back gracefully.
        # We keep the same primitive but with a single synthetic channel = scalar reward sign.
        k = 1
        vector_env = False
    else:
        k = int(np.asarray(info_probe["vector"]).reshape(-1).shape[0])
        vector_env = True

    # Reset for clean episode start
    obs_arr, _ = env.reset(seed=seed + 1)
    obs_arr = np.asarray(obs_arr, dtype=np.float32).reshape(-1)

    # Networks
    trunk = PolicyTrunk(obs_dim, n_actions).to(DEVICE)
    p_heads = nn.ModuleList([ChannelHead(obs_dim) for _ in range(k)]).to(DEVICE)
    q_heads = nn.ModuleList([ChannelHead(obs_dim) for _ in range(k)]).to(DEVICE)
    opt = torch.optim.Adam(
        list(trunk.parameters())
        + list(p_heads.parameters())
        + list(q_heads.parameters()),
        lr=LR,
    )

    hasher = TrunkHasher(TRUNK_DIM, HASH_BITS, seed)
    abar = AbarTable(k=k, n_actions=n_actions)
    buf = ReplayBuffer(BUFFER_CAP, obs_dim, k)

    # Diagnostics
    dbg_var_pq = []

    t0 = time.monotonic()
    env_steps = 0
    update_count = 0

    # Episode rollout state
    cur_obs = obs_arr
    cur_ep_indices: list[int] = []
    cur_ep_fires: list[np.ndarray] = []

    def policy_logits_and_cluster(obs_np: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        with torch.no_grad():
            t = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
            logits, h = trunk(t)
            logits_np = logits.squeeze(0).cpu().numpy()
            h_np = h.squeeze(0).cpu().numpy()
        c = hasher.hash(h_np)
        return logits_np, c, h_np

    def act_with_nudge(obs_np: np.ndarray) -> tuple[int, int]:
        logits_np, c, _h = policy_logits_and_cluster(obs_np)
        if env_steps >= WARMUP_STEPS:
            M, counts = abar.row_matrix(c)
            nudge = pareto_nudge(M, counts)
            logits_np = logits_np + ALPHA * nudge
        # entropy floor: blend with uniform
        probs = _softmax_with_floor(logits_np, ENTROPY_FLOOR)
        a = int(rng.choice(n_actions, p=probs))
        return a, c

    while True:
        if time.monotonic() - t0 > time_budget_s:
            break

        a, c = act_with_nudge(cur_obs)
        next_obs, _r, term, trunc, info = env.step(a)
        next_obs_np = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        if vector_env:
            v = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
            fire_now = (np.abs(v) > 1e-9).astype(np.float32)
        else:
            r_scalar = float(_r)
            fire_now = np.array([1.0 if abs(r_scalar) > 1e-9 else 0.0], dtype=np.float32)

        i = buf.add(cur_obs, next_obs_np, a, fire_now, c)
        cur_ep_indices.append(i)
        cur_ep_fires.append(fire_now)

        # Update Ā[c, a, :] with the signed phase-area on this observed transition,
        # using the *current* p_m, q_m heads (no gradient).
        if vector_env or k == 1:
            with torch.no_grad():
                ot = torch.from_numpy(cur_obs.astype(np.float32)).unsqueeze(0)
                ont = torch.from_numpy(next_obs_np.astype(np.float32)).unsqueeze(0)
                p_t = np.array([float(p_heads[m](ot).item()) for m in range(k)])
                p_n = np.array([float(p_heads[m](ont).item()) for m in range(k)])
                q_t = np.array([float(q_heads[m](ot).item()) for m in range(k)])
                q_n = np.array([float(q_heads[m](ont).item()) for m in range(k)])
            # signed phase-area cross-product: p_t * q_n - p_n * q_t
            area = p_t * q_n - p_n * q_t  # shape (k,)
            abar.update(c, a, area)

        env_steps += 1
        cur_obs = next_obs_np
        done = bool(term) or bool(trunc)

        if done:
            # backfill long-horizon labels for this episode and push to replay
            if len(cur_ep_indices) > 0:
                fires_arr = np.stack(cur_ep_fires, axis=0)
                buf.fill_horizon_for_episode(cur_ep_indices, fires_arr)
            cur_ep_indices = []
            cur_ep_fires = []
            obs_reset, _ = env.reset(seed=seed + 100 + env_steps)
            cur_obs = np.asarray(obs_reset, dtype=np.float32).reshape(-1)

        # Train heads
        if buf.size >= BATCH and env_steps % UPDATE_EVERY == 0:
            obs_b, _next_b, fnow_b, fh_b = buf.sample(BATCH, rng)
            loss = 0.0
            # Channel-specific heads: each head trained independently.
            for m in range(k):
                p_pred = p_heads[m](obs_b)
                q_pred = q_heads[m](obs_b)
                p_loss = F.binary_cross_entropy(p_pred, fnow_b[:, m])
                q_loss = F.binary_cross_entropy(q_pred, fh_b[:, m])
                loss = loss + p_loss + q_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_count += 1

            # diagnostic: Var(p_m - q_m) over the batch
            if update_count % 50 == 0 and k > 0:
                with torch.no_grad():
                    diffs = []
                    for m in range(k):
                        diffs.append((p_heads[m](obs_b) - q_heads[m](obs_b)).var().item())
                    dbg_var_pq.append(diffs)

    env.close()
    train_s = time.monotonic() - t0
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} "
        f"updates={update_count} k={k} clusters={len(abar.cells)} "
        f"var_pq_last={dbg_var_pq[-1] if dbg_var_pq else None}",
        flush=True,
    )

    # Build deterministic policy_fn for evaluation: argmax of (logits + alpha * nudge)
    trunk_eval = trunk
    p_eval = p_heads
    q_eval = q_heads

    def policy_fn(obs: np.ndarray) -> int:
        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            t = torch.from_numpy(obs_np).unsqueeze(0)
            logits, h = trunk_eval(t)
            logits_np = logits.squeeze(0).cpu().numpy()
            h_np = h.squeeze(0).cpu().numpy()
        c = hasher.hash(h_np)
        M, counts = abar.row_matrix(c)
        nudge = pareto_nudge(M, counts)
        scored = logits_np + ALPHA * nudge
        return int(np.argmax(scored))

    return policy_fn


def _softmax_with_floor(logits: np.ndarray, eps: float) -> np.ndarray:
    z = logits - logits.max()
    p = np.exp(z)
    p = p / p.sum()
    n = p.shape[0]
    # blend with uniform to enforce entropy floor
    p = (1.0 - eps) * p + eps * (1.0 / n)
    p = p / p.sum()
    return p


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
