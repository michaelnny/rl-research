"""HRC: Horizon-Recursive Concordance.

Run 20260606-29-auto. See hypothesis.md for full description.

Two reward-independent statistics maintained per (state-cluster, action):
  K[s,a,:]  in [0,1]^L  -- self-concordance fractions at L exponentially-spaced horizons
  P[s,a,:]  in [-1,1]^k -- terminal-sign-conditional channel propensity (variance-gated)

Improvement operator (per decision step):
  Delta_K = pareto-non-dominance margin over actions on the L-vector K[s,*,:]
  Delta_P = pareto-non-dominance margin over actions on the k-vector P[s,*,:]
  Logit nudge: alpha * Delta_K * max(Delta_P, 1)

Sample action ~ softmax(z + nudge), where z are base policy logits.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

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


# ---------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------

def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr


def _obs_dim(env) -> int:
    sample, _ = env.reset()
    return _flatten_obs(sample).size


# ---------------------------------------------------------------------
# Policy network: small MLP returning logits
# ---------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, embed_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, embed_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(embed_dim, n_actions)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.trunk(x)
        logits = self.head(emb)
        return logits, emb

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


# ---------------------------------------------------------------------
# Online k-means on policy-trunk embeddings (transition geometry)
# ---------------------------------------------------------------------

class OnlineKMeans:
    def __init__(self, n_clusters: int, dim: int, lr: float = 0.05, rng: np.random.Generator | None = None):
        self.n_clusters = n_clusters
        self.dim = dim
        self.lr = lr
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.centroids = self.rng.normal(0.0, 0.1, size=(n_clusters, dim)).astype(np.float32)
        self.counts = np.zeros(n_clusters, dtype=np.int64)
        self.initialized = np.zeros(n_clusters, dtype=bool)

    def assign(self, emb: np.ndarray) -> int:
        # emb: (dim,) — return nearest centroid index
        diffs = self.centroids - emb[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        return int(np.argmin(d2))

    def update(self, emb: np.ndarray) -> int:
        # find a slot to seed if any uninitialized centroid exists
        if not self.initialized.all():
            idx = int(np.argmin(self.initialized.astype(np.int64)))
            self.centroids[idx] = emb
            self.initialized[idx] = True
            self.counts[idx] += 1
            return idx
        c = self.assign(emb)
        # slow-rate update — use 1/sqrt(count) damping
        self.counts[c] += 1
        eta = max(self.lr, 1.0 / np.sqrt(self.counts[c]))
        self.centroids[c] = (1 - eta) * self.centroids[c] + eta * emb
        return c


# ---------------------------------------------------------------------
# Pareto non-dominance margin
# ---------------------------------------------------------------------

def pareto_margins(M: np.ndarray) -> np.ndarray:
    """Given M of shape (A, D), return a vector of length A.

    For each action a, count
        n_a = number of actions a' such that a Pareto-non-dominates a' (>= on all, > on some)
        m_a = number of actions a' such that a is Pareto-dominated by a'
    Return n_a - m_a.
    """
    A = M.shape[0]
    if A == 0:
        return np.zeros(0, dtype=np.float32)
    # ge[i,j] = M[i] >= M[j] componentwise all
    diff = M[:, None, :] - M[None, :, :]  # (A, A, D)
    ge_all = np.all(diff >= 0.0, axis=-1)  # (A, A) i dominates-eq j
    gt_any = np.any(diff > 0.0, axis=-1)
    dominates = ge_all & gt_any  # i Pareto-dominates j
    # n_a: i strictly dominates how many j (excluding self by gt_any requirement)
    n = dominates.sum(axis=1).astype(np.float32)
    # m_a: how many i Pareto-dominate a — count over rows
    m = dominates.sum(axis=0).astype(np.float32)
    return n - m


# ---------------------------------------------------------------------
# HRC algorithm
# ---------------------------------------------------------------------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed + 12345)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 4

    # probe obs dim
    obs0, _ = env.reset(seed=seed)
    flat = _flatten_obs(obs0)
    obs_dim = flat.size

    # detect vector channels
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    # determine k by stepping once
    _, _, _, _, info0 = env.step(env.action_space.sample())
    if is_vector and "vector" in info0:
        k_channels = int(np.asarray(info0["vector"]).reshape(-1).size)
    else:
        # for scalar envs, treat scalar reward as single channel
        k_channels = 1
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, n_actions, embed_dim=32).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Horizons: exponentially spaced
    horizons = [1, 4, 16, 64]  # L = 4 (kept modest for sub-120s budgets)
    L = len(horizons)
    H_max = max(horizons)

    # State clustering
    n_clusters = 64
    kmeans = OnlineKMeans(n_clusters, dim=policy.embed_dim, rng=rng)

    # K and P statistics — we keep numerator/denominator for K (concordance counts),
    # and channel-side aggregates for P
    K_match = np.zeros((n_clusters, n_actions, L), dtype=np.float64)
    K_count = np.zeros((n_clusters, n_actions, L), dtype=np.float64)
    # P side: for each (cluster, action), per-channel terminal cumulants list
    # We accumulate trajectory-level aggregates for variance-gated sign propensity
    # Use running per-channel (sum, sumsq, n) for global median estimate via running median
    # And per (s,a,m): count_above, count_below
    P_above = np.zeros((n_clusters, n_actions, k_channels), dtype=np.float64)
    P_below = np.zeros((n_clusters, n_actions, k_channels), dtype=np.float64)
    # global per-channel reservoir for median + variance estimate
    chan_reservoir: list[np.ndarray] = []  # of trajectory terminal cumulants
    RESERVOIR_MAX = 256

    # alpha: nudge magnitude
    alpha = 1.0
    entropy_floor = 0.05  # add small noise to logits to avoid premature collapse

    # Replay buffer of recent trajectories (for K computation)
    # store: list of (cluster, action, obs_flat) per step; compute K via batch snapshot pass
    traj_buffer: deque = deque(maxlen=200)

    # PG learning: simple REINFORCE-style update on trajectory return for the base policy
    # so that policy net learns reward signal too — but the IMPROVEMENT OPERATOR is the
    # logit nudge from K & P at *decision* time. Per the hypothesis, the operator is the
    # primitive; the policy net is the base z-logits. We need *some* base to nudge.
    # The hypothesis does not name a base learner, but rule 4 says z = base policy network
    # logits. We use a tiny REINFORCE update so that the network is not random; the
    # research claim is about the nudge.

    t0 = time.monotonic()
    env = harness.make_env(env_id, seed)
    obs, _ = env.reset(seed=seed)
    obs_flat = _flatten_obs(obs)

    total_steps = 0
    n_episodes = 0
    margin_fired_K = 0
    margin_fired_P = 0
    margin_total_decisions = 0

    cur_traj_clusters: list[int] = []
    cur_traj_actions: list[int] = []
    cur_traj_logprobs: list[torch.Tensor] = []
    cur_traj_rewards: list[float] = []
    cur_traj_vectors: list[np.ndarray] = []
    cur_traj_obs: list[np.ndarray] = []

    LOG_INTERVAL = 2000
    MAX_TIME_FRACTION = 0.92  # stop training a bit early to leave time for eval

    while time.monotonic() - t0 < MAX_TIME_FRACTION * time_budget_s:
        # forward pass
        x = torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            emb = policy.embed(x).cpu().numpy().reshape(-1)
        logits_t, _ = policy(x)
        logits_np = logits_t.detach().cpu().numpy().reshape(-1)

        # cluster assignment
        cluster = kmeans.update(emb)

        # Compute Pareto margins on K and P slices for this cluster
        with np.errstate(invalid="ignore"):
            denom = np.maximum(K_count[cluster], 1.0)
            K_slice = K_match[cluster] / denom  # (n_actions, L)
            # actions with zero counts -> default to 0.5 (neutral) so they don't dominate or get dominated
            zero_mask = (K_count[cluster] == 0)
            K_slice = np.where(zero_mask, 0.5, K_slice)

        # P slice: sign-conditional propensity per channel, variance-gated
        # variance estimated from global reservoir
        if len(chan_reservoir) >= 4:
            R = np.stack(chan_reservoir, axis=0)  # (T, k)
            chan_var = R.var(axis=0)
            # variance gate: use median variance as threshold
            v_thresh = max(1e-6, 0.1 * float(np.median(chan_var)))
            chan_active = chan_var > v_thresh
        else:
            chan_active = np.zeros(k_channels, dtype=bool)

        # propensity = (above - below) / (above + below) clipped to [-1,1], hard zero on inactive channels
        denom_p = P_above[cluster] + P_below[cluster]
        denom_p_safe = np.where(denom_p > 0, denom_p, 1.0)
        P_slice = (P_above[cluster] - P_below[cluster]) / denom_p_safe
        P_slice = np.where(denom_p[..., None] if False else (denom_p > 0)[..., None] if False else (denom_p > 0), P_slice, 0.0)
        # zero-out inactive channels (hard clip)
        P_slice = P_slice * chan_active[None, :].astype(np.float64)

        Delta_K = pareto_margins(K_slice)  # (n_actions,)
        Delta_P = pareto_margins(P_slice)  # (n_actions,)

        # nudge: alpha * Delta_K * max(Delta_P, 1)
        # — interpret max(Delta_P, 1) elementwise: at least 1, so horizon side always fires
        Delta_P_floored = np.maximum(Delta_P, 1.0)
        nudge = alpha * Delta_K * Delta_P_floored  # shape (n_actions,)

        # add small entropy noise
        noise = entropy_floor * rng.standard_normal(n_actions)

        nudged_logits_np = logits_np + nudge.astype(np.float32) + noise.astype(np.float32)

        # diagnostics
        if np.any(Delta_K != 0):
            margin_fired_K += 1
        if np.any(Delta_P != 0):
            margin_fired_P += 1
        margin_total_decisions += 1

        # sample
        nl = nudged_logits_np - nudged_logits_np.max()
        probs = np.exp(nl)
        probs /= probs.sum()
        action = int(rng.choice(n_actions, p=probs))

        # log-prob through the BASE policy at the sampled action — used for REINFORCE
        # on the underlying network only (the nudge is a non-parametric operator)
        logp = F.log_softmax(logits_t, dim=-1).reshape(-1)[action]

        # store
        cur_traj_clusters.append(cluster)
        cur_traj_actions.append(action)
        cur_traj_logprobs.append(logp)
        cur_traj_obs.append(obs_flat.copy())

        # step
        obs, reward, term, trunc, info = env.step(action)
        if is_vector and "vector" in info:
            vec = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
        else:
            vec = np.array([float(reward)], dtype=np.float64)
        cur_traj_rewards.append(float(reward))
        cur_traj_vectors.append(vec)

        obs_flat = _flatten_obs(obs)
        total_steps += 1
        done = bool(term) or bool(trunc)

        if done:
            n_episodes += 1
            T = len(cur_traj_actions)
            # terminal cumulant per channel
            terminal = np.sum(np.stack(cur_traj_vectors, axis=0), axis=0)  # (k,)

            # update channel reservoir
            chan_reservoir.append(terminal.copy())
            if len(chan_reservoir) > RESERVOIR_MAX:
                chan_reservoir.pop(0)

            # compute current per-channel median for sign assignment
            R = np.stack(chan_reservoir, axis=0)
            med = np.median(R, axis=0)
            sign_above = (terminal > med).astype(np.float64)  # (k,)
            sign_below = (terminal < med).astype(np.float64)

            # Update P: for each (cluster, action) on this trajectory,
            # add channel sign propensity contributions
            for c, a in zip(cur_traj_clusters, cur_traj_actions):
                P_above[c, a, :] += sign_above
                P_below[c, a, :] += sign_below

            # Update K: snapshot policy argmax on each obs in trajectory
            # Compute argmax in batch
            obs_batch = np.stack(cur_traj_obs, axis=0)
            with torch.no_grad():
                xb = torch.from_numpy(obs_batch).float().to(device)
                logits_b = policy.logits(xb).cpu().numpy()
            argmax_b = np.argmax(logits_b, axis=1)  # (T,)

            actions_arr = np.array(cur_traj_actions, dtype=np.int64)
            clusters_arr = np.array(cur_traj_clusters, dtype=np.int64)

            for li, h in enumerate(horizons):
                if T <= h:
                    continue
                # pairs (t, t+h)
                src_t = np.arange(T - h)
                tgt_t = src_t + h
                src_c = clusters_arr[src_t]
                src_a = actions_arr[src_t]
                # concordance: snapshot argmax at obs_{t+h} matches actually-taken a_{t+h}
                match = (argmax_b[tgt_t] == actions_arr[tgt_t]).astype(np.float64)
                # accumulate
                # use np.add.at for index-aware accumulation
                np.add.at(K_match[:, :, li], (src_c, src_a), match)
                np.add.at(K_count[:, :, li], (src_c, src_a), 1.0)

            # REINFORCE update on the BASE policy net using cumulative scalar return
            # (the nudge is the research primitive; the base is just to give z meaning)
            G = sum(cur_traj_rewards)
            # baseline: simple running mean
            traj_buffer.append(G)
            baseline = float(np.mean(traj_buffer)) if len(traj_buffer) > 0 else 0.0
            adv = G - baseline

            if abs(adv) > 1e-8 and len(cur_traj_logprobs) > 0:
                logp_sum = torch.stack(cur_traj_logprobs).sum()
                loss = -adv * logp_sum
                # entropy bonus on the most recent batch is omitted — entropy floor is in nudge
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

            # reset trajectory buffers
            cur_traj_clusters = []
            cur_traj_actions = []
            cur_traj_logprobs = []
            cur_traj_rewards = []
            cur_traj_vectors = []
            cur_traj_obs = []

            obs, _ = env.reset(seed=seed + n_episodes)
            obs_flat = _flatten_obs(obs)

        if total_steps % LOG_INTERVAL == 0:
            fr_K = margin_fired_K / max(1, margin_total_decisions)
            fr_P = margin_fired_P / max(1, margin_total_decisions)
            print(
                f"[train] env={env_id} seed={seed} steps={total_steps} eps={n_episodes} "
                f"K_fire={fr_K:.3f} P_fire={fr_P:.3f} train_s={time.monotonic()-t0:.1f}",
                flush=True,
            )

    env.close()

    fr_K = margin_fired_K / max(1, margin_total_decisions)
    fr_P = margin_fired_P / max(1, margin_total_decisions)
    print(
        f"[train] FINAL env={env_id} seed={seed} steps={total_steps} eps={n_episodes} "
        f"K_fire={fr_K:.3f} P_fire={fr_P:.3f} train_s={time.monotonic()-t0:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Build a deterministic policy_fn that uses base logits + (frozen) nudge
    # via the trained statistics. We freeze K, P, kmeans state.
    K_match_f = K_match.copy()
    K_count_f = K_count.copy()
    P_above_f = P_above.copy()
    P_below_f = P_below.copy()
    chan_active_f = chan_active.copy() if len(chan_reservoir) >= 4 else np.zeros(k_channels, dtype=bool)
    centroids_f = kmeans.centroids.copy()
    initialized_f = kmeans.initialized.copy()

    policy.eval()

    def assign_cluster(emb: np.ndarray) -> int:
        if not initialized_f.any():
            return 0
        valid = np.where(initialized_f)[0]
        diffs = centroids_f[valid] - emb[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        return int(valid[int(np.argmin(d2))])

    def policy_fn(obs_in: np.ndarray):
        flat = _flatten_obs(obs_in)
        x = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            emb = policy.embed(x).cpu().numpy().reshape(-1)
            logits_v = policy.logits(x).cpu().numpy().reshape(-1)
        c = assign_cluster(emb)
        denom = np.maximum(K_count_f[c], 1.0)
        K_slice = K_match_f[c] / denom
        zero_mask = (K_count_f[c] == 0)
        K_slice = np.where(zero_mask, 0.5, K_slice)
        denom_p = P_above_f[c] + P_below_f[c]
        denom_p_safe = np.where(denom_p > 0, denom_p, 1.0)
        P_slice = (P_above_f[c] - P_below_f[c]) / denom_p_safe
        P_slice = np.where((denom_p > 0), P_slice, 0.0)
        P_slice = P_slice * chan_active_f[None, :].astype(np.float64)
        Delta_K = pareto_margins(K_slice)
        Delta_P = pareto_margins(P_slice)
        Delta_P_floored = np.maximum(Delta_P, 1.0)
        nudge = alpha * Delta_K * Delta_P_floored
        nl = logits_v + nudge.astype(np.float32)
        return int(np.argmax(nl))

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
