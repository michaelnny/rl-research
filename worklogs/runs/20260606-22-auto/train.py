"""KTAC: Kendall-Tau Action-Concordance with Per-Channel Order Consensus.

Substrate contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120

Realizes the hypothesis 20260606-22-auto faithfully:

  * Experience object: replay buffer of complete trajectories. Each step
    records (o_t, a_t, v_t, h_t) and the trajectory records terminal
    cumulant c_T = sum_t v_t per channel.
  * Context-cluster index: online k-means over policy penultimate-layer
    features h(o_t), refit periodically.
  * Core primitive: P_m[s, a, a'] -- per-channel pairwise dominance
    probabilities accumulated over trajectory pairs that share a context
    cluster s at some step and chose different actions there. The vote is
    the SIGN of the channel-m terminal-cumulant difference; magnitudes
    never enter.
  * Improvement operator: Kemeny consensus order sigma*(s) over the k
    per-channel pairwise matrices, computed via Borda-initialized swap
    improvement.
  * Execution rule: a ~ softmax(logit_theta(o) + alpha * score_sigma*(cluster(o))(.)).
  * Vector feedback rule: each channel m yields its own P_m. Channels
    are aggregated only via Kemeny consensus on rankings -- never by any
    scalar combination wᵀr of channel cumulants.

For scalar envs (DoorKey, KeyCorridor, Craftax) we treat the scalar
reward as a single channel so the same primitive applies.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Observation flattening
# ----------------------------------------------------------------------------


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr.reshape(-1)


# ----------------------------------------------------------------------------
# Policy network: small MLP returning logits and a penultimate-layer feature.
# ----------------------------------------------------------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, feat: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, feat)
        self.head = nn.Linear(feat, n_actions)
        self.feat_dim = feat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        logits = self.head(h2)
        return logits, h2


# ----------------------------------------------------------------------------
# Online k-means context-cluster store (Euclidean over penultimate features).
# ----------------------------------------------------------------------------


class OnlineKMeans:
    def __init__(self, k: int, dim: int, rng: np.random.Generator):
        self.k = k
        self.dim = dim
        self.rng = rng
        self.centroids: np.ndarray | None = None
        self.counts = np.zeros(k, dtype=np.int64)

    def fit(self, points: np.ndarray, n_iters: int = 5) -> None:
        if len(points) < self.k:
            # Not enough points to seed clusters; fall back to noisy duplicates.
            base = points.mean(axis=0, keepdims=True) if len(points) else np.zeros((1, self.dim), np.float32)
            self.centroids = base.repeat(self.k, axis=0) + 0.01 * self.rng.standard_normal((self.k, self.dim)).astype(np.float32)
            return
        idx = self.rng.choice(len(points), size=self.k, replace=False)
        centroids = points[idx].copy()
        for _ in range(n_iters):
            # Assign
            d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            assign = d2.argmin(axis=1)
            new_centroids = centroids.copy()
            for c in range(self.k):
                mask = assign == c
                if mask.any():
                    new_centroids[c] = points[mask].mean(axis=0)
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        self.centroids = centroids

    def assign(self, point: np.ndarray) -> int:
        if self.centroids is None:
            return 0
        d2 = ((self.centroids - point[None, :]) ** 2).sum(axis=1)
        return int(d2.argmin())

    def assign_batch(self, points: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            return np.zeros(len(points), dtype=np.int64)
        d2 = ((points[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return d2.argmin(axis=1)


# ----------------------------------------------------------------------------
# Kemeny consensus: Borda-initialized swap improvement on pairwise prob matrices.
# ----------------------------------------------------------------------------


def kemeny_consensus(p_matrices: list[np.ndarray]) -> np.ndarray:
    """Compute Kemeny consensus permutation across k channels.

    p_matrices: list of (A, A) arrays, p[a, a'] = P(a beats a' on channel m).
    Returns a permutation (rank order) sigma of length A; sigma[0] is the
    top-ranked action.
    """
    A = p_matrices[0].shape[0]
    if A <= 1:
        return np.arange(A, dtype=np.int64)

    # Aggregate disagreement matrix D[a, b] = sum_m (1 - p_m[a, b]); placing
    # a before b incurs cost D[b, a]. We minimize total cost.
    D = np.zeros((A, A), dtype=np.float64)
    for P in p_matrices:
        # Cost of placing a above b = (1 - P[a, b]) since P[a, b] is "a beats b"
        D += 1.0 - P

    # Borda initialization: rank actions by row-sum of (sum_m P_m[a, .]).
    borda = np.zeros(A, dtype=np.float64)
    for P in p_matrices:
        borda += P.sum(axis=1)
    sigma = np.argsort(-borda)  # descending => best first

    def total_cost(perm: np.ndarray) -> float:
        cost = 0.0
        for i in range(A):
            for j in range(i + 1, A):
                # perm[i] is placed above perm[j]; cost is sum_m (1 - P_m[perm[i], perm[j]])
                cost += D[perm[j], perm[i]]
        return cost

    # Local search: try adjacent swaps until no improvement.
    improved = True
    cur_cost = total_cost(sigma)
    iters = 0
    max_iters = 4 * A * A  # bounded so we never spin
    while improved and iters < max_iters:
        improved = False
        for i in range(A - 1):
            a = sigma[i]
            b = sigma[i + 1]
            # Swap delta: only the (a,b) pair changes orientation.
            delta = D[a, b] - D[b, a]
            if delta < -1e-12:
                sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
                cur_cost += delta
                improved = True
                iters += 1
                if iters >= max_iters:
                    break
    return sigma


# ----------------------------------------------------------------------------
# KTAC training loop
# ----------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s) - 5  # leave headroom for return

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed + 31337)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    obs0, _ = env.reset(seed=seed)
    obs_dim = int(np.prod(np.asarray(obs0).shape))
    if not hasattr(env.action_space, "n"):
        # Continuous action env not supported by KTAC; fall back to random.
        n_actions = 1

        def policy_fn(_obs):
            return env.action_space.sample()

        env.close()
        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 budget_s={time_budget_s} note=continuous-fallback",
            flush=True,
        )
        return policy_fn

    n_actions = int(env.action_space.n)

    # Probe vector channel count by stepping once.
    probe_act = int(np_rng.integers(n_actions))
    _o, _r, term, trunc, info = env.step(probe_act)
    if is_vector:
        n_channels = int(np.asarray(info["vector"]).reshape(-1).shape[0])
    else:
        n_channels = 1
    # Reset after probe.
    obs, _ = env.reset(seed=seed)

    # Hyperparameters (kept conservative; not tuned to chase numbers).
    K_CLUSTERS = 16
    HIDDEN = 64
    FEAT = 32
    ALPHA = 1.0  # logit nudge magnitude
    LR = 3e-3
    REFIT_EVERY_EPISODES = 4
    KEMENY_EVERY_EPISODES = 2
    MAX_BUFFER_TRAJS = 256
    MAX_PAIRS_PER_UPDATE = 4096

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyNet(obs_dim, n_actions, hidden=HIDDEN, feat=FEAT).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # Replay buffer of trajectories. Each trajectory:
    #   feats: (T, FEAT) numpy
    #   actions: (T,) int
    #   c_T: (n_channels,) numpy float -- terminal cumulant
    buffer: deque[dict] = deque(maxlen=MAX_BUFFER_TRAJS)

    # Online k-means.
    kmeans = OnlineKMeans(K_CLUSTERS, FEAT, np_rng)

    # Per-(cluster) Kemeny consensus order. None until enough data.
    sigma_star: dict[int, np.ndarray] = {}
    # Per-cluster rank-position score, centered: score[a] = (|A| - rank(a)) / (|A| - 1) - 0.5
    score_table: dict[int, np.ndarray] = {}

    def compute_score_from_sigma(sigma: np.ndarray) -> np.ndarray:
        A = len(sigma)
        score = np.zeros(A, dtype=np.float32)
        if A <= 1:
            return score
        for rank, act in enumerate(sigma):
            # rank 0 => best
            score[act] = (A - 1 - rank) / (A - 1) - 0.5
        return score

    # Sample-action helper: feed-forward, add nudge, sample.
    def act_with_nudge(obs_np: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        x = torch.from_numpy(flatten_obs(obs_np)).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, feat = net(x)
        feat_np = feat.squeeze(0).cpu().numpy()
        cluster = kmeans.assign(feat_np)
        nudge = score_table.get(cluster)
        logits_np = logits.squeeze(0).cpu().numpy().astype(np.float64)
        if nudge is not None:
            logits_np = logits_np + ALPHA * nudge.astype(np.float64)
        # softmax-sample
        m = logits_np.max()
        probs = np.exp(logits_np - m)
        probs = probs / probs.sum()
        a = int(np_rng.choice(n_actions, p=probs))
        return a, feat_np, logits_np

    # Recompute Kemeny consensus from buffer.
    def update_kemeny():
        if len(buffer) < 4:
            return
        # Refit k-means on a sample of feature points.
        feats_all = []
        for traj in buffer:
            feats_all.append(traj["feats"])
        feats_concat = np.concatenate(feats_all, axis=0)
        if len(feats_concat) > 4096:
            sub = np_rng.choice(len(feats_concat), size=4096, replace=False)
            feats_concat = feats_concat[sub]
        kmeans.fit(feats_concat, n_iters=5)

        # Re-assign clusters for every step in every trajectory.
        for traj in buffer:
            traj["clusters"] = kmeans.assign_batch(traj["feats"])

        # Build pairwise count tensor: counts[s, m, a, a'] = # pairs where action a was chosen
        # at a step in cluster s, the other trajectory chose a' at a step in same cluster s,
        # and sign(c_T_self[m] - c_T_other[m]) > 0.
        # We sample random pairs of trajectories.
        N = np.zeros((K_CLUSTERS, n_channels, n_actions, n_actions), dtype=np.float64)
        T = len(buffer)
        if T < 2:
            return
        # Build cluster->list of (traj_idx, step_idx, action) for fast pair sampling.
        cluster_records: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for ti, traj in enumerate(buffer):
            clusters = traj["clusters"]
            actions = traj["actions"]
            for step_idx in range(len(clusters)):
                cluster_records[int(clusters[step_idx])].append((ti, step_idx, int(actions[step_idx])))
        cT_arr = np.stack([traj["c_T"] for traj in buffer], axis=0)  # (T, n_channels)

        n_pairs_drawn = 0
        budget_per_cluster = max(8, MAX_PAIRS_PER_UPDATE // max(1, len(cluster_records)))
        for cluster_id, records in cluster_records.items():
            if len(records) < 2:
                continue
            # Sample up to budget_per_cluster pairs of records.
            n_pairs = min(budget_per_cluster, len(records) * (len(records) - 1) // 2)
            if n_pairs <= 0:
                continue
            idxs_i = np_rng.integers(0, len(records), size=n_pairs)
            idxs_j = np_rng.integers(0, len(records), size=n_pairs)
            for i, j in zip(idxs_i, idxs_j, strict=False):
                if i == j:
                    continue
                ti, _si, ai = records[i]
                tj, _sj, aj = records[j]
                if ti == tj:
                    continue
                if ai == aj:
                    continue
                diff = cT_arr[ti] - cT_arr[tj]
                # Per channel sign vote.
                for m in range(n_channels):
                    s = diff[m]
                    if s > 0:
                        N[cluster_id, m, ai, aj] += 1.0
                    elif s < 0:
                        N[cluster_id, m, aj, ai] += 1.0
                    # ties contribute nothing
                n_pairs_drawn += 1
                if n_pairs_drawn >= MAX_PAIRS_PER_UPDATE:
                    break
            if n_pairs_drawn >= MAX_PAIRS_PER_UPDATE:
                break

        # Compute P_m[s, a, a'] = N[s, m, a, a'] / (N[s, m, a, a'] + N[s, m, a', a]).
        # Where denominator is 0, leave as 0.5 (uninformative).
        sigma_new: dict[int, np.ndarray] = {}
        score_new: dict[int, np.ndarray] = {}
        for s in range(K_CLUSTERS):
            # Only build if there is *any* nonzero count at this cluster.
            if N[s].sum() == 0:
                continue
            p_per_channel = []
            any_signal = False
            for m in range(n_channels):
                Nm = N[s, m]
                denom = Nm + Nm.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    Pm = np.where(denom > 0, Nm / np.maximum(denom, 1e-12), 0.5)
                np.fill_diagonal(Pm, 0.5)
                if denom.sum() > 0:
                    any_signal = True
                p_per_channel.append(Pm)
            if not any_signal:
                continue
            sigma = kemeny_consensus(p_per_channel)
            sigma_new[s] = sigma
            score_new[s] = compute_score_from_sigma(sigma)

        sigma_star.clear()
        sigma_star.update(sigma_new)
        score_table.clear()
        score_table.update(score_new)

    # Policy gradient style update toward the Kemeny consensus order:
    # The hypothesis specifies a logit nudge alpha * score_sigma*(s)(.) at execution.
    # That nudge alone (without learning) does not propagate gradients through the
    # network. To realize "logit nudge ... by gradient descent on a smooth surrogate
    # of Kendall distance" (item 10), we periodically train the network so that its
    # logits *also* move toward sigma* via cross-entropy with the soft consensus
    # distribution exp(score_sigma*(.)). This is the smooth-Kendall surrogate.
    def train_policy_toward_consensus():
        if not score_table:
            return
        # Collect per-cluster centroids->target distribution from score_table.
        targets = {}
        for s, score in score_table.items():
            t = np.exp(score / max(1e-6, 1.0))
            t = t / t.sum()
            targets[s] = t
        # Sample a batch of (obs, target) from the buffer using assigned clusters.
        all_obs_flat = []
        all_targets = []
        for traj in buffer:
            clusters = traj["clusters"]
            for step_idx in range(len(clusters)):
                c = int(clusters[step_idx])
                if c in targets:
                    all_obs_flat.append(traj["obs_flat"][step_idx])
                    all_targets.append(targets[c])
        if not all_obs_flat:
            return
        # Subsample.
        n = len(all_obs_flat)
        batch = min(512, n)
        idxs = np_rng.choice(n, size=batch, replace=False) if n > batch else np.arange(n)
        x = torch.from_numpy(np.stack([all_obs_flat[i] for i in idxs], axis=0)).to(device)
        tgt = torch.from_numpy(np.stack([all_targets[i] for i in idxs], axis=0)).to(device).float()
        for _ in range(2):
            logits, _ = net(x)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(tgt * log_probs).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    # Rollout loop.
    episode_count = 0
    total_steps = 0
    cur_obs = obs
    cur_traj_obs_flat: list[np.ndarray] = []
    cur_traj_feats: list[np.ndarray] = []
    cur_traj_actions: list[int] = []
    cur_traj_vec_sum = np.zeros(n_channels, dtype=np.float64)
    ep_steps = 0

    MAX_EPISODE_STEPS = 1000

    while time.monotonic() < deadline:
        a, feat_np, _logits = act_with_nudge(cur_obs)
        try:
            next_obs, reward, term, trunc, info = env.step(a)
        except Exception:
            break
        ep_steps += 1
        total_steps += 1

        cur_traj_obs_flat.append(flatten_obs(cur_obs))
        cur_traj_feats.append(feat_np)
        cur_traj_actions.append(a)
        if is_vector:
            v = np.asarray(info.get("vector", np.zeros(n_channels)), dtype=np.float64).reshape(-1)
            if v.shape[0] != n_channels:
                v = np.zeros(n_channels, dtype=np.float64)
        else:
            v = np.array([float(reward)], dtype=np.float64)
        cur_traj_vec_sum += v

        done = bool(term) or bool(trunc) or ep_steps >= MAX_EPISODE_STEPS
        cur_obs = next_obs
        if done:
            buffer.append(
                {
                    "obs_flat": np.stack(cur_traj_obs_flat, axis=0).astype(np.float32),
                    "feats": np.stack(cur_traj_feats, axis=0).astype(np.float32),
                    "actions": np.asarray(cur_traj_actions, dtype=np.int64),
                    "c_T": cur_traj_vec_sum.copy(),
                }
            )
            cur_traj_obs_flat = []
            cur_traj_feats = []
            cur_traj_actions = []
            cur_traj_vec_sum = np.zeros(n_channels, dtype=np.float64)
            ep_steps = 0
            episode_count += 1

            if episode_count % KEMENY_EVERY_EPISODES == 0:
                update_kemeny()
            if episode_count % REFIT_EVERY_EPISODES == 0:
                train_policy_toward_consensus()

            try:
                cur_obs, _ = env.reset(seed=seed + 1000 + episode_count)
            except Exception:
                break

    env.close()

    train_s = time.monotonic() - t_start
    print(
        f"[train] env={env_id} seed={seed} env_steps={total_steps} train_s={train_s:.1f} "
        f"budget_s={time_budget_s} episodes={episode_count} clusters_with_sigma={len(sigma_star)} "
        f"channels={n_channels}",
        flush=True,
    )

    # Build deterministic-ish policy_fn for evaluation.
    # The hypothesis's execution rule is sample softmax(logits + alpha*score). We honor
    # that here too (returning argmax of the nudged logits would be a different rule).
    def policy_fn(obs_np: np.ndarray):
        x = torch.from_numpy(flatten_obs(obs_np)).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, feat = net(x)
        feat_np = feat.squeeze(0).cpu().numpy()
        cluster = kmeans.assign(feat_np)
        nudge = score_table.get(cluster)
        logits_np = logits.squeeze(0).cpu().numpy().astype(np.float64)
        if nudge is not None:
            logits_np = logits_np + ALPHA * nudge.astype(np.float64)
        m = logits_np.max()
        probs = np.exp(logits_np - m)
        probs = probs / probs.sum()
        return int(np_rng.choice(n_actions, p=probs))

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
