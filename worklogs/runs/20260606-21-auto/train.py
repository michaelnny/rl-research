"""CSD: Channel-Conditional Successor Disagreement.

Realizes hypothesis 20260606-21-auto.

Primitive:
    For each (cluster c, action a, channel m), compute the empirical
    next-cluster distributions p_m^+ and p_m^- conditional on whether
    channel m fired in the K-step post-action window. The primitive is
    S[c,a,m] = TV(p_m^+, p_m^-) (Laplace smoothed).

Improvement operator:
    At cluster c, logit nudge Delta logit[c,a] = alpha * (n_a - m_a)
    where n_a (resp. m_a) is the count of other actions a' such that
    S[c,a,:] strictly Pareto-dominates (resp. is dominated by) S[c,a',:]
    in the coordinate-wise partial order on R_+^k.

Execution:
    Sample from softmax(base_logits[c,:] + Delta_logits[c,:]). The base
    logits are zero-initialized (uniform Dirichlet prior) ensuring an
    entropy floor. No critic, no Bellman backup, no return-to-go.

Vector feedback:
    Channels participate independently as partition variables. Channels
    with degenerate partitions (one side empty) are dropped from that
    (c,a) row's Pareto comparison. No scalar collapse.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

import harness


# ---------- pre-commit diagnostics knobs ----------
N_CLUSTERS = 64                 # cluster vocabulary size; <= 256 per hypothesis
K_WINDOW = 8                    # K-step post-action window
ALPHA = 2.0                     # logit nudge magnitude
LAPLACE_EPS = 0.5               # Laplace smoothing on cluster bins
REFRESH_EVERY = 512             # transitions between full S recomputes
BUFFER_CAPACITY = 50_000        # per-env transition buffer cap
KMEANS_LR = 0.05                # online k-means centroid update rate
WARMUP_TRANSITIONS = 256        # uniform sampling before any nudge


# ---------- arg parsing ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ---------- observation embedding ----------
def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    o = np.asarray(obs).astype(np.float32).reshape(-1)
    # Normalize MiniGrid uint8 obs to [0,1] approx
    if o.size > 32:
        o = o / 16.0
    return o


# ---------- online mini-batch k-means clusterer ----------
class OnlineKMeans:
    def __init__(self, n_clusters: int, dim: int, rng: np.random.Generator, lr: float = 0.05):
        self.n_clusters = n_clusters
        self.dim = dim
        self.rng = rng
        self.lr = lr
        # Centroids initialized small-random; first n_clusters fresh observations
        # overwrite seed centroids opportunistically.
        self.centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 0.01
        self.counts = np.zeros(n_clusters, dtype=np.int64)
        self.initialized = np.zeros(n_clusters, dtype=bool)

    def assign(self, x: np.ndarray) -> int:
        # Lazy initialization: pick first empty centroid for new obs
        if not self.initialized.all():
            empty_idx = int(np.argmin(self.initialized.astype(np.int32)))
            if not self.initialized[empty_idx]:
                self.centroids[empty_idx] = x
                self.initialized[empty_idx] = True
                self.counts[empty_idx] += 1
                return empty_idx
        # nearest centroid (squared L2)
        diffs = self.centroids - x[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        c = int(np.argmin(d2))
        # online update
        self.centroids[c] += self.lr * (x - self.centroids[c])
        self.counts[c] += 1
        return c


# ---------- channel construction ----------
def _channel_vector(reward: float, info: dict, env_id: str) -> np.ndarray:
    """Return the per-step channel vector. The 'step-penalty' channel is the
    constant-1 channel that fires every step (so the K-step window always has
    f=1 for it; falsifier specifically watches its dominance).

    For vector envs, we use info["vector"] components and define each channel's
    'firing' as |component| > 0 at that step.

    For scalar envs, we use the scalar reward sign as one channel, an absolute
    reward magnitude channel, and the constant step-penalty channel.
    """
    if "vector" in info:
        v = np.asarray(info["vector"], dtype=np.float64)
        # channel m fires if |v[m]| > 0 at that step
        fires = (np.abs(v) > 1e-9).astype(np.float32)
        # Append step-penalty channel (constant-1 firing)
        fires = np.concatenate([fires, np.array([1.0], dtype=np.float32)])
        return fires
    # scalar env: positive-reward channel, any-reward channel, step-penalty
    pos = 1.0 if reward > 1e-9 else 0.0
    nonzero = 1.0 if abs(reward) > 1e-9 else 0.0
    return np.array([pos, nonzero, 1.0], dtype=np.float32)


def _n_channels(env_id: str) -> int:
    # match channel_vector layout
    if env_id == "deep-sea-treasure-concave-v0":
        return 2 + 1  # 2 components + step
    if env_id == "resource-gathering-v0":
        return 3 + 1
    if env_id.startswith("Craftax-"):
        return 3
    # MiniGrid scalar
    return 3


# ---------- transition buffer ----------
class CSDBuffer:
    """Stores rollout transitions and computes S[c,a,m] by buffer scan."""

    def __init__(self, n_clusters: int, n_actions: int, n_channels: int, capacity: int):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.capacity = capacity
        # parallel arrays
        self.c = np.zeros(capacity, dtype=np.int32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.cn = np.zeros(capacity, dtype=np.int32)  # next cluster
        self.fires = np.zeros((capacity, n_channels), dtype=np.float32)  # K-step firing window
        self.size = 0
        self.cursor = 0

    def append(self, c: int, a: int, cn: int, fires: np.ndarray) -> None:
        i = self.cursor
        self.c[i] = c
        self.a[i] = a
        self.cn[i] = cn
        self.fires[i] = fires
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def compute_S(self, eps: float = LAPLACE_EPS) -> np.ndarray:
        """Return S[c,a,m] of shape (n_clusters, n_actions, n_channels).

        For each (c, a, m): partition transitions with that c,a by whether
        f_m=1; compute cluster histograms over cn for each side; smooth with
        Laplace prior eps; compute total-variation distance.
        """
        S = np.zeros((self.n_clusters, self.n_actions, self.n_channels), dtype=np.float32)
        valid = np.zeros((self.n_clusters, self.n_actions, self.n_channels), dtype=bool)
        if self.size == 0:
            return S, valid

        c = self.c[: self.size]
        a = self.a[: self.size]
        cn = self.cn[: self.size]
        fires = self.fires[: self.size]
        n_c = self.n_clusters
        n_a = self.n_actions
        n_m = self.n_channels

        # Bucket transitions by (c, a): build a flat key
        key = (c.astype(np.int64) * n_a) + a.astype(np.int64)
        # for each unique (c,a), gather indices
        unique_keys, inv = np.unique(key, return_inverse=True)
        for ki, uk in enumerate(unique_keys):
            mask = inv == ki
            cur_cn = cn[mask]
            cur_fires = fires[mask]
            cc = int(uk // n_a)
            aa = int(uk % n_a)
            for m in range(n_m):
                fm = cur_fires[:, m] > 0.5
                pos = cur_cn[fm]
                neg = cur_cn[~fm]
                if pos.size == 0 or neg.size == 0:
                    # degenerate partition: dropped
                    S[cc, aa, m] = 0.0
                    valid[cc, aa, m] = False
                    continue
                # Laplace-smoothed histograms over cluster vocabulary
                p_pos = np.bincount(pos, minlength=n_c).astype(np.float64) + eps
                p_neg = np.bincount(neg, minlength=n_c).astype(np.float64) + eps
                p_pos /= p_pos.sum()
                p_neg /= p_neg.sum()
                S[cc, aa, m] = float(0.5 * np.abs(p_pos - p_neg).sum())
                valid[cc, aa, m] = True
        return S, valid


# ---------- pareto vote ----------
def pareto_vote(S_row: np.ndarray, valid_row: np.ndarray) -> np.ndarray:
    """Given S_row (n_actions, n_channels) and valid_row (n_actions, n_channels),
    return per-action (n_a - m_a) score.

    For a pairwise comparison between actions a and a', restrict to channels
    valid for *both* rows. Strict coordinate-wise dominance: all_geq AND any_gt.
    """
    n_a, n_m = S_row.shape
    delta = np.zeros(n_a, dtype=np.float32)
    if n_a < 2:
        return delta
    for a in range(n_a):
        n_dom = 0  # number of a' that a dominates
        n_dom_by = 0  # number of a' that dominate a
        for ap in range(n_a):
            if ap == a:
                continue
            both_valid = valid_row[a] & valid_row[ap]
            if not both_valid.any():
                continue
            sa = S_row[a, both_valid]
            sap = S_row[ap, both_valid]
            # a dominates a' if sa >= sap on all valid channels and > on at least one
            if np.all(sa >= sap) and np.any(sa > sap):
                n_dom += 1
            elif np.all(sap >= sa) and np.any(sap > sa):
                n_dom_by += 1
        delta[a] = float(n_dom - n_dom_by)
    return delta


# ---------- training loop ----------
def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n)
    n_channels = _n_channels(env_id)

    # Determine obs dim from a reset
    obs0, _ = env.reset(seed=seed)
    obs_dim = int(np.prod(np.asarray(obs0).shape))

    clusterer = OnlineKMeans(
        n_clusters=N_CLUSTERS, dim=obs_dim, rng=rng, lr=KMEANS_LR
    )
    buffer = CSDBuffer(
        n_clusters=N_CLUSTERS,
        n_actions=n_actions,
        n_channels=n_channels,
        capacity=BUFFER_CAPACITY,
    )

    # S table — recomputed periodically
    S = np.zeros((N_CLUSTERS, n_actions, n_channels), dtype=np.float32)
    S_valid = np.zeros((N_CLUSTERS, n_actions, n_channels), dtype=bool)
    # Cached per-cluster Delta logits (n_clusters, n_actions)
    delta_logits = np.zeros((N_CLUSTERS, n_actions), dtype=np.float32)

    # Diagnostic accumulators
    n_pareto_nonzero = 0
    n_decision_steps_after_warmup = 0
    n_dead_in_loop = 0  # debug counter

    t_start = time.monotonic()
    deadline = t_start - 0.5  # leave 0.5s slack
    train_deadline = t_start + max(1, time_budget_s - 5)

    obs = obs0
    obs_emb = _flatten_obs(obs)
    c_t = clusterer.assign(obs_emb)

    # episode-local pending (waiting for K-step firing window)
    pending = deque()  # entries: (c_t, a_t, c_tp1, fires_accum, steps_remaining)
    total_transitions = 0
    refresh_count = 0
    last_refresh_size = 0

    def _flush_pending(force_all: bool = False):
        # Drain pending entries whose K-step window is closed (remaining<=0)
        while pending and (force_all or pending[0][4] <= 0):
            c_a, a_a, cn_a, fires_a, _ = pending.popleft()
            buffer.append(c_a, a_a, cn_a, fires_a)

    last_diag_print = t_start
    DIAG_INTERVAL = 30.0

    while time.monotonic() < train_deadline:
        # --- decision time ---
        if total_transitions < WARMUP_TRANSITIONS:
            action = int(rng.integers(n_actions))
        else:
            row_S = S[c_t]
            row_valid = S_valid[c_t]
            # If no valid info at all for this cluster, sample uniformly.
            if not row_valid.any():
                action = int(rng.integers(n_actions))
            else:
                logits = ALPHA * delta_logits[c_t]
                # softmax
                logits = logits - logits.max()
                p = np.exp(logits)
                p /= p.sum()
                action = int(rng.choice(n_actions, p=p))
                if np.any(np.abs(delta_logits[c_t]) > 0):
                    n_pareto_nonzero += 1
                n_decision_steps_after_warmup += 1

        # --- env step ---
        try:
            obs_next, reward, term, trunc, info = env.step(action)
        except Exception:
            # extremely defensive
            n_dead_in_loop += 1
            try:
                obs, _ = env.reset(seed=seed + total_transitions)
            except Exception:
                break
            obs_emb = _flatten_obs(obs)
            c_t = clusterer.assign(obs_emb)
            pending.clear()
            continue

        obs_next_emb = _flatten_obs(obs_next)
        c_tp1 = clusterer.assign(obs_next_emb)
        ch_fires_now = _channel_vector(float(reward), info, env_id)

        # advance K-window for previous pending entries
        for i in range(len(pending)):
            c_a, a_a, cn_a, fires_a, rem = pending[i]
            # OR-in current step's firing
            fires_a = np.maximum(fires_a, ch_fires_now)
            pending[i] = (c_a, a_a, cn_a, fires_a, rem - 1)

        # add new pending entry (the just-taken action)
        pending.append((
            int(c_t),
            int(action),
            int(c_tp1),
            np.zeros(n_channels, dtype=np.float32) + ch_fires_now,  # step-after-action firings
            K_WINDOW - 1,  # we just consumed step 1 of the K-window
        ))

        # flush any that hit K
        _flush_pending(force_all=False)
        total_transitions += 1

        # episode reset
        if bool(term) or bool(trunc):
            # close out remaining pending entries (their K-window is truncated)
            _flush_pending(force_all=True)
            try:
                obs_next, _ = env.reset(seed=seed + 1000 + total_transitions)
            except Exception:
                break
            obs_next_emb = _flatten_obs(obs_next)
            c_tp1 = clusterer.assign(obs_next_emb)
            pending.clear()

        c_t = c_tp1
        obs = obs_next

        # --- periodic refresh of S and delta_logits ---
        if (
            total_transitions - last_refresh_size >= REFRESH_EVERY
            and buffer.size > 0
        ):
            last_refresh_size = total_transitions
            S, S_valid = buffer.compute_S()
            # recompute delta logits per cluster
            for cc in range(N_CLUSTERS):
                delta_logits[cc] = pareto_vote(S[cc], S_valid[cc])
            refresh_count += 1
            # Pre-commit diagnostic: per-channel mean S across populated cells
            now = time.monotonic()
            if now - last_diag_print >= DIAG_INTERVAL:
                last_diag_print = now
                # per-channel mean over (c,a) where any-channel valid
                any_valid_cell = S_valid.any(axis=2)  # (n_clusters, n_actions)
                if any_valid_cell.any():
                    mean_per_m = []
                    for m in range(n_channels):
                        cell_m_valid = S_valid[..., m]
                        if cell_m_valid.any():
                            mean_per_m.append(float(S[..., m][cell_m_valid].mean()))
                        else:
                            mean_per_m.append(0.0)
                    pop = int(any_valid_cell.sum())
                    cluster_occ = clusterer.counts.sum()
                    eff_clusters = int((clusterer.counts > 0).sum())
                    print(
                        f"[csd] env={env_id} t={now - t_start:.1f}s "
                        f"transitions={total_transitions} "
                        f"valid_cells={pop} "
                        f"eff_clusters={eff_clusters} "
                        f"mean_S_per_m={['%.4f' % x for x in mean_per_m]} "
                        f"pareto_fire_rate={n_pareto_nonzero / max(1, n_decision_steps_after_warmup):.3f}",
                        flush=True,
                    )

    env.close()

    # Final S refresh
    if buffer.size > 0:
        S, S_valid = buffer.compute_S()
        for cc in range(N_CLUSTERS):
            delta_logits[cc] = pareto_vote(S[cc], S_valid[cc])

    # Final diagnostic line
    pareto_rate = n_pareto_nonzero / max(1, n_decision_steps_after_warmup)
    eff_clusters = int((clusterer.counts > 0).sum())
    print(
        f"[csd-final] env={env_id} transitions={total_transitions} "
        f"refreshes={refresh_count} eff_clusters={eff_clusters} "
        f"pareto_fire_rate={pareto_rate:.3f} "
        f"n_dead_in_loop={n_dead_in_loop}",
        flush=True,
    )

    # ---------- deterministic policy_fn (uses argmax over logits to keep eval reproducible) ----------
    # We use argmax of (delta_logits[c]) at decision time. If a cluster has no
    # valid information, fall back to a fixed seeded rng action so behavior is
    # deterministic across calls within a single evaluation.
    eval_rng = np.random.default_rng(seed + 99_991)

    def policy_fn(obs_e):
        x = _flatten_obs(np.asarray(obs_e))
        # nearest centroid (no centroid update at eval)
        if not clusterer.initialized.any():
            return int(eval_rng.integers(n_actions))
        # only compare initialized centroids
        init_idx = np.where(clusterer.initialized)[0]
        diffs = clusterer.centroids[init_idx] - x[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        cc = int(init_idx[int(np.argmin(d2))])
        if not S_valid[cc].any():
            return int(eval_rng.integers(n_actions))
        scores = delta_logits[cc]
        if np.all(scores == scores[0]):
            return int(eval_rng.integers(n_actions))
        return int(np.argmax(scores))

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
