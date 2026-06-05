"""BLIC: Block-Lookback Imminence Concordance.

Realizes worklogs/runs/20260606-32-auto/hypothesis.md.

Primitive: per-(cluster, action, channel) running empirical mean
    IC[s, a, m] = E[ q_m(o_{t+1}) - q_m(o_t) | cluster(o_t)=s, a_t=a ]
where q_m is a small supervised binary classifier predicting whether
channel m fires somewhere in the next H steps.

Improvement operator: at decision time, in cluster s, compute for each
action a the signed Pareto-non-dominance margin
    m_a = n_a^dom - n_a^sub
over the k-vector IC[s, a, :] vs. IC[s, a', :] for all a' != a, and
nudge logit(a) by alpha * m_a.

Execution rule: sample from softmax(logits + nudge); base policy net
trained by cross-entropy on selected action vs. the nudged distribution
(stop-gradient on the nudge).

Vector feedback rule: vector outcomes enter ONLY as binary windowed
labels y_m(t) = 1 iff channel m fires in {t+1, ..., t+H}, used to
supervise q_m. No scalarization, no weighted sum of channels.
"""

from __future__ import annotations

import argparse
import time

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


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class PolicyNet(nn.Module):
    """Small MLP with categorical action head and an exposed penultimate
    activation used as the cluster feature."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.head(h2)
        return logits, h2  # logits, penultimate features


class QClassifier(nn.Module):
    """k binary heads: q_m(o) -> P(channel m fires within next H steps)."""

    def __init__(self, obs_dim: int, k: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.head(h))


class OnlineKMeans:
    """Lightweight online k-means with cell-occupancy floor.

    K is fixed up front but cells may stay unused; cluster lookup returns
    the nearest occupied centroid (or assigns the next free centroid if
    the closest occupancy is below floor).
    """

    def __init__(self, K: int, dim: int, lr: float = 0.05, occ_floor: int = 4, seed: int = 0):
        self.K = K
        self.dim = dim
        self.lr = lr
        self.occ_floor = occ_floor
        self.rng = np.random.default_rng(seed)
        self.centroids = self.rng.standard_normal((K, dim)).astype(np.float32) * 0.01
        self.counts = np.zeros(K, dtype=np.int64)
        self.initialized = np.zeros(K, dtype=bool)

    def assign(self, feat: np.ndarray) -> int:
        # If any centroid uninitialized, claim it.
        free = np.where(~self.initialized)[0]
        if len(free) > 0:
            k = int(free[0])
            self.centroids[k] = feat.astype(np.float32)
            self.initialized[k] = True
            self.counts[k] += 1
            return k
        # Otherwise nearest centroid (Euclidean).
        diff = self.centroids - feat[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        k = int(np.argmin(d2))
        self.counts[k] += 1
        # Online update.
        self.centroids[k] += self.lr * (feat.astype(np.float32) - self.centroids[k])
        return k


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, k: int, n_actions: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity,), dtype=np.int64)
        # binary channel-fire mask at the transition itself: did m fire on this step?
        self.fires = np.zeros((capacity, k), dtype=np.float32)
        self.ep_idx = np.zeros((capacity,), dtype=np.int64)
        self.t_in_ep = np.zeros((capacity,), dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add(self, o, a, no, fire_vec, ep, t):
        i = self.ptr
        self.obs[i] = o
        self.next_obs[i] = no
        self.acts[i] = a
        self.fires[i] = fire_vec
        self.ep_idx[i] = ep
        self.t_in_ep[i] = t
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.size, size=n)


# ---------------------------------------------------------------------------
# Channel-firing labelling
# ---------------------------------------------------------------------------


def channel_fire_from_vector(vec: np.ndarray) -> np.ndarray:
    """Binary 'did this channel fire on this step' indicator. We treat
    nonzero entries (in either sign) as a firing event so step-penalty
    channels (always-on) and rare-event reward channels are both
    captured."""
    return (np.abs(vec) > 1e-12).astype(np.float32)


def windowed_labels_for_buffer(buf: ReplayBuffer, indices: np.ndarray, H: int) -> np.ndarray:
    """For each sampled transition i, label_m = 1 iff channel m fires in
    any of {i+1, ..., i+H} that share buf.ep_idx[i]."""
    k = buf.fires.shape[1]
    labels = np.zeros((len(indices), k), dtype=np.float32)
    size = buf.size
    for j, i in enumerate(indices):
        ep = buf.ep_idx[i]
        # walk forward up to H steps
        agg = np.zeros(k, dtype=np.float32)
        # The ring buffer makes "next index" tricky, so we just walk by storage
        # order; transitions within an episode are stored contiguously by
        # construction (we add one transition per env step and never interleave).
        for h in range(1, H + 1):
            ii = (i + h) % size
            if buf.ep_idx[ii] != ep:
                break
            agg = np.maximum(agg, buf.fires[ii])
        labels[j] = agg
    return labels


# ---------------------------------------------------------------------------
# Observation flattening
# ---------------------------------------------------------------------------


def flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = harness.make_env(env_id, seed)
    obs0, _ = env.reset(seed=seed)
    obs_flat = flatten_obs(obs0)
    obs_dim = obs_flat.shape[0]
    if not hasattr(env.action_space, "n"):
        env.close()
        raise RuntimeError("BLIC requires discrete action space")
    n_actions = int(env.action_space.n)

    is_vector = harness.ENV_TYPE[env_id] == "vector"
    # Probe k via one step on a fresh env to avoid disturbing the main env.
    if is_vector:
        probe = harness.make_env(env_id, seed + 1)
        probe.reset(seed=seed + 1)
        _, _, _, _, info0 = probe.step(probe.action_space.sample())
        k = int(np.asarray(info0["vector"]).shape[0])
        probe.close()
    else:
        # On scalar envs we still need k>=1; we treat the scalar as a
        # single channel (rebadge boundary, logged explicitly).
        k = 1

    # ----- Hyperparameters (chosen up front; not retuned) -----
    hidden = 64
    H = 6                  # imminence window
    K_clusters = 32        # online k-means K
    alpha = 1.5            # logit nudge scale
    qm_lr = 3e-3
    pol_lr = 3e-3
    qm_batch = 64
    pol_batch = 64
    update_every = 4       # env-steps between learner updates
    warmup_steps = 64      # env-steps before any nudge / qm update
    cell_floor = 4         # minimum (s,a) samples before that cell's IC is trusted
    log_every_s = 30.0

    # ----- Components -----
    policy = PolicyNet(obs_dim, n_actions, hidden=hidden)
    qm = QClassifier(obs_dim, k, hidden=hidden)
    pol_opt = torch.optim.Adam(policy.parameters(), lr=pol_lr)
    qm_opt = torch.optim.Adam(qm.parameters(), lr=qm_lr)
    kmeans = OnlineKMeans(K=K_clusters, dim=hidden, lr=0.05, occ_floor=cell_floor, seed=seed)

    # ----- Primitive state: IC tensor and counts -----
    ic_sum = np.zeros((K_clusters, n_actions, k), dtype=np.float64)
    ic_cnt = np.zeros((K_clusters, n_actions), dtype=np.int64)

    buf = ReplayBuffer(capacity=20_000, obs_dim=obs_dim, k=k, n_actions=n_actions)

    # Scratch features for q_m delta computation are kept in float32.
    obs = obs0
    obs_arr = flatten_obs(obs)
    ep_idx = 0
    t_in_ep = 0

    env_steps = 0
    nudge_fire_count = 0
    nudge_total_count = 0
    last_log_t = time.monotonic()
    t_start = time.monotonic()

    # Base softmax sampling rng (torch-controlled via manual_seed above).
    while time.monotonic() - t_start < time_budget_s:
        # ---- Decision-time: cluster lookup + Pareto nudge ----
        with torch.no_grad():
            ot = torch.from_numpy(obs_arr).unsqueeze(0)
            logits, feat = policy(ot)
            base_logits_np = logits.squeeze(0).numpy().copy()
            feat_np = feat.squeeze(0).numpy()
        s_idx = kmeans.assign(feat_np)

        # Compute per-action IC mean estimates (NaN where cell has too few
        # samples; in that case the action is treated as having no
        # comparable vector and contributes neither to dom nor sub counts).
        ic_means = np.full((n_actions, k), np.nan, dtype=np.float64)
        for a in range(n_actions):
            c = ic_cnt[s_idx, a]
            if c >= cell_floor:
                ic_means[a] = ic_sum[s_idx, a] / c

        # Pareto-non-dominance margin per action across observed
        # alternatives in this cluster.
        margins = np.zeros(n_actions, dtype=np.float64)
        valid = ~np.isnan(ic_means).any(axis=1)
        n_valid = int(valid.sum())
        if env_steps >= warmup_steps and n_valid >= 2:
            for a in range(n_actions):
                if not valid[a]:
                    continue
                ica = ic_means[a]
                dom = 0
                sub = 0
                for ap in range(n_actions):
                    if ap == a or not valid[ap]:
                        continue
                    icap = ic_means[ap]
                    # a Pareto-dominates a' iff coord-wise >= and strict on >=1
                    ge = ica >= icap
                    gt = ica > icap
                    if np.all(ge) and np.any(gt):
                        dom += 1
                    le = ica <= icap
                    lt = ica < icap
                    if np.all(le) and np.any(lt):
                        sub += 1
                margins[a] = dom - sub

        if np.any(margins != 0):
            nudge_fire_count += 1
        nudge_total_count += 1

        nudged_logits = base_logits_np + alpha * margins.astype(np.float32)
        # Numerically-stable softmax sample.
        nl = nudged_logits - nudged_logits.max()
        probs = np.exp(nl)
        probs = probs / probs.sum()
        a_t = int(rng.choice(n_actions, p=probs))

        # ---- Step env ----
        next_obs, reward, term, trunc, info = env.step(a_t)
        next_arr = flatten_obs(next_obs)
        if is_vector:
            vec = np.asarray(info["vector"], dtype=np.float64)
        else:
            vec = np.array([float(reward)], dtype=np.float64)
        fire_vec = channel_fire_from_vector(vec)

        # ---- Compute delta_m using q_m ON OBSERVED transitions only ----
        with torch.no_grad():
            qpair = qm(torch.from_numpy(np.stack([obs_arr, next_arr])))
        delta = (qpair[1] - qpair[0]).numpy().astype(np.float64)  # (k,)

        # ---- Update IC running mean ----
        ic_sum[s_idx, a_t] += delta
        ic_cnt[s_idx, a_t] += 1

        # ---- Push to replay ----
        buf.add(obs_arr.copy(), a_t, next_arr.copy(), fire_vec, ep_idx, t_in_ep)

        # ---- Update components every few steps ----
        if env_steps >= warmup_steps and env_steps % update_every == 0 and buf.size > qm_batch + H + 1:
            # q_m supervised step on windowed labels
            idxs = buf.sample_indices(qm_batch, rng)
            labels = windowed_labels_for_buffer(buf, idxs, H)
            obs_b = torch.from_numpy(buf.obs[idxs])
            lab_b = torch.from_numpy(labels)
            preds = qm(obs_b)
            qm_loss = F.binary_cross_entropy(preds.clamp(1e-6, 1 - 1e-6), lab_b)
            qm_opt.zero_grad()
            qm_loss.backward()
            qm_opt.step()

            # Policy supervised step: cross-entropy of selected action vs.
            # nudged categorical distribution computed for replayed states
            # using the current IC tensor and current cluster assignments.
            # Stop-gradient on the nudge term.
            pidxs = buf.sample_indices(pol_batch, rng)
            pol_obs = torch.from_numpy(buf.obs[pidxs])
            pol_acts = torch.from_numpy(buf.acts[pidxs]).long()
            logits_b, feats_b = policy(pol_obs)

            # Build per-sample nudge using current IC & current cluster.
            with torch.no_grad():
                feats_np = feats_b.numpy()
                nudge_b = np.zeros((pol_batch, n_actions), dtype=np.float32)
                for j in range(pol_batch):
                    sj = kmeans.assign(feats_np[j])  # this also updates centroid
                    ic_m = np.full((n_actions, k), np.nan, dtype=np.float64)
                    for a in range(n_actions):
                        c = ic_cnt[sj, a]
                        if c >= cell_floor:
                            ic_m[a] = ic_sum[sj, a] / c
                    valid_j = ~np.isnan(ic_m).any(axis=1)
                    if valid_j.sum() < 2:
                        continue
                    for a in range(n_actions):
                        if not valid_j[a]:
                            continue
                        ica = ic_m[a]
                        dom = 0
                        sub = 0
                        for ap in range(n_actions):
                            if ap == a or not valid_j[ap]:
                                continue
                            icap = ic_m[ap]
                            ge = ica >= icap
                            gt = ica > icap
                            if np.all(ge) and np.any(gt):
                                dom += 1
                            le = ica <= icap
                            lt = ica < icap
                            if np.all(le) and np.any(lt):
                                sub += 1
                        nudge_b[j, a] = alpha * (dom - sub)
            nudge_t = torch.from_numpy(nudge_b)  # detached tensor (no grad)

            # Cross-entropy of selected action against the nudged distribution
            # (target distribution = softmax(logits.detach() + nudge), so
            # there is a stop-gradient on the nudge term as required by the
            # hypothesis; we use a soft cross-entropy form to match the
            # "supervised on nudged categorical" execution rule).
            target_logits = logits_b.detach() + nudge_t
            target_probs = F.softmax(target_logits, dim=-1)
            log_probs = F.log_softmax(logits_b, dim=-1)
            # Hard CE on selected action (the recorded a_t) plus a small
            # KL-to-target term for the nudged-categorical signal.
            ce_selected = F.nll_loss(log_probs, pol_acts)
            kl_to_target = -(target_probs * log_probs).sum(dim=-1).mean()
            pol_loss = 0.5 * ce_selected + 0.5 * kl_to_target
            pol_opt.zero_grad()
            pol_loss.backward()
            pol_opt.step()

        # ---- Episode bookkeeping ----
        obs_arr = next_arr
        env_steps += 1
        t_in_ep += 1
        if term or trunc or t_in_ep >= harness.MAX_EPISODE_STEPS:
            obs, _ = env.reset(seed=seed + env_steps)
            obs_arr = flatten_obs(obs)
            ep_idx += 1
            t_in_ep = 0

        # ---- Periodic diagnostics ----
        now = time.monotonic()
        if now - last_log_t >= log_every_s:
            last_log_t = now
            occ = (kmeans.counts > 0).sum()
            frac_fire = nudge_fire_count / max(1, nudge_total_count)
            ic_abs_mean = np.abs(ic_sum / np.maximum(ic_cnt[..., None], 1)).mean()
            print(
                f"[blic] env={env_id} t={now - t_start:.1f}s steps={env_steps} "
                f"clusters_used={int(occ)}/{K_clusters} nudge_fire_frac={frac_fire:.3f} "
                f"ic_abs_mean={ic_abs_mean:.4g} k={k}",
                flush=True,
            )

    env.close()

    elapsed = time.monotonic() - t_start
    frac_fire = nudge_fire_count / max(1, nudge_total_count)
    print(
        f"[blic] FINAL env={env_id} train_s={elapsed:.1f} steps={env_steps} "
        f"nudge_fire_frac={frac_fire:.3f} clusters_used="
        f"{int((kmeans.counts > 0).sum())}/{K_clusters} k={k}",
        flush=True,
    )

    # ---- Build deterministic policy_fn ----
    # At eval time we apply the same Pareto nudge (with frozen IC tensor
    # and frozen kmeans) and take argmax of the nudged logits.
    frozen_ic_sum = ic_sum.copy()
    frozen_ic_cnt = ic_cnt.copy()
    frozen_centroids = kmeans.centroids.copy()
    frozen_init = kmeans.initialized.copy()

    def policy_fn(obs_in):
        oa = flatten_obs(obs_in)
        with torch.no_grad():
            logits, feat = policy(torch.from_numpy(oa).unsqueeze(0))
        base = logits.squeeze(0).numpy()
        feat_np = feat.squeeze(0).numpy()
        # Cluster lookup against frozen centroids (no online updates).
        if not frozen_init.any():
            return int(np.argmax(base))
        diff = frozen_centroids[frozen_init] - feat_np[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        # Map back to original cluster index.
        idx_in_init = int(np.argmin(d2))
        all_idx = np.where(frozen_init)[0]
        s_idx_e = int(all_idx[idx_in_init])

        ic_m = np.full((n_actions, k), np.nan, dtype=np.float64)
        for a in range(n_actions):
            c = frozen_ic_cnt[s_idx_e, a]
            if c >= cell_floor:
                ic_m[a] = frozen_ic_sum[s_idx_e, a] / c
        valid_e = ~np.isnan(ic_m).any(axis=1)
        margins_e = np.zeros(n_actions, dtype=np.float64)
        if valid_e.sum() >= 2:
            for a in range(n_actions):
                if not valid_e[a]:
                    continue
                ica = ic_m[a]
                dom = 0
                sub = 0
                for ap in range(n_actions):
                    if ap == a or not valid_e[ap]:
                        continue
                    icap = ic_m[ap]
                    ge = ica >= icap
                    gt = ica > icap
                    if np.all(ge) and np.any(gt):
                        dom += 1
                    le = ica <= icap
                    lt = ica < icap
                    if np.all(le) and np.any(lt):
                        sub += 1
                margins_e[a] = dom - sub
        nudged = base + alpha * margins_e.astype(np.float32)
        return int(np.argmax(nudged))

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
