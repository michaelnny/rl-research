"""CRP — Channel Rank-Position Concordance.

Per (state-cluster, action, channel) running mean of within-trajectory
rank-percentile of the channel's firing magnitude, restricted to firing
steps. Improvement operator: at each visited cluster (with at least two
visited actions), mark actions whose trend-corrected rank vector
R_tilde[s, a, :] is Pareto-non-dominated by every other visited action;
apply a fixed-magnitude logit nudge alpha toward those preferred actions.
Greedy execution on nudged logits with a small uniform exploration floor.

No critic, no Bellman backup, no scalar advantage, no scalarized weight.

Vector feedback (info["vector"]) supplies the per-step channel reading.
For scalar envs, we synthesize a 1-channel vector from the scalar reward
(the operator then degenerates to "prefer early-firing actions" — the
hypothesis explicitly flags this k=1 regime as scalarization-equivalent;
we do not claim novelty there).
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cluster hash: coarse observation -> integer bucket id.
# ---------------------------------------------------------------------------

def _cluster_hash(obs, n_buckets: int) -> int:
    arr = np.asarray(obs)
    if arr.dtype.kind == "f":
        # Quantize floats to reduce singleton cells on continuous-ish obs.
        q = np.round(arr * 4.0).astype(np.int32)
        b = q.tobytes()
    else:
        b = arr.astype(np.int32, copy=False).tobytes()
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "little") % n_buckets


# ---------------------------------------------------------------------------
# Within-trajectory rank-percentile per channel, restricted to firing steps.
# ---------------------------------------------------------------------------

def _within_traj_rank_percentile(v: np.ndarray, eps_zero: float = 1e-12) -> np.ndarray:
    """rho[t, m] = ordinal rank of |v[t, m]| within firing steps of channel m,
    divided by |S_m|. nan when v[t, m] is not a firing step (|v| <= eps_zero).
    """
    T, k = v.shape
    rho = np.full((T, k), np.nan, dtype=np.float64)
    mag = np.abs(v)
    for m in range(k):
        firing = mag[:, m] > eps_zero
        idx = np.flatnonzero(firing)
        if idx.size == 0:
            continue
        vals = mag[idx, m]
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(idx.size, dtype=np.float64)
        ranks[order] = np.arange(1, idx.size + 1, dtype=np.float64)
        rho[idx, m] = ranks / float(idx.size)
    return rho


# ---------------------------------------------------------------------------
# CRP trainer.
# ---------------------------------------------------------------------------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n)

    # Fixed CRP hyperparameters (not tuned).
    N_BUCKETS = 4096
    ALPHA = 1.0           # fixed logit nudge magnitude
    EPS_FLOOR = 0.05      # uniform exploration floor at execution
    EPS_TRAIN = 0.30      # higher exploration during training
    N_SEED_EPISODES = 8   # seeding phase before operator fires
    EPS_ZERO = 1e-9       # firing-step threshold

    # Probe one step to detect vector channel and infer k.
    obs, _ = env.reset(seed=seed)
    probe_action = int(rng.integers(n_actions))
    _, _, _, _, info = env.step(probe_action)
    if "vector" in info:
        k = int(np.asarray(info["vector"]).shape[0])
        has_vector = True
    else:
        k = 1
        has_vector = False
    obs, _ = env.reset(seed=seed)

    logit_nudge = np.zeros((N_BUCKETS, n_actions), dtype=np.float64)
    R_sum = np.zeros((N_BUCKETS, n_actions, k), dtype=np.float64)
    R_cnt = np.zeros((N_BUCKETS, n_actions, k), dtype=np.int64)
    visit_cnt = np.zeros((N_BUCKETS, n_actions), dtype=np.int64)
    # Buffer-wide signed sum/count of channel firing magnitudes (for trend sigma).
    chan_signed_sum = np.zeros(k, dtype=np.float64)
    chan_signed_cnt = np.zeros(k, dtype=np.int64)

    t0 = time.monotonic()
    n_episodes = 0
    n_steps = 0
    n_nudge_applications = 0
    n_pareto_marked = 0
    sigma = np.ones(k, dtype=np.float64)

    def policy_train(obs_arr):
        s = _cluster_hash(obs_arr, N_BUCKETS)
        if rng.random() < EPS_TRAIN:
            return int(rng.integers(n_actions)), s
        logits = logit_nudge[s]
        if not np.any(logits):
            return int(rng.integers(n_actions)), s
        noise = 1e-9 * rng.standard_normal(n_actions)
        return int(np.argmax(logits + noise)), s

    while True:
        if time.monotonic() - t0 >= time_budget_s:
            break
        try:
            obs, _ = env.reset(seed=int(seed + 1 + n_episodes))
        except TypeError:
            obs, _ = env.reset()
        traj_clusters: list[int] = []
        traj_actions: list[int] = []
        traj_vec: list[np.ndarray] = []
        ep_steps = 0
        while True:
            if time.monotonic() - t0 >= time_budget_s:
                break
            a, s = policy_train(obs)
            try:
                next_obs, reward, term, trunc, info = env.step(a)
            except Exception:
                break
            if has_vector and "vector" in info:
                v = np.asarray(info["vector"], dtype=np.float64)
                if v.shape[0] != k:
                    vf = np.zeros(k, dtype=np.float64)
                    vf[: min(k, v.shape[0])] = v[: min(k, v.shape[0])]
                    v = vf
            else:
                v = np.array([float(reward)], dtype=np.float64)
            traj_clusters.append(s)
            traj_actions.append(a)
            traj_vec.append(v)
            obs = next_obs
            ep_steps += 1
            n_steps += 1
            if term or trunc or ep_steps >= harness.MAX_EPISODE_STEPS:
                break

        if len(traj_actions) == 0:
            n_episodes += 1
            continue

        v_arr = np.stack(traj_vec, axis=0)  # (T, k)
        s_arr = np.asarray(traj_clusters, dtype=np.int64)
        a_arr = np.asarray(traj_actions, dtype=np.int64)
        rho = _within_traj_rank_percentile(v_arr, eps_zero=EPS_ZERO)
        firing_mask = ~np.isnan(rho)

        # Update R-cell sums/counts and signed channel accumulator.
        for m in range(k):
            mask_m = firing_mask[:, m]
            if not mask_m.any():
                continue
            ts = np.flatnonzero(mask_m)
            ss = s_arr[ts]
            aa = a_arr[ts]
            vals = rho[ts, m]
            np.add.at(R_sum[:, :, m], (ss, aa), vals)
            np.add.at(R_cnt[:, :, m], (ss, aa), 1)
            chan_signed_sum[m] += float(v_arr[ts, m].sum())
            chan_signed_cnt[m] += int(ts.size)
        np.add.at(visit_cnt, (s_arr, a_arr), 1)

        n_episodes += 1

        if n_episodes < N_SEED_EPISODES:
            continue

        # Trend sigma per channel — sign of buffer-wide mean firing v.
        # Channels that haven't fired stay at sigma = +1 (no inversion).
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_signed = np.where(
                chan_signed_cnt > 0,
                chan_signed_sum / np.maximum(chan_signed_cnt, 1),
                0.0,
            )
        sigma = np.where(mean_signed >= 0, 1.0, -1.0)

        # Operator: Pareto non-domination on trend-corrected rank vector,
        # applied at every cluster visited this episode that has >= 2 actions
        # observed.
        visited_clusters = np.unique(s_arr)
        for s_c in visited_clusters:
            visited_actions = np.flatnonzero(visit_cnt[s_c] > 0)
            if visited_actions.size < 2:
                continue
            R_local = np.full((visited_actions.size, k), np.nan, dtype=np.float64)
            for i, a_v in enumerate(visited_actions):
                cnt = R_cnt[s_c, a_v]
                defined = cnt > 0
                if defined.any():
                    R_local[i, defined] = R_sum[s_c, a_v, defined] / cnt[defined]
            R_tilde = np.where(sigma[None, :] >= 0, R_local, 1.0 - R_local)

            n_a = R_tilde.shape[0]
            non_dominated = np.ones(n_a, dtype=bool)
            for i in range(n_a):
                if not non_dominated[i]:
                    continue
                for j in range(n_a):
                    if i == j:
                        continue
                    both = ~np.isnan(R_tilde[i]) & ~np.isnan(R_tilde[j])
                    if not both.any():
                        continue
                    ge = R_tilde[j, both] >= R_tilde[i, both]
                    gt = R_tilde[j, both] > R_tilde[i, both]
                    if ge.all() and gt.any():
                        non_dominated[i] = False
                        break
            for i, a_v in enumerate(visited_actions):
                if non_dominated[i]:
                    logit_nudge[s_c, a_v] += ALPHA
                    n_nudge_applications += 1
            n_pareto_marked += int(non_dominated.sum())

    env.close()

    eval_rng = np.random.default_rng(seed + 1234567)

    def policy_fn(obs_arr):
        s = _cluster_hash(obs_arr, N_BUCKETS)
        if eval_rng.random() < EPS_FLOOR:
            return int(eval_rng.integers(n_actions))
        logits = logit_nudge[s]
        if not np.any(logits):
            return int(eval_rng.integers(n_actions))
        return int(np.argmax(logits))

    n_visited_cells = int((visit_cnt > 0).sum())
    n_eligible = int(((visit_cnt > 0).sum(axis=1) > 1).sum())
    n_R_populated = int((R_cnt > 0).sum())
    sigma_str = sigma.tolist() if n_episodes >= N_SEED_EPISODES else "pre-seed"
    print(
        f"[crp] env={env_id} seed={seed} eps={n_episodes} steps={n_steps} "
        f"visited_cells={n_visited_cells} eligible_clusters={n_eligible} "
        f"R_cells_populated={n_R_populated} pareto_marked={n_pareto_marked} "
        f"nudges={n_nudge_applications} k={k} has_vector={has_vector} sigma={sigma_str}",
        flush=True,
    )

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
