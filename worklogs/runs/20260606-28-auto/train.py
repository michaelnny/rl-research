"""PRAR: Policy-Regime Antisymmetric Residual.

Realizes the hypothesis at worklogs/runs/20260606-28-auto/hypothesis.md.

Core primitive:
    F[r, a, m] = (1/N[r,a]) sum_E sum_{t : regime(pi_E, o_t)=r, a_t=a}
                  w(t/T_E) * (v_t[m] - mu_E[m])
where
    w(s) = 2s - 1                  (antisymmetric within-episode position weight)
    mu_E[m] = (1/T_E) sum_s v_s[m] (per-episode channel mean)
    regime(pi, o) = (entropy_bin, top1_prob_bin, top1_minus_top2_gap_bin)

Improvement operator (decision time):
    nudge[a] = D_+(a) - D_-(a)
where D_+(a) and D_-(a) are coordinate-wise Pareto dominance counts
on F[r_t, a, :] across the legal action set; logits = base_logits + alpha * nudge,
with alpha = 0 for cells with N[r,a] < 5.

Substrate contract:
    train(env_id, seed, time_budget_s) -> policy_fn(obs) -> action.
    For vector envs, training consumes info["vector"].
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import harness


# ---------------------------------------------------------------------------
# Regime tagging
# ---------------------------------------------------------------------------

# 3 bins per axis -> 27 regimes.
N_BINS = 3


def _regime_tag(probs: np.ndarray) -> int:
    """Map a discrete action distribution to a regime in [0, 27)."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    # entropy normalized to [0, 1]
    n = max(len(p), 2)
    H = -float(np.sum(p * np.log(p))) / np.log(n)
    sorted_p = np.sort(p)[::-1]
    top1 = float(sorted_p[0])
    top2 = float(sorted_p[1]) if len(sorted_p) >= 2 else 0.0
    gap = top1 - top2

    # Fixed-edge bins (chosen so that all three axes span [0, 1])
    # Entropy: thirds of [0,1].
    h_bin = min(int(H * N_BINS), N_BINS - 1)
    # top1 prob: thirds of [1/n, 1] mapped to thirds of [0,1].
    p_bin = min(int(top1 * N_BINS), N_BINS - 1)
    # gap: thirds of [0, 1].
    g_bin = min(int(gap * N_BINS), N_BINS - 1)
    return (h_bin * N_BINS + p_bin) * N_BINS + g_bin


N_REGIMES = N_BINS * N_BINS * N_BINS


# ---------------------------------------------------------------------------
# Pareto dominance vote on F[r_t, :, :]
# ---------------------------------------------------------------------------


def _pareto_nudge(slice_ra: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Compute D_+(a) - D_-(a) on coordinate-wise partial order.

    slice_ra: (|A|, k) F[r_t, :, :]
    valid_mask: (|A|,) bool — actions with N[r, a] >= 5
    Returns nudge: (|A|,) float; entries with valid_mask=False are 0.
    """
    A, k = slice_ra.shape
    nudge = np.zeros(A, dtype=np.float64)
    if not valid_mask.any():
        return nudge

    # For each pair (a, b), a dominates b iff
    #   all_m F[a,m] >= F[b,m]  and  any_m F[a,m] > F[b,m].
    # D_+(a) = sum_b dominates(a, b); D_-(a) = sum_b dominates(b, a).
    F = slice_ra
    # Pairwise: ge[a,b,m] = F[a,m] >= F[b,m]; gt[a,b,m] = F[a,m] > F[b,m]
    ge = F[:, None, :] >= F[None, :, :]
    gt = F[:, None, :] > F[None, :, :]
    dominates = ge.all(axis=-1) & gt.any(axis=-1)  # (A, A)

    # Mask off invalid actions on either side of the comparison.
    vm = valid_mask
    pair_mask = vm[:, None] & vm[None, :]
    dominates = dominates & pair_mask

    D_plus = dominates.sum(axis=1).astype(np.float64)  # row a dominates how many b
    D_minus = dominates.sum(axis=0).astype(np.float64)  # col a is dominated by how many
    nudge = D_plus - D_minus
    return nudge


# ---------------------------------------------------------------------------
# Observation-to-feature: tiny linear policy with hashed observation features
# ---------------------------------------------------------------------------


def _obs_to_feat(obs, feat_dim: int, rng_proj: np.random.Generator) -> np.ndarray:
    """Project obs to a fixed-dim feature via a stable random projection.

    The policy itself is a small linear model: logits = W @ feat.
    For Craftax we get float arrays; for MiniGrid ImgObsWrapper we get a uint8
    image (H, W, 3); we just flatten and project.
    """
    arr = np.asarray(obs).reshape(-1).astype(np.float32)
    # Pad/truncate to a max raw dim, then project.
    max_raw = 4096
    if arr.size > max_raw:
        # sub-sample stride
        step = max(1, arr.size // max_raw)
        arr = arr[::step][:max_raw]
    if arr.size < max_raw:
        pad = np.zeros(max_raw - arr.size, dtype=np.float32)
        arr = np.concatenate([arr, pad])
    # Cached projection matrix (set once).
    P = rng_proj  # actually a (max_raw, feat_dim) ndarray — sentinel by name
    feat = arr @ P
    feat = np.tanh(feat * 0.01)
    return feat


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # Action space cardinality
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        # Fall back to a random policy if we somehow get a non-discrete env.
        env.close()

        def policy_fn(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} note=non-discrete-fallback",
            flush=True,
        )
        return policy_fn

    A = int(action_space.n)

    # Feature projection (fixed at init).
    max_raw = 4096
    feat_dim = 32
    proj = rng.standard_normal((max_raw, feat_dim)).astype(np.float32) / np.sqrt(max_raw)

    # Linear policy: logits = W @ feat + b
    W = rng.standard_normal((A, feat_dim)).astype(np.float32) * 0.01
    b = np.zeros(A, dtype=np.float32)

    def base_probs(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = W @ feat + b
        # Numerically stable softmax
        m = logits.max()
        e = np.exp(logits - m)
        p = e / e.sum()
        return logits, p

    def featurize(obs):
        arr = np.asarray(obs).reshape(-1).astype(np.float32)
        if arr.size > max_raw:
            step = max(1, arr.size // max_raw)
            arr = arr[::step][:max_raw]
        if arr.size < max_raw:
            pad = np.zeros(max_raw - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        feat = arr @ proj
        return np.tanh(feat * 0.01)

    # k = vector channel count: probe by stepping once if vector env, else k=1.
    if is_vector:
        # Probe number of channels from HV_REF dimensionality.
        k = harness.HV_REF[env_id].shape[0]
    else:
        # Scalar envs: synthesize a 1-channel "vector" from the scalar reward.
        # The hypothesis is targeted at vector envs; on scalar envs we still
        # follow the same data flow with k=1, which makes Pareto vote a scalar
        # comparison (acceptable per the hypothesis).
        k = 1

    # F tensor and counts
    F = np.zeros((N_REGIMES, A, k), dtype=np.float64)
    N = np.zeros((N_REGIMES, A), dtype=np.int64)

    # Hyperparameters from hypothesis: alpha <= 0.05 / k
    alpha = 0.05 / max(k, 1)
    min_n = 5

    # Online policy: REINFORCE-like is NOT used. The policy parameters W, b
    # are NOT updated by any gradient — the only way decisions change is via
    # the F-driven logit nudge at decision time. This matches the
    # hypothesis's "operator updates only the policy logits" rule. The
    # randomized W, b act as a fixed "base prior"; after enough episodes,
    # F dominates per-step action selection through the nudge.

    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave headroom for eval

    env_steps = 0
    n_episodes = 0
    # Diagnostics
    regime_hist = np.zeros(N_REGIMES, dtype=np.int64)

    while time.monotonic() < deadline:
        try:
            obs, _info = env.reset(seed=seed + 1_000 + n_episodes)
        except TypeError:
            obs, _info = env.reset()
        # Episode buffer
        ep_regimes: list[int] = []
        ep_actions: list[int] = []
        ep_vecs: list[np.ndarray] = []
        steps_this_ep = 0
        done = False
        while not done and steps_this_ep < harness.MAX_EPISODE_STEPS:
            feat = featurize(obs)
            logits, p_pi = base_probs(feat)
            r_t = _regime_tag(p_pi)
            # F-driven nudge over the action set.
            valid_mask = N[r_t] >= min_n
            nudge = _pareto_nudge(F[r_t], valid_mask)
            adj_logits = logits + alpha * nudge.astype(np.float32)
            m = adj_logits.max()
            e = np.exp(adj_logits - m)
            p_adj = e / e.sum()
            a = int(rng.choice(A, p=p_adj))

            try:
                obs_next, reward, term, trunc, info = env.step(a)
            except Exception:
                # Robust to any env-side anomaly.
                break

            if is_vector:
                v = np.asarray(info.get("vector"), dtype=np.float64)
                if v.shape != (k,):
                    # Fallback: pad/truncate to k
                    v_fixed = np.zeros(k, dtype=np.float64)
                    common = min(k, v.size)
                    v_fixed[:common] = v.reshape(-1)[:common]
                    v = v_fixed
            else:
                v = np.array([float(reward)], dtype=np.float64)

            ep_regimes.append(r_t)
            ep_actions.append(a)
            ep_vecs.append(v)
            regime_hist[r_t] += 1

            obs = obs_next
            steps_this_ep += 1
            env_steps += 1
            done = bool(term) or bool(trunc)

            if time.monotonic() >= deadline:
                break

        # End of episode: update F with the antisymmetric position-weighted
        # residual. We use this episode's data to update the running mean
        # F[r,a,m] += (contrib - F[r,a,m]) / N_new[r,a].
        T_E = len(ep_vecs)
        if T_E >= 2:
            V = np.stack(ep_vecs, axis=0)  # (T_E, k)
            mu_E = V.mean(axis=0)  # (k,)
            # Position weight w(t/T_E) for t = 0..T_E-1 (use (t+1)/T_E so
            # the last step has s = 1 and w = +1).
            ts = (np.arange(T_E) + 1.0) / T_E
            w = 2.0 * ts - 1.0  # (T_E,)
            dev = V - mu_E[None, :]  # (T_E, k)
            contrib = w[:, None] * dev  # (T_E, k)
            for t in range(T_E):
                r = ep_regimes[t]
                a_idx = ep_actions[t]
                N[r, a_idx] += 1
                # Online running mean
                F[r, a_idx] += (contrib[t] - F[r, a_idx]) / float(N[r, a_idx])

        n_episodes += 1

    env.close()

    # Diagnostics: regime occupancy and cell population
    regime_mass = regime_hist.astype(np.float64)
    total_mass = float(regime_mass.sum()) if regime_mass.sum() > 0 else 1.0
    regime_frac = regime_mass / total_mass
    n_active_regimes = int((regime_frac >= 0.05).sum())
    n_active_cells = int((N >= min_n).sum())
    nonzero_per_channel = (np.abs(F).sum(axis=(0, 1)) > 1e-12).astype(int)

    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={n_episodes} train_s={time.monotonic() - t_start:.1f} "
        f"budget_s={time_budget_s} k={k} A={A} "
        f"active_regimes_ge5pct={n_active_regimes} "
        f"active_cells_ge5={n_active_cells} "
        f"nonzero_channels={nonzero_per_channel.tolist()}",
        flush=True,
    )

    # Frozen policy: same logit nudge readout from frozen F tensor.
    F_frozen = F.copy()
    N_frozen = N.copy()

    def policy_fn(obs):
        feat = featurize(obs)
        logits, p_pi = base_probs(feat)
        r_t = _regime_tag(p_pi)
        valid_mask = N_frozen[r_t] >= min_n
        nudge = _pareto_nudge(F_frozen[r_t], valid_mask)
        adj_logits = logits + alpha * nudge.astype(np.float32)
        m = adj_logits.max()
        e = np.exp(adj_logits - m)
        p_adj = e / e.sum()
        # Sampling at evaluation, temperature 1, per the hypothesis.
        return int(rng.choice(A, p=p_adj))

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
