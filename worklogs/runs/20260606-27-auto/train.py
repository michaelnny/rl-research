"""PHI: Path-Homology Invariants — candidate algorithm.

Realizes hypothesis 20260606-27-auto: per-(state-cluster, action) Lévy-area
contributions in observation-embedding space, projected channel-wise through
k independent linear regressors fit against terminal vector cumulants, used
via a Pareto-non-dominance count operator (no scalar collapse).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.reshape(-1).astype(np.float32)


def _cluster_id(emb: np.ndarray, projector: np.ndarray, n_buckets: int) -> int:
    # Simple coarse clustering: project embedding to a few sign bits.
    sig = (projector @ emb) > 0.0
    h = 0
    for b in sig:
        h = (h << 1) | int(b)
    return int(h % n_buckets)


def _levy_area_step(e_t: np.ndarray, e_tp1: np.ndarray) -> np.ndarray:
    # Δa_{ij} = (1/2)(e_i(o_t) e_j(o_{t+1}) - e_j(o_t) e_i(o_{t+1}))
    outer = np.outer(e_t, e_tp1)
    return 0.5 * (outer - outer.T)


# --------------------------------------------------------------------------
# Core training
# --------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    rng = np.random.default_rng(seed + 12345)

    # Determine action_space size.
    if hasattr(env.action_space, "n"):
        n_actions = int(env.action_space.n)
    else:
        env.close()
        # Continuous action — outside hypothesis scope. Fall back to random.
        action_space = env.action_space

        def policy_fn(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} (continuous-action fallback)",
            flush=True,
        )
        return policy_fn

    # Embedding dimension: hypothesis says d >= 8 to avoid rank-deficient area.
    d = 12

    # Initial reset to learn observation shape.
    obs0, _ = env.reset(seed=seed)
    obs0_flat = _flatten_obs(obs0)
    obs_dim = obs0_flat.shape[0]

    # Frozen random projection for embedding e(o) = R o, normalized.
    R_proj = rng.standard_normal((d, obs_dim)).astype(np.float32) / np.sqrt(obs_dim)

    # Cluster projector — coarse 4-bit sign hash → up to 16 cells per action.
    n_cluster_bits = 4
    n_clusters = 2**n_cluster_bits
    cluster_proj = rng.standard_normal((n_cluster_bits, d)).astype(np.float32)

    # Determine k (number of vector channels).
    # Probe one step to get info["vector"] shape if vector env.
    if is_vector:
        # take a no-op-like step to probe vector dim
        try:
            probe_action = 0
            _, _, term, trunc, info = env.step(probe_action)
            vec_probe = np.asarray(info.get("vector", np.zeros(1)), dtype=np.float64)
            k = int(vec_probe.shape[0])
            # Reset for clean rollouts.
            obs0, _ = env.reset(seed=seed)
        except Exception:
            k = 2
            obs0, _ = env.reset(seed=seed)
    else:
        # Treat scalar reward as the single channel.
        k = 1

    # Per-(cluster, action) running-sum and count of one-step Lévy increments.
    # Stored as a (n_clusters, n_actions, d, d) tensor + count.
    sa_sum = np.zeros((n_clusters, n_actions, d, d), dtype=np.float32)
    sa_count = np.zeros((n_clusters, n_actions), dtype=np.int32)

    # Per-trajectory buffer: store per-trajectory area A(τ) ∈ R^{d×d} and
    # terminal cumulant c_T ∈ R^k for regression.
    traj_areas: list[np.ndarray] = []
    traj_cumulants: list[np.ndarray] = []

    # k regressors P_m ∈ R^{d×d} (flattened as length d²) — fit by least squares.
    P = np.zeros((k, d, d), dtype=np.float32)
    P_R2 = np.zeros(k, dtype=np.float32)

    # Base policy ℓ_θ — small linear logit head, init zero (uniform).
    W = np.zeros((n_actions, d), dtype=np.float32)
    b = np.zeros(n_actions, dtype=np.float32)
    lr = 0.05  # supervised distillation step size

    # Operator parameters
    alpha = 1.0  # logit nudge gain

    # Diagnostics
    n_decisions = 0
    n_pareto_fired = 0
    pareto_fire_steps = 0  # decisions where Δℓ has any non-zero entry

    # --------------------------------------------------------------
    # Refit regressors via least squares on flattened area entries.
    # --------------------------------------------------------------
    def refit_regressors():
        nonlocal P, P_R2
        if len(traj_areas) < 4:
            return
        # Build feature matrix X (N, d²) and target Y (N, k)
        X = np.stack([a.reshape(-1) for a in traj_areas], axis=0).astype(np.float32)
        Y = np.stack(traj_cumulants, axis=0).astype(np.float32)
        # Center X and Y
        Xm = X.mean(axis=0, keepdims=True)
        Ym = Y.mean(axis=0, keepdims=True)
        Xc = X - Xm
        Yc = Y - Ym
        # Independent ridge regression per channel (no scalarization).
        ridge = 1e-2
        # Solve (Xc^T Xc + ridge I) w = Xc^T Yc[:, m]  for each m
        XtX = Xc.T @ Xc
        reg = XtX + ridge * np.eye(XtX.shape[0], dtype=np.float32)
        try:
            inv = np.linalg.solve(reg, Xc.T @ Yc)  # (d², k)
        except np.linalg.LinAlgError:
            return
        for m in range(k):
            w = inv[:, m]
            P[m] = w.reshape(d, d)
            # Compute R² for this channel
            y = Yc[:, m]
            yhat = Xc @ w
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum(y**2)) + 1e-12
            P_R2[m] = float(max(0.0, 1.0 - ss_res / ss_tot))

    # --------------------------------------------------------------
    # Operator: per-state Pareto-non-dominance vote.
    # --------------------------------------------------------------
    def operator_logit_nudge(cluster_id: int) -> np.ndarray:
        # g[a, m] = <P_m, E[Δa | cluster, a]>
        # Only consider actions that have been visited in this cluster.
        visited = sa_count[cluster_id] > 0
        if visited.sum() < 2:
            return np.zeros(n_actions, dtype=np.float32)
        g = np.zeros((n_actions, k), dtype=np.float32)
        for a in range(n_actions):
            if not visited[a]:
                # Unvisited actions excluded from the vote — they have no signal.
                continue
            mean_da = sa_sum[cluster_id, a] / float(sa_count[cluster_id, a])
            for m in range(k):
                g[a, m] = float(np.sum(P[m] * mean_da))
        # Pareto-non-dominance counts among visited actions.
        nudge = np.zeros(n_actions, dtype=np.float32)
        visited_idx = np.where(visited)[0]
        if len(visited_idx) < 2:
            return nudge
        for a in visited_idx:
            n_dom = 0
            n_sub = 0
            for ap in visited_idx:
                if ap == a:
                    continue
                # a strictly dominates ap iff g[a] >= g[ap] elementwise and >.
                ge = np.all(g[a] >= g[ap])
                gt = np.any(g[a] > g[ap])
                le = np.all(g[a] <= g[ap])
                lt = np.any(g[a] < g[ap])
                if ge and gt:
                    n_dom += 1
                if le and lt:
                    n_sub += 1
            nudge[a] = alpha * (n_dom - n_sub)
        return nudge

    # --------------------------------------------------------------
    # Action sampling: a ~ softmax(ℓ_θ(o) + Δℓ(cluster, ·))
    # --------------------------------------------------------------
    def sample_action(emb: np.ndarray, cluster_id: int, rng_):
        base_logits = W @ emb + b
        nudge = operator_logit_nudge(cluster_id)
        logits = base_logits + nudge
        # Softmax with stability
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        p = ex / (np.sum(ex) + 1e-12)
        a = int(rng_.choice(n_actions, p=p))
        return a, p, nudge

    # --------------------------------------------------------------
    # Distillation training step on base policy:
    # match the operator-biased softmax distribution.
    # --------------------------------------------------------------
    def distill_step(emb: np.ndarray, target_p: np.ndarray):
        nonlocal W, b
        base_logits = W @ emb + b
        m = float(np.max(base_logits))
        ex = np.exp(base_logits - m)
        p = ex / (np.sum(ex) + 1e-12)
        # Cross-entropy gradient: dL/dlogit = p - target_p
        grad_logits = p - target_p
        # dL/dW = grad_logits[:, None] * emb[None, :]
        gW = np.outer(grad_logits, emb)
        gb = grad_logits
        W -= lr * gW
        b -= lr * gb

    # --------------------------------------------------------------
    # Main rollout loop until time budget exhausted.
    # --------------------------------------------------------------
    t0 = time.monotonic()
    deadline = t0 - 5.0 + time_budget_s  # leave headroom for eval cleanup
    env_steps = 0
    n_traj_done = 0
    refit_interval = 4  # refit P every N completed trajectories
    max_buffer = 256

    obs = obs0
    emb_t = R_proj @ _flatten_obs(obs)
    cluster_t = _cluster_id(emb_t, cluster_proj, n_clusters)

    cur_area = np.zeros((d, d), dtype=np.float32)
    cur_cumulant = np.zeros(k, dtype=np.float64)
    ep_steps = 0
    max_ep_steps = harness.MAX_EPISODE_STEPS

    try:
        while time.monotonic() < deadline:
            # Sample action under operator-biased policy.
            a, p_target, nudge = sample_action(emb_t, cluster_t, rng)
            n_decisions += 1
            if np.any(nudge != 0):
                pareto_fire_steps += 1

            # Step env.
            try:
                step_out = env.step(a)
            except Exception:
                # Resilience: if env errors, reset.
                obs, _ = env.reset(seed=seed + env_steps)
                emb_t = R_proj @ _flatten_obs(obs)
                cluster_t = _cluster_id(emb_t, cluster_proj, n_clusters)
                cur_area = np.zeros((d, d), dtype=np.float32)
                cur_cumulant = np.zeros(k, dtype=np.float64)
                ep_steps = 0
                continue

            if len(step_out) == 5:
                obs_next, reward, term, trunc, info = step_out
            else:
                obs_next, reward, done_legacy, info = step_out
                term, trunc = done_legacy, False

            env_steps += 1
            ep_steps += 1

            # Vector cumulant accumulation.
            if is_vector:
                v = np.asarray(info.get("vector", np.zeros(k)), dtype=np.float64)
                if v.shape[0] != k:
                    v = np.resize(v, (k,))
                cur_cumulant += v
            else:
                cur_cumulant[0] += float(reward)

            # Embedding for next obs.
            emb_tp1 = R_proj @ _flatten_obs(obs_next)
            cluster_tp1 = _cluster_id(emb_tp1, cluster_proj, n_clusters)

            # Per-step Lévy area increment.
            da = _levy_area_step(emb_t, emb_tp1)
            cur_area += da

            # Update per-(cluster, action) running sum.
            sa_sum[cluster_t, a] += da
            sa_count[cluster_t, a] += 1

            # Distillation step on base policy toward biased target.
            distill_step(emb_t, p_target)

            # Advance state.
            emb_t = emb_tp1
            cluster_t = cluster_tp1

            done = bool(term) or bool(trunc) or ep_steps >= max_ep_steps
            if done:
                # Commit trajectory to buffer.
                traj_areas.append(cur_area.copy())
                traj_cumulants.append(cur_cumulant.astype(np.float32).copy())
                if len(traj_areas) > max_buffer:
                    traj_areas.pop(0)
                    traj_cumulants.pop(0)
                n_traj_done += 1
                if n_traj_done % refit_interval == 0:
                    refit_regressors()

                # Reset.
                obs, _ = env.reset(seed=seed + n_traj_done)
                emb_t = R_proj @ _flatten_obs(obs)
                cluster_t = _cluster_id(emb_t, cluster_proj, n_clusters)
                cur_area = np.zeros((d, d), dtype=np.float32)
                cur_cumulant = np.zeros(k, dtype=np.float64)
                ep_steps = 0

    except Exception as exc:
        print(f"[train] training-loop exception: {exc!r}", flush=True)

    train_s = time.monotonic() - t0
    fire_rate = pareto_fire_steps / max(1, n_decisions)

    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} "
        f"trajs={n_traj_done} k={k} d={d} R2={P_R2.tolist()} "
        f"pareto_fire_rate={fire_rate:.3f} clusters_used={int((sa_count.sum(axis=1) > 0).sum())}/{n_clusters}",
        flush=True,
    )

    try:
        env.close()
    except Exception:
        pass

    # -------------------------------------------------------------------
    # Deterministic deployment policy: argmax of biased logits.
    # -------------------------------------------------------------------
    def policy_fn(obs_in):
        emb = R_proj @ _flatten_obs(obs_in)
        cid = _cluster_id(emb, cluster_proj, n_clusters)
        base_logits = W @ emb + b
        nudge = operator_logit_nudge(cid)
        logits = base_logits + nudge
        return int(np.argmax(logits))

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
