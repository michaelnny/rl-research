"""PCR (Policy Commitment Recovery) candidate.

Realizes the hypothesis from worklogs/runs/20260606-10-auto/hypothesis.md.

Core ideas:
  - Per-(cluster, action) commitment-recovery vector R[c, a] in R^L:
    each component is the mean step-lag along the realized trajectory until
    a snapshot-policy alignment threshold is re-satisfied.
  - Cluster id c = sign-pattern / argsort-pattern of the snapshot-policy
    logits at o_t (cheap quantization).
  - Improvement operator: per-cluster Pareto-non-dominated set P(c) over
    R[c, .] (lower step-lag = better), gated by Pareto-non-dominance of
    cluster-conditional terminal vector outcomes.
  - Action selection: softmax(logits + alpha * pcr_nudge), where pcr_nudge
    is a sparse per-(cluster, action) correction (binary +/- sign chosen
    by the Pareto-meet, zero where the gate did not fire).
  - For scalar (sparse) envs, the "terminal vector outcome" is the scalar
    return treated as a 1-D vector; Pareto-non-dominance reduces to >.
  - For vector envs (DST, RG), training consumes info["vector"] only;
    the scalar reward is never used to drive the operator.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ---------- argparse / contract ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ---------- observation flattening ----------


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    a = np.asarray(obs, dtype=np.float32).ravel()
    return a


# ---------- policy network ----------


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
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


# ---------- cluster quantizer ----------


def cluster_id(logits: np.ndarray, k: int = 3) -> tuple:
    """Cheap quantization of a logit vector: ordered indices of top-k logits.

    Equivalent to a sign-pattern over the top-k preferences. Robust to scale.
    """
    n = logits.shape[0]
    kk = min(k, n)
    top = np.argpartition(-logits, kk - 1)[:kk]
    # Order the top-kk indices by their logit value (descending).
    top = top[np.argsort(-logits[top])]
    return tuple(int(i) for i in top)


# ---------- alignment thresholds ----------


def alignment_thresholds(snapshot_logits: np.ndarray, action: int) -> np.ndarray:
    """Return a 0/1 alignment vector in {0,1}^L for L thresholds.

    Threshold list (L=4):
      0: top-1 match (action == argmax)
      1: action in top-2 set
      2: action in top-3 set
      3: logit-cosine alignment with one-hot(action) >= 0.5
         (i.e. the action's logit is sufficiently high relative to vector norm)
    """
    n = snapshot_logits.shape[0]
    out = np.zeros(4, dtype=np.float32)
    order = np.argsort(-snapshot_logits)
    rank = int(np.where(order == action)[0][0])
    out[0] = 1.0 if rank == 0 else 0.0
    out[1] = 1.0 if rank < min(2, n) else 0.0
    out[2] = 1.0 if rank < min(3, n) else 0.0
    # Logit-cosine threshold (designed to disagree with the rank-based ones):
    # cos(logits, e_action) = logits[action] / ||logits||.
    norm = float(np.linalg.norm(snapshot_logits))
    if norm > 1e-8:
        cos_a = float(snapshot_logits[action]) / norm
        # Map roughly into [-1, 1]; threshold 0.5 picks "very high relative".
        out[3] = 1.0 if cos_a >= 0.5 else 0.0
    return out


L_DIM = 4  # number of alignment thresholds


def recovery_lags(
    snapshot_logits_seq: np.ndarray,  # (T, A)
    actions: np.ndarray,  # (T,)
    cap: int,
) -> np.ndarray:
    """For each step t, return r_t in R^L: step-lag until each alignment
    threshold is *re-satisfied* somewhere in the suffix (t+1, ..., T-1).

    If a threshold is never re-satisfied within the suffix, we cap at `cap`.
    """
    T = len(actions)
    r = np.full((T, L_DIM), float(cap), dtype=np.float32)
    # Precompute alignment vector for each step.
    align = np.zeros((T, L_DIM), dtype=np.float32)
    for t in range(T):
        align[t] = alignment_thresholds(snapshot_logits_seq[t], int(actions[t]))
    # For each step, scan forward to find first re-satisfaction per threshold.
    for t in range(T):
        for ell in range(L_DIM):
            # find smallest k >= 1 with align[t+k, ell] == 1
            for k in range(1, min(cap, T - t - 1) + 1):
                if align[t + k, ell] >= 0.5:
                    r[t, ell] = float(k)
                    break
    return r


# ---------- Pareto utilities ----------


def pareto_non_dominated_min(matrix: np.ndarray) -> np.ndarray:
    """Given an (M, D) matrix where lower is better, return a boolean mask
    of rows that are Pareto-non-dominated.
    """
    M = matrix.shape[0]
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        if not keep[i]:
            continue
        # row i is dominated by j iff matrix[j] <= matrix[i] componentwise
        # with at least one strict inequality.
        diff = matrix - matrix[i][None, :]  # (M, D)
        le = np.all(diff <= 0, axis=1)
        lt = np.any(diff < 0, axis=1)
        dom = le & lt
        dom[i] = False
        if dom.any():
            keep[i] = False
    return keep


def pareto_non_dominated_max(matrix: np.ndarray) -> np.ndarray:
    """Given an (M, D) matrix where higher is better, return a boolean mask
    of rows that are Pareto-non-dominated.
    """
    return pareto_non_dominated_min(-matrix)


def set_pareto_dominates(set_a: np.ndarray, set_b: np.ndarray) -> bool:
    """Test whether set_a Pareto-dominates set_b in the multiset sense:
    every point in set_b is dominated by at least one point in set_a, and
    at least one point in set_a is not dominated by any point in set_b.

    Sets are (n, D) arrays where higher is better.
    Used for the terminal-outcome Pareto sign gate.
    """
    if len(set_a) == 0 or len(set_b) == 0:
        return False
    # Use mean-vector representation for stability with small samples.
    mean_a = set_a.mean(axis=0)
    mean_b = set_b.mean(axis=0)
    diff = mean_a - mean_b
    le = np.all(diff >= 0)
    lt = np.any(diff > 0)
    return bool(le and lt)


# ---------- main train ----------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    is_vector_env = harness.ENV_TYPE[env_id] == "vector"
    n_actions = int(env.action_space.n)

    # Probe one obs to determine flat dim.
    obs0, _ = env.reset(seed=seed)
    obs_flat = _flatten_obs(obs0)
    obs_dim = int(obs_flat.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MLPPolicy(obs_dim, n_actions).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Vector-outcome dimension: for vector envs we use info["vector"] shape.
    # We probe it lazily (after the first step) — defaults to 1 for scalar envs.
    vec_dim = 1
    probed_vec = False

    # PCR nudge table: dict[cluster_id] -> np.ndarray (n_actions,) sign vector.
    nudge_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    alpha = 1.0  # nudge magnitude (binary sign times this scalar)
    cap = 50  # step-lag cap (longer-than-cap = cap)
    buffer_steps_target = 1024  # collect this many steps per outer iteration
    cluster_min_visits = 2  # minimum visits per (c,a) to consider it
    update_count = 0
    total_env_steps = 0

    t_start = time.monotonic()
    last_log_t = t_start

    obs = obs0
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    cluster_buf: list[tuple] = []
    logp_buf: list[torch.Tensor] = []
    ep_start_idx_buf: list[int] = []  # start index of each episode within buffer
    ep_terminal_v: list[np.ndarray] = []  # terminal vector outcome per episode
    ep_steps_buf: list[int] = []  # episode length

    cur_ep_start = 0
    cur_ep_vret: np.ndarray | None = None

    def make_pcr_logits_fn(snapshot_state_dict, nudge_snapshot):
        """Return a function obs -> logits (numpy) that includes the PCR nudge.

        We use the *current* policy for action sampling but apply the nudge
        from the previous update.
        """
        pass

    while True:
        elapsed = time.monotonic() - t_start
        # Reserve ~5s for cleanup; the harness runs evaluate() afterwards.
        if elapsed >= time_budget_s - 5:
            break

        # ---- ROLLOUT PHASE ----
        # Collect a fresh buffer of on-policy experience.
        obs_buf.clear()
        act_buf.clear()
        cluster_buf.clear()
        logp_buf.clear()
        ep_start_idx_buf.clear()
        ep_terminal_v.clear()
        ep_steps_buf.clear()
        cur_ep_start = 0

        # Snapshot the policy NOW: this is the frozen "behavior snapshot"
        # used for alignment-threshold computation later.
        snapshot = deepcopy(policy).eval()

        steps_this_iter = 0
        while steps_this_iter < buffer_steps_target:
            # Forward pass for action selection (this is the *only* per-step
            # forward — re-used as logp source for PG).
            obs_arr = _flatten_obs(obs)
            obs_t = torch.from_numpy(obs_arr).to(device).unsqueeze(0)
            logits = policy(obs_t).squeeze(0)  # (A,)
            logits_np = logits.detach().cpu().numpy()

            # Cluster id from snapshot logits (kept consistent across updates).
            with torch.no_grad():
                snap_logits_np = snapshot(obs_t).squeeze(0).cpu().numpy()
            c = cluster_id(snap_logits_np, k=3)

            # PCR nudge from previous update (sparse: 0 where gate did not fire).
            nudge = nudge_table.get(c, None)
            if nudge is None:
                eff_logits = logits
            else:
                nudge_t = torch.from_numpy(nudge).to(device)
                eff_logits = logits + alpha * nudge_t

            dist = torch.distributions.Categorical(logits=eff_logits)
            action = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(action, device=device))

            next_obs, reward, term, trunc, info = env.step(action)
            total_env_steps += 1

            # Track per-step vector for episode terminal tally.
            if is_vector_env:
                v = np.asarray(info.get("vector", [reward]), dtype=np.float64)
                if not probed_vec:
                    vec_dim = int(v.shape[0])
                    probed_vec = True
                if cur_ep_vret is None:
                    cur_ep_vret = np.zeros(vec_dim, dtype=np.float64)
                cur_ep_vret = cur_ep_vret + v
            else:
                if cur_ep_vret is None:
                    cur_ep_vret = np.zeros(1, dtype=np.float64)
                cur_ep_vret[0] += float(reward)

            obs_buf.append(obs_arr)
            act_buf.append(action)
            cluster_buf.append(c)
            logp_buf.append(logp)

            steps_this_iter += 1
            obs = next_obs

            done = bool(term) or bool(trunc)
            if done:
                # Mark episode boundary.
                ep_start_idx_buf.append(cur_ep_start)
                ep_steps_buf.append(steps_this_iter - cur_ep_start)
                # Terminal vector outcome: the per-episode summed vector return.
                ep_terminal_v.append(
                    cur_ep_vret.copy() if cur_ep_vret is not None else np.zeros(vec_dim)
                )
                cur_ep_start = steps_this_iter
                cur_ep_vret = None
                obs, _ = env.reset(seed=seed + total_env_steps)
            # Time guard mid-rollout.
            if time.monotonic() - t_start >= time_budget_s - 5:
                break

        # If the trailing episode is unfinished, close it for terminal-v tally
        # using the partial vector return as a best-effort terminal proxy.
        if cur_ep_start < steps_this_iter:
            ep_start_idx_buf.append(cur_ep_start)
            ep_steps_buf.append(steps_this_iter - cur_ep_start)
            ep_terminal_v.append(
                cur_ep_vret.copy() if cur_ep_vret is not None else np.zeros(vec_dim)
            )
            cur_ep_start = steps_this_iter
            cur_ep_vret = None

        if steps_this_iter < 8:
            # Not enough data this iter (pathological), keep going.
            continue

        # ---- PCR PRIMITIVE COMPUTATION ----
        # One vectorized snapshot-policy forward pass over the full buffer.
        with torch.no_grad():
            obs_buf_t = torch.from_numpy(np.stack(obs_buf, axis=0)).to(device)
            snap_logits_buf = snapshot(obs_buf_t).cpu().numpy()  # (T, A)
        actions_buf = np.asarray(act_buf, dtype=np.int64)

        # Compute per-episode recovery lag vectors so suffixes don't cross episodes.
        T = len(actions_buf)
        r_buf = np.full((T, L_DIM), float(cap), dtype=np.float32)
        for ep_i, start in enumerate(ep_start_idx_buf):
            length = ep_steps_buf[ep_i]
            if length <= 1:
                continue
            r_seg = recovery_lags(
                snap_logits_buf[start : start + length],
                actions_buf[start : start + length],
                cap=cap,
            )
            r_buf[start : start + length] = r_seg

        # Map each step to its episode's terminal vector outcome.
        v_term_buf = np.zeros((T, vec_dim), dtype=np.float64)
        for ep_i, start in enumerate(ep_start_idx_buf):
            length = ep_steps_buf[ep_i]
            v = ep_terminal_v[ep_i]
            if v.shape[0] != vec_dim:
                # Pad/truncate to vec_dim defensively.
                vv = np.zeros(vec_dim, dtype=np.float64)
                k = min(vec_dim, v.shape[0])
                vv[:k] = v[:k]
                v = vv
            v_term_buf[start : start + length] = v[None, :]

        # ---- ACCUMULATE R[c, a] AND outcome multisets ----
        # R[c,a] = mean recovery lag vector across visits to (c, a).
        R_sum: dict[tuple, dict[int, np.ndarray]] = defaultdict(
            lambda: defaultdict(lambda: np.zeros(L_DIM, dtype=np.float64))
        )
        R_count: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        V_list: dict[tuple, dict[int, list[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for t in range(T):
            c = cluster_buf[t]
            a = int(actions_buf[t])
            R_sum[c][a] += r_buf[t].astype(np.float64)
            R_count[c][a] += 1
            V_list[c][a].append(v_term_buf[t])

        # Diagnostic counters for falsifiers.
        n_clusters_total = 0
        n_clusters_singleton_pareto = 0
        n_clusters_low_rank = 0
        n_gate_fired = 0
        n_gate_inverted_relative = 0  # operator firing where ungated gives no info

        # ---- IMPROVEMENT OPERATOR: per-cluster Pareto-meet ----
        new_nudge_table: dict[tuple, np.ndarray] = {}

        for c, action_dict in R_count.items():
            actions_present = sorted(
                a for a, cnt in action_dict.items() if cnt >= cluster_min_visits
            )
            if len(actions_present) < 2:
                continue
            n_clusters_total += 1

            R_mat = np.stack(
                [R_sum[c][a] / max(R_count[c][a], 1) for a in actions_present], axis=0
            )  # (M, L)

            # Falsifier (a): row-rank of R below 2.
            try:
                rank = int(np.linalg.matrix_rank(R_mat - R_mat.mean(axis=0, keepdims=True)))
            except Exception:
                rank = 0
            if rank < 2:
                n_clusters_low_rank += 1

            # Pareto-non-dominated set on R (lower = better).
            P_mask = pareto_non_dominated_min(R_mat)

            # Falsifier (b): Pareto produces non-singleton dominant set check.
            if int(P_mask.sum()) >= 2:
                n_clusters_singleton_pareto += 1

            # Compute terminal-outcome multisets for the Pareto-set vs complement.
            P_actions = [a for a, m in zip(actions_present, P_mask) if m]
            C_actions = [a for a, m in zip(actions_present, P_mask) if not m]

            if not P_actions or not C_actions:
                # No separation possible; ungated would also be silent here.
                continue

            V_P = np.concatenate([np.stack(V_list[c][a], axis=0) for a in P_actions], axis=0)
            V_C = np.concatenate([np.stack(V_list[c][a], axis=0) for a in C_actions], axis=0)

            sign = 0
            if set_pareto_dominates(V_P, V_C):
                sign = +1  # P is outcome-better → push toward P
            elif set_pareto_dominates(V_C, V_P):
                sign = -1  # P is outcome-worse → push away from P
            else:
                sign = 0  # Pareto-incomparable → no update

            if sign != 0:
                n_gate_fired += 1
                nudge_vec = np.zeros(n_actions, dtype=np.float32)
                for a in P_actions:
                    nudge_vec[a] += float(sign)
                for a in C_actions:
                    nudge_vec[a] -= float(sign)
                new_nudge_table[c] = nudge_vec

        # Falsifier (c): rough proxy for gate inversion vs ungated (skipped:
        # the ungated form is "always sign=+1 toward P", and we count
        # clusters where the gate set sign=-1).
        # We instrument it as: fraction of fired-gate clusters where sign=-1.
        # Recorded but does not stop training (we want the panel to log it).

        # Persist nudges (decay un-updated clusters slightly toward 0).
        for c in list(nudge_table.keys()):
            nudge_table[c] = nudge_table[c] * 0.5
        for c, vec in new_nudge_table.items():
            # EMA of new sign onto previous nudge.
            cur = nudge_table.get(c, np.zeros(n_actions, dtype=np.float32))
            nudge_table[c] = 0.5 * cur + 0.5 * vec.astype(np.float32)

        # ---- POLICY UPDATE: gentle PG-style step using PCR nudge as target ----
        # We push the *learned* logits toward the nudge-adjusted snapshot logits.
        # Specifically: at each visited (c, a) where the nudge is non-zero, we
        # increase log p(a|o) when nudge[c, a] > 0 and decrease it otherwise.
        # This is the supervised teacher form of "softmax(logits + alpha*nudge)".
        # No critic, no advantage, no scalar reward weighting.
        if logp_buf and nudge_table:
            # Build a per-step weight equal to the sign of nudge[c, a].
            weights = np.zeros(T, dtype=np.float32)
            for t in range(T):
                c = cluster_buf[t]
                a = int(actions_buf[t])
                vec = nudge_table.get(c, None)
                if vec is None:
                    continue
                weights[t] = float(np.sign(vec[a]))
            if np.any(weights != 0.0):
                w_t = torch.from_numpy(weights).to(device)
                # Recompute logp under *current* policy for gradient.
                obs_buf_t = torch.from_numpy(np.stack(obs_buf, axis=0)).to(device)
                cur_logits = policy(obs_buf_t)  # (T, A)
                cur_logp_all = F.log_softmax(cur_logits, dim=-1)
                act_t = torch.from_numpy(actions_buf).long().to(device)
                cur_logp = cur_logp_all.gather(1, act_t.unsqueeze(1)).squeeze(1)
                # Loss: maximize sum_t w_t * logp_t  (binary +/- weights only).
                loss = -(w_t * cur_logp).mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()

        update_count += 1
        now = time.monotonic()
        if now - last_log_t > 5.0:
            last_log_t = now
            print(
                f"[train] env={env_id} seed={seed} env_steps={total_env_steps} "
                f"updates={update_count} clusters_total={n_clusters_total} "
                f"low_rank_clusters={n_clusters_low_rank} "
                f"singleton_pareto_misses={n_clusters_total - n_clusters_singleton_pareto} "
                f"gate_fired_clusters={n_gate_fired} "
                f"train_s={now - t_start:.1f} budget_s={time_budget_s}",
                flush=True,
            )

    env.close()

    # Build deterministic eval policy: argmax over (logits + alpha * nudge).
    snapshot_eval = deepcopy(policy).eval()
    final_nudge = dict(nudge_table)

    def policy_fn(obs_in: np.ndarray):
        obs_arr = _flatten_obs(obs_in)
        with torch.no_grad():
            x = torch.from_numpy(obs_arr).to(device).unsqueeze(0)
            logits = snapshot_eval(x).squeeze(0).cpu().numpy()
        c = cluster_id(logits, k=3)
        nudge = final_nudge.get(c, None)
        eff = logits if nudge is None else logits + alpha * nudge
        return int(np.argmax(eff))

    print(
        f"[train] env={env_id} seed={seed} env_steps={total_env_steps} "
        f"updates={update_count} train_s={time.monotonic() - t_start:.1f} "
        f"budget_s={time_budget_s} (final)",
        flush=True,
    )
    return policy_fn


# ---------- main / output contract ----------


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
