"""LRA: Loop-Return Aversion.

Realizes 20260606-04-auto hypothesis:
  - Primitive: per-(obs_hash, action) running mean of vector cumulant deltas
    accumulated along closed within-episode loops (intra-trajectory hash
    recurrences).
  - Improvement operator: at decision time, suppress logits of any action
    whose loop-signature L[s,a] is Pareto-dominated by the zero vector
    (all channels <= 0, at least one < 0).
  - Execution: sample from softmax(logits - alpha * mask).
  - Vector feedback: Pareto comparison vs zero, no scalarization.
  - Zero counterfactual rollouts; on-policy only.

For scalar (sparse / craftax) envs, we fabricate a 1-D "vector" cumulant
from the scalar reward each step. The operator and primitive are unchanged
structurally; the dominance-vs-zero test on a single channel reduces to
"strictly negative loop-return", which is the natural projection of the
operator.

Contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import defaultdict

import numpy as np

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


def _obs_hash(obs) -> int:
    """Stable hash of an observation for intra-trajectory recurrence detection.

    Uses md5 of the raw bytes; keeps a small int via mod, sufficient for
    dictionary keys.
    """
    arr = np.ascontiguousarray(np.asarray(obs))
    h = hashlib.md5(arr.tobytes()).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _num_actions(action_space) -> int:
    return int(action_space.n)


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    n_actions = _num_actions(action_space)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # Hyperparameters (chosen conservatively per hypothesis guidance; not tuned
    # to chase a number).
    alpha = 3.0           # logit suppression magnitude
    n_min = 2             # minimum loop-witnesses per (s,a) before mask fires
    base_entropy = 1.0    # all logits start at 0 -> uniform softmax

    # Logits per (state-hash, action). Default is zeros (uniform). We do not
    # do gradient updates on un-suppressed actions per hypothesis #3: only
    # suppression mask shapes the policy.
    logits_table: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(n_actions, dtype=np.float64)
    )

    # Per-(s,a) running sum and count of loop-signature Δc vectors.
    L_sum: dict[tuple[int, int], np.ndarray] = {}
    L_count: dict[tuple[int, int], int] = defaultdict(int)

    # Diagnostics
    n_decision_steps = 0
    n_loop_events = 0
    n_pareto_dom_zero = 0
    sign_hist_pos = None  # per channel, how often L is > 0
    sign_hist_neg = None
    sign_hist_zero = None

    rng = np.random.default_rng(seed)

    def policy_logits_for(h: int) -> np.ndarray:
        """Return effective logits for state-hash h after suppression."""
        base = logits_table[h]
        # Build mask from L_sum / L_count for each action.
        mask = np.zeros(n_actions, dtype=np.float64)
        for a in range(n_actions):
            cnt = L_count.get((h, a), 0)
            if cnt < n_min:
                continue
            mean_vec = L_sum[(h, a)] / cnt
            # Pareto-dominated by zero: all channels <= 0, at least one < 0.
            if np.all(mean_vec <= 0.0) and np.any(mean_vec < 0.0):
                mask[a] = 1.0
        return base - alpha * mask

    def sample_action(h: int) -> int:
        eff = policy_logits_for(h)
        # Numerically stable softmax sampling
        eff = eff - eff.max()
        p = np.exp(eff)
        p = p / p.sum()
        return int(rng.choice(n_actions, p=p))

    def update_from_trajectory(traj_hashes, traj_actions, traj_cums):
        """Scan a finished trajectory for intra-episode hash recurrences and
        update L_sum / L_count with per-channel cumulant deltas.

        traj_hashes[t]   : obs hash at decision step t (state agent saw before action)
        traj_actions[t]  : action taken at step t
        traj_cums[t]     : cumulative vector cumulant up to (and including) step t
                           (so c_t' - c_t for t' > t is the loop signature for
                           a loop entered at step t and closed at step t').
                           For t=0 we use a zero anchor before any step has run;
                           we pass cums starting AFTER the first step, with a
                           prepended zero so traj_cums[0]=0 corresponds to
                           "cumulant before the first action".
        """
        nonlocal n_loop_events, n_pareto_dom_zero, sign_hist_pos, sign_hist_neg, sign_hist_zero

        # Group decision-step indices by hash so we can pair up recurrences in O(T).
        # For each pair (t, t') with t < t' and same hash, the loop entered by
        # action a_t closes at t' with signature traj_cums[t'] - traj_cums[t].
        hash_to_indices: dict[int, list[int]] = defaultdict(list)
        for t, h in enumerate(traj_hashes):
            hash_to_indices[h].append(t)

        for _h, idxs in hash_to_indices.items():
            if len(idxs) < 2:
                continue
            # All pairs (t, t') with t < t': potentially O(k^2) per bucket but
            # bounded by trajectory length and bucket size. To keep cost O(T)-
            # ish, pair each occurrence only with its immediate next occurrence
            # (the smallest closed loop entered by a_t). This still gives one
            # loop-update event per recurrence and is faithful to the
            # hypothesis's "scan each new trajectory for hash recurrences"
            # description.
            for j in range(len(idxs) - 1):
                t = idxs[j]
                tp = idxs[j + 1]
                a = traj_actions[t]
                delta = traj_cums[tp] - traj_cums[t]
                key = (_h, a)
                if key not in L_sum:
                    L_sum[key] = np.zeros_like(delta, dtype=np.float64)
                L_sum[key] = L_sum[key] + delta
                L_count[key] += 1
                n_loop_events += 1

                if sign_hist_pos is None:
                    k = delta.shape[0]
                    sign_hist_pos = np.zeros(k, dtype=np.int64)
                    sign_hist_neg = np.zeros(k, dtype=np.int64)
                    sign_hist_zero = np.zeros(k, dtype=np.int64)
                # Update sign histogram on the running mean for this (s,a)
                mean_vec = L_sum[key] / L_count[key]
                sign_hist_pos += (mean_vec > 0).astype(np.int64)
                sign_hist_neg += (mean_vec < 0).astype(np.int64)
                sign_hist_zero += (mean_vec == 0).astype(np.int64)
                if np.all(mean_vec <= 0.0) and np.any(mean_vec < 0.0):
                    n_pareto_dom_zero += 1

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave grace for eval
    env_steps = 0
    episodes = 0
    obs, _ = env.reset(seed=seed)

    # Per-episode buffers
    traj_hashes: list[int] = []
    traj_actions: list[int] = []
    # cums[t] = vector cumulant before action t (cums[0] = 0)
    # We grow it by appending the post-step cumulant.
    if is_vector:
        # Determine k once on the first vector observation
        # We probe by taking a no-op-ish step once; safer to start at zero and
        # shape on first step.
        cur_cum = None  # set on first step
    else:
        cur_cum = np.zeros(1, dtype=np.float64)  # 1-D scalar projection
    cums_list: list[np.ndarray] = []

    while time.monotonic() < deadline:
        h = _obs_hash(obs)
        # Append "cumulant before this action" anchor.
        if cur_cum is None:
            # We do not yet know k; use placeholder; will replace on first step.
            pass
        else:
            cums_list.append(cur_cum.copy())

        a = sample_action(h)
        traj_hashes.append(h)
        traj_actions.append(a)

        next_obs, reward, term, trunc, info = env.step(a)
        env_steps += 1

        if is_vector:
            v = np.asarray(info.get("vector", [reward]), dtype=np.float64)
            if cur_cum is None:
                cur_cum = np.zeros_like(v, dtype=np.float64)
                # Backfill the missing pre-step anchor for this single decision step.
                cums_list.append(cur_cum.copy())
            cur_cum = cur_cum + v
        else:
            cur_cum = cur_cum + np.asarray([reward], dtype=np.float64)

        obs = next_obs
        done = bool(term) or bool(trunc)

        if done:
            # Append the final post-step cumulant so that for any t, cums[t+1]
            # is the cumulant AFTER step t. For loop signature delta we want
            # "cumulant accumulated between decision step t and decision step t'".
            # Decision step t happens BEFORE step t. Closing at decision step
            # t' means we returned to the same hash before taking an action
            # there. So loop signature = cum_before(t') - cum_before(t)
            # = cums_list[t'] - cums_list[t]. cums_list as built has length
            # equal to len(traj_hashes) (one anchor per decision). Good.
            update_from_trajectory(traj_hashes, traj_actions, cums_list)
            episodes += 1

            # Reset buffers
            traj_hashes = []
            traj_actions = []
            cums_list = []
            if is_vector:
                cur_cum = None
            else:
                cur_cum = np.zeros(1, dtype=np.float64)
            obs, _ = env.reset(seed=seed + episodes + 1)
        n_decision_steps += 1

    env.close()

    # Diagnostics
    collision_rate = (n_loop_events / max(1, n_decision_steps))
    pareto_rate = (n_pareto_dom_zero / max(1, n_loop_events))
    print(
        f"[lra] env={env_id} seed={seed} env_steps={env_steps} episodes={episodes} "
        f"train_s={time.monotonic() - t_start:.1f} budget_s={time_budget_s} "
        f"loop_events={n_loop_events} collision_rate={collision_rate:.3f} "
        f"pareto_dom_zero_rate={pareto_rate:.3f} unique_states={len(logits_table)} "
        f"masked_keys={sum(1 for k in L_count if L_count[k] >= n_min)}",
        flush=True,
    )
    if sign_hist_pos is not None:
        print(
            f"[lra] sign_hist pos={sign_hist_pos.tolist()} "
            f"neg={sign_hist_neg.tolist()} zero={sign_hist_zero.tolist()}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Frozen deployment policy (no extra rollouts; consults static lookup).
    # We sample from the masked softmax with a fresh deterministic RNG.
    # ------------------------------------------------------------------
    eval_rng = np.random.default_rng(seed + 7777)

    # Snapshot the L tables into immutable arrays for fast eval.
    masked_actions: dict[int, np.ndarray] = {}
    seen_state_logits: dict[int, np.ndarray] = dict(logits_table)
    for (h, a), cnt in L_count.items():
        if cnt < n_min:
            continue
        mean_vec = L_sum[(h, a)] / cnt
        if np.all(mean_vec <= 0.0) and np.any(mean_vec < 0.0):
            arr = masked_actions.setdefault(h, np.zeros(n_actions, dtype=np.float64))
            arr[a] = 1.0

    def policy_fn(obs_in):
        h = _obs_hash(obs_in)
        base = seen_state_logits.get(h)
        if base is None:
            base = np.zeros(n_actions, dtype=np.float64)
        mask = masked_actions.get(h)
        eff = base if mask is None else (base - alpha * mask)
        eff = eff - eff.max()
        p = np.exp(eff)
        p = p / p.sum()
        return int(eval_rng.choice(n_actions, p=p))

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
