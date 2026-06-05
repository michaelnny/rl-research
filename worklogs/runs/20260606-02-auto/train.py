"""DPC: Divergent-Prefix Concordance.

Realizes the hypothesis at worklogs/runs/20260606-02-auto/hypothesis.md.

Core primitive:
    V[s_div, a, a', m]  per-channel sign-vote tensor over divergence events
    S[s_div, a, a', m]  raw cumulant-sum-difference tensor (Pareto gate)

For every pair of trajectories (tau_i, tau_j) currently in the buffer, find
their longest common ACTION prefix. Let k be its length, s_div = obs_hash at
step k, a_i, a_j be the two first-divergent actions. Compute the terminal
suffix vector cumulants c_T(tau_i), c_T(tau_j) (sum over per-step vector
signal from step k onward, taken from info["vector"] for vector envs, or
from a 1-D per-step scalar reward for scalar envs).

Update:
    V[s_div, a_i, a_j, m] += sign((c_i - c_j)[m])
    V[s_div, a_j, a_i, m] += sign((c_j - c_i)[m])
    S[s_div, a_i, a_j, m] += (c_i - c_j)[m]
    S[s_div, a_j, a_i, m] += (c_j - c_i)[m]

Improvement operator at s_div: for each candidate action a, compute
    mu(a) = sum_{a' != a} 1[
        V[s_div, a, a', :] componentwise-dominates 0     # all signs > 0
        AND S[s_div, a, a', :] is Pareto-non-dominated by S[s_div, a', a, :]
    ]
Then take a small SGD step on cross-entropy between current policy
pi(.|s_div) and softmax(alpha * mu(.)).

Execution rule: sample from current policy at fixed temperature; the policy
itself drives all rollouts. For scalar envs we synthesize a 1-D per-step
"vector" cumulant from the scalar reward signal (the per-channel sign-vote
machinery is identical with d=1; the hypothesis's failure-mode 2 explicitly
predicts collapse on collinear/1-D channel envs).

Buffer is bounded; pairs are subsampled per the Reviewer's compute risk note.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Callable

import numpy as np

import harness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_hash(obs) -> int:
    """Cheap, near-free prefix-matching key. Used ONLY to identify s_div."""
    arr = np.asarray(obs)
    # Quantize floats lightly to make hashing robust across float reps.
    if np.issubdtype(arr.dtype, np.floating):
        q = np.round(arr * 1000.0).astype(np.int64)
    else:
        q = arr.astype(np.int64, copy=False)
    return int(hash(q.tobytes()))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(x) / len(x)
    return ex / s


# ---------------------------------------------------------------------------
# DPC trainer
# ---------------------------------------------------------------------------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    rng = np.random.default_rng(seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # Action-space size.
    if not hasattr(env.action_space, "n"):
        # Continuous action spaces are out of scope for this primitive.
        env.close()
        action_space = env.action_space

        def policy_fn_random(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} unsupported action space; random fallback",
            flush=True,
        )
        return policy_fn_random

    n_actions = int(env.action_space.n)

    # Logits per s_div hash. Default: zero -> uniform policy. Unknown s_div
    # at decision time falls back to uniform.
    logits: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    # Tensor V and S, sparse-keyed by (s_div, a, a').
    # Keep d as the channel dim. Vector envs: from info["vector"]; scalar: d=1.
    # We discover d on the first step.
    d_holder = {"d": None}

    def _ensure_d(v):
        if d_holder["d"] is None:
            d_holder["d"] = int(np.asarray(v).shape[0]) if np.asarray(v).ndim else 1

    V: dict[tuple, np.ndarray] = {}
    S: dict[tuple, np.ndarray] = {}

    # Hyper-params (per hypothesis: small SGD step size, fixed temperature).
    temperature = 1.0
    alpha = 1.0      # operator scale on mu
    lr = 0.5         # SGD step on cross-entropy of logits at s_div
    buffer_max = 32  # keep recent on-policy trajectories; bound compute
    pair_cap = 64    # subsample pairs per update (Reviewer compute note)

    # Diagnostics.
    diag_decisions = 0
    diag_nudges_fired = 0       # decision steps where mu(a) sum > 0 anywhere
    diag_pairs_processed = 0
    diag_unique_sdiv_aa = set()
    diag_updates = 0

    # On-policy buffer of trajectories.
    # Each entry: dict with "actions", "obs_hashes", "vec_per_step" (T, d)
    buffer: list[dict] = []

    # ---- Policy ----
    def sample_action(obs_hash: int) -> int:
        lg = logits.get(obs_hash)
        if lg is None:
            probs = np.ones(n_actions, dtype=np.float64) / n_actions
        else:
            probs = _softmax(lg / max(temperature, 1e-6))
        return int(rng.choice(n_actions, p=probs))

    # ---- One trajectory rollout ----
    def rollout_once() -> dict | None:
        try:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        except Exception:
            return None
        actions: list[int] = []
        hashes: list[int] = []
        vecs: list[np.ndarray] = []
        steps = 0
        max_steps = harness.MAX_EPISODE_STEPS
        while steps < max_steps:
            h = _obs_hash(obs)
            a = sample_action(h)
            try:
                step_out = env.step(a)
            except Exception:
                break
            if len(step_out) == 5:
                obs2, reward, term, trunc, info = step_out
            else:
                # Defensive: some envs may use 4-tuple; treat trunc=False.
                obs2, reward, term, info = step_out
                trunc = False
            if is_vector:
                v = np.asarray(info.get("vector", np.array([float(reward)])), dtype=np.float64)
            else:
                v = np.asarray([float(reward)], dtype=np.float64)
            _ensure_d(v)
            actions.append(int(a))
            hashes.append(int(h))
            vecs.append(v)
            steps += 1
            if bool(term) or bool(trunc):
                break
            obs = obs2
        if not actions:
            return None
        return {
            "actions": np.asarray(actions, dtype=np.int64),
            "obs_hashes": np.asarray(hashes, dtype=np.int64),
            "vec_per_step": np.stack(vecs, axis=0),  # (T, d)
        }

    # ---- Pair update: build/refresh divergence-event tensor for one new tau ----
    def update_from_pairs(new_traj: dict, others: list[dict]) -> int:
        nonlocal diag_pairs_processed
        d = d_holder["d"]
        if d is None:
            return 0
        sdiv_set: set[tuple] = set()
        # Subsample pair partners.
        if len(others) > pair_cap:
            idxs = rng.choice(len(others), size=pair_cap, replace=False)
            partners = [others[i] for i in idxs]
        else:
            partners = others
        a_i_arr = new_traj["actions"]
        h_i_arr = new_traj["obs_hashes"]
        v_i_arr = new_traj["vec_per_step"]
        T_i = len(a_i_arr)
        for tj in partners:
            a_j_arr = tj["actions"]
            h_j_arr = tj["obs_hashes"]
            v_j_arr = tj["vec_per_step"]
            T_j = len(a_j_arr)
            # Find longest common ACTION prefix; require shared start hash too
            # (the s_div key must be a real shared decision context).
            k = 0
            T_min = min(T_i, T_j)
            while k < T_min and a_i_arr[k] == a_j_arr[k] and h_i_arr[k] == h_j_arr[k]:
                k += 1
            if k >= T_min:
                # No divergence in the overlapping prefix (one is a prefix of
                # the other or they coincide). Skip.
                continue
            # Verify they share the obs hash at position k (same s_div); this
            # is required for the divergence event to be well-defined.
            if h_i_arr[k] != h_j_arr[k]:
                continue
            s_div = int(h_i_arr[k])
            ai = int(a_i_arr[k])
            aj = int(a_j_arr[k])
            if ai == aj:
                # Tie at position k means the divergence was actually later;
                # by our prefix check this shouldn't occur. Defensive skip.
                continue
            # Terminal suffix vector cumulants from step k onward.
            c_i = v_i_arr[k:].sum(axis=0)
            c_j = v_j_arr[k:].sum(axis=0)
            delta = (c_i - c_j).astype(np.float64)
            sgn = np.sign(delta)
            key_ij = (s_div, ai, aj)
            key_ji = (s_div, aj, ai)
            if key_ij not in V:
                V[key_ij] = np.zeros(d, dtype=np.float64)
                S[key_ij] = np.zeros(d, dtype=np.float64)
            if key_ji not in V:
                V[key_ji] = np.zeros(d, dtype=np.float64)
                S[key_ji] = np.zeros(d, dtype=np.float64)
            V[key_ij] += sgn
            V[key_ji] += -sgn
            S[key_ij] += delta
            S[key_ji] += -delta
            sdiv_set.add((s_div, ai, aj))
            sdiv_set.add((s_div, aj, ai))
            diag_unique_sdiv_aa.add(key_ij)
            diag_unique_sdiv_aa.add(key_ji)
            diag_pairs_processed += 1
        # Return set of s_div nodes that received any update.
        return len({k_[0] for k_ in sdiv_set})

    # ---- Improvement operator: cross-entropy SGD step on logits at s_div ----
    def apply_operator(touched_sdivs: set[int]) -> None:
        nonlocal diag_decisions, diag_nudges_fired, diag_updates
        for s_div in touched_sdivs:
            mu = np.zeros(n_actions, dtype=np.float64)
            for a in range(n_actions):
                count = 0
                for a2 in range(n_actions):
                    if a == a2:
                        continue
                    k_aa = (s_div, a, a2)
                    k_aa_rev = (s_div, a2, a)
                    if k_aa not in V:
                        continue
                    sign_vec = V[k_aa]
                    # Componentwise-dominates 0: all signs strictly > 0.
                    if not np.all(sign_vec > 0):
                        continue
                    # Pareto-non-dominance gate on raw cumulant sums S:
                    # S[a,a'] is Pareto-non-dominated by S[a',a] iff it is
                    # NOT the case that S[a',a] >= S[a,a'] componentwise with
                    # at least one strict inequality.
                    s_aa = S[k_aa]
                    s_rev = S.get(k_aa_rev, -s_aa)  # by construction equal to -s_aa
                    dominated = bool(np.all(s_rev >= s_aa) and np.any(s_rev > s_aa))
                    if dominated:
                        continue
                    count += 1
                mu[a] = float(count)
            diag_decisions += 1
            if mu.sum() > 0:
                diag_nudges_fired += 1
            else:
                continue  # no signal at this s_div
            target = _softmax(alpha * mu)
            # Cross-entropy between target and softmax(logits): grad on logits
            # is (current - target). SGD step: logits -= lr * (current - target).
            cur = _softmax(logits[s_div] / max(temperature, 1e-6))
            grad = cur - target
            logits[s_div] = logits[s_div] - lr * grad
            diag_updates += 1

    # ---- Main loop ----
    t0 = time.monotonic()
    deadline = t0 + max(1.0, time_budget_s - 5.0)  # leave headroom for eval
    n_rollouts = 0
    total_env_steps = 0
    while time.monotonic() < deadline:
        traj = rollout_once()
        if traj is None:
            break
        n_rollouts += 1
        total_env_steps += int(traj["actions"].shape[0])
        # Pair the new trajectory with all in buffer; update tensor; fire op.
        touched_sdivs_count = update_from_pairs(traj, buffer)
        # Collect the s_divs touched, but update_from_pairs already
        # accumulated into V/S; re-derive touched set from new pair partners
        # by recomputing (cheap because we only need s_div nodes).
        touched_sdivs: set[int] = set()
        for tj in buffer[-min(pair_cap, len(buffer)):]:
            T_min = min(len(traj["actions"]), len(tj["actions"]))
            k = 0
            ai = traj["actions"]; aj = tj["actions"]
            hi = traj["obs_hashes"]; hj = tj["obs_hashes"]
            while k < T_min and ai[k] == aj[k] and hi[k] == hj[k]:
                k += 1
            if k < T_min and hi[k] == hj[k] and ai[k] != aj[k]:
                touched_sdivs.add(int(hi[k]))
        apply_operator(touched_sdivs)
        # Maintain bounded buffer.
        buffer.append(traj)
        if len(buffer) > buffer_max:
            buffer.pop(0)
        _ = touched_sdivs_count  # silence linter
        if time.monotonic() >= deadline:
            break

    # ---- Diagnostics ----
    inv_rate = (diag_nudges_fired / max(diag_decisions, 1)) if diag_decisions else 0.0
    pairs_per_unique = (
        diag_pairs_processed / max(len(diag_unique_sdiv_aa), 1)
        if diag_unique_sdiv_aa
        else 0.0
    )
    print(
        f"[train] env={env_id} seed={seed} env_steps={total_env_steps} "
        f"rollouts={n_rollouts} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} d={d_holder['d']} "
        f"sdiv_nodes={len(logits)} V_cells={len(V)} "
        f"updates={diag_updates} decisions={diag_decisions} "
        f"nudges_fired={diag_nudges_fired} invocation_rate={inv_rate:.3f} "
        f"pairs={diag_pairs_processed} mean_pairs_per_unique={pairs_per_unique:.3f}",
        flush=True,
    )

    # Snapshot final logits as a plain dict for closure (avoid defaultdict
    # mutation at evaluation time).
    final_logits: dict[int, np.ndarray] = {k: v.copy() for k, v in logits.items()}
    final_temperature = temperature
    eval_rng = np.random.default_rng(seed + 7777)

    def policy_fn(obs) -> int:
        h = _obs_hash(obs)
        lg = final_logits.get(h)
        if lg is None:
            probs = np.ones(n_actions, dtype=np.float64) / n_actions
        else:
            probs = _softmax(lg / max(final_temperature, 1e-6))
        return int(eval_rng.choice(n_actions, p=probs))

    try:
        env.close()
    except Exception:
        pass
    return policy_fn


# ---------------------------------------------------------------------------
# CLI shell (kept identical to substrate floor)
# ---------------------------------------------------------------------------

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
