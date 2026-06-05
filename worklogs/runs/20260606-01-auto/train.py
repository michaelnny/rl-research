"""CEC: Continuation-Endpoint Concordance.

Per-(state-hash, action) Pareto-bucket multiset of vector cumulants, indexed
by exit-observation-hash bucket. Logit updates are signed concordance counts:
the number of exit-hash buckets where action a's bucket-conditional mean
cumulant Pareto-dominates a's, minus the reverse.

Contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120

For vector envs, training MUST consume info["vector"]. CEC keeps the cumulant
vector-valued throughout; the only Boolean reduction is per-bucket Pareto
dominance, then *counted* across exit-hash buckets (never summed over channels).
"""

from __future__ import annotations

import argparse
import hashlib
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


def _obs_hash(obs) -> bytes:
    """Stable hash of an observation (used both for state-key and exit-key)."""
    arr = np.asarray(obs)
    # On dict/object obs, fall back to repr; on array obs use bytes of contents.
    if arr.dtype == object:
        return hashlib.blake2b(repr(obs).encode("utf-8"), digest_size=16).digest()
    return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()


def _pareto_strict_dominates(u: np.ndarray, v: np.ndarray) -> bool:
    """Strict coordinate-wise Pareto dominance: u >= v on all coords, > on some."""
    return bool(np.all(u >= v) and np.any(u > v))


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    n_actions = int(action_space.n) if hasattr(action_space, "n") else None
    if n_actions is None:
        # CEC's primitive operates on a finite action set; fall back to random.
        rng = np.random.default_rng(seed + 99_999)

        def policy_fn(_obs):
            return action_space.sample()

        env.close()
        print(
            f"[train] env={env_id} seed={seed} CEC: continuous action space, fallback random",
            flush=True,
        )
        return policy_fn

    is_vector = harness.ENV_TYPE[env_id] == "vector"

    rng = np.random.default_rng(seed + 12_345)
    t_start = time.monotonic()

    # --- CEC primitive state ---
    # logits[state_hash] -> np.ndarray shape (n_actions,)
    logits: dict[bytes, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    # samples[state_hash][action][exit_hash] -> list of vector cumulants Δc (np.ndarray, shape (k,))
    samples: dict[bytes, dict[int, dict[bytes, list[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # CEC hyperparameters (pinned, not tuned-to-target):
    alpha = 0.1            # logit step size for concordance vote
    temperature = 1.0      # softmax temperature for action sampling
    min_bucket_samples = 2 # threshold for bucket inclusion in concordance count
    seed_episodes = 8      # warm-up episodes before logit nudges fire
    max_logit_abs = 10.0   # clamp logits to keep softmax numerically stable

    # Determine vector dim k from a single dummy step on vector envs.
    # On non-vector envs, fabricate a 2-channel diagnostic vector:
    #   channel 0: scalar reward
    #   channel 1: -1 per step (a step counter / shorter-is-better proxy)
    # This is the reset-structural / vector-diagnostic side-info the
    # hypothesis explicitly relies on for sparse envs.
    obs0, _ = env.reset(seed=seed)
    if is_vector:
        # Probe one step to discover k.
        a0 = int(rng.integers(n_actions))
        _, _, term, trunc, info = env.step(a0)
        v_probe = np.asarray(info["vector"], dtype=np.float64)
        k = int(v_probe.shape[0])
        # Reset since we consumed a step.
        obs0, _ = env.reset(seed=seed + 1)
    else:
        k = 2

    # --- Episode loop ---
    n_episodes = 0
    n_logit_updates = 0
    n_steps_total = 0
    bucket_size_diag: list[int] = []  # for falsifier diagnostic

    def softmax_action(state_hash: bytes) -> int:
        l = logits[state_hash]
        l = np.clip(l, -max_logit_abs, max_logit_abs)
        z = (l - l.max()) / max(temperature, 1e-6)
        p = np.exp(z)
        p = p / p.sum()
        return int(rng.choice(n_actions, p=p))

    while True:
        if time.monotonic() - t_start > max(1.0, time_budget_s - 3.0):
            break

        # --- Collect one episode ---
        obs, _ = env.reset(seed=seed + 2_000 + n_episodes)
        traj_state_hashes: list[bytes] = []
        traj_actions: list[int] = []
        traj_v: list[np.ndarray] = []  # per-step vector signal v_u

        done = False
        steps = 0
        max_steps = harness.MAX_EPISODE_STEPS
        last_obs = obs
        while not done and steps < max_steps:
            s_hash = _obs_hash(obs)
            traj_state_hashes.append(s_hash)
            if n_episodes < seed_episodes:
                a = int(rng.integers(n_actions))
            else:
                a = softmax_action(s_hash)
            traj_actions.append(a)
            obs_next, reward, term, trunc, info = env.step(a)
            if is_vector:
                v_u = np.asarray(info["vector"], dtype=np.float64)
                if v_u.shape[0] != k:
                    # If vector dimension somehow drifts, re-pad/truncate.
                    tmp = np.zeros(k, dtype=np.float64)
                    tmp[: min(k, v_u.shape[0])] = v_u[: min(k, v_u.shape[0])]
                    v_u = tmp
            else:
                # Synthetic per-step vector for non-vector envs:
                # (scalar_reward, -1.0) — keeps the cumulant vector-valued
                # so the partial-order comparison is meaningful (shorter
                # paths Pareto-dominate longer ones once reward is matched).
                v_u = np.array([float(reward), -1.0], dtype=np.float64)
            traj_v.append(v_u)
            done = bool(term) or bool(trunc)
            last_obs = obs_next
            obs = obs_next
            steps += 1
            n_steps_total += 1
            if time.monotonic() - t_start > max(1.0, time_budget_s - 3.0):
                break

        if len(traj_state_hashes) == 0:
            n_episodes += 1
            continue

        # Exit hash := hash of the terminal observation.
        exit_hash = _obs_hash(last_obs)

        # Compute Δc = Σ_{u=t}^{T-1} v_u for every step t (suffix sums).
        T = len(traj_v)
        suffix = np.zeros((T + 1, k), dtype=np.float64)
        for u in range(T - 1, -1, -1):
            suffix[u] = suffix[u + 1] + traj_v[u]

        # Record (s, a, exit_hash, Δc_t) into the sample bag.
        # Only one Δc per (s, a, exit) pair from a given trajectory — that
        # is the natural "one episode contributes one Δc per (s,a) visit"
        # rule the hypothesis describes (a step that visits (s,a) records
        # its own continuation Δc; multiple visits in one episode are
        # treated as separate samples within the same exit bucket).
        touched_states: set[bytes] = set()
        for t in range(T):
            s_h = traj_state_hashes[t]
            a_t = traj_actions[t]
            dc = suffix[t].copy()
            samples[s_h][a_t][exit_hash].append(dc)
            touched_states.add(s_h)

        n_episodes += 1

        # --- Apply CEC concordance update for visited states ---
        if n_episodes <= seed_episodes:
            continue

        for s_h in touched_states:
            sa_buckets = samples[s_h]
            actions_seen = list(sa_buckets.keys())
            if len(actions_seen) < 2:
                continue

            # For each pair (a, a'), compute concordance C(s; a, a') as the
            # signed count of exit-hash buckets where mean Δc(a) Pareto-
            # dominates mean Δc(a'), minus the reverse. Only include buckets
            # with ≥ min_bucket_samples in BOTH actions (robust dominance).
            concordance = np.zeros(n_actions, dtype=np.float64)
            for ai in actions_seen:
                ci_total = 0
                for aj in actions_seen:
                    if aj == ai:
                        continue
                    bi = sa_buckets[ai]
                    bj = sa_buckets[aj]
                    shared_exits = set(bi.keys()) & set(bj.keys())
                    c_ij = 0
                    for ex in shared_exits:
                        if len(bi[ex]) < min_bucket_samples:
                            continue
                        if len(bj[ex]) < min_bucket_samples:
                            continue
                        mi = np.mean(np.stack(bi[ex], axis=0), axis=0)
                        mj = np.mean(np.stack(bj[ex], axis=0), axis=0)
                        if _pareto_strict_dominates(mi, mj):
                            c_ij += 1
                        elif _pareto_strict_dominates(mj, mi):
                            c_ij -= 1
                    ci_total += c_ij
                concordance[ai] = ci_total

            if np.all(concordance == 0):
                continue

            # Diagnostic: track median bucket size at this state.
            for ai in actions_seen:
                for ex, bag in sa_buckets[ai].items():
                    bucket_size_diag.append(len(bag))

            # Logit update: logit(s, a) += alpha * C(s; a, ·) summed over a' != a,
            # which is exactly `concordance[a]` divided by |A|.
            logits[s_h] += alpha * concordance / max(1, n_actions)
            logits[s_h] = np.clip(logits[s_h], -max_logit_abs, max_logit_abs)
            n_logit_updates += 1

    # --- Falsifier diagnostic ---
    if bucket_size_diag:
        med_bucket = float(np.median(bucket_size_diag))
    else:
        med_bucket = 0.0
    n_states = len(samples)
    n_logit_states = len(logits)

    env.close()

    print(
        f"[train] env={env_id} seed={seed} CEC episodes={n_episodes} "
        f"steps={n_steps_total} states={n_states} logit_states={n_logit_states} "
        f"logit_updates={n_logit_updates} median_bucket_size={med_bucket:.2f} "
        f"k={k} train_s={time.monotonic() - t_start:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Snapshot logits as a frozen lookup; unseen states fall back to argmax over
    # zero-vector logits (i.e. action 0). To keep deployment a single forward
    # pass per the hypothesis's rollout-cost discipline, use deterministic argmax.
    frozen_logits = {k_: v.copy() for k_, v in logits.items()}

    def policy_fn(obs):
        s_h = _obs_hash(obs)
        l = frozen_logits.get(s_h)
        if l is None:
            return 0
        return int(np.argmax(l))

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
