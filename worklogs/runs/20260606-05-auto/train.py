"""TPP — Terminal-Postfix Pairing.

Realizes the hypothesis in worklogs/runs/20260606-05-auto/hypothesis.md.

Primitive: per-(observation-hash, action) postfix-Pareto-vote count W[s, a].
For every unordered pair of completed trajectories (tau, tau') whose terminal
observation-hashes match, walk both backward in lockstep until their
observation sequences first disagree. The "postfix-divergence anchor" is the
observation just before that disagreement; the "postfix-divergence actions"
are the actions chosen at that anchor on both trajectories. If the terminal
vector outcomes are Pareto-comparable, increment W[anchor, a] for the action
on the Pareto-non-dominated side (the strict-dominator side). Pareto-
incomparable pairs abstain. The improvement operator is a logit nudge at
each anchor toward argmax of W[s, .].

Vector feedback: Pareto comparison componentwise on terminal vectors c_T,
no scalarization. For scalar envs (sparse axis), c_T is the trajectory's
sum scalar reward (1-D vector); Pareto comparison reduces to scalar
comparison there but the structural object is the same.

Falsifier diagnostics logged to stdout:
  - terminal-hash collision rate among trajectory pairs
  - Pareto-comparable rate among matched-terminal-hash pairs
  - distribution of postfix-shared depth Delta
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


def _obs_hash(obs) -> int:
    arr = np.ascontiguousarray(np.asarray(obs))
    h = hashlib.blake2b(arr.tobytes(), digest_size=8)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return int.from_bytes(h.digest(), "big", signed=False)


def _pareto_relation(a: np.ndarray, b: np.ndarray, atol: np.ndarray) -> int:
    """Return +1 if a strictly Pareto-dominates b, -1 if b dominates a,
    0 if Pareto-incomparable (or near-equal within per-channel atol).
    Uses MAD-standardized tolerances passed in via `atol` for tie-breaking
    under near-equality.
    """
    diff = a - b
    a_ge = diff >= -atol
    b_ge = diff <= atol
    a_strict = diff > atol
    b_strict = diff < -atol
    if a_ge.all() and a_strict.any():
        return 1
    if b_ge.all() and b_strict.any():
        return -1
    return 0


def _postfix_anchor(traj_a, traj_b):
    """Walk two trajectories backward in lockstep from their (matched) terminal
    observations. Find largest Delta such that hash(obs_{T-d}) ==
    hash(obs'_{T'-d}) for all 0 <= d <= Delta. The postfix-divergence anchor
    is then at index (T - Delta - 1) in tau and (T' - Delta - 1) in tau'.

    Each trajectory is a list of (obs_hash, action) tuples for steps
    0..T-1, plus a terminal_obs_hash. We need the action chosen at the
    pre-divergence step.

    Returns (anchor_obs_hash, action_a, action_b, Delta) or None if no valid
    anchor exists (e.g., one trajectory is shorter than the shared depth).

    Special case: if at some depth d the obs hashes still match but the
    actions chosen differ, that observation is itself the anchor and the
    postfix-divergence actions are those two differing actions.
    """
    # traj is list of (obs_hash, action) for t=0..T-1, terminal_obs_hash matches
    # backward step depth d=0 corresponds to the terminal observation
    # (which always matches by construction)
    # depth d=k corresponds to obs at index T-k for the per-step list:
    #   the observation BEFORE action a_{T-k} was taken is obs at index T-k.
    # We track per-step (obs_hash, action). The terminal observation is
    # appended as a separate entry with no action.
    # So traj is: [(obs_0, a_0), (obs_1, a_1), ..., (obs_{T-1}, a_{T-1})]
    # and terminal_obs_hash is the obs after a_{T-1}. They must match across
    # the two trajectories at depth 0.

    Ta = len(traj_a)
    Tb = len(traj_b)
    max_walk = min(Ta, Tb)

    # Special case: if at depth d the agent chose different actions but the
    # observation was the same, the anchor is that observation. We need to
    # check this for each step.
    # The action chosen at obs_{T-1-d} (zero-indexed step T-1-d) is
    # traj[T-1-d][1]; that's what brought us to obs_{T-d} (or terminal if
    # d == 0).
    #
    # Backward walk: for d = 1, 2, ...:
    #   compare obs at index Ta-d in traj_a vs obs at index Tb-d in traj_b
    #   (these are obs_{T-d}, the observation at depth d back from terminal).
    # As long as they match, increment Delta. When they first differ,
    # the anchor observation is the previous step's obs (depth d-1).
    # The action at the anchor on each trajectory is the action that took
    # the trajectory from the anchor toward the next (still-shared) state.

    Delta = 0
    for d in range(1, max_walk + 1):
        oa = traj_a[Ta - d][0]
        ob = traj_b[Tb - d][0]
        if oa == ob:
            Delta = d
        else:
            break

    # Now: depth Delta is the deepest backward step where obs hashes still
    # match. The postfix-divergence anchor is at depth Delta + 1 (one step
    # further back). But we also need to handle the same-obs-different-action
    # case: at depth Delta, the obs at that step is shared; if the actions
    # taken FROM that obs differ, the anchor is that obs itself.
    #
    # Action taken from obs at depth d: the action at traj[T-d][1] is the
    # action that produced obs_{T-d+1} (i.e., the depth-(d-1) observation).
    # Equivalently: at the step whose obs is obs_{T-d}, the action chosen
    # is at index Ta-d in traj_a.
    #
    # Actually for depth d shared, the actions chosen at that shared obs
    # are: traj_a[Ta - d][1] and traj_b[Tb - d][1]. If those differ AND
    # depth d+1 is where divergence happened (or d == max_walk), we can use
    # this shared obs as the anchor.

    # We want the *deepest* (largest d) point at which the postfix walks
    # diverged. The walks diverge first at depth Delta+1 (the obs differs)
    # OR at depth Delta where the action differs (same obs, different action
    # taken). In the latter case, the anchor IS the depth-Delta observation
    # and the actions are the differing actions taken at that step.

    # Case 1: anchor is at the action level at depth Delta (same obs at
    # Delta, different action chosen).
    # Case 2: anchor is at depth Delta+1 (the previous step before obs
    # diverged).

    # For Case 2 to apply, depth Delta+1 must exist on both trajectories
    # (i.e., max_walk >= Delta+1) AND the obs at that depth must be
    # different across the two trajectories. We also need to know the
    # anchor obs itself; in Case 2 the anchor obs is at depth Delta on
    # one trajectory but at depth Delta+1 the obs differs, so the
    # "anchor" interpreted as "last shared obs" is at depth Delta. The
    # action chosen at the anchor is the one that led to the differing
    # next obs.

    # Both cases have anchor obs at depth Delta (or depth 0 = terminal if
    # Delta == 0). Action at anchor on each trajectory is the action chosen
    # at the depth-Delta step:
    #   action_a_at_anchor = traj_a[Ta - Delta - 1][1]   (if Delta < Ta)
    #   action_b_at_anchor = traj_b[Tb - Delta - 1][1]   (if Delta < Tb)
    #
    # The anchor obs is the obs at the depth-Delta step (which is shared
    # by definition):
    #   anchor_obs_hash = traj_a[Ta - Delta - 1][0]      (if Delta < Ta)
    # OR if Delta == max_walk and one trajectory is exactly that length,
    # the anchor obs is the terminal obs itself? No -- if Delta equals
    # min(Ta, Tb) entirely, the shorter trajectory has no "depth Delta+1"
    # step to compare; we treat this as no qualifying anchor (the shorter
    # trajectory was a true postfix of the longer one and the divergence
    # is at the root of the shorter one).

    if Delta >= Ta or Delta >= Tb:
        # One trajectory is a strict postfix of the other; the anchor would
        # be at or before the start of the shorter one. We require both
        # trajectories to have an action at the anchor step.
        return None

    anchor_idx_a = Ta - Delta - 1
    anchor_idx_b = Tb - Delta - 1
    anchor_obs_hash_a, action_a = traj_a[anchor_idx_a]
    anchor_obs_hash_b, action_b = traj_b[anchor_idx_b]

    # If Delta > 0, then the obs at depth Delta is shared by definition
    # (we walked back that far successfully). If Delta == 0, the shared
    # depth is just the terminal; the anchor is the last per-step obs
    # before terminal, which need not match. In that case there is no
    # shared anchor obs and we abstain.
    if Delta == 0 and anchor_obs_hash_a != anchor_obs_hash_b:
        return None

    # If Delta > 0 and the anchor obs hashes don't match, that contradicts
    # the backward walk; but we computed Delta = d-1 where d was the first
    # mismatch, so the obs at index Ta-d != obs at index Tb-d, but the
    # obs at the SAME-DEPTH-d-1 step (anchor index) DO match.
    # Actually no: depth d=1 means index Ta-1 and Tb-1. If those match,
    # Delta becomes 1. The anchor index for Delta=1 is Ta-2 and Tb-2,
    # which we have NOT verified match. The anchor obs hash on trajectory
    # a is at index Ta-Delta-1 = Ta-2; we want it to be the obs from which
    # the action at depth Delta was taken.

    # CORRECTION: re-derive. The per-step entry traj[t] = (obs_t, a_t)
    # records the obs SEEN before action a_t was taken; obs_{t+1} is in
    # entry traj[t+1] (or is the terminal_obs if t+1 == T).
    # Backward depth d=0 is terminal_obs (entry "off the end").
    # Backward depth d=1 is obs_{T-1} = traj[T-1][0].
    # Backward depth d is obs_{T-d} = traj[T-d][0] for d in [1, T].
    # We want the largest Delta such that obs at depth d match for all
    # 0 <= d <= Delta. d=0 matches by construction (terminal_obs match).
    # The loop above checks d in [1, max_walk] and sets Delta to largest
    # matching d. So Delta in [0, max_walk].
    #
    # The postfix-divergence ACTION is the action taken at depth Delta+1
    # (the action that consumed the FIRST DIFFERING obs). Equivalently:
    # the anchor obs is the obs at depth Delta+1, and the action chosen
    # there is what determined whether we ended up at the same depth-Delta
    # state or somewhere else. So:
    #   anchor_idx = T - (Delta + 1) = T - Delta - 1
    #   anchor_obs_hash = traj[anchor_idx][0]   (this is obs_{T-Delta-1})
    #   action_at_anchor = traj[anchor_idx][1]
    # The two anchor obs hashes need NOT match -- they're literally the
    # obs at the deeper-than-shared step. The hypothesis says the anchor
    # is the "last-shared-from-the-end observation" and the actions are
    # at that shared observation.
    #
    # Re-reading hypothesis: "find the largest Delta such that
    # hash(s_{T-delta}) = hash(s'_{T'-delta}) for all 0 <= delta <= Delta.
    # The postfix-divergence point is (s_{T-Delta-1}, a_{T-Delta-1}) for
    # tau and (s'_{T'-Delta-1}, a'_{T'-Delta-1}) for tau'; these are the
    # last actions before the trajectories' backward-walked observation
    # sequences first disagreed."
    #
    # So the anchor is two DIFFERENT observations s_{T-Delta-1} and
    # s'_{T'-Delta-1} (these are the points ONE STEP DEEPER than the
    # shared region). The "anchor" in the hypothesis is the per-trajectory
    # obs at depth Delta+1, NOT a shared obs.
    #
    # But then the next clause: "When the divergence is at the action
    # only -- same observation, different action chosen -- that observation
    # is itself the postfix-divergence anchor."
    #
    # So the standard case has DIFFERENT anchor observations on each side,
    # and W[s, a] is updated separately on each side using its own anchor
    # obs. The Pareto-non-dominated side gets +1 at its (anchor_obs, action)
    # pair. The dominated side does NOT contribute.
    #
    # This makes more sense and matches the hypothesis text. Let me
    # implement it that way.

    # We already have anchor_idx_a = Ta - Delta - 1 and similarly for b.
    # These are valid as long as Delta < Ta and Delta < Tb (checked above).
    return (
        anchor_obs_hash_a,
        action_a,
        anchor_obs_hash_b,
        action_b,
        Delta,
    )


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    is_vector_env = harness.ENV_TYPE[env_id] == "vector"
    is_craftax = env_id == "Craftax-Symbolic-v1"
    action_space = env.action_space
    n_actions = int(action_space.n) if hasattr(action_space, "n") else None

    rng = np.random.default_rng(seed)

    # Logit table per observation hash. Default is uniform (zeros).
    # W[s, a] is the postfix-Pareto-vote count.
    logits: dict[int, np.ndarray] = {}
    W: dict[int, np.ndarray] = {}

    # Hyperparameters (fixed; not tuned).
    NUDGE_ALPHA = 0.5  # logit nudge per pair contributing
    PAIR_BUFFER_CAP = 256  # reservoir cap for completed trajectories
    MAX_TRAJ_LEN = 600  # cap per-trajectory step storage
    MAX_PAIRS_PER_UPDATE = 4000  # cap on pair comparisons per update batch
    BATCH_TRAJS = 8  # rollouts before each update

    # Pair buffer: list of (terminal_obs_hash, [(obs_hash, action), ...], c_T)
    # with reservoir sampling (recency-weighted: keep most recent + reservoir).
    buffer: list[tuple[int, list[tuple[int, int]], np.ndarray]] = []

    # Diagnostics
    n_pairs_total = 0
    n_pairs_matched_terminal = 0
    n_pairs_pareto_comparable = 0
    n_nudges_applied = 0
    deltas_seen: list[int] = []

    # MAD tolerance for vector channels (per-env).
    # We'll estimate per-channel range from observed terminal vectors.
    vec_observed: list[np.ndarray] = []

    def _get_logits(h: int) -> np.ndarray:
        if h not in logits:
            logits[h] = np.zeros(n_actions, dtype=np.float64)
        return logits[h]

    def _sample_action(obs_h: int) -> int:
        if n_actions is None:
            return action_space.sample()
        l = logits.get(obs_h)
        if l is None:
            # Uniform random for unseen states.
            return int(rng.integers(n_actions))
        # Softmax sampling.
        m = l.max()
        e = np.exp(l - m)
        p = e / e.sum()
        return int(rng.choice(n_actions, p=p))

    def _greedy_action(obs_h: int) -> int:
        if n_actions is None:
            return action_space.sample()
        l = logits.get(obs_h)
        if l is None:
            return int(rng.integers(n_actions))
        # Argmax with random tie-break.
        m = l.max()
        ties = np.flatnonzero(l == m)
        if len(ties) == 1:
            return int(ties[0])
        return int(rng.choice(ties))

    def _terminal_vector(scalar_ret: float, vec_ret: np.ndarray | None) -> np.ndarray:
        if vec_ret is not None:
            return np.asarray(vec_ret, dtype=np.float64)
        return np.asarray([float(scalar_ret)], dtype=np.float64)

    def _atol_per_channel() -> np.ndarray:
        if not vec_observed:
            return np.array([1e-9])
        arr = np.stack(vec_observed)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        rng_ch = arr.max(axis=0) - arr.min(axis=0)
        # MAD-based tolerance, fallback to range, fallback to 1e-9.
        atol = np.maximum(1e-9 * np.ones_like(mad), 1e-3 * np.maximum(mad, 1e-9 * rng_ch))
        return atol

    def _do_update(buf):
        nonlocal n_pairs_total, n_pairs_matched_terminal, n_pairs_pareto_comparable
        nonlocal n_nudges_applied
        N = len(buf)
        if N < 2:
            return

        # Bucket by terminal_obs_hash to find pairs cheaply.
        by_term: dict[int, list[int]] = defaultdict(list)
        for i, (term_h, _steps, _c) in enumerate(buf):
            by_term[term_h].append(i)

        atol = _atol_per_channel()
        # Compute total pair count.
        total_pairs = N * (N - 1) // 2
        n_pairs_total += total_pairs

        # Iterate over matched-terminal-hash pairs.
        comparisons_done = 0
        for term_h, idxs in by_term.items():
            if len(idxs) < 2 or comparisons_done >= MAX_PAIRS_PER_UPDATE:
                continue
            # Iterate pairs within this bucket.
            for ii in range(len(idxs)):
                for jj in range(ii + 1, len(idxs)):
                    if comparisons_done >= MAX_PAIRS_PER_UPDATE:
                        break
                    comparisons_done += 1
                    i = idxs[ii]
                    j = idxs[jj]
                    n_pairs_matched_terminal += 1
                    _, steps_a, c_a = buf[i]
                    _, steps_b, c_b = buf[j]
                    rel = _pareto_relation(c_a, c_b, atol)
                    if rel == 0:
                        continue
                    n_pairs_pareto_comparable += 1
                    res = _postfix_anchor(steps_a, steps_b)
                    if res is None:
                        continue
                    anchor_h_a, act_a, anchor_h_b, act_b, Delta = res
                    deltas_seen.append(Delta)
                    # The Pareto-non-dominated side gets the nudge.
                    # rel == +1 means c_a dominates c_b -> nudge a-side.
                    # rel == -1 means c_b dominates c_a -> nudge b-side.
                    if rel == 1:
                        anchor_h, act = anchor_h_a, act_a
                    else:
                        anchor_h, act = anchor_h_b, act_b
                    if anchor_h not in W:
                        W[anchor_h] = np.zeros(n_actions, dtype=np.float64)
                    W[anchor_h][act] += 1.0
                    # Apply logit nudge proportional to margin in W.
                    l = _get_logits(anchor_h)
                    w = W[anchor_h]
                    max_w = w.max()
                    margin = w[act] - max_w  # <= 0; 0 means it's the (or a) max
                    # Hypothesis: logit(a | s) += alpha * sign(W[s, a] - max W[s, *])
                    # weighted by margin. sign(0) = 0 means non-max actions
                    # would not get pushed. Use: nudge = alpha * (w[a] - mean(w))
                    # which is mirror-descent toward softmax(W).
                    mean_w = w.mean()
                    l[act] += NUDGE_ALPHA * (w[act] - mean_w) / max(1.0, max_w)
                    n_nudges_applied += 1
                if comparisons_done >= MAX_PAIRS_PER_UPDATE:
                    break

        # Re-anchor logits to softmax(W) -- a stronger version of the
        # nudge: each visited anchor's logits = log-softmax of W (which is
        # mirror-descent's exact projection). Scaled by an effective
        # learning rate via NUDGE_ALPHA.
        for h, w in W.items():
            target = NUDGE_ALPHA * (w - w.mean())
            # Blend toward target rather than overwrite to avoid jumps.
            cur = _get_logits(h)
            logits[h] = 0.5 * cur + 0.5 * target

    # Rollout loop.
    t_start = time.monotonic()
    deadline = t_start - 1.0  # leave 1s safety margin
    deadline = t_start + max(1.0, time_budget_s - 5.0)
    env_steps = 0
    n_rollouts = 0

    while time.monotonic() < deadline:
        # One rollout episode.
        try:
            obs, _ = env.reset(seed=seed + 1000 + n_rollouts)
        except Exception:
            break
        steps_list: list[tuple[int, int]] = []
        scalar_ret = 0.0
        vec_ret: np.ndarray | None = None
        done = False
        ep_steps = 0
        terminal_obs_hash: int | None = None
        last_obs = obs
        while not done and ep_steps < harness.MAX_EPISODE_STEPS and ep_steps < MAX_TRAJ_LEN:
            obs_h = _obs_hash(obs)
            if n_actions is None:
                action = action_space.sample()
            else:
                action = _sample_action(obs_h)
            steps_list.append((obs_h, int(action) if n_actions is not None else 0))
            try:
                step_out = env.step(action)
            except Exception:
                break
            if len(step_out) == 5:
                next_obs, reward, term, trunc, info = step_out
            else:
                next_obs, reward, term, trunc = step_out[:4]
                info = {}
            scalar_ret += float(reward)
            if is_vector_env and "vector" in info:
                v = np.asarray(info["vector"], dtype=np.float64)
                if vec_ret is None:
                    vec_ret = v.copy()
                else:
                    vec_ret += v
            done = bool(term) or bool(trunc)
            obs = next_obs
            last_obs = next_obs
            ep_steps += 1
            env_steps += 1
            if time.monotonic() >= deadline:
                break

        if not steps_list:
            n_rollouts += 1
            continue

        terminal_obs_hash = _obs_hash(last_obs)
        c_T = _terminal_vector(scalar_ret, vec_ret)
        if is_vector_env and vec_ret is not None:
            vec_observed.append(vec_ret.copy())

        # If n_actions is None (continuous) we skip W updates entirely;
        # primitive is defined for categorical actions only.
        if n_actions is not None:
            buffer.append((terminal_obs_hash, steps_list, c_T))
            # Reservoir cap with recency: keep last PAIR_BUFFER_CAP entries.
            if len(buffer) > PAIR_BUFFER_CAP:
                # Drop random older entries (simple reservoir).
                drop_idx = int(rng.integers(len(buffer) - 1))
                buffer.pop(drop_idx)

        n_rollouts += 1

        if (
            n_actions is not None
            and n_rollouts % BATCH_TRAJS == 0
            and len(buffer) >= 2
            and time.monotonic() < deadline
        ):
            _do_update(buffer)

    # Final update.
    if n_actions is not None and len(buffer) >= 2:
        try:
            _do_update(buffer)
        except Exception:
            pass

    env.close()

    # Diagnostics.
    if deltas_seen:
        deltas_arr = np.asarray(deltas_seen)
        delta_summary = (
            f"n={len(deltas_arr)} mean={deltas_arr.mean():.2f} "
            f"min={int(deltas_arr.min())} max={int(deltas_arr.max())} "
            f"p50={int(np.median(deltas_arr))}"
        )
    else:
        delta_summary = "n=0"
    pair_match_rate = (
        n_pairs_matched_terminal / n_pairs_total if n_pairs_total > 0 else 0.0
    )
    pareto_rate = (
        n_pairs_pareto_comparable / n_pairs_matched_terminal
        if n_pairs_matched_terminal > 0
        else 0.0
    )
    elapsed = time.monotonic() - t_start
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"rollouts={n_rollouts} train_s={elapsed:.1f} budget_s={time_budget_s}",
        flush=True,
    )
    print(
        f"[tpp] n_pairs_total={n_pairs_total} matched_terminal={n_pairs_matched_terminal} "
        f"pareto_comparable={n_pairs_pareto_comparable} nudges_applied={n_nudges_applied} "
        f"pair_match_rate={pair_match_rate:.4f} pareto_rate={pareto_rate:.4f} "
        f"|W|={len(W)} |logits|={len(logits)} delta_dist={delta_summary}",
        flush=True,
    )

    # Build the deployment policy: greedy w.r.t. learned logits;
    # fallback uniform-random for unseen obs.
    def policy_fn(obs):
        if n_actions is None:
            return action_space.sample()
        h = _obs_hash(obs)
        return _greedy_action(h)

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
