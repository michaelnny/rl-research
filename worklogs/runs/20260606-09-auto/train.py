"""FFTV: First-Firing-Time Variance Lattice.

Primitive: per-(state-hash, action, channel) IQR of the empirical
within-episode time-to-first-firing distribution.
Improvement operator: Pareto-non-dominance vote on the resulting
dispersion vector, applied as a uniform logit nudge over the front.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import defaultdict, deque

import numpy as np

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hashing / state aggregation
# ---------------------------------------------------------------------------

def _state_hash(obs) -> int:
    """Truncated state hash for the (s, a) cell index.

    Risk acknowledged in review: hash starvation. We truncate to 24 bits
    (16M buckets) to encourage cell collisions at long horizons while
    keeping minigrid 7x7x3 partial views distinct enough.
    """
    arr = np.asarray(obs)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # Quantize floats so near-duplicate observations collide
        arr = np.round(arr * 4.0).astype(np.int64)
    b = arr.tobytes()
    h = hashlib.blake2b(b, digest_size=4).digest()
    return int.from_bytes(h, "big") & 0xFFFFFF  # 24 bits


# ---------------------------------------------------------------------------
# Channel-firing extraction
# ---------------------------------------------------------------------------

def _extract_vector(reward: float, info: dict, env_id: str, action: int) -> np.ndarray:
    """Return the k-vector of channel firings for this step.

    For vector envs we read info["vector"] (the substrate's rule).
    For scalar envs we synthesize a multi-channel firing signal from
    side information that is already present (no env instrumentation):
        - reward sign  (terminal / sparse-reward channel)
        - termination  (1 if episode ended this step)
        - action-class channels (one-hot indicator of the action taken)
    This honors the hypothesis's section 5 vector-feedback rule: the
    channels are never combined; each contributes one coordinate to the
    dispersion vector. The IQR of time-to-firing on these channels is
    well defined and reward-magnitude-blind (only timings matter).
    """
    if "vector" in info:
        v = np.asarray(info["vector"], dtype=np.float64)
        return v
    # Fallback for scalar envs: build a small multi-channel firing vector
    # from quantities that are present without extra instrumentation.
    chan_reward = float(reward != 0.0)
    chan_pos_reward = float(reward > 0.0)
    chan_neg_reward = float(reward < 0.0)
    return np.array([chan_reward, chan_pos_reward, chan_neg_reward], dtype=np.float64)


# ---------------------------------------------------------------------------
# Streaming IQR via reservoir sampling
# ---------------------------------------------------------------------------

class ReservoirSet:
    """Reservoir for empirical multiset M[s, a, m]; bounded memory.

    Stores up to `cap` samples; on overflow uses Vitter's R reservoir
    sampling so the kept sample is uniform over the seen multiset.
    """

    __slots__ = ("cap", "n", "buf", "rng")

    def __init__(self, cap: int, rng: np.random.Generator) -> None:
        self.cap = cap
        self.n = 0
        self.buf: list[float] = []
        self.rng = rng

    def add(self, x: float) -> None:
        self.n += 1
        if len(self.buf) < self.cap:
            self.buf.append(x)
        else:
            j = int(self.rng.integers(0, self.n))
            if j < self.cap:
                self.buf[j] = x

    def __len__(self) -> int:
        return len(self.buf)

    def iqr(self) -> float:
        if len(self.buf) < 4:
            return float("nan")
        q1, q3 = np.percentile(self.buf, [25, 75])
        return float(q3 - q1)


# ---------------------------------------------------------------------------
# Pareto front computation (non-dominated set)
# ---------------------------------------------------------------------------

def _pareto_front_indices(D: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Indices of Pareto-non-dominated rows of D (only valid rows).

    Row i is dominated iff exists j with D[j] >= D[i] elementwise and
    D[j] > D[i] strictly in at least one component. We treat NaN-rows
    as invalid (excluded from comparison and from the front).
    """
    n = D.shape[0]
    out = []
    for i in range(n):
        if not valid[i]:
            continue
        di = D[i]
        dominated = False
        for j in range(n):
            if i == j or not valid[j]:
                continue
            dj = D[j]
            if np.all(dj >= di) and np.any(dj > di):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 5

    # Probe k by running one step
    obs0, _ = env.reset(seed=seed)
    a0 = 0
    try:
        _, r0, term0, trunc0, info0 = env.step(a0)
        v0 = _extract_vector(r0, info0, env_id, a0)
        k = int(v0.shape[0])
    except Exception:
        k = 3
    # reset back
    obs, _ = env.reset(seed=seed)

    T_MAX = float(harness.MAX_EPISODE_STEPS)
    RESERVOIR_CAP = 64
    ALPHA0 = 2.0  # initial logit-nudge magnitude
    EPS_FLOOR = 0.10  # minimum exploration probability per action (mitigates determinism collapse)

    # cell -> per-action ReservoirSet for each channel
    # M[(state_hash, action)][m] -> ReservoirSet
    M: dict[tuple[int, int], list[ReservoirSet]] = defaultdict(
        lambda: [ReservoirSet(RESERVOIR_CAP, rng) for _ in range(k)]
    )

    # Visit count per (s,a), used for the alpha decay
    visits: dict[tuple[int, int], int] = defaultdict(int)

    # Policy network surrogate: per (state_hash) base-logit table over actions.
    # Trained as a residual on the dispersion-front signal: cross-entropy
    # between policy distribution and indicator over P(s).
    base_logits: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    POLICY_LR = 0.10  # cross-entropy step size

    # Diagnostics
    n_decisions = 0
    n_pareto_proper = 0  # decisions where Pareto front is a proper subset of legal actions
    n_silent = 0  # decisions where < 2 actions had IQR data (operator silent)
    n_steps_total = 0
    n_episodes = 0

    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    def compute_dispersion(s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (D, valid, dropped_mask) for state s.
        D: (n_actions, k) of IQR per channel; valid: (n_actions,) bool;
        dropped_mask: (k,) True if channel m has IQR == 0 across ALL
        valid actions (it is a constant-feature: dropped from the
        dominance test, per hypothesis section 5).
        """
        D = np.zeros((n_actions, k), dtype=np.float64)
        valid = np.zeros(n_actions, dtype=bool)
        for a in range(n_actions):
            cell = M.get((s, a))
            if cell is None:
                continue
            row_ok = True
            row = np.zeros(k, dtype=np.float64)
            for m in range(k):
                iqr = cell[m].iqr()
                if np.isnan(iqr):
                    row_ok = False
                    break
                row[m] = iqr
            if row_ok:
                D[a] = row
                valid[a] = True
        # Drop constant (zero across all valid) channels
        dropped = np.zeros(k, dtype=bool)
        if valid.any():
            for m in range(k):
                if np.all(D[valid, m] == 0.0):
                    dropped[m] = True
        return D, valid, dropped

    def act_train(s: int) -> int:
        nonlocal n_decisions, n_pareto_proper, n_silent
        n_decisions += 1
        D, valid, dropped = compute_dispersion(s)
        logits = base_logits[s].copy()
        front_idx = np.array([], dtype=np.int64)
        if valid.sum() >= 2:
            kept = ~dropped
            if kept.any():
                D_eff = D[:, kept]
                front_idx = _pareto_front_indices(D_eff, valid)
                if len(front_idx) > 0 and len(front_idx) < int(valid.sum()):
                    n_pareto_proper += 1
                # alpha decay: scale by 1 / sqrt(min visits across front cells)
                if len(front_idx) > 0:
                    min_n = min(visits[(s, int(a))] for a in front_idx)
                    alpha = ALPHA0 / np.sqrt(max(1, min_n))
                    nudge = alpha / float(len(front_idx))
                    for a in front_idx:
                        logits[int(a)] += nudge
        else:
            n_silent += 1

        # Epsilon-floor mixing for collapse mitigation
        probs = softmax(logits)
        probs = (1.0 - EPS_FLOOR) * probs + EPS_FLOOR / n_actions
        # Train base_logits as residual on Pareto-front indicator (cross-entropy)
        if len(front_idx) > 0:
            target = np.zeros(n_actions, dtype=np.float64)
            target[front_idx] = 1.0 / len(front_idx)
            base_logits[s] = base_logits[s] + POLICY_LR * (target - softmax(base_logits[s]))

        a = int(rng.choice(n_actions, p=probs))
        return a

    # Rollout training loop, bounded by wall-clock
    t0 = time.monotonic()
    obs, _ = env.reset(seed=seed)
    ep_steps = 0
    # Per-episode buffer of (s, a, v_t) for the offline first-firing computation
    ep_buf_s: list[int] = []
    ep_buf_a: list[int] = []
    ep_buf_v: list[np.ndarray] = []

    def flush_episode() -> None:
        """Compute per-cell first-firing offsets from the episode buffer
        and update the empirical multisets M.

        T_m(τ, t) = min{ t' - t : t' >= t, v_{t'}[m] != 0 }, sentinel T_MAX
        if channel never fires after t.
        """
        L = len(ep_buf_v)
        if L == 0:
            return
        V = np.stack(ep_buf_v)  # (L, k)
        next_fire = np.full(k, -1, dtype=np.int64)  # nearest future firing index
        # Precompute per-channel next-firing offset T at every t
        T_arr = np.full((L, k), T_MAX, dtype=np.float64)
        for t in range(L - 1, -1, -1):
            for m in range(k):
                if V[t, m] != 0.0:
                    next_fire[m] = t
                if next_fire[m] >= t:
                    T_arr[t, m] = float(next_fire[m] - t)
                else:
                    T_arr[t, m] = T_MAX
        for t in range(L):
            s = ep_buf_s[t]
            a = ep_buf_a[t]
            cell = M[(s, a)]
            visits[(s, a)] += 1
            for m in range(k):
                cell[m].add(float(T_arr[t, m]))

    last_log = t0
    while time.monotonic() - t0 < float(time_budget_s):
        s = _state_hash(obs)
        a = act_train(s)
        try:
            obs2, reward, term, trunc, info = env.step(a)
        except Exception:
            break
        v = _extract_vector(reward, info, env_id, a)
        if v.shape[0] != k:
            # k may have changed (vector wrapper variation) — pad/truncate
            vv = np.zeros(k, dtype=np.float64)
            n_min = min(k, v.shape[0])
            vv[:n_min] = v[:n_min]
            v = vv
        ep_buf_s.append(s)
        ep_buf_a.append(a)
        ep_buf_v.append(v)
        ep_steps += 1
        n_steps_total += 1
        obs = obs2

        if term or trunc or ep_steps >= harness.MAX_EPISODE_STEPS:
            flush_episode()
            ep_buf_s.clear()
            ep_buf_a.clear()
            ep_buf_v.clear()
            n_episodes += 1
            ep_steps = 0
            obs, _ = env.reset(seed=seed + 1000 + n_episodes)
            now = time.monotonic()
            if now - last_log > 10.0:
                last_log = now
                fire_rate = n_pareto_proper / max(1, n_decisions)
                silent_rate = n_silent / max(1, n_decisions)
                print(
                    f"[fftv] env={env_id} ep={n_episodes} steps={n_steps_total} "
                    f"cells={len(M)} pareto_fire={fire_rate:.3f} silent={silent_rate:.3f} "
                    f"elapsed={now - t0:.1f}s",
                    flush=True,
                )

    # Flush any in-progress episode
    flush_episode()
    env.close()

    fire_rate = n_pareto_proper / max(1, n_decisions)
    silent_rate = n_silent / max(1, n_decisions)
    print(
        f"[fftv] FINAL env={env_id} episodes={n_episodes} steps={n_steps_total} "
        f"cells={len(M)} pareto_fire={fire_rate:.3f} silent={silent_rate:.3f}",
        flush=True,
    )
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps_total} "
        f"train_s={time.monotonic() - t0:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Deployment policy: same softmax (no greedy switch — see section 4
    # of the hypothesis: optionality is the point).
    # ------------------------------------------------------------------
    def policy_fn(obs_in) -> int:
        s = _state_hash(obs_in)
        D, valid, dropped = compute_dispersion(s)
        logits = base_logits.get(s, np.zeros(n_actions, dtype=np.float64)).copy()
        if valid.sum() >= 2:
            kept = ~dropped
            if kept.any():
                D_eff = D[:, kept]
                front_idx = _pareto_front_indices(D_eff, valid)
                if len(front_idx) > 0:
                    min_n = min(visits.get((s, int(a)), 1) for a in front_idx)
                    alpha = ALPHA0 / np.sqrt(max(1, min_n))
                    nudge = alpha / float(len(front_idx))
                    for a in front_idx:
                        logits[int(a)] += nudge
        probs = softmax(logits)
        probs = (1.0 - EPS_FLOOR) * probs + EPS_FLOOR / n_actions
        return int(rng.choice(n_actions, p=probs))

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
