"""FED — Frontier-Expanding Dispersion.

Realizes the hypothesis at worklogs/runs/20260605-04-auto/hypothesis.md:

  primitive: per-context, per-action empirical multiset of vector-outcome
  suffix signatures F(u, a), with its componentwise Pareto front P(u, a).
  improvement_operator: logit nudge
      pi(a|u) <- pi(a|u) * exp(eta * I[F(u,a) extends P(u) outward])
  where the indicator is True iff the conditional outcome multiset of (u, a)
  contributes at least one new Pareto-non-dominated point to P(u) without
  removing any existing point above tolerance.

No scalar collapse: vector signatures stay in R^k throughout; the channel
interaction is the partial Pareto order. No critic, no return-to-go, no w^T r.

For sparse envs (single-channel), k=1 and the partial order degenerates to
empirical max-extension; the hypothesis explicitly predicts weakness on
DoorKey-style envs.

Contract:
    uv run train.py --env ENV --seed S --time-budget-s T
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


# --------------------------------------------------------------------------- #
# observation hashing — produces context bucket id u
# --------------------------------------------------------------------------- #


def _obs_hash(obs) -> int:
    """Stable, fast hash of an observation into a bucket id.

    Hash collisions ARE the bucketing — coarse grouping is required for any
    per-bucket multiset to accumulate sample mass. We deliberately downsample
    the raw observation before hashing so that perceptually-similar states
    map to the same bucket.
    """
    arr = np.asarray(obs)
    if arr.dtype == np.object_:
        # fallback: stringify
        b = repr(obs).encode("utf-8")
    elif arr.size == 0:
        b = b"\x00"
    else:
        # Compress: convert to int8 if integer-typed (MiniGrid: small ints), or
        # quantize floats to 8-bit so that near-duplicates collide.
        if np.issubdtype(arr.dtype, np.integer):
            payload = arr.astype(np.int16, copy=False).tobytes()
        else:
            # quantize floats to 16 levels per dim; larger arrays get heavier
            # quantization to keep collision rate sane.
            qstep = 16
            lo = np.minimum(arr.min(), 0.0)
            hi = np.maximum(arr.max(), 1.0)
            rng = max(hi - lo, 1e-6)
            q = np.clip(((arr - lo) / rng * qstep).astype(np.int16), 0, qstep)
            payload = q.tobytes()
        b = payload
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


# --------------------------------------------------------------------------- #
# Pareto utilities — strict componentwise partial order
# --------------------------------------------------------------------------- #


def _dominates(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """True iff a Pareto-dominates b (>= on all channels, > on at least one)."""
    return bool(np.all(a >= b - tol) and np.any(a > b + tol))


def _pareto_front_indices(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Return indices of non-dominated rows in `points` (each row in R^k)."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        pi = points[i]
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if _dominates(points[j], pi, tol):
                keep[i] = False
                break
    return np.where(keep)[0]


def _front_extends_outward(
    front: np.ndarray, candidates: np.ndarray, tol: float = 1e-6
) -> bool:
    """True iff merging `candidates` into `front` yields at least one new
    non-dominated point and removes none above tolerance.

    Concretely: there exists c in candidates such that no f in front dominates
    c, AND no c in candidates dominates any f in front (so no removal). In the
    "no contraction above tolerance" reading from the hypothesis, we forbid
    a candidate from dominating an existing front point — additions only.
    """
    if len(front) == 0:
        return len(candidates) > 0
    # No candidate dominates any existing front point (no contraction).
    for c in candidates:
        for f in front:
            if _dominates(c, f, tol):
                # candidate would push f off the front -> contraction
                return False
    # At least one candidate is non-dominated by current front (extension).
    for c in candidates:
        dominated = False
        for f in front:
            if _dominates(f, c, tol):
                dominated = True
                break
        if not dominated:
            # also require strict novelty: c is not equal to a front point
            equal = any(np.all(np.abs(c - f) <= tol) for f in front)
            if not equal:
                return True
    return False


# --------------------------------------------------------------------------- #
# Per-bucket multiset store
# --------------------------------------------------------------------------- #


class FrontierStore:
    """Tracks F(u) and F(u, a) suffix-signature multisets, plus P(u)."""

    def __init__(self, k: int, max_per_bucket: int = 32):
        self.k = k
        self.max_per_bucket = max_per_bucket
        # F(u): bounded ring buffer of suffix signatures observed at u
        self.F_u: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_per_bucket))
        # F(u, a): same, conditional on the action taken at u
        self.F_ua: dict[tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=max_per_bucket)
        )

    def push_visit(self, u: int, a: int, suffix_sig: np.ndarray) -> None:
        self.F_u[u].append(suffix_sig)
        self.F_ua[(u, a)].append(suffix_sig)

    def pareto_front(self, u: int) -> np.ndarray:
        if u not in self.F_u or len(self.F_u[u]) == 0:
            return np.zeros((0, self.k), dtype=np.float64)
        pts = np.asarray(self.F_u[u], dtype=np.float64)
        idx = _pareto_front_indices(pts)
        return pts[idx]

    def conditional_set(self, u: int, a: int) -> np.ndarray:
        key = (u, a)
        if key not in self.F_ua or len(self.F_ua[key]) == 0:
            return np.zeros((0, self.k), dtype=np.float64)
        return np.asarray(self.F_ua[key], dtype=np.float64)

    def n_samples(self, u: int) -> int:
        return len(self.F_u[u]) if u in self.F_u else 0


# --------------------------------------------------------------------------- #
# Logit-nudge policy table
# --------------------------------------------------------------------------- #


class PolicyTable:
    """pi(a|u) parameterized as a logits table; softmax sample, greedy at eval."""

    def __init__(self, n_actions: int, eta: float = 0.5):
        self.n_actions = n_actions
        self.eta = eta
        self.logits: dict[int, np.ndarray] = {}

    def get(self, u: int) -> np.ndarray:
        if u not in self.logits:
            self.logits[u] = np.zeros(self.n_actions, dtype=np.float64)
        return self.logits[u]

    def probs(self, u: int) -> np.ndarray:
        z = self.get(u)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, u: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_actions, p=self.probs(u)))

    def greedy(self, u: int) -> int:
        return int(np.argmax(self.get(u)))

    def nudge(self, u: int, a: int) -> None:
        # pi(a|u) <- pi(a|u) * exp(eta) -> in logit space: logit_a += eta
        self.get(u)[a] += self.eta


# --------------------------------------------------------------------------- #
# Action-space probing
# --------------------------------------------------------------------------- #


def _n_actions(env) -> int:
    sp = env.action_space
    if hasattr(sp, "n"):
        return int(sp.n)
    raise ValueError("FED requires a discrete action space")


def _vector_dim(env_id: str, env, info_sample: dict | None) -> int:
    if harness.ENV_TYPE[env_id] == "vector":
        # mo_gymnasium envs report a vector reward; harness wrapper places it
        # in info["vector"]. Probe HV_REF for size.
        return int(harness.HV_REF[env_id].shape[0])
    # scalar envs (sparse / craftax): k = 1, partial order = max-extension
    return 1


# --------------------------------------------------------------------------- #
# Training loop — pure FED
# --------------------------------------------------------------------------- #


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed)
    n_actions = _n_actions(env)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    k = _vector_dim(env_id, env, None)

    store = FrontierStore(k=k, max_per_bucket=32)
    policy = PolicyTable(n_actions=n_actions, eta=0.5)

    t0 = time.monotonic()
    train_deadline = t0 + max(1, time_budget_s - 5)

    n_episodes = 0
    n_steps = 0
    n_updates = 0
    n_extensions = 0
    bucket_visits: dict[int, int] = defaultdict(int)

    obs, _info = env.reset(seed=seed)
    while time.monotonic() < train_deadline:
        # ---- collect one episode ----
        traj_u: list[int] = []
        traj_a: list[int] = []
        traj_v: list[np.ndarray] = []  # per-step vector contribution
        steps_in_ep = 0
        ep_done = False
        while not ep_done and steps_in_ep < harness.MAX_EPISODE_STEPS:
            u = _obs_hash(obs)
            bucket_visits[u] += 1
            a = policy.sample(u, rng)
            traj_u.append(u)
            traj_a.append(a)
            try:
                obs, reward, term, trunc, info = env.step(a)
            except Exception:
                # any env step failure: end episode
                break
            if is_vector:
                v = np.asarray(info.get("vector", np.zeros(k)), dtype=np.float64)
                if v.shape != (k,):
                    v = np.zeros(k, dtype=np.float64)
            else:
                v = np.array([float(reward)], dtype=np.float64)
            traj_v.append(v)
            ep_done = bool(term) or bool(trunc)
            steps_in_ep += 1
            n_steps += 1
            if time.monotonic() >= train_deadline:
                break

        n_episodes += 1
        if not traj_v:
            obs, _ = env.reset(seed=seed + n_episodes)
            continue

        # ---- compute suffix signatures sigma_i = sum_{j>=i} v_j ----
        T = len(traj_v)
        suffix = np.zeros((T, k), dtype=np.float64)
        running = np.zeros(k, dtype=np.float64)
        for t in range(T - 1, -1, -1):
            running = running + traj_v[t]
            suffix[t] = running.copy()

        # ---- store per-bucket and per-(bucket, action) suffix signatures ----
        for t in range(T):
            store.push_visit(traj_u[t], traj_a[t], suffix[t])

        # ---- improvement operator: per visited context u, find actions a
        #      whose conditional set F(u, a) extends the front P(u) outward;
        #      nudge logits at u toward those actions. ----
        # Use the set of *unique* contexts visited this episode to avoid O(T*A)
        # work on every step.
        unique_u = set(traj_u)
        for u in unique_u:
            if store.n_samples(u) < 2:
                continue
            front = store.pareto_front(u)
            if len(front) == 0:
                continue
            for a in range(n_actions):
                cond = store.conditional_set(u, a)
                if len(cond) == 0:
                    continue
                if _front_extends_outward(front, cond):
                    policy.nudge(u, a)
                    n_updates += 1
                    n_extensions += 1

            if time.monotonic() >= train_deadline:
                break

        obs, _ = env.reset(seed=seed + n_episodes)

    env.close()

    # collision-rate diagnostic
    total_visits = sum(bucket_visits.values())
    n_buckets = len(bucket_visits)
    collision_rate = (
        1.0 - (n_buckets / total_visits) if total_visits > 0 else 0.0
    )

    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} "
        f"episodes={n_episodes} budget_s={time_budget_s} "
        f"buckets={n_buckets} collision_rate={collision_rate:.3f} "
        f"updates={n_updates} extensions={n_extensions} k={k}",
        flush=True,
    )

    # ---- frozen greedy policy_fn for evaluation ----
    # Snapshot logits so the eval is deterministic and independent of the
    # training data structures.
    snapshot = {u: arr.copy() for u, arr in policy.logits.items()}

    def policy_fn(obs: np.ndarray):
        u = _obs_hash(obs)
        if u in snapshot:
            return int(np.argmax(snapshot[u]))
        # unseen bucket: deterministic fallback action 0 (greedy of zero logits)
        return 0

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
