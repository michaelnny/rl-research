"""TRAC: Transition-Refractive Action Channels.

Primitive: per-(state-cluster c, action a, channel m) Jensen-Shannon
divergence between two empirical successor-cluster distributions,
partitioned by whether channel m fired within window [t+1, t+W] after
taking action a at state-cluster c.

Improvement operator: at decision time, Pareto-non-dominance count over
the per-action k-vector R[c,a,:] (no scalar collapse, no reward weighting).

Substrate contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import harness


# -------------------------- TRAC config -------------------------- #

CLUSTER_COUNT = 64        # number of state clusters (LSH buckets)
PROJ_DIM = 16             # random-projection dim (sign hash -> bits)
WINDOW = 4                # look-ahead window W for channel-firing partition
ALPHA = 1.0               # logit nudge magnitude
TEMP = 1.0                # softmax temperature
MIN_MASS = 1              # minimum count per side before a JSD term contributes
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# -------------------------- helpers -------------------------- #

def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs).ravel().astype(np.float32)
    return arr


class LSHCluster:
    """Random-projection sign hash -> integer cluster index in [0, C)."""

    def __init__(self, n_clusters: int, proj_dim: int, seed: int):
        self.n_clusters = n_clusters
        self.proj_dim = proj_dim
        self.rng = np.random.default_rng(seed + 7)
        self.W = None  # lazy: shape (obs_dim, proj_dim)
        self.bit_weights = (1 << np.arange(proj_dim)).astype(np.int64)

    def _ensure(self, obs_dim: int):
        if self.W is None or self.W.shape[0] != obs_dim:
            # Gaussian random projection.
            self.W = self.rng.standard_normal((obs_dim, self.proj_dim)).astype(np.float32)

    def index(self, obs) -> int:
        x = _flatten_obs(obs)
        self._ensure(x.shape[0])
        # Normalize (avoid scale-blow-up).
        nx = np.linalg.norm(x) + 1e-8
        z = (x / nx) @ self.W
        bits = (z > 0).astype(np.int64)
        h = int((bits * self.bit_weights).sum())
        return h % self.n_clusters


def _channels_from_step(env_id: str, reward: float, info: dict) -> np.ndarray:
    """Return a length-k float32 vector representing this step's channel
    activations. A channel "fires" at step t iff its value is nonzero.
    """
    if "vector" in info and info["vector"] is not None:
        return np.asarray(info["vector"], dtype=np.float64)
    # Scalar-reward env (MiniGrid / Craftax): synthesize channels.
    # ch0: positive-reward event; ch1: negative-reward event; ch2: any step.
    return np.array(
        [1.0 if reward > 0 else 0.0,
         1.0 if reward < 0 else 0.0,
         1.0],
        dtype=np.float64,
    )


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) between two count vectors."""
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum < MIN_MASS or q_sum < MIN_MASS:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    # KL(p || m): only sum where p > 0
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + EPS) - np.log(b[mask] + EPS))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _pareto_dominates(u: np.ndarray, v: np.ndarray) -> bool:
    """Strict Pareto dominance: u >= v elementwise and u > v on at least one axis."""
    return bool(np.all(u >= v) and np.any(u > v))


# -------------------------- TRAC core -------------------------- #

class TRAC:
    def __init__(self, n_clusters: int, n_actions: int, n_channels: int, seed: int):
        self.C = n_clusters
        self.A = n_actions
        self.K = n_channels
        # H_fire[c,a,m,c'] and H_nofire[c,a,m,c'] are sparse — use dicts keyed by (c,a,m).
        # Each entry is a length-C int64 array; lazy-allocated.
        self.H_fire: dict[tuple[int, int, int], np.ndarray] = {}
        self.H_nofire: dict[tuple[int, int, int], np.ndarray] = {}
        # Cached R[c,a,m] table: dict (c,a) -> np.ndarray length K.
        # We invalidate by recomputing lazily on policy queries (cheap: K is small).
        self.dirty: dict[tuple[int, int], bool] = {}
        # Logits per cluster (float32). Initialized to zero -> uniform sampling.
        self.logits = np.zeros((n_clusters, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed + 31)

    def _hist(self, table, key) -> np.ndarray:
        h = table.get(key)
        if h is None:
            h = np.zeros(self.C, dtype=np.int64)
            table[key] = h
        return h

    def record_window(self, c: int, a: int, c_next: int, fired: np.ndarray):
        """fired: length-K bool array — did channel m fire anywhere in
        [t+1, t+W] after the (c, a) event?
        """
        for m in range(self.K):
            key = (c, a, m)
            tbl = self.H_fire if fired[m] else self.H_nofire
            self._hist(tbl, key)[c_next] += 1
        self.dirty[(c, a)] = True

    def row_R(self, c: int, a: int) -> np.ndarray:
        """Per-channel JSD vector R[c,a,:] in R^K_{>=0}."""
        out = np.zeros(self.K, dtype=np.float64)
        for m in range(self.K):
            hf = self.H_fire.get((c, a, m))
            hn = self.H_nofire.get((c, a, m))
            if hf is None or hn is None:
                continue
            out[m] = _jsd(hf, hn)
        return out

    def pareto_nudge(self, c: int) -> np.ndarray:
        """Return per-action additive logit nudge alpha * (n_a - m_a)."""
        rows = np.stack([self.row_R(c, a) for a in range(self.A)], axis=0)
        # If all rows are identically zero, no signal — return zeros.
        if not np.any(rows > 0):
            return np.zeros(self.A, dtype=np.float32)
        n = np.zeros(self.A, dtype=np.int64)  # actions that a dominates
        m = np.zeros(self.A, dtype=np.int64)  # actions that dominate a
        for a in range(self.A):
            for ap in range(self.A):
                if a == ap:
                    continue
                if _pareto_dominates(rows[a], rows[ap]):
                    n[a] += 1
                if _pareto_dominates(rows[ap], rows[a]):
                    m[a] += 1
        return ALPHA * (n - m).astype(np.float32)

    def policy_logits(self, c: int) -> np.ndarray:
        """Effective logits at cluster c: stored logits + Pareto nudge."""
        return self.logits[c] + self.pareto_nudge(c)

    def sample_action(self, c: int) -> int:
        logits = self.policy_logits(c) / TEMP
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return int(self.rng.choice(self.A, p=p))

    def diagnostics(self) -> dict:
        """Per-cell fill-rate diagnostics for the falsifier check."""
        n_cells_total = self.C * self.A * self.K
        # cells with both sides having >= 8 distinct entries summed
        ok = 0
        for key, hf in self.H_fire.items():
            hn = self.H_nofire.get(key)
            if hn is None:
                continue
            if hf.sum() >= 8 and hn.sum() >= 8:
                ok += 1
        return {
            "n_cells_with_minimum_mass": ok,
            "n_cells_total": n_cells_total,
            "fill_rate": ok / max(n_cells_total, 1),
            "n_fire_keys": len(self.H_fire),
            "n_nofire_keys": len(self.H_nofire),
        }


# -------------------------- Trajectory loop -------------------------- #

def _collect_and_update(trac: TRAC, env, env_id: str, cluster: LSHCluster,
                        n_actions: int, max_steps: int) -> tuple[int, int]:
    """Run one trajectory, register (c, a, c_next, fired) entries with
    correct windowing.
    Returns (n_steps, episode_terminated_flag).
    """
    obs, _ = env.reset()
    c_t = cluster.index(obs)
    # Pending events: list of dicts each with c, a, t_event, fired_so_far (bool array).
    # On step t we look at channels fired in (t_event, t_event + W].
    pending: list[dict] = []
    steps = 0

    while steps < max_steps:
        a_t = trac.sample_action(c_t)
        try:
            obs_next, reward, term, trunc, info = env.step(a_t)
        except Exception:
            return steps, 1
        # Channel fired at this future step:
        ch = _channels_from_step(env_id, reward, info)
        fired_now = (ch != 0.0)

        c_next = cluster.index(obs_next)

        # Update all pending events: include this step's firings if within window.
        new_pending = []
        for ev in pending:
            ev["fired_in_window"] = ev["fired_in_window"] | fired_now
            ev["age"] += 1
            if ev["age"] >= WINDOW:
                # Finalize: c_next at finalization time (W steps after event).
                trac.record_window(ev["c"], ev["a"], c_next, ev["fired_in_window"])
            else:
                new_pending.append(ev)
        pending = new_pending

        # Register new pending event for (c_t, a_t) with c_next-stub (we'll
        # overwrite c_next on finalization; the cluster recorded is the one
        # observed W steps later).
        pending.append({
            "c": c_t,
            "a": a_t,
            "age": 0,
            "fired_in_window": np.zeros(trac.K, dtype=bool),
        })

        c_t = c_next
        steps += 1

        if term or trunc:
            # Finalize all remaining pending events using the terminal cluster.
            for ev in pending:
                # Only finalize if at least one window step elapsed (else
                # the partition is undefined — but it's fine: fired_in_window
                # is the event's fired-mask up to terminal).
                trac.record_window(ev["c"], ev["a"], c_t, ev["fired_in_window"])
            pending = []
            return steps, 1

    # Episode hit max_steps; finalize pending events.
    for ev in pending:
        trac.record_window(ev["c"], ev["a"], c_t, ev["fired_in_window"])
    return steps, 0


# -------------------------- train -------------------------- #

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    np.random.seed(seed)
    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else None
    if n_actions is None:
        # Fallback: random policy floor (TRAC requires discrete actions).
        rng = np.random.default_rng(seed + 99_999)
        action_space = env.action_space
        env.close()
        def policy_fn(_obs):
            return action_space.sample()
        print(f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 budget_s={time_budget_s}",
              flush=True)
        return policy_fn

    # Probe to determine number of channels.
    obs, _ = env.reset(seed=seed)
    obs2, r0, term0, trunc0, info0 = env.step(0)
    sample_ch = _channels_from_step(env_id, r0, info0)
    n_channels = int(sample_ch.shape[0])
    env.close()

    cluster = LSHCluster(CLUSTER_COUNT, PROJ_DIM, seed)
    trac = TRAC(CLUSTER_COUNT, n_actions, n_channels, seed)

    env = harness.make_env(env_id, seed)
    env.reset(seed=seed)

    t0 = time.monotonic()
    total_steps = 0
    n_episodes = 0
    safety_max = harness.MAX_EPISODE_STEPS
    last_log = t0

    while time.monotonic() - t0 < time_budget_s:
        steps, _term = _collect_and_update(trac, env, env_id, cluster, n_actions, safety_max)
        total_steps += steps
        n_episodes += 1
        if time.monotonic() - last_log > 20.0:
            d = trac.diagnostics()
            print(
                f"[train] env={env_id} seed={seed} eps={n_episodes} steps={total_steps} "
                f"fill={d['fill_rate']:.3f} cells_ok={d['n_cells_with_minimum_mass']} "
                f"t={time.monotonic() - t0:.1f}s",
                flush=True,
            )
            last_log = time.monotonic()

    env.close()

    diag = trac.diagnostics()
    print(
        f"[train] env={env_id} seed={seed} env_steps={total_steps} eps={n_episodes} "
        f"train_s={time.monotonic() - t0:.1f} budget_s={time_budget_s} "
        f"fill_rate={diag['fill_rate']:.4f} "
        f"cells_min_mass={diag['n_cells_with_minimum_mass']}/{diag['n_cells_total']}",
        flush=True,
    )

    # Frozen evaluation policy — same TRAC sampler, with seeded RNG so eval is
    # reproducible. The policy is sample-based by hypothesis (no greedy argmax).
    eval_rng = np.random.default_rng(seed + 4242)

    def policy_fn(obs):
        c = cluster.index(obs)
        logits = trac.policy_logits(c) / TEMP
        logits = logits - logits.max()
        p = np.exp(logits)
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
