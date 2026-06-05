"""TCP — Temporal Channel Precedence.

Run: uv run train.py --env ENV --seed 0 --time-budget-s 120

Implements the hypothesis 20260606-08-auto:
    primitive: lag-asymmetry tensor Λ[j,k] over channel pairs, derived
        precedence DAG P (edge k→j iff Λ[j,k] > τ_DAG).
    improvement_operator: at every decision step, restrict per-(cluster,
        action) firing-rate vectors μ[c,a,·] to the prefix-induced
        residual channel set R(τ_{:t}) and compute a signed Pareto-vote
        logit nudge.

Channels are never summed. The agent reads info["vector"] from
env.step() and treats each component as a channel.
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


# -------- helpers ---------------------------------------------------------


def _obs_cluster_key(obs) -> tuple:
    """Coarse observation cluster.

    For DST and Resource-Gathering both have small Discrete-Box
    observations (int tile coords / inventory counts), so the tuple of
    rounded ints is a faithful cluster key — exactly the "transition
    geometry" lens required by the hypothesis.
    """
    arr = np.asarray(obs).reshape(-1)
    return tuple(int(x) for x in arr.astype(np.int64))


def _pareto_compare(x: np.ndarray, y: np.ndarray) -> int:
    """Return +1 if x dominates y, -1 if y dominates x, else 0."""
    if x.size == 0:
        return 0
    ge_x = np.all(x >= y)
    gt_x = np.any(x > y)
    ge_y = np.all(y >= x)
    gt_y = np.any(y > x)
    if ge_x and gt_x and not (ge_y and gt_y):
        return 1
    if ge_y and gt_y and not (ge_x and gt_x):
        return -1
    return 0


# -------- TCP agent -------------------------------------------------------


class TCPAgent:
    def __init__(
        self,
        n_actions: int,
        n_channels: int,
        rng: np.random.Generator,
        tau_dag: float = 0.4,
        alpha: float = 1.5,
        firing_eps: float = 1e-9,
        explore_eps: float = 0.15,
    ):
        self.n_actions = int(n_actions)
        self.n_channels = int(n_channels)
        self.rng = rng
        self.tau_dag = float(tau_dag)
        self.alpha = float(alpha)
        self.firing_eps = float(firing_eps)
        self.explore_eps = float(explore_eps)

        # Λ tensor accumulator. Sum of sign(t_first(j) - t_first(k)) over
        # trajectories, plus count, gives Λ[j,k] = sum / count in [-1,+1].
        self._lam_sum = np.zeros((n_channels, n_channels), dtype=np.float64)
        self._lam_n = 0  # number of trajectories aggregated

        # μ[c,a,m]: empirical mean of channel-m firing on the *next* step
        # after taking action a in cluster c. Stored as (sum, count).
        # cluster c is hashed via a Python dict to keep this online.
        # mu_sum[c][a, m], mu_cnt[c][a]
        self._mu_sum: dict[tuple, np.ndarray] = {}
        self._mu_cnt: dict[tuple, np.ndarray] = {}

        # Number of trajectories observed (for diagnostics).
        self.n_traj = 0
        self.n_pareto_votes_fired = 0
        self.n_decisions = 0
        self.n_residual_full = 0
        self.n_residual_singleton = 0
        self.n_residual_empty = 0

    # ---- statistics updates ----------------------------------------------

    def observe_step(self, c_prev: tuple, a_prev: int, fired_now: np.ndarray) -> None:
        """μ-update from one (cluster, action, next-firing-vector) tuple."""
        if c_prev not in self._mu_sum:
            self._mu_sum[c_prev] = np.zeros((self.n_actions, self.n_channels), dtype=np.float64)
            self._mu_cnt[c_prev] = np.zeros(self.n_actions, dtype=np.float64)
        self._mu_sum[c_prev][a_prev] += fired_now.astype(np.float64)
        self._mu_cnt[c_prev][a_prev] += 1.0

    def observe_trajectory_end(self, t_first: np.ndarray) -> None:
        """Λ-update at end of trajectory.

        t_first[m] = index of first step at which channel m fires;
        np.inf if it never fired in this trajectory.
        Pairwise sign(t_first(j) - t_first(k)): +1 if j fires after k,
        -1 if before, 0 if simultaneous. If exactly one of (j,k) is
        finite the hypothesis treats it as "only one fires" -> 0.
        """
        self.n_traj += 1
        finite = np.isfinite(t_first)
        # default contrib zero
        contrib = np.zeros((self.n_channels, self.n_channels), dtype=np.float64)
        for j in range(self.n_channels):
            for k in range(self.n_channels):
                if j == k:
                    continue
                if finite[j] and finite[k]:
                    diff = t_first[j] - t_first[k]
                    if diff > 0:
                        contrib[j, k] = 1.0
                    elif diff < 0:
                        contrib[j, k] = -1.0
                    else:
                        contrib[j, k] = 0.0
                else:
                    contrib[j, k] = 0.0
        self._lam_sum += contrib
        self._lam_n += 1

    # ---- DAG / residual computation --------------------------------------

    def lambda_matrix(self) -> np.ndarray:
        if self._lam_n == 0:
            return np.zeros((self.n_channels, self.n_channels), dtype=np.float64)
        return self._lam_sum / max(1, self._lam_n)

    def precedence_dag(self) -> np.ndarray:
        """Boolean adjacency: P[k, j] = True iff edge k -> j (Λ[j,k] > τ)."""
        lam = self.lambda_matrix()
        adj = np.zeros((self.n_channels, self.n_channels), dtype=bool)
        for j in range(self.n_channels):
            for k in range(self.n_channels):
                if j == k:
                    continue
                if lam[j, k] > self.tau_dag:
                    adj[k, j] = True  # edge k -> j
        return adj

    def residual_set(self, fired_in_prefix: np.ndarray, P: np.ndarray) -> np.ndarray:
        """R(τ_{:t}) = channels not yet fired AND every predecessor in P
        has fired in prefix."""
        residual = []
        for j in range(self.n_channels):
            if fired_in_prefix[j]:
                continue
            preds = np.where(P[:, j])[0]
            if len(preds) == 0 or all(fired_in_prefix[p] for p in preds):
                residual.append(j)
        return np.array(residual, dtype=np.int64)

    # ---- decision --------------------------------------------------------

    def select_action(
        self,
        cluster: tuple,
        fired_in_prefix: np.ndarray,
        P: np.ndarray,
        base_logits: np.ndarray | None = None,
    ) -> int:
        self.n_decisions += 1
        R = self.residual_set(fired_in_prefix, P)
        if R.size == 0:
            self.n_residual_empty += 1
        elif R.size == self.n_channels:
            self.n_residual_full += 1
        elif R.size == 1:
            self.n_residual_singleton += 1

        if base_logits is None:
            base_logits = np.zeros(self.n_actions, dtype=np.float64)
        logits = base_logits.copy()

        # Always-on epsilon exploration so trajectories diversify
        # enough to populate Λ and μ. (no env-specific tuning)
        if self.rng.random() < self.explore_eps:
            return int(self.rng.integers(self.n_actions))

        if R.size > 0 and cluster in self._mu_sum:
            mu_s = self._mu_sum[cluster]
            mu_c = self._mu_cnt[cluster]
            # Per-action firing-rate vector restricted to R coords.
            mu_R = np.zeros((self.n_actions, R.size), dtype=np.float64)
            valid_action = np.zeros(self.n_actions, dtype=bool)
            for a in range(self.n_actions):
                if mu_c[a] > 0:
                    mu_R[a] = mu_s[a, R] / mu_c[a]
                    valid_action[a] = True
            # Pareto-vote: for each pair, count dominates / dominated.
            valid_idx = np.where(valid_action)[0]
            if valid_idx.size >= 2:
                votes = np.zeros(self.n_actions, dtype=np.float64)
                fired_any = False
                for a in valid_idx:
                    plus = 0
                    minus = 0
                    for b in valid_idx:
                        if a == b:
                            continue
                        cmp = _pareto_compare(mu_R[a], mu_R[b])
                        if cmp > 0:
                            plus += 1
                        elif cmp < 0:
                            minus += 1
                    votes[a] = plus - minus
                    if plus + minus > 0:
                        fired_any = True
                if fired_any and np.any(votes != 0):
                    self.n_pareto_votes_fired += 1
                    logits = logits + self.alpha * votes

        # softmax sample
        z = logits - logits.max()
        e = np.exp(z)
        p = e / e.sum()
        return int(self.rng.choice(self.n_actions, p=p))


# -------- training loop ---------------------------------------------------


def _detect_n_channels(env_id: str, env) -> int:
    """Probe one step to determine vector length (or fall back to 1 for
    scalar envs)."""
    if harness.ENV_TYPE[env_id] == "vector":
        # We've already reset; do one no-op step probe via action 0.
        # But to avoid changing env state we instead use known shapes:
        if env_id == "deep-sea-treasure-concave-v0":
            return 2
        if env_id == "resource-gathering-v0":
            return 3
    # scalar envs: treat scalar reward as a single channel
    return 1


def _vector_from_step(env_id: str, reward: float, info: dict, n_channels: int) -> np.ndarray:
    if harness.ENV_TYPE[env_id] == "vector":
        v = np.asarray(info["vector"], dtype=np.float64)
        if v.shape[0] != n_channels:
            # defensive: pad/truncate
            out = np.zeros(n_channels, dtype=np.float64)
            out[: min(n_channels, v.shape[0])] = v[: min(n_channels, v.shape[0])]
            return out
        return v
    # scalar env -> one channel
    return np.array([float(reward)], dtype=np.float64)


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 4
    n_channels = _detect_n_channels(env_id, env)

    agent = TCPAgent(n_actions=n_actions, n_channels=n_channels, rng=rng)

    t0 = time.monotonic()
    deadline = t0 + max(1, time_budget_s - 5)  # leave grace for evaluate()
    env_steps = 0
    n_episodes = 0

    # outer rollout loop
    while time.monotonic() < deadline:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        cluster = _obs_cluster_key(obs)
        # firing-channel indicator over prefix
        fired_in_prefix = np.zeros(n_channels, dtype=bool)
        # first-firing time per channel
        t_first = np.full(n_channels, np.inf, dtype=np.float64)
        # current snapshot of precedence DAG (refresh per episode)
        P = agent.precedence_dag()

        steps = 0
        done = False
        while not done and steps < harness.MAX_EPISODE_STEPS:
            if time.monotonic() >= deadline:
                break
            a = agent.select_action(cluster, fired_in_prefix, P)
            try:
                obs2, reward, term, trunc, info = env.step(a)
            except Exception:
                break
            v = _vector_from_step(env_id, reward, info, n_channels)
            # channel "fires" if its component is strictly positive
            # (matches hypothesis: 1[v_{t+1}[m] > 0])
            fired_now = (v > agent.firing_eps).astype(np.float64)
            # update μ statistics
            agent.observe_step(cluster, a, fired_now)
            # update prefix firing history + first-firing-time
            for m in range(n_channels):
                if fired_now[m] > 0 and not fired_in_prefix[m]:
                    fired_in_prefix[m] = True
                    t_first[m] = steps
            cluster = _obs_cluster_key(obs2)
            steps += 1
            env_steps += 1
            done = bool(term) or bool(trunc)
        # end of episode -> Λ update
        agent.observe_trajectory_end(t_first)
        n_episodes += 1

    env.close()

    # Build a deterministic-ish policy_fn for evaluation. We use the
    # learned agent statistics. During eval we still maintain the
    # per-episode prefix firing history; we use a closure with mutable
    # state that resets when we detect an "initial cluster" again. To
    # keep the substrate contract (policy_fn(obs) -> action) we re-init
    # the prefix every time we detect a reset signature: a cluster we
    # have already returned an action for previously and the per-call
    # internal "reset detector" sees a step counter exceeding some
    # threshold — simpler: track by id and reset prefix on each eval
    # call that sees the same first-cluster after termination. But
    # harness.evaluate calls reset between episodes; we therefore need
    # to detect resets externally. Instead we reset prefix when we see
    # cluster equality with the very first cluster of the latest run
    # AND the call-counter is > 0. A robust fallback: reset prefix if
    # the cluster hasn't changed for many consecutive identical calls
    # (extremely unlikely in normal play). We use an alternative
    # approach: the policy is queried from harness.evaluate's loop,
    # which calls reset() then step() repeatedly. Each new reset
    # produces the env's initial obs. We detect a reset whenever the
    # current cluster matches the env's initial cluster AND we have
    # stepped at least once since the last reset.

    # Capture initial-cluster signature from a fresh env (no extra
    # rollouts beyond a single reset).
    probe_env = harness.make_env(env_id, seed + 7777)
    init_obs, _ = probe_env.reset(seed=seed + 7777)
    init_cluster = _obs_cluster_key(init_obs)
    probe_env.close()

    state = {
        "fired_in_prefix": np.zeros(n_channels, dtype=bool),
        "steps_since_reset": 0,
        "P": agent.precedence_dag(),
    }

    # Final agent.rng for deterministic eval: use a fresh seeded one so
    # eval is reproducible per seed.
    eval_rng = np.random.default_rng(seed + 12345)
    agent.rng = eval_rng
    # Disable epsilon-exploration at eval (commit to learned policy).
    agent.explore_eps = 0.0

    def policy_fn(obs):
        cluster = _obs_cluster_key(obs)
        # detect a new episode: cluster matches initial-cluster and
        # we've already taken at least one step.
        if cluster == init_cluster and state["steps_since_reset"] > 0:
            state["fired_in_prefix"] = np.zeros(n_channels, dtype=bool)
            state["steps_since_reset"] = 0
        a = agent.select_action(
            cluster,
            state["fired_in_prefix"],
            state["P"],
        )
        state["steps_since_reset"] += 1
        return a

    # Diagnostics (visible in panel.txt for the Reviewer/Curator).
    lam = agent.lambda_matrix()
    P_final = agent.precedence_dag()
    n_dag_edges = int(P_final.sum())
    n_clusters = len(agent._mu_sum)
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={n_episodes} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} n_channels={n_channels} "
        f"n_traj_lam={agent._lam_n} n_dag_edges={n_dag_edges} "
        f"n_clusters={n_clusters} n_decisions={agent.n_decisions} "
        f"n_pareto_votes={agent.n_pareto_votes_fired} "
        f"n_residual_full={agent.n_residual_full} "
        f"n_residual_singleton={agent.n_residual_singleton} "
        f"n_residual_empty={agent.n_residual_empty}",
        flush=True,
    )
    print(f"[train] lambda_matrix=\n{lam}", flush=True)
    print(f"[train] precedence_DAG_adj=\n{P_final.astype(int)}", flush=True)
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
