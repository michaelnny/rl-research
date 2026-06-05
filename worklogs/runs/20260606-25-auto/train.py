"""SDS: Stochastic-Dominance Sketches over Action-Conditional Continuation Distributions.

Realizes hypothesis 20260606-25-auto faithfully:

- Experience object: trajectories with per-step vector feedback v_t in R^k
  (from info["vector"] on vector envs; constructed as [-step_penalty, reward]
  on scalar envs to match the 2-channel structure of "step cost + payoff").
- Primitive: per-(state-cluster s, action a, channel m) compressed quantile
  sketch Q_{s,a,m} of size B of the empirical distribution of c_T - c_t,
  where c_t is the running per-channel cumulant up to step t.
- Improvement operator: at each decision step, for each ordered pair (a,a')
  add an edge a -> a' iff Q_{s,a,m} strictly first-order stochastically
  dominates Q_{s,a',m} on EVERY channel m (test via max over the B-quantile
  gaps). Logit nudge alpha * (D+(a) - D-(a)).
- Execution: action ~ softmax(logits_theta(o_t) + alpha * nudge(o_t)).
- Vector feedback rule: logical conjunction of per-channel SD edges (NOT
  Pareto-mean, NOT scalarization).
- Gates: a cell pair (s,a),(s,a') is compared only if both have >= N_min
  samples; a cluster s contributes a nudge only if at least 2 channels are
  non-degenerate (sketch range > eps) for at least one cell in s.
- KL surrogate: a periodic gradient step pulls pi_theta toward the nudged
  policy on the on-policy minibatch of (obs, nudge) pairs.

Sketches are MC, written at episode close (no Bellman, no bootstrap, no
distributional TD target, no scalar functional of the distribution).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ----------------------------- hyperparameters --------------------------------

B_SKETCH = 64           # quantile sketch size per (cell, channel)
N_MIN = 8               # min samples per cell to participate in SD comparison
EPS_RANGE = 1e-6        # sketch "non-degenerate" range threshold
ALPHA = 1.0             # logit nudge scale
N_CLUSTERS = 16         # online k-means clusters
EMBED_DIM = 16          # observation embedding dim
LR = 3e-4               # Adam lr
ENT_COEF = 0.02         # entropy regularizer on base policy
KL_COEF = 1.0           # KL surrogate coefficient
GRAD_BATCH = 64         # decision-step minibatch for periodic update
UPDATE_EVERY = 32       # gradient step every N collected decision steps
EMBED_BATCH = 64        # embedding-head SSL batch size
EMBED_UPDATE_EVERY = 32
RECENT_BUFFER = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- quantile sketch --------------------------------


class QuantileSketch:
    """Sorted reservoir of size B; quantile readout = ordered samples.

    Insertions O(log B) via bisect into a fixed-size array; once full,
    a uniform-random replacement keeps the reservoir an unbiased sample of
    the empirical distribution. We then use the sorted samples themselves
    as B uniform-quantile estimates.
    """

    __slots__ = ("buf", "n", "rng")

    def __init__(self, rng: np.random.Generator):
        self.buf = np.empty(B_SKETCH, dtype=np.float64)
        self.n = 0
        self.rng = rng

    def insert(self, x: float) -> None:
        if self.n < B_SKETCH:
            self.buf[self.n] = x
            self.n += 1
            if self.n == B_SKETCH:
                self.buf.sort()
        else:
            # reservoir replacement
            j = int(self.rng.integers(0, self.n + 1))
            if j < B_SKETCH:
                # maintain sorted order
                self.buf[j] = x
                self.buf.sort()
            self.n += 1

    def quantiles(self) -> np.ndarray | None:
        """Return B sorted quantile estimates, or None if fewer than N_MIN samples."""
        if self.n < N_MIN:
            return None
        if self.n < B_SKETCH:
            # linearly interpolate the partial reservoir to B quantiles
            sorted_part = np.sort(self.buf[: self.n])
            xs = np.linspace(0.0, 1.0, self.n)
            grid = np.linspace(0.0, 1.0, B_SKETCH)
            return np.interp(grid, xs, sorted_part)
        return self.buf  # already sorted


def sd_strictly_dominates(qa: np.ndarray, qb: np.ndarray) -> bool:
    """Strict first-order stochastic dominance: qa >=_SD qb (qa is "stochastically larger").

    Equivalent on equal-probability quantile grids to: qa[i] >= qb[i] for all i,
    with strict inequality somewhere. Channels are pre-oriented so "more is
    better" (cost channels are pre-negated at insert time).
    """
    if qa is None or qb is None:
        return False
    diff = qa - qb
    if diff.min() >= 0.0 and diff.max() > 0.0:
        return True
    return False


# ----------------------------- online k-means ---------------------------------


class OnlineKMeans:
    """Mini-batch online k-means on a fixed-dim embedding."""

    def __init__(self, k: int, dim: int, rng: np.random.Generator):
        self.k = k
        self.dim = dim
        self.rng = rng
        self.centers = rng.standard_normal((k, dim)).astype(np.float32) * 0.1
        self.counts = np.zeros(k, dtype=np.int64)
        self.initialized = False
        self.init_buf: list[np.ndarray] = []

    def assign(self, x: np.ndarray) -> int:
        # x: (dim,)
        d = self.centers - x[None, :]
        return int(np.argmin(np.einsum("kd,kd->k", d, d)))

    def update(self, x: np.ndarray) -> int:
        if not self.initialized:
            self.init_buf.append(x.copy())
            if len(self.init_buf) >= self.k:
                pts = np.stack(self.init_buf, axis=0)
                idx = self.rng.choice(len(pts), size=self.k, replace=False)
                self.centers = pts[idx].astype(np.float32)
                self.initialized = True
            return int(len(self.init_buf) - 1) % self.k
        c = self.assign(x)
        self.counts[c] += 1
        lr = 1.0 / float(self.counts[c])
        self.centers[c] = (1.0 - lr) * self.centers[c] + lr * x.astype(np.float32)
        return c


# ----------------------------- networks ---------------------------------------


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    a = np.asarray(obs)
    return a.astype(np.float32).reshape(-1)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class EmbedNet(nn.Module):
    """Self-supervised observation embedding head trained on next-obs prediction."""

    def __init__(self, obs_dim: int, n_actions: int, embed_dim: int = EMBED_DIM, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, embed_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + n_actions, hidden), nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )
        self.n_actions = n_actions

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict(self, emb: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(a, num_classes=self.n_actions).float()
        return self.predictor(torch.cat([emb, a_oh], dim=-1))


# ----------------------------- vector channel orientation ---------------------


def channel_signs_for(env_id: str, k: int) -> np.ndarray:
    """Pre-orient channels so larger values mean "more favorable".

    For DST: channels = (treasure, time-penalty); time penalty is negative-good
    (less negative = better) -> sign is +1 already. Treasure is +1.
    For RG: channels = (time-penalty, gold, gem); time-penalty is negative-good.
    Both DST and RG vectors as emitted by mo_gymnasium are already such that
    "larger is better" on every channel (penalties are negative numbers, so
    larger=less-negative=better). So a uniform +1 sign works.

    For scalar-env synthesized 2-channel vector (step-cost, reward) we set
    both signs to +1 (we negate step-cost into "step-progress" upstream).
    """
    return np.ones(k, dtype=np.float64)


# ----------------------------- main train -------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1_234_567)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 4

    obs0, _ = env.reset(seed=seed)
    obs_flat = _flatten_obs(obs0)
    obs_dim = obs_flat.shape[0]

    # determine k (# vector channels)
    if is_vector:
        # take a probe step to discover vector dim
        a_probe = 0
        _, _, term, trunc, info = env.step(a_probe)
        k_channels = int(np.asarray(info["vector"]).shape[0])
        env.close()
        env = harness.make_env(env_id, seed)
        env.reset(seed=seed)
    else:
        # scalar env -> synthesize a 2-channel vector
        # channel 0: step-progress (= 0 normally, but we register a constant -1
        # alive-cost so each step is "more negative is worse" -> pre-negated to +1 each step
        # making the channel constant; instead use channel 0 = -1 (alive cost)
        # and channel 1 = scalar reward. We negate channel 0 at insert time so
        # "larger = better" holds.
        k_channels = 2

    signs = channel_signs_for(env_id, k_channels)

    # nets
    policy = PolicyNet(obs_dim, n_actions).to(DEVICE)
    embed_net = EmbedNet(obs_dim, n_actions).to(DEVICE)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR)
    opt_emb = torch.optim.Adam(embed_net.parameters(), lr=LR)

    kmeans = OnlineKMeans(N_CLUSTERS, EMBED_DIM, rng)

    # sketches: dict[(s, a, m)] -> QuantileSketch
    sketches: dict[tuple[int, int, int], QuantileSketch] = {}

    def sk(s: int, a: int, m: int) -> QuantileSketch:
        key = (s, a, m)
        q = sketches.get(key)
        if q is None:
            q = QuantileSketch(rng)
            sketches[key] = q
        return q

    # per-cluster cell-presence index for fast nudge computation
    # cluster_actions[s] = set of actions with any sketch
    cluster_actions: dict[int, set[int]] = defaultdict(set)

    # rollout state
    obs, _ = env.reset(seed=seed)
    ep_obs: list[np.ndarray] = []
    ep_actions: list[int] = []
    ep_clusters: list[int] = []
    ep_vec: list[np.ndarray] = []  # per-step vector cumulant contribution

    # buffers for KL surrogate
    decision_obs: deque[np.ndarray] = deque(maxlen=RECENT_BUFFER)
    decision_nudges: deque[np.ndarray] = deque(maxlen=RECENT_BUFFER)
    decision_logits: deque[np.ndarray] = deque(maxlen=RECENT_BUFFER)

    # transitions for SSL embedding loss
    ssl_obs: deque[np.ndarray] = deque(maxlen=RECENT_BUFFER)
    ssl_act: deque[int] = deque(maxlen=RECENT_BUFFER)
    ssl_next: deque[np.ndarray] = deque(maxlen=RECENT_BUFFER)

    decisions_since_update = 0
    transitions_since_embed_update = 0
    n_episodes = 0
    n_steps = 0
    n_nudges_nonzero = 0

    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave 5s headroom for evaluation

    def compute_nudge(s: int) -> np.ndarray:
        """Compute D+(a) - D-(a) for each action, given cluster s.

        Gate: at least 2 channels must be non-degenerate (sketch range > EPS_RANGE)
        across the cluster's populated cells. Otherwise return zero vector.
        """
        actions_here = cluster_actions.get(s)
        if not actions_here or len(actions_here) < 2:
            return np.zeros(n_actions, dtype=np.float32)

        # Prefetch quantile arrays per (a, m); only retain those with >= N_MIN samples.
        per_action_q: dict[int, list[np.ndarray | None]] = {}
        non_degenerate_channels = 0
        channel_has_any = [False] * k_channels
        channel_nondeg = [False] * k_channels
        for a in actions_here:
            qs: list[np.ndarray | None] = []
            for m in range(k_channels):
                q = sketches.get((s, a, m))
                arr = q.quantiles() if q is not None else None
                qs.append(arr)
                if arr is not None:
                    channel_has_any[m] = True
                    if (arr.max() - arr.min()) > EPS_RANGE:
                        channel_nondeg[m] = True
            per_action_q[a] = qs
        non_degenerate_channels = sum(channel_nondeg)
        if non_degenerate_channels < 2:
            return np.zeros(n_actions, dtype=np.float32)

        # Tournament: edge a -> a' iff for ALL m: qa[m] strictly SD-dominates qb[m].
        # Both cells must have >= N_MIN samples on every channel that's non-degenerate
        # globally; require all k channels populated to honor "every channel".
        nudge = np.zeros(n_actions, dtype=np.float32)
        action_list = list(actions_here)
        for a in action_list:
            qa = per_action_q[a]
            if any(arr is None for arr in qa):
                continue
            for b in action_list:
                if a == b:
                    continue
                qb = per_action_q[b]
                if any(arr is None for arr in qb):
                    continue
                # strict SD on every channel?
                all_dom = True
                any_strict = False
                for m in range(k_channels):
                    diff = qa[m] - qb[m]
                    if diff.min() < 0.0:
                        all_dom = False
                        break
                    if diff.max() > 0.0:
                        any_strict = True
                if all_dom and any_strict:
                    nudge[a] += 1.0  # out-degree (D+)
                    nudge[b] -= 1.0  # in-degree (D-)
        return nudge

    def policy_logits_np(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0).to(DEVICE)
            return policy(t).squeeze(0).cpu().numpy().astype(np.float32)

    def embed_np(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0).to(DEVICE)
            return embed_net.embed(t).squeeze(0).cpu().numpy().astype(np.float32)

    def kl_surrogate_step() -> None:
        if len(decision_obs) < GRAD_BATCH:
            return
        idx = rng.choice(len(decision_obs), size=GRAD_BATCH, replace=False)
        obs_batch = np.stack([decision_obs[i] for i in idx], axis=0)
        nud_batch = np.stack([decision_nudges[i] for i in idx], axis=0)
        old_logits_batch = np.stack([decision_logits[i] for i in idx], axis=0)

        ob_t = torch.from_numpy(obs_batch.astype(np.float32)).to(DEVICE)
        nud_t = torch.from_numpy(nud_batch.astype(np.float32)).to(DEVICE)
        old_l_t = torch.from_numpy(old_logits_batch.astype(np.float32)).to(DEVICE)

        # target distribution: softmax(old_logits + alpha * nudge)
        target_logits = old_l_t + ALPHA * nud_t
        target_log_p = F.log_softmax(target_logits, dim=-1)
        target_p = target_log_p.exp().detach()

        cur_logits = policy(ob_t)
        cur_log_p = F.log_softmax(cur_logits, dim=-1)
        # KL(target || cur)
        kl = (target_p * (target_log_p.detach() - cur_log_p)).sum(dim=-1).mean()
        ent = -(cur_log_p.exp() * cur_log_p).sum(dim=-1).mean()
        loss = KL_COEF * kl - ENT_COEF * ent

        opt_pi.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_pi.step()

    def embed_update_step() -> None:
        if len(ssl_obs) < EMBED_BATCH:
            return
        idx = rng.choice(len(ssl_obs), size=EMBED_BATCH, replace=False)
        ob = np.stack([ssl_obs[i] for i in idx], axis=0).astype(np.float32)
        ac = np.array([ssl_act[i] for i in idx], dtype=np.int64)
        nx = np.stack([ssl_next[i] for i in idx], axis=0).astype(np.float32)

        ob_t = torch.from_numpy(ob).to(DEVICE)
        ac_t = torch.from_numpy(ac).to(DEVICE)
        nx_t = torch.from_numpy(nx).to(DEVICE)

        emb = embed_net.embed(ob_t)
        pred = embed_net.predict(emb, ac_t)
        loss = F.mse_loss(pred, nx_t)
        opt_emb.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(embed_net.parameters(), 1.0)
        opt_emb.step()

    while time.monotonic() < deadline:
        # ---- decision time ----
        o_flat = _flatten_obs(obs)
        emb = embed_np(o_flat)
        s = kmeans.update(emb)
        logits = policy_logits_np(o_flat)
        nudge = compute_nudge(s)
        if np.any(nudge != 0):
            n_nudges_nonzero += 1
        eff_logits = logits + ALPHA * nudge
        # softmax sample
        eff_logits = eff_logits - eff_logits.max()
        probs = np.exp(eff_logits)
        probs = probs / probs.sum()
        action = int(rng.choice(n_actions, p=probs))

        # store for KL surrogate
        decision_obs.append(o_flat.copy())
        decision_nudges.append(nudge.astype(np.float32).copy())
        decision_logits.append(logits.astype(np.float32).copy())

        # step env
        next_obs, reward, term, trunc, info = env.step(action)
        next_flat = _flatten_obs(next_obs)
        if is_vector:
            v = np.asarray(info.get("vector"), dtype=np.float64)
        else:
            # synthesized 2-channel vector: alive-cost (negate to "step-progress" so larger=better)
            # channel 0: -(-1.0) = +1.0 if step occurred (always 1.0 here);
            # to keep "more is better" alignment with episode-length, instead encode
            # alive cost as -1 so longer episodes have more negative cumulant on
            # channel 0 (worse). Channel 1: scalar reward.
            v = np.array([-1.0, float(reward)], dtype=np.float64)
        # apply per-channel orientation sign so SD on every channel means "stochastically larger == favorable"
        v = v * signs

        # SSL transition
        ssl_obs.append(o_flat.copy())
        ssl_act.append(action)
        ssl_next.append(next_flat.copy())

        ep_obs.append(o_flat.copy())
        ep_actions.append(action)
        ep_clusters.append(s)
        ep_vec.append(v.copy())

        n_steps += 1
        decisions_since_update += 1
        transitions_since_embed_update += 1

        if decisions_since_update >= UPDATE_EVERY:
            kl_surrogate_step()
            decisions_since_update = 0
        if transitions_since_embed_update >= EMBED_UPDATE_EVERY:
            embed_update_step()
            transitions_since_embed_update = 0

        done = bool(term) or bool(trunc)
        if done:
            # episode close: write sketches
            # c_t = cumulative vector reward from step 0 to t-1; c_T = sum of all v
            vec_arr = np.stack(ep_vec, axis=0)  # (T, k)
            c_T = vec_arr.sum(axis=0)  # (k,)
            # cumulative up through step t (inclusive of v_t):
            c_cum_incl = np.cumsum(vec_arr, axis=0)
            # c_t (cumulative BEFORE step t) = c_cum_incl - vec_arr (i.e. shifted)
            c_before = c_cum_incl - vec_arr
            # continuation delta from step t to T: c_T - c_before[t]
            delta = c_T[None, :] - c_before  # (T, k)
            for t_idx in range(len(ep_actions)):
                s_t = ep_clusters[t_idx]
                a_t = ep_actions[t_idx]
                cluster_actions[s_t].add(a_t)
                d = delta[t_idx]
                for m in range(k_channels):
                    sk(s_t, a_t, m).insert(float(d[m]))

            ep_obs.clear()
            ep_actions.clear()
            ep_clusters.clear()
            ep_vec.clear()
            n_episodes += 1

            obs, _ = env.reset()
        else:
            obs = next_obs

        if n_steps >= 500_000:
            break

    # Drain any partial episode (do not write sketches for un-finished trajectories;
    # the hypothesis specifies write on episode end).
    env.close()

    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} "
        f"n_episodes={n_episodes} n_clusters_pop={int((kmeans.counts > 0).sum())} "
        f"n_sketch_cells={len(sketches)} n_decisions_with_nudge={n_nudges_nonzero} "
        f"train_s={time.monotonic() - t_start:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Final policy: deterministic argmax over (logits + alpha * nudge).
    policy.eval()
    embed_net.eval()

    def policy_fn(obs_in: np.ndarray) -> int:
        o_flat = _flatten_obs(obs_in)
        emb = embed_np(o_flat)
        s = kmeans.assign(emb) if kmeans.initialized else 0
        logits = policy_logits_np(o_flat)
        nudge = compute_nudge(s)
        eff = logits + ALPHA * nudge
        return int(np.argmax(eff))

    return policy_fn


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
