"""PICAV: Path-Integrated Channel-Asymmetry Voting.

Realizes the candidate from worklogs/runs/20260605-06-auto/hypothesis.md.

Primitive: per-(obs-hash bucket b, action a) running mean of the signed
antisymmetric pair-contribution vector
    delta_{jk,t} = v_t[j] * n_t[k] - v_t[k] * n_t[j]
where n_t[k] = sum_{s>t} v_s[k] is the future-cumulant of channel k.
The trajectory integral A_{jk} = sum_t delta_{jk,t} is antisymmetric in
(j,k) and measures whether channel j tended to fire before channel k.

(Note: the hypothesis writes the per-step pair-contribution using
"Delta n_t[k]" inside the cross-multiplication; Delta(future-cumulant)
literally equals v_t[k], which would make every delta_{jk,t} identically
zero and contradict the prose, the trajectory integral identity, and the
predicted nonzero firing rate. We therefore implement the only
self-consistent reading of the mechanism: pair-contributions cross the
current per-step vector with the *level* of the future-cumulant of the
other channel. This is the standard antisymmetric temporal-ordering
moment described in the hypothesis prose.)

Improvement operator: at each obs-hash bucket b with samples across
multiple actions, compute the upper-orthant Pareto frontier F(b) over
the per-action mean pair-contribution vectors mu(b, a). For each
non-frontier action a, find the frontier action a* that coordinate-wise
dominates a on the most pair-coordinates D(a*, a). Add to the policy
loss the logit-shift
    alpha * D(a*, a) * (log pi(a*|s) - log pi(a|s))
summed across decisions in bucket b. No reward weight, no Bellman, no
critic.

Side-information: vector diagnostics (info["vector"]) and obs-hash.
Vector envs only; on single-channel envs k(k-1)/2=0 and the operator
deliberately no-ops (declared failure mode #1).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def obs_hash(obs: np.ndarray) -> tuple:
    """Lightweight observation-hash bucket.

    For DST and Resource-Gathering, observations are small integer vectors
    so a tuple-of-ints gives a perfect (no-collision) bucket. For larger
    observation spaces we coarsen via int8 truncation.
    """
    arr = np.asarray(obs)
    if np.issubdtype(arr.dtype, np.integer):
        return tuple(int(x) for x in arr.flatten())
    # coarse bucket for non-integer obs
    return tuple(np.round(arr.flatten()).astype(np.int32).tolist())


def obs_to_features(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).flatten()
    return arr


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class Policy(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PICAV statistics store
# ---------------------------------------------------------------------------


class PICAVStats:
    """Per-(bucket, action) running mean of pair-contribution vectors."""

    def __init__(self, n_actions: int, n_pairs: int):
        self.n_actions = n_actions
        self.n_pairs = n_pairs
        # bucket -> ndarray[n_actions, n_pairs] mean
        self._mean: dict[tuple, np.ndarray] = {}
        # bucket -> ndarray[n_actions] count
        self._count: dict[tuple, np.ndarray] = {}

    def update(self, bucket: tuple, action: int, delta: np.ndarray) -> None:
        if self.n_pairs == 0:
            return
        if bucket not in self._mean:
            self._mean[bucket] = np.zeros(
                (self.n_actions, self.n_pairs), dtype=np.float64
            )
            self._count[bucket] = np.zeros(self.n_actions, dtype=np.int64)
        c = self._count[bucket][action]
        m = self._mean[bucket][action]
        c_new = c + 1
        # incremental running mean
        m += (delta - m) / c_new
        self._count[bucket][action] = c_new

    def buckets_with_multi_action_coverage(self, min_samples: int = 2):
        """Yield (bucket, mean[n_actions,n_pairs], count[n_actions]) where
        at least 2 distinct actions have >= min_samples visits."""
        if self.n_pairs == 0:
            return
        for b, count in self._count.items():
            covered = count >= min_samples
            if int(covered.sum()) >= 2:
                yield b, self._mean[b], count


def upper_orthant_frontier(mu: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return boolean mask over actions indicating frontier membership.

    mu: [n_actions, n_pairs]
    mask: [n_actions] of which actions are eligible (have enough samples).
    Action a is on the frontier iff there is no other eligible action a'
    with mu[a'] >= mu[a] coordinate-wise and strictly greater on at least
    one coordinate.
    """
    n_actions = mu.shape[0]
    on_front = mask.copy()
    for a in range(n_actions):
        if not mask[a]:
            on_front[a] = False
            continue
        for ap in range(n_actions):
            if ap == a or not mask[ap]:
                continue
            ge = np.all(mu[ap] >= mu[a])
            gt = np.any(mu[ap] > mu[a])
            if ge and gt:
                on_front[a] = False
                break
    return on_front


def dominance_count(mu_dominator: np.ndarray, mu_dominated: np.ndarray) -> int:
    """Number of pair-coordinates on which mu_dominator >= mu_dominated."""
    return int(np.sum(mu_dominator >= mu_dominated))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    obs0, _ = env.reset(seed=seed)
    feat0 = obs_to_features(obs0)
    in_dim = int(feat0.shape[0])
    n_actions = int(env.action_space.n)

    # Determine k (vector channel count) by stepping once on a fresh env
    if is_vector:
        env_probe = harness.make_env(env_id, seed + 1)
        env_probe.reset(seed=seed + 1)
        _, _, _, _, info0 = env_probe.step(env_probe.action_space.sample())
        k = int(np.asarray(info0["vector"]).shape[0])
        env_probe.close()
    else:
        k = 1

    n_pairs = (k * (k - 1)) // 2
    pair_idx = [(j, kk) for j in range(k) for kk in range(j + 1, k)]

    policy = Policy(in_dim=in_dim, n_actions=n_actions, hidden=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)

    stats = PICAVStats(n_actions=n_actions, n_pairs=n_pairs)

    # Hyperparameters (declared in candidate; not tuned per-env)
    alpha = 0.1  # logit-shift weight
    entropy_coef = 0.01
    update_every_episodes = 8
    min_samples_per_cell = 2
    fire_count = 0
    decision_count = 0
    sign_pos = 0
    sign_neg = 0

    t0 = time.monotonic()
    env_steps = 0
    train_s = 0.0

    # rolling buffer of (feat, action, log_prob, entropy_term, bucket) per
    # decision, flushed at update time
    ep_records: list[tuple] = []

    n_episodes = 0
    while time.monotonic() - t0 < time_budget_s - 2.0:
        # one trajectory
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        feat = obs_to_features(obs)
        traj_feats: list[np.ndarray] = []
        traj_actions: list[int] = []
        traj_buckets: list[tuple] = []
        traj_vectors: list[np.ndarray] = []
        traj_logp_entries: list[tuple[torch.Tensor, torch.Tensor]] = []

        done = False
        steps = 0
        while not done and steps < harness.MAX_EPISODE_STEPS and (
            time.monotonic() - t0 < time_budget_s - 1.0
        ):
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
            logits = policy(x)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            dist = torch.distributions.Categorical(probs=probs)
            action_t = dist.sample()
            action = int(action_t.item())

            bucket = obs_hash(obs)
            traj_feats.append(feat)
            traj_actions.append(action)
            traj_buckets.append(bucket)
            traj_logp_entries.append((log_probs.squeeze(0), dist.entropy().squeeze(0)))

            obs, reward, term, trunc, info = env.step(action)
            if is_vector:
                v = np.asarray(info["vector"], dtype=np.float64)
            else:
                v = np.array([float(reward)], dtype=np.float64)
            traj_vectors.append(v)
            feat = obs_to_features(obs)
            env_steps += 1
            steps += 1
            done = bool(term) or bool(trunc)
            decision_count += 1

        # ---- compute per-step pair-contribution vectors via single
        # backward pass over the trajectory (future cumulants n_t).
        T = len(traj_vectors)
        if T == 0:
            continue
        V = np.stack(traj_vectors, axis=0)  # [T, k]
        # n_t[k] = sum_{s>t} v_s[k]; n_{T-1} = 0; iterate backward.
        n_t = np.zeros((T, k), dtype=np.float64)
        # n_t[t] = V[t+1] + V[t+2] + ... + V[T-1]
        running = np.zeros(k, dtype=np.float64)
        for t in range(T - 1, -1, -1):
            n_t[t] = running
            running = running + V[t]

        # delta_{jk,t} = V[t,j] * n_t[t,k] - V[t,k] * n_t[t,j]
        if n_pairs > 0:
            deltas = np.zeros((T, n_pairs), dtype=np.float64)
            for p, (j, kk) in enumerate(pair_idx):
                deltas[:, p] = V[:, j] * n_t[:, kk] - V[:, kk] * n_t[:, j]
            # diagnostics: any nonzero?
            for t in range(T):
                nz = np.any(np.abs(deltas[t]) > 1e-12)
                if nz:
                    fire_count += 1
                # sign distribution for k=2 audit
                if n_pairs == 1:
                    if deltas[t, 0] > 0:
                        sign_pos += 1
                    elif deltas[t, 0] < 0:
                        sign_neg += 1

            # update PICAV stats
            for t in range(T):
                stats.update(traj_buckets[t], traj_actions[t], deltas[t])

        # store decision records for later loss computation
        for t in range(T):
            ep_records.append(
                (
                    traj_buckets[t],
                    traj_actions[t],
                    traj_logp_entries[t][0],
                    traj_logp_entries[t][1],
                )
            )

        n_episodes += 1

        # ---- improvement operator: every N episodes, build the logit-shift loss
        if n_episodes % update_every_episodes == 0 and n_pairs > 0:
            # cache frontier info per bucket
            bucket_frontier: dict[tuple, tuple] = {}
            for b, mu, count in stats.buckets_with_multi_action_coverage(
                min_samples=min_samples_per_cell
            ):
                mask = count >= min_samples_per_cell
                front = upper_orthant_frontier(mu, mask)
                bucket_frontier[b] = (mu, count, mask, front)

            if bucket_frontier:
                t_train = time.monotonic()
                loss_terms: list[torch.Tensor] = []
                for bucket, action, log_probs, entropy in ep_records:
                    if bucket not in bucket_frontier:
                        continue
                    mu, count, mask, front = bucket_frontier[bucket]
                    if not mask[action] or front[action]:
                        # already on frontier or below sample threshold
                        continue
                    # find frontier action that dominates `action` on most coords
                    best_a = -1
                    best_dom = -1
                    mu_a = mu[action]
                    for ap in range(n_actions):
                        if not front[ap]:
                            continue
                        d = dominance_count(mu[ap], mu_a)
                        if d > best_dom:
                            best_dom = d
                            best_a = ap
                    if best_a < 0 or best_dom <= 0:
                        continue
                    # logit-shift loss: alpha * D * (log pi(a*|s) - log pi(a|s))
                    # we want to *increase* this difference, so loss = -that
                    shift = log_probs[best_a] - log_probs[action]
                    loss_terms.append(-alpha * float(best_dom) * shift)
                    # entropy bonus
                    loss_terms.append(-entropy_coef * entropy)

                if loss_terms:
                    optimizer.zero_grad()
                    loss = torch.stack(loss_terms).sum() / max(1, len(loss_terms))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                train_s += time.monotonic() - t_train

            ep_records.clear()

    env.close()

    fire_rate = fire_count / max(1, decision_count)
    sign_total = sign_pos + sign_neg
    sign_pos_frac = sign_pos / max(1, sign_total)
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} k={k} n_pairs={n_pairs} "
        f"buckets={len(stats._mean) if n_pairs>0 else 0} "
        f"decisions={decision_count} fire_rate={fire_rate:.3f} "
        f"sign_pos_frac={sign_pos_frac:.3f}",
        flush=True,
    )

    # deterministic policy: argmax of logits
    policy.eval()

    def policy_fn(obs: np.ndarray):
        feat = obs_to_features(obs)
        with torch.no_grad():
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
            logits = policy(x)
            return int(torch.argmax(logits, dim=-1).item())

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
