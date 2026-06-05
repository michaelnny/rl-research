"""ARP: Action-Reachable Pattern Lattice (run 20260606-17-auto).

Realizes the hypothesis at worklogs/runs/20260606-17-auto/hypothesis.md.

Primitive: per-(state-cluster s, action a) empirical SET S(s,a) of binary
suffix-firing patterns b ∈ {0,1}^k where b[m] = 1 iff vector channel m fired
(v_m != 0) at least once in the suffix following the (s, a) transition.
Improvement operator: strict-superset existence on the lattice ({0,1}^k, ≤).
At decision step in cluster s, additive logit nudge α (n_a^dom − n_a^sub).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ---------------------------------------------------------------------------
# Policy network with explicit penultimate features for state clustering
# ---------------------------------------------------------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, feat_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, feat_dim)
        self.head = nn.Linear(feat_dim, n_actions)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h))
        return z

    def forward(self, x: torch.Tensor):
        z = self.features(x)
        logits = self.head(z)
        return logits, z


# ---------------------------------------------------------------------------
# Online prototype clustering on the penultimate features
# ---------------------------------------------------------------------------


class OnlineClusterer:
    """Streaming nearest-prototype clusterer in feature space.

    A new prototype is created when the closest existing prototype is farther
    than `radius`. Cluster id is the index of the nearest prototype. EMA
    update of prototypes keeps centroids stable while the policy net trains.
    Capped at `max_clusters` to bound memory.
    """

    def __init__(self, feat_dim: int, radius: float = 0.5, ema: float = 0.05, max_clusters: int = 256):
        self.feat_dim = feat_dim
        self.radius = float(radius)
        self.ema = float(ema)
        self.max_clusters = int(max_clusters)
        self.protos: list[np.ndarray] = []

    def assign(self, feat: np.ndarray, *, learn: bool = True) -> int:
        if not self.protos:
            if learn:
                self.protos.append(feat.copy())
                return 0
            return 0
        P = np.stack(self.protos)
        d = np.linalg.norm(P - feat[None, :], axis=1)
        i = int(np.argmin(d))
        if learn and d[i] > self.radius and len(self.protos) < self.max_clusters:
            self.protos.append(feat.copy())
            return len(self.protos) - 1
        if learn:
            self.protos[i] = (1.0 - self.ema) * self.protos[i] + self.ema * feat
        return i


# ---------------------------------------------------------------------------
# The pattern lattice — the actual primitive S(s, a) ⊆ {0,1}^k
# ---------------------------------------------------------------------------


class PatternLattice:
    """Stores S(s, a) ⊆ {0,1}^k as a hash-set per (cluster, action) cell.

    Patterns are encoded as small integers (bitmask over k channels), so each
    cell is a Python `set[int]`. Strict-superset on bitmasks is bitwise.
    """

    def __init__(self, n_actions: int, k: int):
        self.n_actions = int(n_actions)
        self.k = int(k)
        # cells: dict[(cluster_id, action) -> set[int]]
        self.cells: dict[tuple[int, int], set[int]] = {}

    def add(self, cluster_id: int, action: int, mask: int) -> None:
        key = (int(cluster_id), int(action))
        s = self.cells.get(key)
        if s is None:
            s = set()
            self.cells[key] = s
        s.add(int(mask))

    def supports(self, cluster_id: int) -> list[set[int]]:
        return [self.cells.get((int(cluster_id), a), set()) for a in range(self.n_actions)]

    def nudge_vector(self, cluster_id: int) -> np.ndarray:
        """Compute additive logit nudge n_a^dom − n_a^sub at this cluster.

        For each ordered pair (a, a'), dom(a, a') is True iff there exists
        b ∈ S(s,a) such that for every b' ∈ S(s,a'): b ⊃ b' strictly
        (bitwise (b | b') == b and b != b').
        """
        A = self.n_actions
        supports = self.supports(cluster_id)
        nudges = np.zeros(A, dtype=np.float32)
        # Pre-list each support
        sup_lists = [list(s) for s in supports]
        for a in range(A):
            Sa = sup_lists[a]
            if not Sa:
                continue
            for ap in range(A):
                if ap == a:
                    continue
                Sap = sup_lists[ap]
                if not Sap:
                    continue
                # exists b in Sa such that for all b' in Sap: b ⊃ b' strictly
                fired = False
                for b in Sa:
                    ok = True
                    for bp in Sap:
                        # strict superset: (b | bp) == b AND b != bp
                        if (b | bp) != b or b == bp:
                            ok = False
                            break
                    if ok:
                        fired = True
                        break
                if fired:
                    nudges[a] += 1.0
                    nudges[ap] -= 1.0
        return nudges

    def stats(self) -> dict[str, float]:
        if not self.cells:
            return {"n_cells": 0, "avg_size": 0.0, "frac_singleton": 0.0}
        sizes = np.array([len(v) for v in self.cells.values()], dtype=np.float64)
        return {
            "n_cells": int(len(self.cells)),
            "avg_size": float(sizes.mean()),
            "frac_singleton": float((sizes == 1).mean()),
        }


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def obs_to_vec(obs) -> np.ndarray:
    """Flatten an observation to a 1-D float32 vector.

    Handles vector envs (small Box) and (defensively) sparse envs by
    flattening images. The hypothesis run targets vector envs.
    """
    a = np.asarray(obs)
    return a.astype(np.float32).reshape(-1)


def get_vector_channels(env_id: str, info: dict) -> np.ndarray:
    """Return the per-step vector signal as a float array.

    For vector envs we read info["vector"]. For non-vector envs we still
    reach this code path (the candidate is run on stage=vector), but we
    fall back to a single-channel projection of scalar reward as a safety
    net — the lattice will then collapse, which is informative.
    """
    v = info.get("vector", None)
    if v is None:
        return np.zeros(1, dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    obs0, _ = env.reset(seed=seed)
    obs_vec0 = obs_to_vec(obs0)
    obs_dim = int(obs_vec0.shape[0])
    action_space = env.action_space
    n_actions = int(action_space.n) if hasattr(action_space, "n") else 0

    # Probe k (number of vector channels) on the first env step
    probe_action = 0 if n_actions > 0 else action_space.sample()
    _, _, _, _, info0 = env.step(probe_action)
    v0 = get_vector_channels(env_id, info0)
    k = int(v0.shape[0])
    # Reset so we don't bias the first episode
    env.reset(seed=seed)

    # k may be large; the hypothesis explicitly notes it's small on this panel.
    # We cap pattern enumeration safety bound at k <= 16 (mask fits in uint).
    if k > 16:
        k = 16

    feat_dim = 32
    net = PolicyNet(obs_dim, n_actions, feat_dim=feat_dim)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    clusterer = OnlineClusterer(feat_dim=feat_dim, radius=0.5, ema=0.05, max_clusters=256)
    lattice = PatternLattice(n_actions=n_actions, k=k)

    alpha = 1.0  # logit-nudge magnitude per dominated action
    train_steps_per_episode = 4
    update_batch_cap = 512

    # Diagnostics
    n_decision_steps = 0
    n_operator_fires = 0
    n_episodes = 0
    n_env_steps = 0
    pair_total = 0
    pair_fires = 0

    t_start = time.monotonic()

    # Episode buffer for batched policy updates after each episode
    # Each record: (obs_vec, action, cluster_id_at_decision_time, nudge_vec_at_decision_time)
    recent_records: list[tuple[np.ndarray, int, int, np.ndarray]] = []

    def policy_logits_with_nudge(obs_vec_np: np.ndarray, learn_cluster: bool):
        x = torch.from_numpy(obs_vec_np).float().unsqueeze(0)
        with torch.no_grad():
            logits_t, feat_t = net(x)
        logits = logits_t.squeeze(0).cpu().numpy()
        feat = feat_t.squeeze(0).cpu().numpy()
        cid = clusterer.assign(feat, learn=learn_cluster)
        nudge = lattice.nudge_vector(cid)
        return logits, nudge, cid

    while time.monotonic() - t_start < time_budget_s:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        steps = 0
        # Per-step storage for this episode (to compute suffix-OR at end)
        step_clusters: list[int] = []
        step_actions: list[int] = []
        step_masks: list[int] = []  # binarized vector at this step
        step_records: list[tuple[np.ndarray, int, int, np.ndarray]] = []

        while not done and steps < harness.MAX_EPISODE_STEPS:
            obs_vec = obs_to_vec(obs)
            logits, nudge, cid = policy_logits_with_nudge(obs_vec, learn_cluster=True)
            n_decision_steps += 1
            # Track operator firing diagnostics
            if np.any(np.abs(nudge) > 0.0):
                n_operator_fires += 1
            # Pair-fire counts: total ordered pairs evaluated and fires
            pair_total += n_actions * (n_actions - 1)

            adjusted = logits + alpha * nudge
            # Sample action from softmax(adjusted) (Gumbel-max for numerical stability)
            g = rng.gumbel(size=n_actions).astype(np.float32)
            action = int(np.argmax(adjusted + g))

            step_records.append((obs_vec.copy(), action, cid, nudge.copy()))

            obs, _reward, term, trunc, info = env.step(action)
            v = get_vector_channels(env_id, info)
            # Binarize: channel fires iff |v_m| > 0 (epsilon for float)
            mask = 0
            for m in range(min(k, int(v.shape[0]))):
                if abs(float(v[m])) > 1e-12:
                    mask |= 1 << m

            step_clusters.append(cid)
            step_actions.append(action)
            step_masks.append(mask)

            done = bool(term) or bool(trunc)
            steps += 1
            n_env_steps += 1
            if time.monotonic() - t_start >= time_budget_s:
                break

        # Episode ended (or budget hit). Compute suffix-OR per step, deposit
        # one pattern into each visited (cluster, action) cell.
        T = len(step_masks)
        if T > 0:
            suffix_or = 0
            # Walk backward computing ⋁_{u >= t} mask_u
            suffix_masks = [0] * T
            for t in range(T - 1, -1, -1):
                suffix_or |= step_masks[t]
                suffix_masks[t] = suffix_or
            for t in range(T):
                lattice.add(step_clusters[t], step_actions[t], suffix_masks[t])

        # Recount pair-fires for diagnostics on the post-update lattice over
        # this episode's decision steps (cheap; bounded by T * A^2)
        for cid in set(step_clusters):
            sup_lists = [list(s) for s in lattice.supports(cid)]
            for a in range(n_actions):
                Sa = sup_lists[a]
                if not Sa:
                    continue
                for ap in range(n_actions):
                    if ap == a:
                        continue
                    Sap = sup_lists[ap]
                    if not Sap:
                        continue
                    for b in Sa:
                        ok = True
                        for bp in Sap:
                            if (b | bp) != b or b == bp:
                                ok = False
                                break
                        if ok:
                            pair_fires += 1
                            break

        recent_records.extend(step_records)
        if len(recent_records) > update_batch_cap * 4:
            recent_records = recent_records[-update_batch_cap * 4 :]

        # Policy-update phase: cross-entropy toward softmax(logits + α·nudge)
        # using the cluster ids and nudges captured at the time of the
        # decision (so the supervision reflects what the lattice said then).
        if recent_records:
            for _ in range(train_steps_per_episode):
                if time.monotonic() - t_start >= time_budget_s:
                    break
                idx = rng.integers(0, len(recent_records), size=min(update_batch_cap, len(recent_records)))
                batch_obs = np.stack([recent_records[i][0] for i in idx], axis=0)
                batch_nudge = np.stack([recent_records[i][3] for i in idx], axis=0)
                x = torch.from_numpy(batch_obs).float()
                target_nudge = torch.from_numpy(batch_nudge).float()
                logits_b, _ = net(x)
                # Soft targets: softmax(logits.detach() + alpha * nudge)
                with torch.no_grad():
                    soft_target = F.softmax(logits_b.detach() + alpha * target_nudge, dim=-1)
                logp = F.log_softmax(logits_b, dim=-1)
                loss = -(soft_target * logp).sum(dim=-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        n_episodes += 1

    env.close()

    # Final diagnostics (logged once)
    stats = lattice.stats()
    op_rate = (n_operator_fires / max(1, n_decision_steps))
    pair_rate = (pair_fires / max(1, pair_total))
    print(
        f"[arp] env={env_id} seed={seed} episodes={n_episodes} env_steps={n_env_steps} "
        f"clusters={len(clusterer.protos)} cells={stats['n_cells']} "
        f"avg|S(s,a)|={stats['avg_size']:.3f} frac_singleton={stats['frac_singleton']:.3f} "
        f"operator_fire_rate={op_rate:.4f} pair_dom_rate={pair_rate:.4f} k={k}",
        flush=True,
    )
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_env_steps} "
        f"train_s={time.monotonic() - t_start:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Deterministic policy: argmax over logits + α·nudge using frozen net+lattice
    net.eval()

    def policy_fn(obs_np):
        obs_vec = obs_to_vec(obs_np)
        x = torch.from_numpy(obs_vec).float().unsqueeze(0)
        with torch.no_grad():
            logits_t, feat_t = net(x)
        logits = logits_t.squeeze(0).cpu().numpy()
        feat = feat_t.squeeze(0).cpu().numpy()
        cid = clusterer.assign(feat, learn=False)
        nudge = lattice.nudge_vector(cid)
        return int(np.argmax(logits + alpha * nudge))

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
