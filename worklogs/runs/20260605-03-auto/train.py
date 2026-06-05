"""RSD — Reconvergent Segment Dominance.

Faithful realization of run 20260605-03-auto's hypothesis:
- Primitive: directed multigraph over observation-hash nodes; parallel
  edges are vector-tagged action segments extracted from past trajectories.
- Improvement operator: for each (u, v) with >= 2 parallel edges, find the
  Pareto-dominant edge under coordinate-wise dominance over the segment's
  accumulated vector signal; behavior-clone the policy toward the dominant
  segment's action sequence with weight = count of strictly-dominated peers.
- Vector channels are NEVER collapsed; on terminal-only-reward sparse envs
  we append a constant +1 step-cost channel so the partial order is well
  defined (the operator then reduces to "imitate the shorter path between
  recurring observations" as the hypothesis explicitly predicts).
- No critic, no advantage, no return-weighted log-prob, no Bellman backup,
  no rollout verification, no reward-magnitude in the gradient weight —
  only the *count of dominated peers*.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# --------------------------------------------------------------------------- #
# Observation hashing                                                         #
# --------------------------------------------------------------------------- #


def obs_hash(obs: np.ndarray) -> bytes:
    """Deterministic content hash of an observation array.

    For MiniGrid `ImgObsWrapper` outputs (uint8 H x W x 3), hashing the raw
    bytes yields exact equivalence on the agent-centric symbolic image.
    For mo-gymnasium discrete-state envs (DST, Resource-Gathering) the
    obs is a small integer or 1-D vector; we hash bytes of the canonical
    array form.
    """
    arr = np.ascontiguousarray(obs)
    return hashlib.blake2b(arr.tobytes(), digest_size=12).digest()


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """Flatten obs to a float32 1-D tensor."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return torch.from_numpy(arr)


# --------------------------------------------------------------------------- #
# Policy                                                                      #
# --------------------------------------------------------------------------- #


class MLPPolicy(nn.Module):
    """Discrete-action MLP. Input is flat obs; output is action logits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def logits(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.net(obs_flat)


# --------------------------------------------------------------------------- #
# RSD core                                                                    #
# --------------------------------------------------------------------------- #


def pareto_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff `a` Pareto-dominates `b`: a >= b in all channels, > in some."""
    return bool(np.all(a >= b) and np.any(a > b))


def collect_episode(
    env,
    policy: MLPPolicy,
    is_vector: bool,
    n_actions: int,
    temperature: float,
    rng: np.random.Generator,
    seed: int,
    max_steps: int,
):
    """Run one episode under the current policy. Returns the trajectory as
    parallel arrays plus per-step vector signals.

    Vector signal:
      - vector envs: info["vector"] from each step.
      - scalar envs: a 2-channel vector [reward, -1] where -1 is the
        step-cost channel (negative so that "fewer steps" is dominant).
    """
    obs, _ = env.reset(seed=seed)
    obs_list = [np.asarray(obs).copy()]
    actions: list[int] = []
    vec_list: list[np.ndarray] = []
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        ot = obs_to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = policy.logits(ot).squeeze(0).numpy()
        # Softmax with temperature, then sample.
        z = logits / max(temperature, 1e-3)
        z = z - z.max()
        probs = np.exp(z)
        probs = probs / probs.sum()
        a = int(rng.choice(n_actions, p=probs))
        next_obs, reward, term, trunc, info = env.step(a)
        if is_vector:
            v = np.asarray(info["vector"], dtype=np.float64).copy()
        else:
            # 2-channel vector for scalar envs: [reward, step-cost(-1)].
            v = np.array([float(reward), -1.0], dtype=np.float64)
        actions.append(a)
        vec_list.append(v)
        obs_list.append(np.asarray(next_obs).copy())
        total_reward += float(reward)
        done = bool(term) or bool(trunc)
        obs = next_obs
        steps += 1
    return obs_list, actions, vec_list, total_reward, steps


def build_segment_index(
    obs_list: list[np.ndarray],
    actions: list[int],
    vec_list: list[np.ndarray],
    edge_store: dict,
    max_segment_len: int,
    obs_index_in_traj: dict,
):
    """Decompose this trajectory into all contiguous segments up to length
    `max_segment_len`, group them by (hash(o_i), hash(o_j)) into edge_store.

    edge_store: dict[(u_hash, v_hash)] -> list of dicts with keys
        {accum_vec, action_seq, start_obs}
    """
    T = len(actions)
    if T == 0:
        return
    hashes = [obs_hash(o) for o in obs_list]
    # cumulative vector
    cum = np.zeros((T + 1, vec_list[0].shape[0]), dtype=np.float64)
    for t in range(T):
        cum[t + 1] = cum[t] + vec_list[t]
    # All segments (i, j), 0 <= i < j <= T, j - i <= max_segment_len
    for i in range(T):
        u = hashes[i]
        jmax = min(T, i + max_segment_len)
        for j in range(i + 1, jmax + 1):
            v = hashes[j]
            if u == v:
                continue  # zero-length cycle in hash space; skip
            accum = cum[j] - cum[i]
            seg = {
                "accum": accum,
                "actions": np.asarray(actions[i:j], dtype=np.int64),
                "start_obs": obs_list[i],
            }
            edge_store[(u, v)].append(seg)


def find_dominated_pairs(edges: list[dict]) -> list[tuple[int, int]]:
    """Given the list of edges in E(u,v), return list of (dominant_idx, count)
    where count = number of strictly-dominated peers.
    Each edge that dominates at least one other is returned.
    """
    n = len(edges)
    if n < 2:
        return []
    accs = np.stack([e["accum"] for e in edges], axis=0)
    out: list[tuple[int, int]] = []
    for i in range(n):
        cnt = 0
        for k in range(n):
            if i == k:
                continue
            if pareto_dominates(accs[i], accs[k]):
                cnt += 1
        if cnt > 0:
            out.append((i, cnt))
    return out


def rsd_update(
    policy: MLPPolicy,
    optimizer: torch.optim.Optimizer,
    edge_store: dict,
    max_pairs_per_update: int,
    rng: np.random.Generator,
):
    """One RSD policy update: scan edges in edge_store, find Pareto-dominant
    peers, add a count-weighted BC loss term for each.

    The supervised target is the dominant segment's full action sequence,
    conditioned on the start observation `o_i`. We treat the action sequence
    autoregressively-as-iid given `o_i` (no learned context model — the
    policy is feedforward; the conditioning is purely on the start obs).
    This matches the hypothesis: ``∇_θ log π_θ(a^{s*}_t | context_t)``
    summed over the dominant segment's actions, weighted by dominance count.
    """
    # Collect per-(u,v) dominant pairs.
    items = []
    for key, edges in edge_store.items():
        if len(edges) < 2:
            continue
        doms = find_dominated_pairs(edges)
        for di, count in doms:
            items.append((edges[di], count))
    if not items:
        return 0, 0
    # Optionally subsample to bound compute per update.
    if len(items) > max_pairs_per_update:
        idx = rng.choice(len(items), size=max_pairs_per_update, replace=False)
        items = [items[i] for i in idx]

    # Build a single batch of (start_obs, action) supervised pairs,
    # weighted by the dominance count.
    obs_list = []
    act_list = []
    weights = []
    for seg, count in items:
        start = obs_to_tensor(seg["start_obs"])
        # Each action in the dominant segment is supervised against the
        # SAME start observation (feedforward policy = no learned context).
        for a in seg["actions"]:
            obs_list.append(start)
            act_list.append(int(a))
            weights.append(float(count))

    if not obs_list:
        return 0, 0

    obs_batch = torch.stack(obs_list, dim=0)
    act_batch = torch.tensor(act_list, dtype=torch.long)
    w_batch = torch.tensor(weights, dtype=torch.float32)

    logits = policy.logits(obs_batch)
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(1, act_batch.unsqueeze(-1)).squeeze(-1)
    # Behavior-clone loss: maximize weighted log-prob of dominant action.
    loss = -(w_batch * chosen).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return len(items), len(obs_list)


# --------------------------------------------------------------------------- #
# train()                                                                     #
# --------------------------------------------------------------------------- #


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    n_actions = int(env.action_space.n)

    # Probe obs to learn flat dim.
    probe_obs, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(probe_obs, dtype=np.float32).reshape(-1).shape[0])

    policy = MLPPolicy(obs_dim, n_actions, hidden=128)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Edge store: (u_hash, v_hash) -> list[segment dict]
    edge_store: dict = defaultdict(list)

    # Configuration
    max_segment_len = 16  # cap parallel-edge length to bound the O(T*L) decomp
    max_pairs_per_update = 256
    max_episode_steps = harness.MAX_EPISODE_STEPS
    edge_store_cap = 50_000  # memory cap on total stored segments

    # Diagnostics
    episodes = 0
    total_env_steps = 0
    total_updates = 0
    max_edge_mult = 1
    n_pair_updates_total = 0

    # Temperature schedule: gently anneal to keep diversity.
    temp_start, temp_end = 1.5, 0.7

    while time.monotonic() - t0 < time_budget_s * 0.95:
        # Anneal temperature linearly with elapsed fraction.
        frac = min(1.0, (time.monotonic() - t0) / max(1.0, time_budget_s))
        temperature = temp_start + (temp_end - temp_start) * frac

        ep_seed = seed + 100 + episodes
        obs_list, actions, vec_list, total_reward, steps = collect_episode(
            env,
            policy,
            is_vector,
            n_actions,
            temperature,
            rng,
            ep_seed,
            max_episode_steps,
        )
        episodes += 1
        total_env_steps += steps

        if steps > 0:
            build_segment_index(
                obs_list,
                actions,
                vec_list,
                edge_store,
                max_segment_len,
                obs_index_in_traj={},
            )
            # Track diagnostic: max edge multiplicity.
            for k, v in edge_store.items():
                if len(v) > max_edge_mult:
                    max_edge_mult = len(v)

        # Memory cap: if too many segments, prune buckets back to their
        # 8 most-recent witnesses (keep diversity but bound size).
        total_segments = sum(len(v) for v in edge_store.values())
        if total_segments > edge_store_cap:
            for k in list(edge_store.keys()):
                if len(edge_store[k]) > 8:
                    edge_store[k] = edge_store[k][-8:]

        # RSD improvement step: behavior-clone toward Pareto-dominant edges.
        n_pairs, n_grads = rsd_update(
            policy, optimizer, edge_store, max_pairs_per_update, rng
        )
        if n_pairs > 0:
            total_updates += 1
            n_pair_updates_total += n_pairs

        if episodes % 10 == 0 or episodes <= 3:
            print(
                f"[rsd] ep={episodes} env_steps={total_env_steps} "
                f"edges={len(edge_store)} max_mult={max_edge_mult} "
                f"updates={total_updates} pairs_seen={n_pair_updates_total} "
                f"temp={temperature:.2f} elapsed={time.monotonic() - t0:.1f}s "
                f"last_ret={total_reward:.3f}",
                flush=True,
            )

    env.close()

    print(
        f"[rsd-final] env={env_id} seed={seed} episodes={episodes} "
        f"env_steps={total_env_steps} edges={len(edge_store)} "
        f"max_edge_mult={max_edge_mult} updates={total_updates} "
        f"train_s={time.monotonic() - t0:.1f}",
        flush=True,
    )

    # Build deterministic greedy policy for evaluation.
    def policy_fn(obs: np.ndarray):
        ot = obs_to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = policy.logits(ot).squeeze(0).numpy()
        return int(np.argmax(logits))

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
