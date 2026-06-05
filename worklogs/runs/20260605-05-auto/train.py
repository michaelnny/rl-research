"""VCC — Vector-Cumulant Confluence (run 20260605-05-auto).

Faithful realization of the hypothesis:
  * index by quantized running per-channel cumulant c_t = sum_{s<=t} v_s
  * per (cumulant-bucket, action) maintain Pareto front of continuation
    deltas Delta = c_T - c_t
  * decision-time logit nudge = alpha * dominance_margin(a; b(s))
  * distill nudged logits into a small neural policy via supervised KL
  * epsilon-fraction front-discovery exploration via least-visited bucket
  * NO scalar collapse, NO Bellman backup, NO critic, NO advantage,
    NO learned weighting w of channels.

Vector envs supply info["vector"] directly. Scalar-reward envs are
augmented with cheap structural channels (step-parity,
action-repeat-streak, terminal-flag) per the hypothesis.

Contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from typing import Any

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


# -----------------------------------------------------------------------------
# Pareto-front utilities
# -----------------------------------------------------------------------------

def pareto_front(points: np.ndarray) -> np.ndarray:
    """Strict-dominance Pareto front (maximization, channel-wise)."""
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        # any other point that strictly dominates p?
        # x dominates p iff all(x >= p) and any(x > p)
        ge = np.all(points >= p, axis=1)
        gt = np.any(points > p, axis=1)
        dominates = ge & gt
        dominates[i] = False
        if dominates.any():
            keep[i] = False
    return points[keep]


def count_dominated(front_a: np.ndarray, front_b: np.ndarray) -> int:
    """Number of points in front_b that are strictly dominated by some point in front_a."""
    if len(front_a) == 0 or len(front_b) == 0:
        return 0
    # Broadcast: for each b, exists a with a >= b on all channels and a > b on some
    # front_a: (Na, k); front_b: (Nb, k)
    a = front_a[:, None, :]  # (Na, 1, k)
    b = front_b[None, :, :]  # (1, Nb, k)
    ge = np.all(a >= b, axis=2)
    gt = np.any(a > b, axis=2)
    dom = ge & gt  # (Na, Nb)
    return int(np.any(dom, axis=0).sum())


# -----------------------------------------------------------------------------
# Observation featurization (input to the small policy net)
# -----------------------------------------------------------------------------

def flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


# -----------------------------------------------------------------------------
# Tiny MLP policy
# -----------------------------------------------------------------------------

class TinyPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# VCC store: cumulant-prefix-bucket -> per-action Pareto front of continuations
# -----------------------------------------------------------------------------

class VCCStore:
    def __init__(self, n_actions: int, k_channels: int, max_per_action: int = 64):
        self.n_actions = n_actions
        self.k = k_channels
        self.max_per_action = max_per_action
        # bucket -> action -> deque of continuation-delta vectors (recency window)
        self.bucket_action_deltas: dict[tuple, list[deque]] = {}
        # bucket -> visit count (for least-visited exploration)
        self.bucket_visits: dict[tuple, int] = defaultdict(int)
        # cached fronts per (bucket, action)
        self.fronts: dict[tuple[tuple, int], np.ndarray] = {}

    def _ensure_bucket(self, bucket: tuple) -> None:
        if bucket not in self.bucket_action_deltas:
            self.bucket_action_deltas[bucket] = [
                deque(maxlen=self.max_per_action) for _ in range(self.n_actions)
            ]

    def add(self, bucket: tuple, action: int, delta: np.ndarray) -> None:
        self._ensure_bucket(bucket)
        self.bucket_action_deltas[bucket][action].append(np.asarray(delta, dtype=np.float64))
        self.fronts.pop((bucket, action), None)

    def visit(self, bucket: tuple) -> None:
        self.bucket_visits[bucket] += 1

    def front(self, bucket: tuple, action: int) -> np.ndarray:
        if bucket not in self.bucket_action_deltas:
            return np.zeros((0, self.k), dtype=np.float64)
        key = (bucket, action)
        if key in self.fronts:
            return self.fronts[key]
        dq = self.bucket_action_deltas[bucket][action]
        if len(dq) == 0:
            f = np.zeros((0, self.k), dtype=np.float64)
        else:
            pts = np.stack(list(dq), axis=0)
            f = pareto_front(pts)
        self.fronts[key] = f
        return f

    def dominance_margin(self, bucket: tuple) -> np.ndarray:
        """Per-action margin m(a) = #(other-fronts pts dominated by F(b,a))
                                  - #(F(b,a) pts dominated by other-fronts).
        """
        m = np.zeros(self.n_actions, dtype=np.float64)
        if bucket not in self.bucket_action_deltas:
            return m
        fronts = [self.front(bucket, a) for a in range(self.n_actions)]
        for a in range(self.n_actions):
            others = [fronts[b] for b in range(self.n_actions) if b != a]
            other_pts = (
                np.concatenate(others, axis=0) if any(len(x) for x in others)
                else np.zeros((0, self.k), dtype=np.float64)
            )
            m[a] = count_dominated(fronts[a], other_pts) - count_dominated(other_pts, fronts[a])
        return m

    def least_visited_action(self, bucket: tuple, rng: np.random.Generator) -> int:
        """Action whose (bucket, action) pair has the fewest stored continuations.

        This is bucket-coverage exploration over the cumulant manifold:
        we prefer the action that has been *least sampled* from this
        cumulant prefix.
        """
        if bucket not in self.bucket_action_deltas:
            return int(rng.integers(self.n_actions))
        counts = np.array(
            [len(self.bucket_action_deltas[bucket][a]) for a in range(self.n_actions)],
            dtype=np.int64,
        )
        m = counts.min()
        cand = np.where(counts == m)[0]
        return int(rng.choice(cand))


# -----------------------------------------------------------------------------
# Cumulant quantization
# -----------------------------------------------------------------------------

class CumulantQuantizer:
    """Quantize per-channel cumulants by per-channel adaptive bin width.

    Bucket = tuple of int per channel. Width chosen from a running estimate
    of per-channel scale; defaults to 1.0 if scale is unknown.
    """

    def __init__(self, k: int, target_width: np.ndarray | None = None):
        self.k = k
        if target_width is None:
            target_width = np.ones(k, dtype=np.float64)
        self.width = np.maximum(np.asarray(target_width, dtype=np.float64), 1e-6)

    def __call__(self, c: np.ndarray) -> tuple:
        q = np.floor(c / self.width).astype(np.int64)
        return tuple(int(x) for x in q)


# -----------------------------------------------------------------------------
# Per-step vector signal
# -----------------------------------------------------------------------------

def vector_step_signal(
    is_vector_env: bool,
    info: dict,
    action: int,
    last_action: int | None,
    repeat_streak: int,
    step_idx: int,
    done: bool,
) -> np.ndarray:
    """Return per-step vector v_s.

    Vector envs: use info['vector'] directly.
    Scalar envs: cheap structural channels (no task semantics):
      - step-parity in {-1,+1}
      - action-repeat-streak (post-update)
      - terminal-flag
      - unique-step indicator (always +1 since each step is unique;
        accumulates as plain step count — used as a coverage channel)
    """
    if is_vector_env:
        v = np.asarray(info["vector"], dtype=np.float64)
        return v
    parity = 1.0 if (step_idx % 2 == 0) else -1.0
    repeat = float(repeat_streak)
    term = 1.0 if done else 0.0
    unique = 1.0
    return np.array([parity, repeat, term, unique], dtype=np.float64)


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave a margin

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    is_vector_env = harness.ENV_TYPE[env_id] == "vector"

    # action / observation discovery
    action_space = env.action_space
    if hasattr(action_space, "n"):
        n_actions = int(action_space.n)
    else:
        env.close()
        raise RuntimeError("VCC requires a discrete action space.")

    # k = number of vector-feedback channels
    k_channels: int
    bucket_widths: np.ndarray
    if is_vector_env:
        # peek one step to learn k
        obs0, _ = env.reset(seed=seed)
        a0 = int(rng.integers(n_actions))
        _, _, _, _, info0 = env.step(a0)
        v0 = np.asarray(info0["vector"], dtype=np.float64)
        k_channels = int(v0.shape[0])
        # use width 1.0 per channel by default (DST treasure values are O(1..125),
        # time penalty is -1/step; resource-gathering uses {-1, 0, 1} signals)
        bucket_widths = np.ones(k_channels, dtype=np.float64)
    else:
        # auxiliary structural channels
        k_channels = 4
        # parity is +/-1; widths 1 means each int parity step is its own bucket
        # repeat-streak grows; bucket every 4 steps; terminal is 0/1; unique grows.
        bucket_widths = np.array([2.0, 4.0, 1.0, 8.0], dtype=np.float64)

    quantizer = CumulantQuantizer(k_channels, target_width=bucket_widths)
    store = VCCStore(n_actions=n_actions, k_channels=k_channels, max_per_action=64)

    # observation featurization probe
    obs0, _ = env.reset(seed=seed + 1)
    feat0 = flatten_obs(obs0)
    obs_dim = int(feat0.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TinyPolicy(obs_dim, n_actions, hidden=64).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    # Hyperparameters
    alpha = 1.0          # logit nudge magnitude
    epsilon = 0.10       # fraction of front-discovery exploration
    distill_steps_per_episode = 4
    distill_batch = 64

    # Episode buffer for distillation
    state_buf: deque = deque(maxlen=4096)
    target_logits_buf: deque = deque(maxlen=4096)

    n_episodes = 0
    n_env_steps = 0
    bucket_count_high_water = 0
    nonzero_margin_episodes = 0

    while time.monotonic() < deadline:
        obs, _ = env.reset(seed=seed + 1000 + n_episodes)
        cumulant = np.zeros(k_channels, dtype=np.float64)
        ep_traj: list[tuple[tuple, int, np.ndarray]] = []  # (bucket, action, cumulant_at_step)
        cum_seq: list[np.ndarray] = [cumulant.copy()]

        last_action: int | None = None
        repeat_streak = 0
        step_idx = 0
        done = False
        ep_nonzero_margin = False

        while not done and step_idx < harness.MAX_EPISODE_STEPS:
            if time.monotonic() > deadline:
                break
            bucket = quantizer(cumulant)
            store.visit(bucket)

            # Decision-time: dominance-nudged logits from VCC
            margin = store.dominance_margin(bucket)
            if np.any(margin != 0):
                ep_nonzero_margin = True

            # Neural policy gives the base logits
            with torch.no_grad():
                feat = torch.from_numpy(flatten_obs(obs)).to(device).unsqueeze(0)
                base_logits = policy(feat).cpu().numpy()[0]
            nudged = base_logits + alpha * margin

            # Front-discovery epsilon: pick least-visited (bucket, action)
            if rng.random() < epsilon:
                action = store.least_visited_action(bucket, rng)
            else:
                # sample from softmax of nudged logits
                probs = _softmax(nudged)
                action = int(rng.choice(n_actions, p=probs))

            # store the (state, target_logits) pair for distillation
            state_buf.append(flatten_obs(obs))
            target_logits_buf.append(nudged.astype(np.float32).copy())

            ep_traj.append((bucket, action, cumulant.copy()))

            # update repeat streak
            if last_action is not None and action == last_action:
                repeat_streak += 1
            else:
                repeat_streak = 0
            last_action = action

            obs, _, term, trunc, info = env.step(action)
            done = bool(term) or bool(trunc)
            step_idx += 1
            n_env_steps += 1

            v = vector_step_signal(
                is_vector_env=is_vector_env,
                info=info,
                action=action,
                last_action=last_action,
                repeat_streak=repeat_streak,
                step_idx=step_idx,
                done=done,
            )
            cumulant = cumulant + v
            cum_seq.append(cumulant.copy())

        # End of episode: c_T is cumulant; install continuation deltas into store
        c_T = cumulant.copy()
        for bucket, action, c_t in ep_traj:
            delta = c_T - c_t
            store.add(bucket, action, delta)

        # Distillation: small policy is trained to match nudged-logit targets via KL
        if len(state_buf) >= distill_batch:
            for _ in range(distill_steps_per_episode):
                if time.monotonic() > deadline:
                    break
                idx = rng.integers(0, len(state_buf), size=distill_batch)
                s_batch = np.stack([state_buf[i] for i in idx], axis=0)
                t_batch = np.stack([target_logits_buf[i] for i in idx], axis=0)
                s_t = torch.from_numpy(s_batch).to(device)
                t_t = torch.from_numpy(t_batch).to(device)
                logits = policy(s_t)
                # KL(target || student) with target = softmax(t_t)
                target_probs = F.softmax(t_t, dim=-1)
                log_student = F.log_softmax(logits, dim=-1)
                loss = -(target_probs * log_student).sum(dim=-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        n_episodes += 1
        bucket_count_high_water = max(bucket_count_high_water, len(store.bucket_visits))
        if ep_nonzero_margin:
            nonzero_margin_episodes += 1

    env.close()

    # Diagnostics
    train_s = time.monotonic() - t_start
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_env_steps} "
        f"episodes={n_episodes} buckets={bucket_count_high_water} "
        f"nonzero_margin_eps={nonzero_margin_episodes} "
        f"train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Build the deterministic policy_fn used at evaluation time.
    # We use the distilled neural policy, taking argmax of its logits.
    policy.eval()

    def policy_fn(obs: np.ndarray):
        feat = torch.from_numpy(flatten_obs(obs)).to(device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(feat).cpu().numpy()[0]
        return int(np.argmax(logits))

    return policy_fn


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(x) / len(x)
    return e / s


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
