"""ACCD — Action-Conditional Channel Dissociation.

Realizes hypothesis 20260606-26-auto:
- Two small offline-supervised classifiers predict H-step per-channel firing.
  - psi_full(o) -> [0,1]^k  (action-marginal)
  - psi_a(o, a) -> [0,1]^k  (action-conditional, shared trunk + action one-hot)
- Primitive: S[o, a, m] = psi_a(o, a)[m] - psi_full(o)[m].
- Improvement operator: Pareto-non-dominance count over rows S[o, a, :],
  k_eff gated (drop channels whose max_a S - min_a S < eps for that o).
  Logit nudge: Delta_logit_a = alpha * (n_dom_a - n_sub_a) / |A|.
- Policy update: BC toward shifted-logit target via cross-entropy.
- Channels: from info["vector"] on vector envs (thresholded |v|>0).
  On scalar envs (DoorKey, KeyCorridor), single-channel binary
  "scalar reward fired in window" indicator (acknowledged k=1 degenerate).
- Craftax: thresholded reward as single channel (used only if stage hits it).
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# -----------------------------------------------------------------------------
# Hyperparameters (single config; not tuned per env)
# -----------------------------------------------------------------------------
H_HORIZON = 8                     # H-step horizon for channel firing target
ALPHA_NUDGE = 2.0                 # logit nudge scale
EPS_NOISE = 0.02                  # k_eff noise floor on max-min spread
BUFFER_CAP = 20_000               # replay capacity
BATCH_SIZE = 64
N_UPDATES_PER_ROUND = 8
ROUND_STEPS = 256                 # env steps per training round
HIDDEN = 64
LR_PSI = 1e-3
LR_POLICY = 5e-4
EVAL_GREEDY = True


# -----------------------------------------------------------------------------
# Observation flattening
# -----------------------------------------------------------------------------
def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


# -----------------------------------------------------------------------------
# Channel extraction (vector signal -> binary per-step k-vector)
# -----------------------------------------------------------------------------
def _extract_channels(env_id: str, reward: float, info: dict, k: int) -> np.ndarray:
    """Return a k-dim {0,1}^k indicator of which channels fired this step."""
    if "vector" in info:
        v = np.asarray(info["vector"], dtype=np.float64)
        # threshold |v|>0; sign-agnostic (firing is "non-zero event").
        ch = (np.abs(v) > 1e-9).astype(np.float32)
        # pad/truncate to k
        out = np.zeros(k, dtype=np.float32)
        n = min(k, ch.shape[0])
        out[:n] = ch[:n]
        return out
    # scalar fallback: single-channel "reward fired this step"
    out = np.zeros(k, dtype=np.float32)
    if abs(float(reward)) > 1e-9:
        out[0] = 1.0
    return out


def _channel_dim(env_id: str) -> int:
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    return 1  # DoorKey, KeyCorridor, Craftax: degenerate k=1


# -----------------------------------------------------------------------------
# Policy network (logits over discrete actions)
# -----------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN):
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
# Two predictive classifiers psi_full(o) and psi_a(o,a)
# Shared embedding trunk for compute economy; separate output heads;
# psi_a takes one-hot action concatenated AFTER trunk so the marginal trunk
# output is independent of action (avoids leaky shared trunk failure mode).
# -----------------------------------------------------------------------------
class PsiFull(nn.Module):
    """Action-marginal H-step channel-firing classifier."""

    def __init__(self, obs_dim: int, k: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, k),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(o))


class PsiA(nn.Module):
    """Action-conditional H-step channel-firing classifier."""

    def __init__(self, obs_dim: int, n_actions: int, k: int, hidden: int = HIDDEN):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, k),
        )

    def forward(self, o: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a_onehot], dim=-1)
        return torch.sigmoid(self.net(x))


# -----------------------------------------------------------------------------
# Pareto-non-dominance vote on S[a, :] (numpy)
# -----------------------------------------------------------------------------
def pareto_dom_minus_sub(S: np.ndarray, eps: float = EPS_NOISE) -> np.ndarray:
    """S: [n_actions, k] signed shifts. Return per-action (n_dom - n_sub).

    k_eff gating: drop channels where max_a S - min_a S < eps.
    Then for each action a: count actions a' such that S[a] strictly Pareto-
    dominates S[a'] on the kept channels (n_dom_a), and count those a' that
    dominate a (n_sub_a). Ties (no strict dominance either way) contribute 0.
    """
    n_a, k = S.shape
    if n_a == 0 or k == 0:
        return np.zeros(n_a, dtype=np.float32)
    spread = S.max(axis=0) - S.min(axis=0)
    keep = spread > eps
    if not np.any(keep):
        return np.zeros(n_a, dtype=np.float32)
    Sk = S[:, keep]
    # geq[i, j, m] = Sk[i, m] >= Sk[j, m]
    diff = Sk[:, None, :] - Sk[None, :, :]
    geq = diff >= 0.0
    gt = diff > 0.0
    # i dominates j iff all m: i>=j and any m: i>j
    dom = geq.all(axis=2) & gt.any(axis=2)  # [n_a, n_a] bool
    n_dom = dom.sum(axis=1).astype(np.float32)  # how many j i dominates
    n_sub = dom.sum(axis=0).astype(np.float32)  # how many i dominate j (per j)
    return n_dom - n_sub


# -----------------------------------------------------------------------------
# Replay buffer: stores per-step (o, a, reward, info-channel-vec, done)
# in trajectory order so H-step labels can be derived offline.
# -----------------------------------------------------------------------------
class TrajBuffer:
    def __init__(self, cap: int, obs_dim: int, k: int):
        self.cap = cap
        self.obs_dim = obs_dim
        self.k = k
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.act = np.zeros(cap, dtype=np.int64)
        self.ch = np.zeros((cap, k), dtype=np.float32)
        self.ep = np.zeros(cap, dtype=np.int64)  # episode id (resets break H window)
        self.size = 0
        self.idx = 0
        self.cur_ep = 0

    def push(self, o, a, ch, done: bool):
        i = self.idx
        self.obs[i] = o
        self.act[i] = a
        self.ch[i] = ch
        self.ep[i] = self.cur_ep
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        if done:
            self.cur_ep += 1

    def sample_with_horizon(self, batch: int, H: int, rng: np.random.Generator):
        """Sample indices i and compute H-step OR-aggregated channel labels.

        label[i, m] = 1 iff any j in [i+1, min(i+H, last_in_episode)] has ch[j, m] = 1.
        Only sample i whose episode has at least one valid following step (so we
        have non-trivial horizon information).
        """
        if self.size < 2:
            return None
        # build linear index list (handle ring buffer)
        if self.size < self.cap:
            valid = np.arange(self.size - 1)
        else:
            # all indices except the most recently written one (no successor info)
            valid = np.array([j for j in range(self.cap) if j != (self.idx - 1) % self.cap])
        if len(valid) == 0:
            return None
        n = min(batch, len(valid))
        picks = rng.choice(valid, size=n, replace=False)
        labels = np.zeros((n, self.k), dtype=np.float32)
        for bi, i in enumerate(picks):
            ep_i = self.ep[i]
            agg = np.zeros(self.k, dtype=np.float32)
            for h in range(1, H + 1):
                j = (i + h) % self.cap if self.size == self.cap else i + h
                if j >= self.size and self.size < self.cap:
                    break
                # ring crossover: stop if the slot is currently the next-write slot
                if self.size == self.cap and j == self.idx:
                    break
                if self.ep[j] != ep_i:
                    break
                agg = np.maximum(agg, self.ch[j])
            labels[bi] = agg
        return picks.astype(np.int64), labels


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = harness.make_env(env_id, seed)
    if not hasattr(env.action_space, "n"):
        # only discrete supported in this candidate; fall back to random
        action_space = env.action_space
        env.close()

        def _rand(_obs):
            return action_space.sample()

        print(f"[train] env={env_id} non-discrete; random fallback", flush=True)
        return _rand

    n_actions = int(env.action_space.n)
    k = _channel_dim(env_id)

    obs0, _ = env.reset(seed=seed)
    obs_flat = _flatten_obs(obs0)
    obs_dim = obs_flat.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, n_actions).to(device)
    psi_full = PsiFull(obs_dim, k).to(device)
    psi_a = PsiA(obs_dim, n_actions, k).to(device)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR_POLICY)
    opt_full = torch.optim.Adam(psi_full.parameters(), lr=LR_PSI)
    opt_a = torch.optim.Adam(psi_a.parameters(), lr=LR_PSI)

    buf = TrajBuffer(BUFFER_CAP, obs_dim, k)

    # Quick eye on diagnostics
    n_steps = 0
    n_updates = 0
    n_episodes = 0
    keff_history: deque = deque(maxlen=64)

    obs = obs_flat
    ep_return = 0.0

    def policy_logits(o_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(o_np).float().to(device).unsqueeze(0)
            return policy(t).cpu().numpy()[0]

    def sample_action(o_np: np.ndarray) -> int:
        logits = policy_logits(o_np)
        # subtract max for stability
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        # numerical guard
        if not np.all(np.isfinite(p)) or p.sum() <= 0:
            return int(rng.integers(n_actions))
        return int(rng.choice(n_actions, p=p))

    while time.monotonic() - t0 < time_budget_s - 5.0:
        # ---- collect a round ----
        for _ in range(ROUND_STEPS):
            if time.monotonic() - t0 >= time_budget_s - 5.0:
                break
            a = sample_action(obs)
            try:
                next_obs, reward, term, trunc, info = env.step(a)
            except Exception:
                # reset on env crash
                next_obs, _ = env.reset(seed=seed + n_episodes + 1)
                term, trunc, reward, info = True, False, 0.0, {}
            ch = _extract_channels(env_id, float(reward), info, k)
            done = bool(term) or bool(trunc)
            buf.push(obs, a, ch, done)
            n_steps += 1
            ep_return += float(reward)

            if done:
                next_obs, _ = env.reset(seed=seed + n_episodes + 1)
                n_episodes += 1
                ep_return = 0.0
            obs = _flatten_obs(next_obs)

        # ---- train classifiers + policy on buffered samples ----
        for _ in range(N_UPDATES_PER_ROUND):
            sample = buf.sample_with_horizon(BATCH_SIZE, H_HORIZON, rng)
            if sample is None:
                break
            idx, labels = sample
            o_np = buf.obs[idx]
            a_np = buf.act[idx]
            o_t = torch.from_numpy(o_np).to(device)
            a_t = torch.from_numpy(a_np).long().to(device)
            y_t = torch.from_numpy(labels).to(device)
            a_oh = F.one_hot(a_t, num_classes=n_actions).float()

            # psi_full BCE
            p_full = psi_full(o_t).clamp(1e-6, 1 - 1e-6)
            loss_full = F.binary_cross_entropy(p_full, y_t)
            opt_full.zero_grad()
            loss_full.backward()
            opt_full.step()

            # psi_a BCE
            p_a = psi_a(o_t, a_oh).clamp(1e-6, 1 - 1e-6)
            loss_a = F.binary_cross_entropy(p_a, y_t)
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()

            # ---- policy update via BC toward shifted-logit target ----
            # For each sampled obs, compute S[o, a', m] across all actions
            # using current psi heads, derive Pareto vote, build target.
            with torch.no_grad():
                p_full_eval = psi_full(o_t)  # [B, k]
                # tile obs for each action
                B = o_t.shape[0]
                o_rep = o_t.unsqueeze(1).expand(-1, n_actions, -1).reshape(B * n_actions, -1)
                a_all = torch.arange(n_actions, device=device).repeat(B)
                a_all_oh = F.one_hot(a_all, num_classes=n_actions).float()
                p_a_all = psi_a(o_rep, a_all_oh).reshape(B, n_actions, k)
                S = p_a_all - p_full_eval.unsqueeze(1)  # [B, n_actions, k]

                # Pareto vote per obs (numpy)
                S_np = S.cpu().numpy()
                votes = np.zeros((B, n_actions), dtype=np.float32)
                for bi in range(B):
                    votes[bi] = pareto_dom_minus_sub(S_np[bi], eps=EPS_NOISE)
                    # diagnostic: k_eff
                    spread = S_np[bi].max(axis=0) - S_np[bi].min(axis=0)
                    keff_history.append(int((spread > EPS_NOISE).sum()))

                base_logits = policy(o_t)
                shifted = base_logits + ALPHA_NUDGE * torch.from_numpy(
                    votes / max(1, n_actions)
                ).to(device)
                target_logp = F.log_softmax(shifted, dim=-1)

            cur_logits = policy(o_t)
            cur_logp = F.log_softmax(cur_logits, dim=-1)
            # BC: cross-entropy between target distribution and current policy
            target_p = target_logp.exp()
            policy_loss = -(target_p * cur_logp).sum(dim=-1).mean()
            opt_pi.zero_grad()
            policy_loss.backward()
            opt_pi.step()
            n_updates += 1

    env.close()

    # ---- build evaluation policy ----
    def eval_policy(obs_np):
        of = _flatten_obs(obs_np)
        with torch.no_grad():
            t = torch.from_numpy(of).float().to(device).unsqueeze(0)
            logits = policy(t).cpu().numpy()[0]
        if EVAL_GREEDY:
            return int(np.argmax(logits))
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        if not np.all(np.isfinite(p)) or p.sum() <= 0:
            return int(rng.integers(n_actions))
        return int(rng.choice(n_actions, p=p))

    keff_mean = float(np.mean(keff_history)) if keff_history else 0.0
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} "
        f"updates={n_updates} eps={n_episodes} k={k} keff_mean={keff_mean:.2f} "
        f"train_s={time.monotonic() - t0:.1f} budget_s={time_budget_s}",
        flush=True,
    )
    return eval_policy


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
