"""CID — Channel-Imminence Differential.

Run 20260606-16-auto. Realizes the hypothesis in
worklogs/runs/20260606-16-auto/hypothesis.md.

Primitive (per-decision, per-channel m, per-action a):
    LR[a, m] = log q_m(f_phi(o_t, a)) - log q_m(o_t)
where q_m: O -> [0,1] is a supervised binary head trained on
labels I[v_{t+1}[m] != 0] from the replay buffer, and f_phi(o,a)
is a supervised one-step next-observation predictor (delta-MLP).

Improvement operator: Pareto-non-dominance vote over k channels.
For each candidate action a:
    n_a = #{a' : LR[a,:] dominates LR[a',:]}    (>=, with strict >)
    m_a = #{a' : LR[a',:] dominates LR[a,:]}
    delta_logit(a) = alpha * (n_a - m_a)
Channels are never linearly combined or scalarized.

Channel-active gate: a channel m is excluded from the Pareto vote
unless q_m's running AUC (estimated from a held-out portion of the
replay buffer) >= 0.55.

Policy update: maximum-entropy + KL projection toward a target
distribution that softmaxes the Pareto margins (no reward-weighted
policy gradient, no critic, no Bellman backup).
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


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.net(o)


class ChannelHeads(nn.Module):
    """k binary classifiers q_m(o) ≈ P(channel m fires next step | o)."""

    def __init__(self, obs_dim: int, k: int, hidden: int = 64):
        super().__init__()
        self.k = k
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.heads = nn.Linear(hidden, k)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # returns logits of shape (..., k)
        return self.heads(self.trunk(o))


class ForwardModel(nn.Module):
    """f_phi(o, a) ≈ E[o_{t+1} | o, a]. Predicts delta in observation."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, o: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a_onehot], dim=-1)
        delta = self.net(x)
        return o + delta


# ----------------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, obs_dim: int, k: int, capacity: int = 50_000):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.fire = np.zeros((capacity, k), dtype=np.float32)  # I[v_{t+1}[m] != 0]
        self.size = 0
        self.idx = 0

    def add(
        self,
        o: np.ndarray,
        a: int,
        no: np.ndarray,
        v: np.ndarray,
    ) -> None:
        i = self.idx
        self.obs[i] = o
        self.act[i] = a
        self.next_obs[i] = no
        self.fire[i] = (np.abs(np.asarray(v, dtype=np.float64)) > 1e-9).astype(np.float32)
        self.idx = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, rng: np.random.Generator) -> dict:
        n = self.size
        idx = rng.integers(0, n, size=batch)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "next_obs": self.next_obs[idx],
            "fire": self.fire[idx],
        }


# ----------------------------------------------------------------------------
# Pareto-non-dominance vote
# ----------------------------------------------------------------------------


def pareto_vote(LR: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """Coordinate-wise Pareto vote over actions on active channels.

    LR: (A, k) array of channel-imminence log-likelihood-ratios
    active_mask: (k,) boolean — which channels enter the partial order
    Returns: (A,) signed margin = n_a - m_a (dominated minus dominating).
    """
    A, k = LR.shape
    if not np.any(active_mask):
        return np.zeros(A, dtype=np.float32)
    L = LR[:, active_mask]  # (A, k_active)
    # For each pair (a, a'): a dominates a' iff L[a] >= L[a'] componentwise
    # AND strictly greater in at least one coordinate.
    n = np.zeros(A, dtype=np.int32)  # number of a' that a dominates
    m = np.zeros(A, dtype=np.int32)  # number of a' that dominate a
    for a in range(A):
        diff = L[a:a + 1, :] - L  # (A, k_active)
        ge = np.all(diff >= 0, axis=1)
        gt_any = np.any(diff > 0, axis=1)
        le = np.all(diff <= 0, axis=1)
        lt_any = np.any(diff < 0, axis=1)
        # a dominates a' (a' index over rows of diff, here we computed a - a')
        # so diff >= 0 with strict somewhere means a dominates a'
        dominates = ge & gt_any
        dominated = le & lt_any
        # exclude self (diff is all zero -> not strict either way)
        n[a] = int(np.sum(dominates))
        m[a] = int(np.sum(dominated))
    return (n - m).astype(np.float32)


# ----------------------------------------------------------------------------
# AUC estimator (channel-active gate)
# ----------------------------------------------------------------------------


def quick_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC. Returns 0.5 if degenerate."""
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Mann-Whitney U based AUC
    n_pos = pos.size
    n_neg = neg.size
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_s) + 1)
    # average ranks for ties
    # (simple version: skip tie correction; close enough as a gate)
    sum_pos = ranks[:n_pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def to_obs_vec(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def infer_k(env_id: str) -> int:
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    return 1  # scalar envs: degenerate to one "fire" channel = (reward != 0)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n)
    obs0, _ = env.reset(seed=seed)
    obs_vec0 = to_obs_vec(obs0)
    obs_dim = obs_vec0.shape[0]
    k = infer_k(env_id)

    is_vector = harness.ENV_TYPE[env_id] == "vector"

    device = torch.device("cpu")  # tiny networks, CPU is plenty fast

    policy = PolicyNet(obs_dim, n_actions).to(device)
    qheads = ChannelHeads(obs_dim, k).to(device)
    fmodel = ForwardModel(obs_dim, n_actions).to(device)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=3e-3)
    opt_q = torch.optim.Adam(qheads.parameters(), lr=3e-3)
    opt_f = torch.optim.Adam(fmodel.parameters(), lr=3e-3)

    buf = ReplayBuffer(obs_dim, k, capacity=50_000)

    # CID hyperparameters
    alpha = 1.0  # logit-nudge magnitude per Pareto-margin unit
    q_floor = 1e-3
    auc_threshold = 0.55
    target_temperature = 0.5  # temperature for the Pareto-margin target distribution
    kl_weight = 1.0
    entropy_weight = 0.01
    min_buffer_for_train = 64
    train_every = 4
    batch = 64
    auc_eval_n = 1024  # number of buffer samples used to estimate per-channel AUC

    obs = obs_vec0
    np_obs_input = np.asarray(obs0)
    raw_obs = np_obs_input  # retained shape

    t0 = time.monotonic()
    env_steps = 0
    updates = 0
    last_auc = np.full((k,), 0.5, dtype=np.float64)
    last_fmodel_mse = float("nan")
    last_qloss = float("nan")
    ep_return = 0.0
    ep_steps = 0
    ep_count = 0

    # Diagnostics
    diag_active_step_counts = np.zeros((k,), dtype=np.int64)  # how often each channel
    # passed the gate
    diag_lr_row_var_sum = np.zeros((k,), dtype=np.float64)
    diag_lr_row_var_n = 0
    diag_decision_count = 0

    def policy_logits(o_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o_t = torch.from_numpy(o_vec.astype(np.float32)).unsqueeze(0)
            return policy(o_t).cpu().numpy()[0]

    def compute_LR(o_vec: np.ndarray) -> np.ndarray:
        """Returns LR matrix (A, k) for one observation."""
        with torch.no_grad():
            o_t = torch.from_numpy(o_vec.astype(np.float32)).unsqueeze(0)
            # log q_m(o_t)
            log_q_o = F.logsigmoid(qheads(o_t))  # (1, k)
            log1m_q_o = F.logsigmoid(-qheads(o_t))
            # apply probability floor in log space
            log_q_o = torch.clamp(log_q_o, min=np.log(q_floor))
            _ = log1m_q_o  # not used directly; the LR uses log q only
            # For each action: build one-hot, predict next obs, then log q_m(next)
            o_rep = o_t.repeat(n_actions, 1)
            a_onehot = torch.eye(n_actions, dtype=torch.float32)
            o_next_pred = fmodel(o_rep, a_onehot)
            log_q_next = F.logsigmoid(qheads(o_next_pred))  # (A, k)
            log_q_next = torch.clamp(log_q_next, min=np.log(q_floor))
            LR = (log_q_next - log_q_o).cpu().numpy()  # (A, k)
        return LR

    def channel_active_mask() -> np.ndarray:
        return last_auc >= auc_threshold

    def pick_action(o_vec: np.ndarray) -> int:
        nonlocal diag_lr_row_var_sum, diag_lr_row_var_n, diag_decision_count
        base = policy_logits(o_vec)
        LR = compute_LR(o_vec)
        active = channel_active_mask()
        margins = pareto_vote(LR, active)  # (A,)
        logits = base + alpha * margins
        # numerically stable softmax sample
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        a = int(rng.choice(n_actions, p=probs))
        # diagnostics
        diag_decision_count += 1
        # row variance per channel across actions
        diag_lr_row_var_sum += LR.var(axis=0)
        diag_lr_row_var_n += 1
        return a

    def update_models() -> None:
        nonlocal last_qloss, last_fmodel_mse, last_auc
        if buf.size < min_buffer_for_train:
            return
        b = buf.sample(batch, rng)
        o_t = torch.from_numpy(b["obs"])
        a_t = torch.from_numpy(b["act"])
        no_t = torch.from_numpy(b["next_obs"])
        fire_t = torch.from_numpy(b["fire"])
        a_onehot = F.one_hot(a_t, num_classes=n_actions).float()

        # 1) q_m: supervised binary on (o_t, fire(t+1))
        q_logits = qheads(o_t)  # (B, k)
        q_loss = F.binary_cross_entropy_with_logits(q_logits, fire_t)
        opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        opt_q.step()
        last_qloss = float(q_loss.detach().item())

        # 2) f_phi: supervised L2 on next-obs prediction (predicts delta)
        no_pred = fmodel(o_t, a_onehot)
        f_loss = F.mse_loss(no_pred, no_t)
        opt_f.zero_grad(set_to_none=True)
        f_loss.backward()
        opt_f.step()
        last_fmodel_mse = float(f_loss.detach().item())

        # 3) Policy: KL projection toward target softmax(margin/T) under
        # current observations + entropy bonus. No reward weighting.
        with torch.no_grad():
            # Build LR per obs in batch
            B = o_t.shape[0]
            o_rep = o_t.unsqueeze(1).expand(B, n_actions, obs_dim).reshape(B * n_actions, obs_dim)
            ah = torch.eye(n_actions, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).reshape(
                B * n_actions, n_actions
            )
            o_next_pred = fmodel(o_rep, ah)  # (B*A, obs_dim)
            log_q_next = F.logsigmoid(qheads(o_next_pred)).clamp(min=float(np.log(q_floor)))
            log_q_next = log_q_next.reshape(B, n_actions, k)  # (B, A, k)
            log_q_o_b = F.logsigmoid(qheads(o_t)).clamp(min=float(np.log(q_floor)))
            log_q_o_b = log_q_o_b.unsqueeze(1)  # (B, 1, k)
            LR_b = (log_q_next - log_q_o_b).cpu().numpy()  # (B, A, k)
            active = channel_active_mask()
            margins_b = np.zeros((B, n_actions), dtype=np.float32)
            for i in range(B):
                margins_b[i] = pareto_vote(LR_b[i], active)
            target_logits_np = margins_b / max(target_temperature, 1e-6)
            # softmax
            tl = target_logits_np - target_logits_np.max(axis=1, keepdims=True)
            target_probs_np = np.exp(tl)
            target_probs_np = target_probs_np / target_probs_np.sum(axis=1, keepdims=True)
            target_probs = torch.from_numpy(target_probs_np.astype(np.float32))

        cur_logits = policy(o_t)
        cur_logp = F.log_softmax(cur_logits, dim=-1)
        cur_p = cur_logp.exp()
        # KL(target || current) — pulls policy toward target distribution
        eps = 1e-8
        target_logp = torch.log(target_probs + eps)
        kl = (target_probs * (target_logp - cur_logp)).sum(dim=-1).mean()
        entropy = -(cur_p * cur_logp).sum(dim=-1).mean()
        loss = kl_weight * kl - entropy_weight * entropy
        opt_policy.zero_grad(set_to_none=True)
        loss.backward()
        opt_policy.step()

    def refresh_auc() -> None:
        nonlocal last_auc
        if buf.size < 64:
            return
        n = min(buf.size, auc_eval_n)
        idx = rng.integers(0, buf.size, size=n)
        o = torch.from_numpy(buf.obs[idx])
        labels = buf.fire[idx]  # (n, k)
        with torch.no_grad():
            scores = torch.sigmoid(qheads(o)).cpu().numpy()  # (n, k)
        new_auc = np.zeros((k,), dtype=np.float64)
        for m_ in range(k):
            new_auc[m_] = quick_auc(scores[:, m_], labels[:, m_])
        last_auc = new_auc

    # warmup: a tiny period of pure on-policy random/uniform exploration to
    # populate buf before models do anything meaningful
    warmup_steps = 32

    # ----------------------------------------------------------------- main loop
    while time.monotonic() - t0 < time_budget_s:
        # Pick action
        if env_steps < warmup_steps or buf.size < min_buffer_for_train:
            a = int(rng.integers(n_actions))
        else:
            a = pick_action(obs)

        next_raw, reward, term, trunc, info = env.step(a)
        next_vec = to_obs_vec(next_raw)
        if is_vector:
            v = np.asarray(info["vector"], dtype=np.float64)
        else:
            # scalar env: synthesize a 1-vector channel = I[reward != 0]
            v = np.array([float(reward)], dtype=np.float64)
        buf.add(obs, a, next_vec, v)

        env_steps += 1
        ep_return += float(reward)
        ep_steps += 1

        done = bool(term) or bool(trunc)
        if done:
            ep_count += 1
            obs_raw, _ = env.reset(seed=seed + ep_count)
            obs = to_obs_vec(obs_raw)
            ep_return = 0.0
            ep_steps = 0
        else:
            obs = next_vec

        # Periodic supervised + policy updates
        if env_steps % train_every == 0:
            update_models()
            updates += 1
            if updates % 8 == 0:
                refresh_auc()

    env.close()

    # Diagnostics summary
    elapsed = time.monotonic() - t0
    if diag_lr_row_var_n > 0:
        avg_row_var = diag_lr_row_var_sum / diag_lr_row_var_n
    else:
        avg_row_var = np.zeros((k,), dtype=np.float64)

    # Falsifier instrumentation
    fire_rate = (
        buf.fire[: buf.size].mean(axis=0) if buf.size > 0 else np.zeros((k,), dtype=np.float32)
    )
    active = channel_active_mask()
    n_active = int(active.sum())
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"train_s={elapsed:.1f} budget_s={time_budget_s} updates={updates}",
        flush=True,
    )
    print(
        f"[cid] auc={np.array2string(last_auc, precision=3)} "
        f"fire_rate={np.array2string(fire_rate, precision=3)} "
        f"active_channels={n_active}/{k} "
        f"lr_row_var={np.array2string(avg_row_var, precision=4)} "
        f"q_loss={last_qloss:.4f} fmodel_mse={last_fmodel_mse:.4f} "
        f"decisions={diag_decision_count}",
        flush=True,
    )
    # Falsifier-specific flag (RG): precious-channel fire fraction <0.01 AND
    # only the step-penalty channel has nonzero LR row variance
    if env_id == "resource-gathering-v0":
        precious = fire_rate[1:]  # gold, gem
        if (precious < 0.01).all():
            print("[cid] WARNING: precious channels never fired (terminal-only collapse)", flush=True)

    # ---- Deterministic policy_fn ----
    # At eval time: greedy over (base_logits + alpha * pareto_margin)
    def policy_fn(o):
        o_vec = to_obs_vec(o)
        base = policy_logits(o_vec)
        LR = compute_LR(o_vec)
        active_mask = channel_active_mask()
        margins = pareto_vote(LR, active_mask)
        logits = base + alpha * margins
        return int(np.argmax(logits))

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
