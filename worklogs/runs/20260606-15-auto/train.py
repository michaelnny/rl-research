"""PCGA: Per-Channel Gradient Alignment.

Hypothesis 20260606-15-auto. Realization:
    - Shared trunk + categorical policy head + k-output auxiliary cumulant head.
    - At each decision step, compute, for each (action a, channel m), the cosine
      similarity in trunk-parameter space between
            g_pi[a]   = ∇_{θ_trunk} log π(a | s)
            g_cum[m]  = ∇_{θ_trunk} ĉ_m(s)
      to populate A[a, m] ∈ R^{|A| x k}.
    - Coordinate-wise Pareto vote across actions:
          n_a = #{a' != a : A[a, :] ≻ A[a', :]}
          m_a = #{a' != a : A[a', :] ≻ A[a, :]}
      Logit nudge: Δ logit(a) = α (n_a - m_a). Sample categorically from softmax.
    - Auxiliary head trained by MSE on observed suffix cumulants
      s_t = Σ_{τ=1..W} v_{t+τ} from a replay buffer, separate optimizer.
    - Policy SGD step on standard log-prob update over a small replay window
      (REINFORCE-style with whitened return) so the trunk evolves; the cumulant
      head's *output* is never used to weight the policy update — only the
      direction of its parameter-space gradient enters at decision time.

Vector envs: each component of info["vector"] is a separate channel; channels
never summed/projected. Scalar envs: reward is treated as a single channel.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# -------- networks --------

class Trunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class PCGAModel(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, k_channels: int, hidden: int = 64):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden)
        self.policy_head = nn.Linear(hidden, n_actions)
        self.cumulant_head = nn.Linear(hidden, k_channels)
        self.n_actions = n_actions
        self.k_channels = k_channels

    def features(self, x):
        return self.trunk(x)

    def logits(self, x):
        return self.policy_head(self.trunk(x))

    def cumulants(self, x):
        return self.cumulant_head(self.trunk(x))


# -------- observation flattening --------

def flatten_obs(obs):
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


# -------- pareto comparisons --------

def pareto_counts(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given A[|A|, k], return (n_a, m_a) per action under coordinate-wise partial order.

    a dominates a' iff all-greater-or-equal AND any-strictly-greater.
    n_a = #{a' : a dominates a'};  m_a = #{a' : a' dominates a}.
    """
    nA = A.shape[0]
    n = np.zeros(nA, dtype=np.float32)
    m = np.zeros(nA, dtype=np.float32)
    for i in range(nA):
        for j in range(nA):
            if i == j:
                continue
            ge = np.all(A[i] >= A[j] - 1e-12)
            gt_any = np.any(A[i] > A[j] + 1e-12)
            if ge and gt_any:
                n[i] += 1
            le = np.all(A[j] >= A[i] - 1e-12)
            lt_any = np.any(A[j] > A[i] + 1e-12)
            if le and lt_any:
                m[i] += 1
    return n, m


# -------- gradient alignment --------

def trunk_grad_vec(loss: torch.Tensor, trunk_params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Backward `loss` w.r.t. trunk_params; return flat gradient vector. Does not retain graph beyond this call."""
    grads = torch.autograd.grad(
        loss, trunk_params, retain_graph=True, create_graph=False, allow_unused=False
    )
    return torch.cat([g.reshape(-1) for g in grads])


def compute_alignment_matrix(
    model: PCGAModel,
    obs_t: torch.Tensor,
    trunk_params: list[torch.nn.Parameter],
) -> np.ndarray:
    """Returns A[|A|, k] cosine similarities in trunk parameter space."""
    nA = model.n_actions
    k = model.k_channels

    # Forward once with a graph that supports both heads w.r.t. trunk params.
    feats = model.trunk(obs_t)  # (hidden,)
    logits = model.policy_head(feats)  # (nA,)
    log_probs = F.log_softmax(logits, dim=-1)  # (nA,)
    cumulants = model.cumulant_head(feats)  # (k,)

    # Collect trunk gradients per action and per channel.
    pi_grads = []
    for a in range(nA):
        g = torch.autograd.grad(
            log_probs[a], trunk_params, retain_graph=True, create_graph=False
        )
        pi_grads.append(torch.cat([gi.reshape(-1) for gi in g]))
    cum_grads = []
    for mi in range(k):
        retain = mi < (k - 1)
        g = torch.autograd.grad(
            cumulants[mi], trunk_params, retain_graph=retain, create_graph=False
        )
        cum_grads.append(torch.cat([gi.reshape(-1) for gi in g]))

    pi_mat = torch.stack(pi_grads, dim=0)   # (nA, P)
    cum_mat = torch.stack(cum_grads, dim=0) # (k,  P)

    pi_norm = pi_mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cum_norm = cum_mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pi_unit = pi_mat / pi_norm
    cum_unit = cum_mat / cum_norm
    A = pi_unit @ cum_unit.t()  # (nA, k)
    return A.detach().cpu().numpy()


# -------- training --------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")  # keep CPU; per-decision cost is small

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    obs0, _ = env.reset(seed=seed)
    obs0_flat = flatten_obs(obs0)
    in_dim = obs0_flat.shape[0]

    # Action space discrete
    action_space = env.action_space
    n_actions = int(action_space.n)

    # Determine number of channels k from a probe step.
    probe_action = 0
    try:
        _, _, _, _, info = env.step(probe_action)
    except Exception:
        info = {}
    if is_vector and "vector" in info:
        k_channels = int(np.asarray(info["vector"]).shape[0])
    else:
        k_channels = 1

    # Reset env again to start training cleanly.
    obs, _ = env.reset(seed=seed)

    model = PCGAModel(in_dim, n_actions, k_channels, hidden=64).to(device)
    trunk_params = list(model.trunk.parameters())

    pi_optim = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.policy_head.parameters()), lr=3e-3
    )
    cum_optim = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.cumulant_head.parameters()), lr=3e-3
    )

    alpha = 1.0   # logit nudge scale
    W = 8         # cumulant-suffix window
    pi_update_every = 32
    cum_update_every = 8
    cum_batch = 64
    pi_max_traj = 256

    # Replay buffers
    replay_obs: deque = deque(maxlen=4096)
    replay_cum_target: deque = deque(maxlen=4096)
    # For trajectories (policy update), track tuples of (obs_flat, action, scalar_reward)
    traj_obs: list[np.ndarray] = []
    traj_actions: list[int] = []
    traj_rewards: list[float] = []
    # Per-step buffer that we'll convert to suffix targets when we have enough trailing steps.
    pending_obs: deque = deque(maxlen=W + 4)
    pending_vec: deque = deque(maxlen=W + 4)

    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave headroom for eval
    env_steps = 0
    decisions_with_pareto = 0
    align_var_acc = 0.0
    align_var_count = 0

    def pop_suffix_targets():
        """When pending has at least W entries past the front, push (obs, suffix_sum) to replay."""
        while len(pending_vec) > W:
            o0 = pending_obs.popleft()
            pending_vec.popleft()  # discard the v at the front (we want suffix AFTER step t)
            # Build suffix from current contents (next W vectors)
            tail = list(pending_vec)[:W]
            if len(tail) < W:
                # restore - shouldn't happen given the while condition
                pending_obs.appendleft(o0)
                break
            target = np.sum(np.stack(tail, axis=0), axis=0)
            replay_obs.append(o0)
            replay_cum_target.append(target.astype(np.float32))

    def cum_sgd_step():
        if len(replay_obs) < 8:
            return
        idx = np.random.randint(0, len(replay_obs), size=min(cum_batch, len(replay_obs)))
        obs_batch = np.stack([replay_obs[i] for i in idx], axis=0)
        tgt_batch = np.stack([replay_cum_target[i] for i in idx], axis=0)
        ob_t = torch.from_numpy(obs_batch).to(device)
        tg_t = torch.from_numpy(tgt_batch).to(device)
        pred = model.cumulants(ob_t)
        # Per-channel MSE summed (channels independent; do not weight)
        loss = F.mse_loss(pred, tg_t)
        cum_optim.zero_grad()
        loss.backward()
        cum_optim.step()

    def pi_sgd_step():
        if len(traj_rewards) < 4:
            return
        # Compute reward-to-go (whitened) for trajectory we have so far.
        obs_arr = np.stack(traj_obs[-pi_max_traj:], axis=0)
        act_arr = np.asarray(traj_actions[-pi_max_traj:], dtype=np.int64)
        rew_arr = np.asarray(traj_rewards[-pi_max_traj:], dtype=np.float32)
        # Discounted return
        gamma = 0.99
        G = np.zeros_like(rew_arr)
        running = 0.0
        for i in range(len(rew_arr) - 1, -1, -1):
            running = rew_arr[i] + gamma * running
            G[i] = running
        # whiten
        if G.std() > 1e-8:
            G = (G - G.mean()) / (G.std() + 1e-8)
        else:
            G = G - G.mean()
        ob_t = torch.from_numpy(obs_arr).to(device)
        ac_t = torch.from_numpy(act_arr).to(device)
        ad_t = torch.from_numpy(G).to(device)
        logits = model.logits(ob_t)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(1, ac_t.unsqueeze(1)).squeeze(1)
        loss = -(chosen * ad_t).mean()
        # small entropy bonus
        ent = -(F.softmax(logits, dim=-1) * logp).sum(dim=-1).mean()
        loss = loss - 1e-3 * ent
        pi_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.trunk.parameters()) + list(model.policy_head.parameters()), 5.0
        )
        pi_optim.step()

    obs_flat = flatten_obs(obs)

    while time.monotonic() < deadline:
        # ---- decision: compute alignment and apply Pareto-vote logit nudge ----
        ob_t = torch.from_numpy(obs_flat).to(device)
        with torch.no_grad():
            base_logits = model.logits(ob_t).cpu().numpy()
        # Need gradients => not under no_grad.
        try:
            A = compute_alignment_matrix(model, ob_t, trunk_params)
        except RuntimeError:
            A = np.zeros((n_actions, k_channels), dtype=np.float32)
        n_dom, m_dom = pareto_counts(A)
        # Diagnostic: variance across actions of A entries.
        if A.size > 0:
            v = float(A.var(axis=0).mean())
            align_var_acc += v
            align_var_count += 1
            if (n_dom + m_dom).max() > 0:
                decisions_with_pareto += 1
        nudged_logits = base_logits + alpha * (n_dom - m_dom)
        # softmax sample
        z = nudged_logits - nudged_logits.max()
        p = np.exp(z)
        p /= p.sum()
        action = int(np.random.choice(n_actions, p=p))

        # ---- step env ----
        obs_next, reward, term, trunc, info = env.step(action)
        obs_next_flat = flatten_obs(obs_next)
        env_steps += 1

        if is_vector and "vector" in info:
            v_vec = np.asarray(info["vector"], dtype=np.float32)
            if v_vec.shape[0] != k_channels:
                # pad/truncate
                tmp = np.zeros(k_channels, dtype=np.float32)
                m = min(k_channels, v_vec.shape[0])
                tmp[:m] = v_vec[:m]
                v_vec = tmp
        else:
            v_vec = np.array([float(reward)], dtype=np.float32)

        traj_obs.append(obs_flat)
        traj_actions.append(action)
        traj_rewards.append(float(reward))

        pending_obs.append(obs_flat)
        pending_vec.append(v_vec)
        pop_suffix_targets()

        # ---- update auxiliary head occasionally ----
        if env_steps % cum_update_every == 0:
            cum_sgd_step()
        # ---- update policy occasionally ----
        if env_steps % pi_update_every == 0:
            pi_sgd_step()
            # cap traj memory
            if len(traj_obs) > 4 * pi_max_traj:
                traj_obs[:] = traj_obs[-pi_max_traj:]
                traj_actions[:] = traj_actions[-pi_max_traj:]
                traj_rewards[:] = traj_rewards[-pi_max_traj:]

        if term or trunc:
            obs, _ = env.reset()
            obs_flat = flatten_obs(obs)
            # flush pending: consume what suffixes we can
            pop_suffix_targets()
            # don't drop: pending may still hold partial; keep for next ep
        else:
            obs_flat = obs_next_flat

    # Final updates
    pi_sgd_step()
    cum_sgd_step()

    env.close()

    # Diagnostics line
    avg_var = align_var_acc / max(1, align_var_count)
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"train_s={time.monotonic() - t_start:.1f} budget_s={time_budget_s} "
        f"k={k_channels} avg_align_var={avg_var:.4f} "
        f"frac_pareto_decisions={decisions_with_pareto / max(1, align_var_count):.3f}",
        flush=True,
    )

    # Frozen policy: greedy on nudged logits at decision time.
    model.eval()
    rng = np.random.default_rng(seed + 7)

    def policy_fn(obs_eval: np.ndarray):
        ob_flat = flatten_obs(obs_eval)
        ob_t = torch.from_numpy(ob_flat)
        with torch.no_grad():
            base_logits = model.logits(ob_t).cpu().numpy()
        try:
            A_eval = compute_alignment_matrix(model, ob_t, trunk_params)
            n_e, m_e = pareto_counts(A_eval)
            nudged = base_logits + alpha * (n_e - m_e)
        except RuntimeError:
            nudged = base_logits
        # Stochastic sampling from softmax (categorical), as the hypothesis specifies.
        z = nudged - nudged.max()
        p = np.exp(z)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return int(rng.integers(n_actions))
        p /= s
        return int(rng.choice(n_actions, p=p))

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
