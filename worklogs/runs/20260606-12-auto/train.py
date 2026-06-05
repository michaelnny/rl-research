"""JFP: Jacobian Firing-Phase candidate (run 20260606-12-auto).

Primitive: per-(state, action, channel) argmax-timestep of the gradient
norm of a learned forward model's cumulant head along an H-step
counterfactual rollout, with a one-shot intervention of action `a` at
step 0 and subsequent actions sampled from the current policy pi_theta.

Improvement operator: Pareto-non-dominance count over the per-channel
firing-phase vector J(s,a) in R^k. Logit nudge:
    Delta_a = alpha * (n_a - m_a)
where n_a = #{a' : J(s,a) dominates J(s,a')}
and   m_a = #{a' : J(s,a') dominates J(s,a)}.
"Dominates" means coordinate-wise <= with at least one strict <.
(Earlier-peaking timing dominates later-peaking timing.)

No critic, no Bellman backup, no scalar advantage, no scalarized vector
reward. The forward model and its cumulant head are trained by ordinary
supervised prediction losses; the policy's only learning signal is the
Pareto nudge.

Substrate contract:
    uv run train.py --env ENV --seed S --time-budget-s T
"""

from __future__ import annotations

import argparse
import math
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


# -------- observation flattening helpers --------

def _flatten_obs(obs):
    a = np.asarray(obs)
    return a.astype(np.float32).reshape(-1)


def _obs_dim(env) -> int:
    obs, _ = env.reset()
    return int(_flatten_obs(obs).shape[0])


# -------- forward-model + cumulant head --------

class ForwardModel(nn.Module):
    """f_theta: (o_t, e(a_t)) -> o_{t+1}; cumulant head g: o -> c in R^k."""

    def __init__(self, obs_dim: int, n_actions: int, k_channels: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.k = k_channels
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.delta_head = nn.Linear(hidden, obs_dim)
        # cumulant head reads next-state latent (we re-encode obs through trunk
        # using a zero action embedding to define a state encoder reused here).
        self.cumulant_head = nn.Linear(obs_dim, k_channels)

    def step(self, o: torch.Tensor, e_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step forward: returns (o_next_pred, c_next_pred).

        o:   (B, obs_dim)
        e_a: (B, n_actions)  -- continuous one-hot or relaxed embedding
        """
        x = torch.cat([o, e_a], dim=-1)
        h = self.trunk(x)
        delta = self.delta_head(h)
        o_next = o + delta
        c_next = self.cumulant_head(o_next)
        return o_next, c_next


# -------- policy --------

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


# -------- replay buffer --------

class Replay:
    def __init__(self, capacity: int, obs_dim: int, k_channels: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.vec = np.zeros((capacity, k_channels), dtype=np.float32)
        self.size = 0
        self.idx = 0

    def add(self, o, a, o2, v):
        i = self.idx
        self.obs[i] = o
        self.act[i] = a
        self.next_obs[i] = o2
        self.vec[i] = v
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch)
        return (
            self.obs[idx],
            self.act[idx],
            self.next_obs[idx],
            self.vec[idx],
        )


# -------- vector channel adapters --------

def vector_dim_for_env(env_id: str, env) -> int:
    """Take a real environment step to determine vector dimension."""
    if harness.ENV_TYPE[env_id] != "vector":
        # We still want a vector signal for cumulant supervision on scalar envs.
        # Use the scalar reward as a 1-D channel.
        return 1
    # Probe by stepping once.
    obs, _ = env.reset()
    a = 0 if hasattr(env.action_space, "n") else env.action_space.sample()
    _, _, _, _, info = env.step(a)
    v = info.get("vector")
    return int(np.asarray(v).reshape(-1).shape[0])


def channel_signal(reward: float, info: dict, k: int, is_vector: bool) -> np.ndarray:
    if is_vector:
        v = np.asarray(info.get("vector"), dtype=np.float32).reshape(-1)
        if v.shape[0] != k:
            v2 = np.zeros((k,), dtype=np.float32)
            n = min(v.shape[0], k)
            v2[:n] = v[:n]
            return v2
        return v
    return np.asarray([float(reward)], dtype=np.float32)


# -------- JFP timing computation --------

def compute_J(
    fm: ForwardModel,
    policy: PolicyNet,
    o0: torch.Tensor,
    n_actions: int,
    H: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute J(s,a) for every action a at state o0.

    Returns a tensor of shape (n_actions, k_channels) with values in
    [1, H], where J[a, m] = argmax_{1<=tau<=H} ||d c_tau[m] / d e(a)||_2.

    The intervention: at step 0 the action embedding is e(a) (one-hot,
    differentiable). At subsequent steps we sample a one-hot from
    pi_theta(.|o_tau). To keep gradients flowing through the unrolled
    forward model w.r.t. e(a), the sampling at later steps uses
    detached, hard one-hot embeddings (the gradient pathway we measure
    is the one-shot intervention at t=0; the policy at t>=1 acts as the
    rollout schedule, not as a differentiable variable).
    """
    k = fm.k
    fm.eval()
    policy.eval()

    # All actions in a batch dimension. We unroll H steps and at each
    # tau collect c_tau, then compute grad norms of c_tau[:, m] w.r.t. e(a).
    # We do this per channel because grad needs a scalar (sum over batch).
    # For efficiency: process per channel, summing across the batch is fine
    # because each batch row's grad lives only in its own e_a slice (the
    # forward model is applied independently per row).

    grad_norms = torch.zeros((n_actions, H, k), device=device)

    for m in range(k):
        # Build a fresh differentiable e_a for the head action of each row.
        e_a = torch.eye(n_actions, device=device, requires_grad=True)
        # state batch: replicate o0 across actions.
        o = o0.unsqueeze(0).expand(n_actions, -1).contiguous()
        # We must record gradients of c_tau[:, m] w.r.t. e_a per tau.
        # To avoid retaining the graph H times, we compute c per step and
        # use grad with retain_graph=True until the last step.
        c_list = []
        cur_e = e_a
        for tau in range(H):
            o, c = fm.step(o, cur_e)
            c_list.append(c)
            if tau < H - 1:
                # sample next action one-hot from policy (no grad pathway)
                with torch.no_grad():
                    logits = policy(o)
                    probs = F.softmax(logits, dim=-1)
                    # categorical sample
                    a_next = torch.multinomial(probs, num_samples=1).squeeze(-1)
                cur_e = F.one_hot(a_next, num_classes=n_actions).float()

        for tau in range(H):
            c_tau_m = c_list[tau][:, m]  # (n_actions,)
            # Per-row gradient of c_tau_m[i] w.r.t. e_a[i,:].
            # Trick: sum c_tau_m gives sum over rows, and grad w.r.t. e_a
            # gives a (n_actions, n_actions) matrix where row i has grad
            # of c_tau_m[i] w.r.t. e_a[i,:] (other entries are zero because
            # row i only feeds row i through fm.step).
            grad = torch.autograd.grad(
                c_tau_m.sum(),
                e_a,
                retain_graph=(tau < H - 1) or (m < k - 1),
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                continue
            # row-wise L2 norm
            gn = grad.norm(dim=-1)  # (n_actions,)
            grad_norms[:, tau, m] = gn

    # argmax over tau (1..H), values are [1..H] (use 1-indexed timesteps)
    J = grad_norms.argmax(dim=1).float() + 1.0  # (n_actions, k)
    return J


def pareto_nudge(J: np.ndarray, alpha: float) -> np.ndarray:
    """Compute logit nudges Delta_a = alpha * (n_a - m_a).

    n_a = #{a' : J[a] dominates J[a']}
    m_a = #{a' : J[a'] dominates J[a]}
    Earlier (smaller) timing dominates: a dominates a' iff J[a] <= J[a']
    coordinate-wise with at least one strict <.
    """
    n_actions = J.shape[0]
    nudges = np.zeros(n_actions, dtype=np.float32)
    for a in range(n_actions):
        n_a = 0
        m_a = 0
        for ap in range(n_actions):
            if a == ap:
                continue
            le = np.all(J[a] <= J[ap])
            lt = np.any(J[a] < J[ap])
            ge = np.all(J[a] >= J[ap])
            gt = np.any(J[a] > J[ap])
            if le and lt:
                n_a += 1
            if ge and gt:
                m_a += 1
        nudges[a] = alpha * (n_a - m_a)
    return nudges


# -------- training --------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t_start = time.monotonic()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 12345)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"
    if not hasattr(env.action_space, "n"):
        # fall back to random for continuous-action envs (none on this panel
        # for vector/sparse, but be safe).
        action_space = env.action_space
        env.close()
        rng2 = np.random.default_rng(seed + 99_999)

        def policy_fn(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} (continuous action: random fallback)",
            flush=True,
        )
        return policy_fn

    n_actions = int(env.action_space.n)
    k_channels = vector_dim_for_env(env_id, env)
    obs_dim = _obs_dim(env)

    # H per spec; choose H based on env footprint
    H = 8
    alpha = 1.0

    fm = ForwardModel(obs_dim, n_actions, k_channels, hidden=64).to(device)
    policy = PolicyNet(obs_dim, n_actions, hidden=64).to(device)

    fm_opt = torch.optim.Adam(fm.parameters(), lr=3e-3)
    pol_opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    replay = Replay(capacity=10_000, obs_dim=obs_dim, k_channels=k_channels)

    # On-policy state
    obs, _ = env.reset(seed=seed)
    o = _flatten_obs(obs)

    env_steps = 0
    grad_steps = 0
    nudge_calls = 0
    fm_loss_running = 0.0
    pareto_constant_count = 0
    last_log_t = t_start

    # We'll keep a small on-policy buffer of (obs, action, logits, nudge)
    # to update policy with a Pareto-nudge-driven cross-entropy:
    # the nudged logits define a target distribution we cross-entropy
    # the policy onto.

    def compute_jfp_nudge(o_np: np.ndarray) -> np.ndarray:
        nonlocal pareto_constant_count, nudge_calls
        o_t = torch.from_numpy(o_np).float().to(device)
        with torch.enable_grad():
            J = compute_J(fm, policy, o_t, n_actions, H, device)
        J_np = J.detach().cpu().numpy()
        nudges = pareto_nudge(J_np, alpha)
        nudge_calls += 1
        if np.allclose(nudges, 0.0):
            pareto_constant_count += 1
        return nudges, J_np

    # warmup with a small batch of random transitions to seed the FM
    warmup_steps = 64
    for _ in range(warmup_steps):
        a = int(rng.integers(0, n_actions))
        obs_next, r, term, trunc, info = env.step(a)
        o_next = _flatten_obs(obs_next)
        v = channel_signal(r, info, k_channels, is_vector)
        replay.add(o, a, o_next, v)
        env_steps += 1
        if term or trunc:
            obs, _ = env.reset(seed=seed + env_steps)
            o = _flatten_obs(obs)
        else:
            o = o_next
        if time.monotonic() - t_start > time_budget_s:
            break

    # --- main loop ---
    fm_batch = 64
    fm_updates_per_step = 2
    nudge_every = 1  # apply nudge at every decision (rollout-cost discipline allows this)

    while time.monotonic() - t_start < time_budget_s:
        # 1. Update forward model from replay (supervised on observed transitions)
        if replay.size >= fm_batch:
            for _ in range(fm_updates_per_step):
                ob, ac, ob2, vc = replay.sample(fm_batch, rng)
                ob_t = torch.from_numpy(ob).to(device)
                ob2_t = torch.from_numpy(ob2).to(device)
                ac_t = torch.from_numpy(ac).to(device)
                vc_t = torch.from_numpy(vc).to(device)
                e_a = F.one_hot(ac_t, num_classes=n_actions).float()
                o_pred, c_pred = fm.step(ob_t, e_a)
                loss_obs = F.mse_loss(o_pred, ob2_t)
                # per-channel MSE preserved (no scalar-collapse over channels)
                loss_c = F.mse_loss(c_pred, vc_t)
                fm_loss = loss_obs + loss_c
                fm_opt.zero_grad()
                fm_loss.backward()
                fm_opt.step()
                fm_loss_running = 0.95 * fm_loss_running + 0.05 * float(fm_loss.item())
                grad_steps += 1

        # 2. Decision step: compute logits, nudge with JFP, sample action
        o_t = torch.from_numpy(o).float().to(device)
        # use the current policy logits as the base distribution
        with torch.no_grad():
            base_logits = policy(o_t).cpu().numpy()

        if env_steps % nudge_every == 0 and replay.size >= fm_batch:
            try:
                nudges, J_np = compute_jfp_nudge(o)
            except RuntimeError:
                nudges = np.zeros(n_actions, dtype=np.float32)
                J_np = None
        else:
            nudges = np.zeros(n_actions, dtype=np.float32)
            J_np = None

        nudged = base_logits + nudges
        probs = np.exp(nudged - nudged.max())
        probs /= probs.sum()
        a = int(rng.choice(n_actions, p=probs))

        # 3. Step env
        obs_next, r, term, trunc, info = env.step(a)
        o_next = _flatten_obs(obs_next)
        v = channel_signal(r, info, k_channels, is_vector)
        replay.add(o, a, o_next, v)
        env_steps += 1

        # 4. Policy update: cross-entropy of policy logits onto the nudged
        # target distribution at state o. This is the "policy's only
        # learning signal is the Pareto nudge" pathway: the target is
        # softmax(base_logits + nudges) detached, and the policy is
        # trained to match. No critic, no Bellman backup.
        target = torch.from_numpy(probs.astype(np.float32)).to(device)
        cur_logits = policy(o_t)
        log_probs = F.log_softmax(cur_logits, dim=-1)
        pol_loss = -(target * log_probs).sum()
        pol_opt.zero_grad()
        pol_loss.backward()
        pol_opt.step()

        if term or trunc:
            obs, _ = env.reset(seed=seed + env_steps)
            o = _flatten_obs(obs)
        else:
            o = o_next

        # 5. Periodic log
        now = time.monotonic()
        if now - last_log_t > 10.0:
            const_frac = (
                pareto_constant_count / max(1, nudge_calls) if nudge_calls else 0.0
            )
            print(
                f"[train] env={env_id} seed={seed} env_steps={env_steps} "
                f"grad_steps={grad_steps} fm_loss={fm_loss_running:.4f} "
                f"nudge_calls={nudge_calls} pareto_const_frac={const_frac:.2f} "
                f"elapsed_s={now - t_start:.1f}",
                flush=True,
            )
            last_log_t = now

    env.close()

    # Final greedy policy: argmax of nudged logits at each state.
    # We freeze the forward model and the policy network. At inference we
    # compute J(s, a) and re-apply the Pareto nudge so execution rule
    # matches training (per spec: "sample actions from the policy whose
    # logits have been nudged by the JFP signal at the current step").
    fm_eval = fm
    pol_eval = policy
    fm_eval.eval()
    pol_eval.eval()

    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"grad_steps={grad_steps} fm_loss={fm_loss_running:.4f} "
        f"nudge_calls={nudge_calls} train_s={time.monotonic() - t_start:.1f} "
        f"budget_s={time_budget_s} (final)",
        flush=True,
    )

    def policy_fn(obs_in):
        o_np = _flatten_obs(obs_in)
        o_t = torch.from_numpy(o_np).float().to(device)
        with torch.no_grad():
            base_logits = pol_eval(o_t).cpu().numpy()
        try:
            with torch.enable_grad():
                J = compute_J(fm_eval, pol_eval, o_t, n_actions, H, device)
            J_np = J.detach().cpu().numpy()
            nudges = pareto_nudge(J_np, alpha)
        except RuntimeError:
            nudges = np.zeros(n_actions, dtype=np.float32)
        nudged = base_logits + nudges
        return int(np.argmax(nudged))

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
