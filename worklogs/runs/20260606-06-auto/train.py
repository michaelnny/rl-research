"""CWAI: Channel-Wise Action Influence.

Hypothesis: 20260606-06-auto

Primitive: per-(state, action, channel) Jacobian column-norm matrix
    G(o_t)[a, m] = || d f_theta(o_t, a)[m] / d e(a) ||_2
where f_theta is a self-supervised forward model trained online with MSE on
the vector observation appended with vector-channel readouts from
info["vector"]. NO reward enters the model loss or the operator.

Improvement operator: at each visited state, compute G(o_t), find the
Pareto-non-dominated rows (coordinate-wise channel dominance), and apply
policy logit nudge:
    Delta logit_a = alpha * (D_plus(a) - D_minus(a))
where D_plus(a) = #{a' : G[a, :] > G[a', :] coordinate-wise} and D_minus is
the reverse count. Logit updates are made with a single SGD-style step on
the policy network so the same-state revisit benefits.

Execution rule: softmax sampling during data collection; eval also softmax
(annealed but never argmax). Dead-channel (G[a, :] = 0) actions deprioritized.

Rollout cost: exactly 1 env step per primitive update.
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


# --- Models ----------------------------------------------------------------


class ForwardModel(nn.Module):
    """Predict next observation + vector channels from (obs, action_embedding).

    Output has dimension obs_dim + vec_dim. The vector channels are appended
    as additional output dimensions; they are NEVER scalarized into reward.
    The Jacobian is taken with respect to the action embedding e(a).
    """

    def __init__(self, obs_dim: int, n_actions: int, vec_dim: int,
                 emb_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.vec_dim = vec_dim
        self.emb_dim = emb_dim
        self.action_emb = nn.Embedding(n_actions, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + emb_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, obs_dim + vec_dim),
        )

    def forward_from_emb(self, obs: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim), e: (B, emb_dim) -> (B, obs_dim + vec_dim)."""
        x = torch.cat([obs, e], dim=-1)
        return self.net(x)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        e = self.action_emb(action)
        return self.forward_from_emb(obs, e)


class PolicyNet(nn.Module):
    """Small MLP producing logits over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# --- Helpers ---------------------------------------------------------------


def flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


def compute_influence_matrix(model: ForwardModel,
                             obs_t: torch.Tensor,
                             n_actions: int,
                             device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute G[a, m] = || d f(obs, e(a))[m] / d e(a) ||_2 for all a in A.

    Returns:
        G: (n_actions, out_dim) tensor of column norms
        out_dim = obs_dim + vec_dim (channels include obs + vector readouts)
    """
    out_dim = model.obs_dim + model.vec_dim
    emb_dim = model.emb_dim
    # batch size n_actions; each row is action a's embedding vector e(a)
    actions = torch.arange(n_actions, device=device)
    e = model.action_emb(actions).detach().clone().requires_grad_(True)  # (A, emb_dim)
    obs_batched = obs_t.unsqueeze(0).expand(n_actions, -1)  # (A, obs_dim)

    out = model.forward_from_emb(obs_batched, e)  # (A, out_dim)

    # Compute Jacobian of out[a, m] w.r.t. e[a, :] for each (a, m).
    # Use a vectorized approach: for each output dim m, sum over batch and
    # take grad w.r.t. e; the gradient at row a is exactly d out[a, m] / d e[a, :]
    # because for any (a', m), d out[a, m] / d e[a', :] = 0 when a' != a.
    G = torch.zeros(n_actions, out_dim, device=device)
    for m in range(out_dim):
        grad = torch.autograd.grad(
            outputs=out[:, m].sum(),
            inputs=e,
            retain_graph=(m < out_dim - 1),
            create_graph=False,
        )[0]  # (A, emb_dim)
        G[:, m] = grad.norm(dim=-1)
    return G.detach(), out.detach()


def pareto_counts(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For matrix G of shape (A, k), compute D_plus(a) and D_minus(a).

    D_plus(a) = #{a' : G[a] strictly dominates G[a']}
        i.e., G[a] >= G[a'] componentwise AND G[a] > G[a'] in at least one dim
    D_minus(a) = #{a' : G[a'] strictly dominates G[a]}
    """
    A = G.shape[0]
    D_plus = np.zeros(A, dtype=np.float32)
    D_minus = np.zeros(A, dtype=np.float32)
    for a in range(A):
        for ap in range(A):
            if a == ap:
                continue
            ge = np.all(G[a] >= G[ap])
            gt_any = np.any(G[a] > G[ap])
            if ge and gt_any:
                D_plus[a] += 1.0
            le = np.all(G[a] <= G[ap])
            lt_any = np.any(G[a] < G[ap])
            if le and lt_any:
                D_minus[a] += 1.0
    return D_plus, D_minus


# --- Train -----------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = harness.make_env(env_id, seed)

    is_vector_env = harness.ENV_TYPE[env_id] == "vector"

    # Determine action space
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        # continuous action: the hypothesis is for discrete action space.
        # Fall back to a simple uniform sampler since CWAI's primitive is
        # defined over a discrete action set. (None of the panel envs are
        # continuous; this is a defensive guard.)
        env.close()
        def policy_fn(_obs):
            return action_space.sample()
        print(f"[train] env={env_id} continuous-action fallback", flush=True)
        return policy_fn
    n_actions = int(action_space.n)

    # Determine obs and vector dims by a probe reset/step
    obs0, _ = env.reset(seed=seed)
    obs_flat0 = flatten_obs(obs0)
    obs_dim = int(obs_flat0.shape[0])

    # Probe info["vector"] to learn vector dim
    if is_vector_env:
        # take a sample step to learn vec dim
        a_probe = int(rng.integers(n_actions))
        obs_probe, _r, term_p, trunc_p, info_p = env.step(a_probe)
        vec_arr = np.asarray(info_p.get("vector", np.zeros(0)), dtype=np.float32)
        vec_dim = int(vec_arr.shape[0])
        # reset env again before main loop
        obs0, _ = env.reset(seed=seed)
        obs_flat0 = flatten_obs(obs0)
    else:
        vec_dim = 0

    # Build models
    model = ForwardModel(obs_dim=obs_dim, n_actions=n_actions, vec_dim=vec_dim,
                         emb_dim=16, hidden=64).to(device)
    policy = PolicyNet(obs_dim=obs_dim, n_actions=n_actions, hidden=64).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    alpha = 0.2  # logit-nudge step size (used as a soft cross-entropy target weight)

    # Diagnostics
    fm_loss_ema = None
    fm_loss_ema_alpha = 0.02
    n_steps = 0
    n_pareto_fires = 0
    n_dead_channel = 0
    n_rank1_steps = 0  # crude rank-1 detector: max sing val ratio

    # For final eval, we'll keep a frozen copy of the policy network params

    obs = obs0
    obs_flat = obs_flat0

    t_start = time.monotonic()
    deadline = t_start + max(1, time_budget_s - 5)  # leave headroom for eval

    # Episode handling
    ep_steps = 0
    max_ep_steps = harness.MAX_EPISODE_STEPS

    while time.monotonic() < deadline:
        n_steps += 1
        obs_t = torch.from_numpy(obs_flat).to(device)

        # ---- 1. Compute influence matrix G at current state ----
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            G_t, _ = compute_influence_matrix(model, obs_t, n_actions, device)
        finally:
            for p in model.parameters():
                p.requires_grad_(True)
            model.train()
        G = G_t.cpu().numpy()  # (A, out_dim)

        # ---- 2. Pareto operator on rows of G ----
        D_plus, D_minus = pareto_counts(G)
        delta = D_plus - D_minus  # (A,) net dominance score
        if np.any(D_plus + D_minus > 0):
            n_pareto_fires += 1

        # Dead-channel deprioritization: actions whose row is all-zero
        row_norms = G.sum(axis=1)
        dead_mask = row_norms <= 1e-10
        if dead_mask.any():
            n_dead_channel += int(dead_mask.sum())

        # ---- 3. Crude rank-1 / collinearity diagnostic ----
        if G.shape[1] >= 2 and np.linalg.norm(G) > 1e-6:
            # SVD of G; if top singular value dominates, channels are collinear
            try:
                s = np.linalg.svd(G, compute_uv=False)
                if s.size >= 2 and s[0] > 0:
                    if s[1] / s[0] < 0.05:
                        n_rank1_steps += 1
            except Exception:
                pass

        # ---- 4. Policy update: nudge logits toward Pareto-non-dominated rows ----
        # We define a target distribution q over actions:
        #   q_a propto exp(alpha * delta_a) with dead-channel actions damped.
        # Then take a single SGD step minimizing CE between policy(obs) and q.
        # This is the operationalization of "Delta logit_a = alpha * (D_plus(a) - D_minus(a))"
        # with the locality of "this state": gradient is at this single obs.
        target_logits = alpha * delta.astype(np.float32)
        # Damp dead-channel actions strongly (subtract a large penalty in target)
        target_logits = target_logits - 5.0 * dead_mask.astype(np.float32)
        target_logits_t = torch.from_numpy(target_logits).to(device)
        target_dist = F.softmax(target_logits_t, dim=-1)

        policy.train()
        cur_logits = policy(obs_t.unsqueeze(0)).squeeze(0)
        log_pi = F.log_softmax(cur_logits, dim=-1)
        ce_loss = -(target_dist * log_pi).sum()
        policy_opt.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_opt.step()

        # ---- 5. Sample action from current policy and step env ----
        with torch.no_grad():
            logits = policy(obs_t.unsqueeze(0)).squeeze(0)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs = np.clip(probs, 1e-8, 1.0)
            probs /= probs.sum()
        action = int(rng.choice(n_actions, p=probs))

        # Step env
        try:
            next_obs, _reward, term, trunc, info = env.step(action)
        except Exception:
            # Defensive: if env crashes, reset
            obs, _ = env.reset(seed=seed + n_steps)
            obs_flat = flatten_obs(obs)
            ep_steps = 0
            continue

        next_flat = flatten_obs(next_obs)
        if is_vector_env:
            vec = np.asarray(info.get("vector", np.zeros(vec_dim)), dtype=np.float32)
            target_arr = np.concatenate([next_flat, vec], axis=0)
        else:
            target_arr = next_flat
        target_t = torch.from_numpy(target_arr).to(device)

        # ---- 6. Forward-model update: MSE on (obs, a) -> next_obs (+ vector channels) ----
        action_t = torch.tensor([action], dtype=torch.long, device=device)
        pred = model(obs_t.unsqueeze(0), action_t).squeeze(0)  # (out_dim,)
        fm_loss = F.mse_loss(pred, target_t)
        model_opt.zero_grad()
        fm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model_opt.step()

        loss_val = float(fm_loss.detach().cpu())
        fm_loss_ema = loss_val if fm_loss_ema is None else (
            (1 - fm_loss_ema_alpha) * fm_loss_ema + fm_loss_ema_alpha * loss_val
        )

        # ---- 7. Episode/reset bookkeeping ----
        ep_steps += 1
        done = bool(term) or bool(trunc) or ep_steps >= max_ep_steps
        if done:
            obs, _ = env.reset(seed=seed + n_steps)
            obs_flat = flatten_obs(obs)
            ep_steps = 0
        else:
            obs = next_obs
            obs_flat = next_flat

    env.close()

    # Periodic diagnostic print (single line)
    train_s = time.monotonic() - t_start
    rank1_frac = (n_rank1_steps / max(1, n_steps))
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} "
        f"fm_loss_ema={fm_loss_ema if fm_loss_ema is not None else float('nan'):.4f} "
        f"pareto_fires={n_pareto_fires} dead_channel_n={n_dead_channel} "
        f"rank1_frac={rank1_frac:.3f}",
        flush=True,
    )

    # Build deterministic-ish policy_fn: softmax sampling at low temperature
    # (annealed but NOT argmax, per execution rule).
    eval_temp = 0.5
    eval_rng = np.random.default_rng(seed + 7)
    policy.eval()

    def policy_fn(o: np.ndarray):
        of = flatten_obs(o)
        with torch.no_grad():
            ot = torch.from_numpy(of).to(device).unsqueeze(0)
            logits = policy(ot).squeeze(0).cpu().numpy()
        # Annealed softmax (temp=eval_temp), never argmax
        scaled = logits / max(eval_temp, 1e-6)
        scaled = scaled - scaled.max()
        p = np.exp(scaled)
        p = p / max(p.sum(), 1e-12)
        return int(eval_rng.choice(n_actions, p=p))

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
