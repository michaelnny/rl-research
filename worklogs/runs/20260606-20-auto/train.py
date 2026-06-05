"""ATP — Action-Tangent Persistence.

Primitive: per-(observation, action, channel) persistence horizon
    h*[o, a, m] = min{h : p_m(o_h) >= tau_m}, computed by H_max forward-model
    rollouts forced to start with action a then continued under current policy.
Improvement operator: Pareto-non-dominance margin (n_a - m_a) over the
    k-vector of horizons (smaller dominates) -> logit nudge of magnitude
    alpha * (n_a - m_a). Policy is updated by REINFORCE-style nudge toward
    a* = argmax_a (n_a - m_a). No critic, no Bellman, no scalar return.

For non-vector envs (sparse MiniGrid, Craftax-Symbolic), we synthesize
channels from sign-of-reward and termination indicators so the structural
pipeline is preserved; the primary axis here is vector envs where ATP's
distinguishing signal lives.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# --------------------------------------------------------------------------
# Observation flattening — keep small and shape-agnostic
# --------------------------------------------------------------------------
def flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs).astype(np.float32).reshape(-1)
    return arr


# --------------------------------------------------------------------------
# Channel extraction from a step. Returns a fixed-length 0/1 firing vector.
# For vector envs, channels are info["vector"] components (binary: did the
# channel fire = is its value non-zero this step).
# For scalar envs, we synthesize a 2-channel signature: (reward != 0,
# terminated). This keeps the pipeline well-defined even when k=1 (the
# hypothesis flags this as "scalar regime, distinction lost"); we keep
# 2 synthetic channels so the operator is not silently degenerate.
# --------------------------------------------------------------------------
def channels_from_step(reward, info, term: bool, env_id: str) -> np.ndarray:
    if "vector" in info:
        v = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
        # binary firing per channel: did this channel produce any signal this step
        return (np.abs(v) > 1e-9).astype(np.float32)
    # synthesized channels for scalar envs (degenerate ATP regime)
    return np.array(
        [float(abs(float(reward)) > 1e-9), float(term)], dtype=np.float32
    )


def n_channels_for_env(env_id: str) -> int:
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    return 2  # synthesized


# --------------------------------------------------------------------------
# Networks: forward-dynamics, channel firing classifiers, policy logits
# All small MLPs.
# --------------------------------------------------------------------------
class ForwardModel(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(action_idx, num_classes=self.n_actions).float()
        x = torch.cat([obs, a_onehot], dim=-1)
        # residual: predict delta then add — easier to learn small step changes
        return obs + self.net(x)


class ChannelHeads(nn.Module):
    """k binary classifiers p_m: o -> [0,1]."""

    def __init__(self, obs_dim: int, k: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs))


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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# --------------------------------------------------------------------------
# Persistence-horizon computation. For each candidate first-action a in [0..|A|),
# unroll the forward model: o_0 = current obs, o_1 = f(o_0, a), then for
# h=2..H_max sample a_h ~ pi(o_{h-1}) and o_h = f(o_{h-1}, a_h). Record p_m(o_h)
# for h=1..H_max. The horizon is min{h : p_m(o_h) >= tau_m} else H_max.
# --------------------------------------------------------------------------
@torch.no_grad()
def persistence_horizons(
    obs_t: torch.Tensor,  # (obs_dim,)
    n_actions: int,
    H_max: int,
    fwd: ForwardModel,
    heads: ChannelHeads,
    policy: PolicyNet,
    tau: torch.Tensor,  # (k,)
    device: torch.device,
) -> torch.Tensor:
    """Return h_star: (n_actions, k) integer tensor of horizons in [1, H_max]."""
    k = tau.shape[0]
    # Batch over actions to amortize forward-model cost.
    obs_batch = obs_t.unsqueeze(0).expand(n_actions, -1).contiguous()
    actions = torch.arange(n_actions, device=device, dtype=torch.long)

    # h_star initialized to H_max (no crossing)
    h_star = torch.full((n_actions, k), H_max, device=device, dtype=torch.long)
    crossed = torch.zeros((n_actions, k), device=device, dtype=torch.bool)

    cur_obs = fwd(obs_batch, actions)  # step 1: forced action a
    for h in range(1, H_max + 1):
        p_m = heads(cur_obs)  # (n_actions, k)
        # check threshold crossings (only for channels not yet crossed)
        new_cross = (p_m >= tau.unsqueeze(0)) & (~crossed)
        if new_cross.any():
            # set h for newly crossed entries
            h_star = torch.where(new_cross, torch.full_like(h_star, h), h_star)
            crossed = crossed | new_cross
        if crossed.all() or h == H_max:
            break
        # sample next action under policy
        logits = policy(cur_obs)
        probs = F.softmax(logits, dim=-1)
        next_a = torch.multinomial(probs, num_samples=1).squeeze(-1)
        cur_obs = fwd(cur_obs, next_a)

    return h_star  # (n_actions, k), integer in [1, H_max]


def pareto_margin(h_star: torch.Tensor) -> torch.Tensor:
    """Pareto-non-dominance margin (n_a - m_a) for each action a.

    Sign convention is anchored by required-candidate-shape #3 of the
    hypothesis: "a* is the action with the LARGEST Pareto-non-dominance
    margin n_a - m_a". Therefore the margin must be HIGH for good actions
    (those that strictly Pareto-dominate many others, are dominated by few),
    where SHORTER horizons dominate (faster onset across channels).

    margin[a] = #{a' : a Pareto-dominates a'} - #{a' : a' Pareto-dominates a}
    where a dominates a' iff h_a <= h_a' coord-wise and strict in some coord.
    """
    h_a = h_star.unsqueeze(1)   # (A, 1, k)
    h_ap = h_star.unsqueeze(0)  # (1, A, k)
    a_dom_ap = (h_a <= h_ap).all(dim=-1) & (h_a < h_ap).any(dim=-1)  # (A, A)
    n_dominates = a_dom_ap.sum(dim=1).long()       # # ap that a dominates
    n_dominated_by = a_dom_ap.sum(dim=0).long()    # # ap that dominate a
    return (n_dominates - n_dominated_by).long()


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
class Replay:
    def __init__(self, capacity: int, obs_dim: int, k: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.fired = np.zeros((capacity, k), dtype=np.float32)  # channel fire labels
        self.size = 0
        self.idx = 0

    def add(self, o, a, no, fired):
        self.obs[self.idx] = o
        self.act[self.idx] = a
        self.next_obs[self.idx] = no
        self.fired[self.idx] = fired
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        n = min(batch, self.size)
        idx = rng.integers(0, self.size, size=n)
        return (
            self.obs[idx],
            self.act[idx],
            self.next_obs[idx],
            self.fired[idx],
        )


# --------------------------------------------------------------------------
# train()
# --------------------------------------------------------------------------
def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n)
    k = n_channels_for_env(env_id)

    # Probe obs_dim with a reset
    obs0, _ = env.reset(seed=seed)
    obs_flat = flatten_obs(obs0)
    obs_dim = obs_flat.shape[0]

    # ATP modules
    fwd = ForwardModel(obs_dim, n_actions, hidden=64).to(device)
    heads = ChannelHeads(obs_dim, k, hidden=64).to(device)
    policy = PolicyNet(obs_dim, n_actions, hidden=64).to(device)

    fwd_opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)
    heads_opt = torch.optim.Adam(heads.parameters(), lr=1e-3)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    capacity = 20_000
    buf = Replay(capacity, obs_dim, k)

    # Hyperparameters consistent with hypothesis (no per-env tuning)
    H_max = 6           # ≤ 8 per #6
    alpha = 1.5         # nudge magnitude
    tau_quantile = 0.75 # per #2: 75th percentile
    tau_floor = 0.05    # lower clamp on tau (#5)
    tau_ceiling = 0.95
    update_every = 8    # SGD step on fwd+heads every k env steps
    sgd_batch = 64
    policy_update_every = 1  # nudge after every decision
    warmup_steps = 200  # uniform random until replay is non-trivial

    obs = obs_flat
    t_start = time.monotonic()
    total_env_steps = 0
    total_updates = 0
    diag_nonzero_delta = 0
    diag_decisions = 0

    # tau initialized at 0.5 (will be re-estimated as data arrives)
    tau = torch.full((k,), 0.5, device=device)

    # replay-bookkeeping for tau quantile re-estimation
    tau_recompute_every = 200

    while True:
        if time.monotonic() - t_start > max(1.0, time_budget_s - 8.0):
            break

        obs_t = torch.from_numpy(obs).to(device)

        # Decide: compute Pareto-margin nudge if past warmup
        used_atp_decision = False
        if buf.size >= warmup_steps:
            with torch.no_grad():
                h_star = persistence_horizons(
                    obs_t, n_actions, H_max, fwd, heads, policy, tau, device
                )
                delta = pareto_margin(h_star).float()
                if (delta != 0).any():
                    diag_nonzero_delta += 1
                logits_base = policy(obs_t)
                logits = logits_base + alpha * delta
                probs = F.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs, num_samples=1).item())
                # Pareto-best action a* (largest margin); ties broken randomly
                max_margin = delta.max()
                a_star_mask = (delta == max_margin)
                a_star_choices = torch.nonzero(a_star_mask, as_tuple=False).flatten()
                a_star = int(
                    a_star_choices[rng.integers(0, len(a_star_choices))].item()
                )
                margin_a_star = float(delta[a_star].item())
            diag_decisions += 1
            used_atp_decision = True
        else:
            # warmup: uniform random for diversity
            action = int(rng.integers(0, n_actions))
            a_star = action  # no learning signal during warmup
            margin_a_star = 0.0

        # Step env
        next_obs_raw, reward, term, trunc, info = env.step(action)
        next_obs = flatten_obs(next_obs_raw)
        fired = channels_from_step(reward, info, bool(term), env_id)
        # store (o, a, o', firing-vector-on-NEXT-step)
        # The hypothesis says p_m: o -> [0,1] is "did channel m fire on this step".
        # We label p_m(o) = 1 if channel m fired AS A RESULT OF the action taken
        # FROM o (i.e. next-step firing). The classifier therefore learns
        # P(channel m fires next | obs_now). That is what the rollout uses
        # downstream (p_m on the predicted future obs).
        # Storage: head label corresponds to obs (the *predictor*) -> fired (effect)
        buf.add(obs, action, next_obs, fired)
        total_env_steps += 1

        # Improvement operator: REINFORCE-style nudge on policy toward a* with
        # weight alpha * margin(a*). Only when buffer is warm and ATP made the
        # decision (so margin_a_star is meaningful).
        if used_atp_decision and total_env_steps % policy_update_every == 0:
            obs_t2 = torch.from_numpy(obs).to(device)
            logits_p = policy(obs_t2)
            logp = F.log_softmax(logits_p, dim=-1)
            # Update only if margin is positive (a* is genuinely Pareto-better)
            if margin_a_star > 0:
                loss = -alpha * margin_a_star * logp[a_star]
                policy_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                policy_opt.step()

        # SGD on forward model + channel heads on a buffer batch
        if total_env_steps % update_every == 0 and buf.size >= 64:
            o_b, a_b, no_b, f_b = buf.sample(sgd_batch, rng)
            o_b_t = torch.from_numpy(o_b).to(device)
            a_b_t = torch.from_numpy(a_b).to(device)
            no_b_t = torch.from_numpy(no_b).to(device)
            f_b_t = torch.from_numpy(f_b).to(device)

            # Forward-dynamics MSE
            pred_no = fwd(o_b_t, a_b_t)
            fwd_loss = F.mse_loss(pred_no, no_b_t)
            fwd_opt.zero_grad(set_to_none=True)
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
            fwd_opt.step()

            # Channel heads: BCE on fired
            p_pred = heads(o_b_t).clamp(1e-6, 1 - 1e-6)
            head_loss = F.binary_cross_entropy(p_pred, f_b_t)
            heads_opt.zero_grad(set_to_none=True)
            head_loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            heads_opt.step()
            total_updates += 1

        # Periodically re-estimate tau as 75th percentile of p_m on the buffer
        if total_env_steps % tau_recompute_every == 0 and buf.size >= 200:
            with torch.no_grad():
                idx = rng.integers(0, buf.size, size=min(buf.size, 1000))
                o_sample = torch.from_numpy(buf.obs[idx]).to(device)
                p_sample = heads(o_sample).cpu().numpy()  # (N, k)
                q = np.quantile(p_sample, tau_quantile, axis=0)
                q = np.clip(q, tau_floor, tau_ceiling)
                tau = torch.from_numpy(q.astype(np.float32)).to(device)

        if term or trunc or total_env_steps % harness.MAX_EPISODE_STEPS == 0:
            obs_raw, _ = env.reset(seed=seed + total_env_steps)
            obs = flatten_obs(obs_raw)
        else:
            obs = next_obs

    env.close()

    # Diagnostics
    nonzero_frac = (diag_nonzero_delta / max(1, diag_decisions))
    print(
        f"[train] env={env_id} seed={seed} env_steps={total_env_steps} "
        f"updates={total_updates} train_s={time.monotonic() - t_start:.1f} "
        f"budget_s={time_budget_s} tau={tau.cpu().numpy().tolist()} "
        f"nonzero_delta_frac={nonzero_frac:.3f}",
        flush=True,
    )

    # Build deterministic eval policy: argmax over (logits + alpha * delta(o))
    fwd.eval()
    heads.eval()
    policy.eval()

    tau_eval = tau.detach().clone()

    def policy_fn(o):
        o_flat = flatten_obs(o)
        ot = torch.from_numpy(o_flat).to(device)
        with torch.no_grad():
            h_star = persistence_horizons(
                ot, n_actions, H_max, fwd, heads, policy, tau_eval, device
            )
            delta = pareto_margin(h_star).float()
            logits = policy(ot) + alpha * delta
            return int(torch.argmax(logits).item())

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
