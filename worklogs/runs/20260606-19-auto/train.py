"""CPR — Channel-Posterior Chebyshev Reweight.

Substrate-contract entry point. Realizes the hypothesis described in
worklogs/runs/20260606-19-auto/hypothesis.md.

Key objects:
  - per-channel reweighted empirical action posterior pihat_m(a|o)
    maintained as an online table indexed by a cluster key c(o)
    with weights w_m(t) = ReLU((Delta c_t[m] - mu_m) / sigma_m)
  - improvement operator: at each gradient step, for each obs in the
    batch take gradient on KL(pihat_{m*} || pi_theta) where m* is the
    per-observation argmax channel. (Optionally smoothed with a
    softmax-temperature beta annealed soft -> hard.)
  - polarity: each channel is tracked with both a positive and a
    sign-flipped (negative) variant; the operator selects the worst
    among 2k empirical posteriors.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict, deque

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


# --- helpers --------------------------------------------------------------


def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _cluster_key(obs) -> tuple:
    """Coarse observation cluster c(o).

    For DST/Resource-Gathering observations are small int tuples already, so
    the raw tuple is the cluster. For higher-dimensional or float observations
    (used on sparse/Craftax stages) we coarsely bin to keep cluster density.
    """
    arr = np.asarray(obs).reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        return tuple(int(x) for x in arr.tolist())
    # bin floats to one decimal place; bin uint8 image grids by raw byte tuple
    if arr.dtype == np.uint8:
        return tuple(int(x) for x in arr.tolist())
    return tuple(int(round(float(x) * 4.0)) for x in arr.tolist())


class _Welford:
    """Online mean/variance for channel-standardisation."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / max(1, self.n - 1))


# --- policy network -------------------------------------------------------


class _Policy(nn.Module):
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


# --- training -------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng_np = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n)

    # determine observation features and dimension
    obs0, _ = env.reset(seed=seed)
    feat0 = _flatten_obs(obs0)
    obs_dim = feat0.shape[0]
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _Policy(obs_dim, n_actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    # --- channel bookkeeping ---------------------------------------------
    # we maintain 2*k posteriors (positive polarity + negative polarity);
    # the operator picks the sup over all of them.
    # For sparse (scalar) envs we synthesise a 2-channel vector:
    #   v[0] = reward (one-hot reward channel)
    #   v[1] = -1 step penalty
    if is_vector:
        # peek vector dim
        a_probe = env.action_space.sample()
        _, _, _, _, info_probe = env.step(a_probe)
        env.reset(seed=seed)
        k = int(np.asarray(info_probe["vector"]).shape[0])
    else:
        k = 2
    n_pol = 2 * k  # positive and negative polarities per channel

    # running stats per (raw, signed) channel
    welford: list[_Welford] = [_Welford() for _ in range(k)]

    # online tables: weighted action counts per (polarity, cluster)
    # S[p][c] = np.array(n_actions,)  ; running sum of w(t) over visits where a_t=a
    S: list[dict[tuple, np.ndarray]] = [defaultdict(lambda: np.zeros(n_actions, dtype=np.float64)) for _ in range(n_pol)]

    # replay of raw deltas per step so we can update tables when stats refine.
    # We just store recent transitions (o, a, dv) for batch sampling.
    replay: deque = deque(maxlen=20_000)

    # ---------------- training loop -------------------------------------
    t_start = time.monotonic()
    deadline = t_start - 0.5 + time_budget_s  # leave a small margin
    env_steps = 0
    grad_steps = 0

    obs, _ = env.reset(seed=seed)
    done = False
    ep_steps = 0

    # Logging buffers for falsifier instrumentation
    dom_counts = np.zeros(n_pol, dtype=np.int64)
    total_dom = 0

    # Schedule: anneal softmax temperature beta from soft (1.0) -> hard (~10.0)
    # used to smoothly blend across channels early then approach argmax.

    # Quick rollout phase: collect some transitions before first gradient step
    warmup_steps = 64
    batch_size = 64
    update_every = 16

    last_print = t_start

    while time.monotonic() < deadline:
        # --- act --------------------------------------------------------
        feat = _flatten_obs(obs)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feat).unsqueeze(0).to(device))
            # entropy floor: mix with uniform with small probability eps_floor
            probs = F.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
        eps_floor = 0.05
        mixed = (1.0 - eps_floor) * probs + eps_floor / n_actions
        mixed = mixed / mixed.sum()
        a = int(rng_np.choice(n_actions, p=mixed))

        # --- step -------------------------------------------------------
        next_obs, reward, term, trunc, info = env.step(a)
        env_steps += 1
        ep_steps += 1

        if is_vector:
            dv = np.asarray(info["vector"], dtype=np.float64)
        else:
            dv = np.array([float(reward), -1.0], dtype=np.float64)

        # update running stats and accumulate online weights
        ck = _cluster_key(obs)
        # for each raw channel m, update both polarities
        for m in range(k):
            welford[m].update(float(dv[m]))
            mu = welford[m].mean
            sd = welford[m].std
            if sd <= 1e-8:
                w_pos = 0.0
                w_neg = 0.0
            else:
                z = (dv[m] - mu) / sd
                w_pos = max(0.0, z)
                w_neg = max(0.0, -z)
            if w_pos > 0.0:
                S[2 * m][ck][a] += w_pos
            if w_neg > 0.0:
                S[2 * m + 1][ck][a] += w_neg

        replay.append((feat, ck, a))

        obs = next_obs
        if term or trunc or ep_steps >= harness.MAX_EPISODE_STEPS:
            obs, _ = env.reset(seed=int(rng_np.integers(1 << 31)))
            done = False
            ep_steps = 0

        # --- gradient update -------------------------------------------
        if env_steps >= warmup_steps and env_steps % update_every == 0 and len(replay) >= batch_size:
            idx = rng_np.integers(0, len(replay), size=batch_size)
            feats_b = np.stack([replay[i][0] for i in idx], axis=0)
            cks_b = [replay[i][1] for i in idx]

            # build per-polarity empirical distributions for this batch
            # shape: (n_pol, batch, n_actions)
            pi_hat = np.zeros((n_pol, batch_size, n_actions), dtype=np.float64)
            valid = np.zeros((n_pol, batch_size), dtype=bool)
            for p in range(n_pol):
                for bi, ck in enumerate(cks_b):
                    counts = S[p].get(ck)
                    if counts is None:
                        continue
                    s = counts.sum()
                    if s <= 1e-12:
                        continue
                    # Laplace smoothing for KL stability
                    smoothed = counts + 1e-3
                    pi_hat[p, bi] = smoothed / smoothed.sum()
                    valid[p, bi] = True

            feats_t = torch.from_numpy(feats_b).to(device)
            logits_t = policy(feats_t)
            log_pi = F.log_softmax(logits_t, dim=-1)  # (B, A)

            pi_hat_t = torch.from_numpy(pi_hat).to(device).float()  # (P, B, A)
            valid_t = torch.from_numpy(valid).to(device)  # (P, B)

            # KL(pihat || pi_theta) per (polarity, batch element)
            # KL = sum_a pihat * (log pihat - log pi_theta)
            log_pihat = torch.log(pi_hat_t.clamp_min(1e-12))
            kl = (pi_hat_t * (log_pihat - log_pi.unsqueeze(0))).sum(dim=-1)  # (P, B)

            # mask invalid entries with -inf so they cannot win argmax
            kl_masked = kl.masked_fill(~valid_t, float("-inf"))

            # only proceed for batch elements that have at least one valid polarity
            any_valid = valid_t.any(dim=0)  # (B,)
            if any_valid.any():
                # per-observation argmax channel m*(o)
                m_star = kl_masked.argmax(dim=0)  # (B,)
                # gather KL of dominant polarity for each obs
                kl_dom = kl.gather(0, m_star.unsqueeze(0)).squeeze(0)  # (B,)
                # only those with any_valid contribute
                loss_terms = kl_dom[any_valid]

                # entropy bonus floor for exploration
                ent = -(F.softmax(logits_t, dim=-1) * log_pi).sum(dim=-1)
                loss = loss_terms.mean() - 1e-3 * ent.mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()

                # falsifier instrumentation: count dominant polarity choices
                m_star_np = m_star[any_valid].detach().cpu().numpy()
                for v in m_star_np:
                    dom_counts[int(v)] += 1
                total_dom += int(any_valid.sum().item())
                grad_steps += 1

        # periodic log
        now = time.monotonic()
        if now - last_print > 20.0:
            last_print = now
            if total_dom > 0:
                shares = (dom_counts / max(1, total_dom)).round(3).tolist()
            else:
                shares = []
            print(
                f"[train] env={env_id} steps={env_steps} grad={grad_steps} "
                f"replay={len(replay)} dom_share={shares} t={now - t_start:.1f}s",
                flush=True,
            )

    env.close()

    # Final report incl. falsifier trace ---------------------------------
    if total_dom > 0:
        shares = dom_counts / total_dom
        max_share = float(shares.max()) if len(shares) else 0.0
    else:
        shares = np.zeros(n_pol)
        max_share = 0.0
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"grad_steps={grad_steps} train_s={time.monotonic() - t_start:.1f} "
        f"budget_s={time_budget_s} dom_shares={shares.round(4).tolist()} "
        f"max_dom_share={max_share:.3f}",
        flush=True,
    )

    # frozen deterministic policy: argmax over policy logits
    policy.eval()

    def policy_fn(o):
        with torch.no_grad():
            x = torch.from_numpy(_flatten_obs(o)).unsqueeze(0).to(device)
            logits = policy(x)
            a = int(torch.argmax(logits, dim=-1).item())
        return a

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
