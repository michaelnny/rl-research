"""ACFC: Action-Frequency / Channel-Frequency Concordance.

Realization of the hypothesis in worklogs/runs/20260606-23-auto/hypothesis.md.

Core primitive:
    C[a, m] = mean_{i<j in buffer} sign(f^i_a - f^j_a) * sign(g^i_m - g^j_m)
where f^i is the per-action frequency vector of episode i and g^i is the
per-channel total firing-count vector of episode i (sum of per-step
info["vector"] over the episode).

Improvement operator (state-independent logit bias):
    b[a] = alpha * (n_a^dom - n_a^sub)
where n_a^dom = #{a' : row C[a,:] Pareto-dominates row C[a',:]}
      n_a^sub = #{a' : row C[a',:] Pareto-dominates row C[a,:]}.

Execution: a ~ softmax(logits_theta(o_t) + b).
Update:    one supervised epoch per completed episode, target =
           softmax(logits_theta(o_t) + sg(b)), cross-entropy.
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


# ----------------------------- policy network -----------------------------


class FlatPolicy(nn.Module):
    """Small MLP over flattened observations -> logits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
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


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    a = np.asarray(obs)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    return a.reshape(-1)


# ----------------------------- ACFC primitive -----------------------------


def compute_C(
    freqs: np.ndarray,  # (B, |A|)
    counts: np.ndarray,  # (B, k)
) -> np.ndarray:
    """C[a, m] = mean_{i<j} sign(f^i_a - f^j_a) * sign(g^i_m - g^j_m)."""
    B = freqs.shape[0]
    if B < 2:
        return np.zeros((freqs.shape[1], counts.shape[1]), dtype=np.float64)
    # df[i,j,a] = f^i_a - f^j_a, dg[i,j,m] = g^i_m - g^j_m
    # Use sign and sum over upper-triangular pairs.
    f = freqs.astype(np.float64)
    g = counts.astype(np.float64)
    A = f.shape[1]
    k = g.shape[1]
    # Pairwise sign differences (B, B, A) and (B, B, k) — vectorized.
    sf = np.sign(f[:, None, :] - f[None, :, :])  # (B,B,A)
    sg = np.sign(g[:, None, :] - g[None, :, :])  # (B,B,k)
    # Use mask i<j to avoid double counting (and the diagonal).
    iu = np.triu_indices(B, k=1)
    sf_pairs = sf[iu[0], iu[1], :]  # (P, A)
    sg_pairs = sg[iu[0], iu[1], :]  # (P, k)
    # C[a,m] = mean over pairs of sf_pa * sg_pm.
    P = sf_pairs.shape[0]
    if P == 0:
        return np.zeros((A, k), dtype=np.float64)
    # (P,A,1) * (P,1,k) summed over P, normalized.
    C = (sf_pairs[:, :, None] * sg_pairs[:, None, :]).mean(axis=0)
    return C  # (A, k)


def pareto_dom_counts(C: np.ndarray) -> np.ndarray:
    """For each a, n_a^dom - n_a^sub.

    a strictly Pareto-dominates a' iff (forall m, C[a,m] >= C[a',m]) and
    (exists m, C[a,m] > C[a',m]).
    """
    A = C.shape[0]
    if A == 0 or C.shape[1] == 0:
        return np.zeros(A, dtype=np.float64)
    # ge[i,j,m] = C[i,m] >= C[j,m]
    ge = C[:, None, :] >= C[None, :, :]
    gt = C[:, None, :] > C[None, :, :]
    dominates = ge.all(axis=2) & gt.any(axis=2)  # (A, A) -> i dominates j
    # diagonal is always False since gt over m is False on diag.
    n_dom = dominates.sum(axis=1)  # # of a' that a dominates
    n_sub = dominates.sum(axis=0)  # # of a' that dominate a
    return (n_dom - n_sub).astype(np.float64)


# ----------------------------- training loop -----------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        # Continuous-action env not in our panel; fall back to random.
        env.close()
        sa = action_space

        def random_policy(_obs):
            return sa.sample()

        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} note=continuous-action-fallback",
            flush=True,
        )
        return random_policy
    n_actions = int(action_space.n)

    # Probe a single reset to determine flat obs dim and channel dim.
    obs0, _ = env.reset(seed=seed)
    obs_flat0 = _flatten_obs(obs0)
    obs_dim = int(obs_flat0.shape[0])

    # Determine channel dim: prefer info["vector"] from first step.
    has_vector = harness.ENV_TYPE.get(env_id, "scalar") == "vector"
    if has_vector:
        # Take a probe step to read the vector dim.
        a0 = action_space.sample() if hasattr(action_space, "sample") else 0
        _, _, term, trunc, info = env.step(int(a0))
        if "vector" in info:
            k_channels = int(np.asarray(info["vector"]).shape[0])
        else:
            k_channels = 1
        env.reset(seed=seed)
    else:
        # Sparse / scalar env: synthesize one channel from per-step reward.
        # This is the explicit k=1 audit point named in the hypothesis.
        k_channels = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = FlatPolicy(obs_dim, n_actions, hidden=128).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Hyperparameters from hypothesis.
    B_BUF = 64  # ring buffer size
    ALPHA = 0.5  # bias scale on (n_dom - n_sub)
    MIN_ENTROPY_FLOOR = 0.05  # mix uniform to avoid identical-frequency degeneracy
    SUP_BATCH = 256  # supervised mini-batch from buffer transitions
    SUP_EPOCHS = 1  # one supervised epoch per completed episode

    # Episode buffer: store (action_seq, vec_seq, obs_seq) for each episode.
    # Cap per-episode tensor sizes to avoid memory blowup on Craftax.
    ep_buf: deque = deque(maxlen=B_BUF)

    # Initial bias = 0.
    bias = np.zeros(n_actions, dtype=np.float64)

    # ---------------------------------------------------------------- loop
    t_start = time.monotonic()
    obs, _ = env.reset(seed=seed + 1)
    cur_actions: list[int] = []
    cur_vecs: list[np.ndarray] = []
    cur_obs: list[np.ndarray] = []
    env_steps = 0
    episodes = 0

    def select_action(o_flat: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.from_numpy(o_flat).float().unsqueeze(0).to(device)
            logits = policy(x).squeeze(0).cpu().numpy().astype(np.float64)
        biased = logits + bias
        # softmax with min-entropy floor: mix in uniform with weight eps.
        # softmax over biased logits.
        m = biased.max()
        p = np.exp(biased - m)
        p = p / p.sum()
        if MIN_ENTROPY_FLOOR > 0.0:
            u = np.full_like(p, 1.0 / n_actions)
            p = (1.0 - MIN_ENTROPY_FLOOR) * p + MIN_ENTROPY_FLOOR * u
        a = int(rng.choice(n_actions, p=p))
        return a

    def end_episode_update():
        """Recompute C, b from the current buffer; one supervised epoch."""
        nonlocal bias
        if len(ep_buf) < 2:
            return
        # Build (B, |A|) frequencies and (B, k) channel counts.
        freqs = np.zeros((len(ep_buf), n_actions), dtype=np.float64)
        counts = np.zeros((len(ep_buf), k_channels), dtype=np.float64)
        for i, (acts, vecs, _obs_arr) in enumerate(ep_buf):
            if len(acts) == 0:
                continue
            counts_i = np.bincount(acts, minlength=n_actions).astype(np.float64)
            freqs[i] = counts_i / max(1, len(acts))
            if len(vecs) > 0:
                # vecs is a list of per-step channel readings (shape k_channels).
                v = np.stack(vecs, axis=0).astype(np.float64)
                counts[i] = v.sum(axis=0)
        C = compute_C(freqs, counts)
        diff = pareto_dom_counts(C)
        bias = ALPHA * diff  # (n_actions,)

        # Supervised imitation: target = softmax(logits + sg(b)).
        # Build training set of (obs_flat, target_dist) over all buffered steps.
        all_obs = []
        for _acts, _vecs, obs_arr in ep_buf:
            if len(obs_arr) > 0:
                all_obs.append(np.stack(obs_arr, axis=0))
        if not all_obs:
            return
        X = np.concatenate(all_obs, axis=0).astype(np.float32)
        N = X.shape[0]
        if N == 0:
            return
        # Subsample if huge, to keep update under budget.
        max_steps = 4096
        if N > max_steps:
            idx = rng.choice(N, size=max_steps, replace=False)
            X = X[idx]
            N = X.shape[0]
        bias_t = torch.from_numpy(bias.astype(np.float32)).to(device)
        Xt = torch.from_numpy(X).to(device)

        for _ in range(SUP_EPOCHS):
            perm = torch.randperm(N, device=device)
            for s in range(0, N, SUP_BATCH):
                mb = perm[s : s + SUP_BATCH]
                xb = Xt[mb]
                logits = policy(xb)
                with torch.no_grad():
                    target = F.softmax(logits + bias_t, dim=-1)
                logp = F.log_softmax(logits, dim=-1)
                loss = -(target * logp).sum(dim=-1).mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                optim.step()

    obs_flat = _flatten_obs(obs)
    while True:
        if time.monotonic() - t_start > time_budget_s:
            break

        cur_obs.append(obs_flat.copy())
        a = select_action(obs_flat)
        try:
            step_out = env.step(a)
        except Exception:
            break
        if len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
        else:
            obs, reward, term, trunc, info = step_out[0], step_out[1], step_out[2], False, step_out[3]
        env_steps += 1

        # Channel reading at this step.
        if has_vector and "vector" in info:
            v = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
            if v.shape[0] != k_channels:
                # Defensive: pad/truncate to declared shape.
                if v.shape[0] > k_channels:
                    v = v[:k_channels]
                else:
                    pad = np.zeros(k_channels - v.shape[0], dtype=np.float64)
                    v = np.concatenate([v, pad])
        else:
            # k=1 fallback: per-step scalar reward as the only channel.
            v = np.array([float(reward)], dtype=np.float64)

        cur_actions.append(int(a))
        cur_vecs.append(v)

        done = bool(term) or bool(trunc)
        obs_flat = _flatten_obs(obs)
        if done:
            ep_buf.append(
                (
                    np.asarray(cur_actions, dtype=np.int64),
                    list(cur_vecs),
                    list(cur_obs),
                )
            )
            episodes += 1
            cur_actions, cur_vecs, cur_obs = [], [], []
            try:
                end_episode_update()
            except Exception as exc:
                print(f"[train] update error (non-fatal): {exc!r}", flush=True)
            obs, _ = env.reset(seed=seed + 1 + episodes)
            obs_flat = _flatten_obs(obs)

    env.close()
    train_s = time.monotonic() - t_start
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} episodes={episodes} "
        f"buf={len(ep_buf)} k_channels={k_channels} alpha={ALPHA} "
        f"bias_min={float(bias.min()):.3f} bias_max={float(bias.max()):.3f} "
        f"train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Deployment: bias is fixed; one forward pass per action; deterministic argmax of biased logits.
    bias_final = bias.copy()
    bias_t_final = torch.from_numpy(bias_final.astype(np.float32)).to(device)

    def policy_fn(o):
        x = torch.from_numpy(_flatten_obs(o)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(x).squeeze(0)
            biased = logits + bias_t_final
        return int(torch.argmax(biased).item())

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
