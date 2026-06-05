"""CSA — Channel-Spectral Action-Influence (run 20260606-30-auto).

Realizes the hypothesis in worklogs/runs/20260606-30-auto/hypothesis.md.

Core primitive: per-(action, channel) Fiedler-ascent Phi[a, m] derived from
the second eigenvector of the channel-m-reweighted observation-transition
Laplacian L_m. Policy logits are nudged additively by a Pareto-non-dominance
count over Phi[a, :]. The policy net is trained only by entropy
regularization (no actor-critic, no Bellman, no log-prob * reward weighting,
no scalarization).
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


# ----------------------------- helpers ------------------------------------ #

def _flat_obs(obs):
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


def _hash_bucket(obs, n_buckets: int, rp_matrix: np.ndarray) -> int:
    """Random-projection hash over a bounded online vocabulary."""
    flat = _flat_obs(obs)
    d = rp_matrix.shape[1]
    if flat.size < d:
        v = np.zeros(d, dtype=np.float32)
        v[: flat.size] = flat
    else:
        v = flat[:d]
    proj = rp_matrix @ v
    # 16-bit signature -> bucket via modulo n_buckets
    bits = (proj > 0).astype(np.uint64)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h % n_buckets)


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fiedler_vector(W_m: np.ndarray, W_backbone: np.ndarray, eps: float = 1e-3):
    """Compute the Fiedler vector and eigenvalue gap for the channel-m
    symmetrized weighted Laplacian L_m = D - (W_sym), with W_sym built from
    (W_m + eps * W_backbone) symmetrized.

    Returns (f_m, gap) where gap = lambda_2 - lambda_1 ~ lambda_2 (since
    lambda_1 is exactly 0 for a connected graph). Falls back to zeros and
    gap=0 when the graph is disconnected/degenerate.
    """
    n = W_m.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32), 0.0
    W_eff = W_m + eps * W_backbone
    W_sym = 0.5 * (W_eff + W_eff.T)
    # zero diagonal
    np.fill_diagonal(W_sym, 0.0)
    deg = W_sym.sum(axis=1)
    if not np.any(deg > 0):
        return np.zeros(n, dtype=np.float32), 0.0
    L = np.diag(deg) - W_sym
    # Symmetric eigendecomposition; n is small (capped). Use numpy.
    try:
        evals, evecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.zeros(n, dtype=np.float32), 0.0
    # eigh sorts ascending. lambda_1 ~ 0 (or near-0); take 2nd smallest
    # nonzero eigenvalue.
    # Use index 1 directly (Fiedler).
    if evals.shape[0] < 2:
        return np.zeros(n, dtype=np.float32), 0.0
    lam1 = float(evals[0])
    lam2 = float(evals[1])
    gap = lam2 - max(lam1, 0.0)
    f = evecs[:, 1].astype(np.float32)
    # check collinearity with all-ones (degenerate connected-component case)
    ones = np.ones(n, dtype=np.float32) / np.sqrt(n)
    if abs(float(np.dot(f, ones))) > 0.999:
        # collinear with constant -> degenerate
        return f, 0.0
    return f, gap


def _pareto_nudge(phi: np.ndarray, alpha: float) -> np.ndarray:
    """phi: [n_actions, k]. Returns delta_logit of shape [n_actions]
    using Pareto-non-dominance count: delta_logit(a) = alpha * (n_a - m_a).
    """
    A, k = phi.shape
    if A == 0:
        return np.zeros(0, dtype=np.float32)
    delta = np.zeros(A, dtype=np.float32)
    for a in range(A):
        n_a = 0
        m_a = 0
        for b in range(A):
            if b == a:
                continue
            ge = np.all(phi[a] >= phi[b])
            gt = np.any(phi[a] > phi[b])
            le = np.all(phi[a] <= phi[b])
            lt = np.any(phi[a] < phi[b])
            if ge and gt:
                n_a += 1
            if le and lt:
                m_a += 1
        delta[a] = alpha * (n_a - m_a)
    return delta


# ----------------------------- training ----------------------------------- #

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))

    env = harness.make_env(env_id, seed)
    is_vector_env = harness.ENV_TYPE[env_id] == "vector"

    # action space
    action_space = env.action_space
    if hasattr(action_space, "n"):
        n_actions = int(action_space.n)
    else:
        env.close()
        # CSA only defined for discrete actions; degrade to random.
        def _random_policy(_obs):
            return action_space.sample()
        print(f"[train] env={env_id} non-discrete action space; CSA falls back", flush=True)
        return _random_policy

    # probe obs to size the random-projection hash and policy input
    obs0, _ = env.reset(seed=seed)
    flat0 = _flat_obs(obs0)
    in_dim = int(flat0.size)

    # Random projection matrix for hashing (16-bit signature)
    rp_dim = min(in_dim, 64)
    sig_bits = 16
    rp_matrix = rng.standard_normal((sig_bits, rp_dim)).astype(np.float32)

    # bucket vocabulary cap
    N_BUCKETS = 256

    # Determine k (number of vector channels)
    # Step once to inspect info['vector'] if present
    a_probe = action_space.sample() if not hasattr(action_space, "n") else 0
    obs1, r1, term1, trunc1, info1 = env.step(a_probe)
    if is_vector_env and "vector" in info1:
        k = int(np.asarray(info1["vector"]).size)
    else:
        # scalar env: build k=2 channels from (reward, step_penalty=-1).
        # This is the substrate-allowed reweighting since info['vector']
        # is not exposed for scalar envs; the hypothesis explicitly relies
        # on per-step k-vector reweighting and predicts channel-degeneracy
        # collapse (failure mode (a)) when reward channel is sparse.
        k = 2
    # reset for fresh training rollout
    obs, _ = env.reset(seed=seed)

    # Channel weight tensors W_m: shape [k, N, N]; W_backbone: [N, N]
    W_m = np.zeros((k, N_BUCKETS, N_BUCKETS), dtype=np.float32)
    W_backbone = np.zeros((N_BUCKETS, N_BUCKETS), dtype=np.float32)

    # Phi accumulators: per (action, channel) running mean, plus counts.
    # We accumulate Fiedler-ascent samples online and recompute f_m every K eps.
    phi_sum = np.zeros((n_actions, k), dtype=np.float64)
    phi_count = np.zeros((n_actions, k), dtype=np.float64)
    phi_mean = np.zeros((n_actions, k), dtype=np.float32)  # for use at decision time

    # Per-channel Fiedler vectors
    fiedler = np.zeros((k, N_BUCKETS), dtype=np.float32)
    eig_gaps = np.zeros(k, dtype=np.float32)

    # Hash bucket cache: map raw signature -> bucket id (we use sig % N_BUCKETS)
    bucket_visits = np.zeros(N_BUCKETS, dtype=np.int64)

    # Policy net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(in_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    ENT_COEF = 0.05
    ALPHA = 1.0  # Pareto nudge magnitude
    K_EPS_REFRESH = 4  # recompute Fiedler vectors every K episodes
    GRAD_CLIP = 1.0

    # Episode buffer for entropy update (no critic, no return weighting)
    log_probs_buf: list[torch.Tensor] = []
    ent_buf: list[torch.Tensor] = []

    episode_count = 0
    step_count = 0
    obs_now = obs
    bucket_now = _hash_bucket(obs_now, N_BUCKETS, rp_matrix)
    bucket_visits[bucket_now] += 1
    ep_steps = 0
    ep_steps_max = harness.MAX_EPISODE_STEPS

    # logging
    last_log_t = t0
    diag_total_revisits = 0
    diag_total_transitions = 0

    def _refresh_fiedler():
        for m in range(k):
            f, gap = _fiedler_vector(W_m[m], W_backbone)
            fiedler[m] = f
            eig_gaps[m] = gap

    def _recompute_phi_means():
        # phi_mean = phi_sum / phi_count (with zero where count==0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(phi_count > 0, phi_sum / np.maximum(phi_count, 1.0), 0.0)
        return mean.astype(np.float32)

    while time.monotonic() - t0 < time_budget_s - 1.0:
        # forward pass for action selection
        obs_t = torch.from_numpy(_flat_obs(obs_now)).to(device).unsqueeze(0)
        logits_theta = policy(obs_t).squeeze(0)
        # delta_logit from Pareto vote on phi_mean
        delta = _pareto_nudge(phi_mean, ALPHA)
        delta_t = torch.from_numpy(delta).to(device)
        logits = logits_theta + delta_t
        dist = torch.distributions.Categorical(logits=logits)
        action = int(dist.sample().item())
        log_p = dist.log_prob(torch.tensor(action, device=device))
        ent = dist.entropy()
        log_probs_buf.append(log_p)
        ent_buf.append(ent)

        # step env
        obs_next, r, term, trunc, info = env.step(action)
        # k-vector for this transition
        if is_vector_env and "vector" in info:
            v = np.asarray(info["vector"], dtype=np.float32).reshape(-1)
            if v.size != k:
                # pad/truncate just in case
                tmp = np.zeros(k, dtype=np.float32)
                tmp[: min(v.size, k)] = v[: min(v.size, k)]
                v = tmp
        else:
            # scalar env: (reward, step-penalty)
            v = np.array([float(r), -1.0], dtype=np.float32)

        bucket_next = _hash_bucket(obs_next, N_BUCKETS, rp_matrix)
        bucket_visits[bucket_next] += 1
        diag_total_transitions += 1
        if bucket_visits[bucket_next] > 1:
            diag_total_revisits += 1

        # Edge-weight accumulation (one edge weight per channel per transition)
        u, v_idx = bucket_now, bucket_next
        W_backbone[u, v_idx] += 1.0
        for m in range(k):
            W_m[m, u, v_idx] += float(v[m])

        # Per-(action, channel) Fiedler-ascent contribution (uses current
        # cached Fiedler vectors; phi accumulates online while f_m is
        # refreshed every K_EPS_REFRESH episodes)
        for m in range(k):
            asc = float(fiedler[m, v_idx] - fiedler[m, u])
            phi_sum[action, m] += asc
            phi_count[action, m] += 1.0

        bucket_now = bucket_next
        obs_now = obs_next
        step_count += 1
        ep_steps += 1
        done = bool(term) or bool(trunc) or ep_steps >= ep_steps_max

        if done:
            episode_count += 1
            # entropy-only policy update (no return weighting, no Bellman)
            if log_probs_buf:
                ent_mean = torch.stack(ent_buf).mean()
                # Loss = -ENT_COEF * entropy (maximize entropy)
                loss = -ENT_COEF * ent_mean
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
                optimizer.step()
            log_probs_buf = []
            ent_buf = []

            # refresh phi means
            phi_mean = _recompute_phi_means()

            # refresh fiedler vectors every K episodes
            if episode_count % K_EPS_REFRESH == 0:
                _refresh_fiedler()
                # log diagnostics
                if time.monotonic() - last_log_t > 5.0:
                    n_visited = int((bucket_visits > 0).sum())
                    mean_revisit = (
                        float(diag_total_revisits) / max(diag_total_transitions, 1)
                    )
                    n_degenerate = int((eig_gaps < 1e-6).sum())
                    print(
                        f"[csa] env={env_id} ep={episode_count} steps={step_count} "
                        f"|V|={n_visited} mean_revisit_frac={mean_revisit:.3f} "
                        f"eig_gaps={np.array2string(eig_gaps, precision=3)} "
                        f"n_degenerate={n_degenerate}/{k} "
                        f"t={time.monotonic()-t0:.1f}s",
                        flush=True,
                    )
                    last_log_t = time.monotonic()

            # reset env
            obs_now, _ = env.reset(seed=seed + 1000 + episode_count)
            bucket_now = _hash_bucket(obs_now, N_BUCKETS, rp_matrix)
            bucket_visits[bucket_now] += 1
            ep_steps = 0

    env.close()

    # Final Fiedler refresh and phi mean
    _refresh_fiedler()
    phi_mean = _recompute_phi_means()
    delta_final = _pareto_nudge(phi_mean, ALPHA)
    delta_final_t = torch.from_numpy(delta_final).to(device)

    print(
        f"[train] env={env_id} seed={seed} env_steps={step_count} "
        f"episodes={episode_count} train_s={time.monotonic()-t0:.1f} "
        f"budget_s={time_budget_s} eig_gaps={np.array2string(eig_gaps, precision=3)}",
        flush=True,
    )

    policy.eval()

    @torch.no_grad()
    def policy_fn(obs):
        x = torch.from_numpy(_flat_obs(obs)).to(device).unsqueeze(0)
        logits_theta = policy(x).squeeze(0)
        logits = logits_theta + delta_final_t
        # deterministic argmax for evaluation
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
