"""CHX — Cumulant-Hull Extremality candidate.

Hypothesis: Within a single trajectory, weight the per-step REINFORCE-style
log-prob update by `(\\hat h_t - 1/T)` where `h_t` is the L2 distance by which
the convex hull of the cumulant-trace `c_t = sum_{s<=t} v_s` would shrink if
point `c_t` were removed (i.e., the leave-one-out hull-shrinkage). Standardize
`\\hat h_t = h_t / sum_t h_t`. The weight is a within-trajectory geometric
quantity in R^k — it is NOT a value, advantage, return, scalar reward, or
linear scalarization w^T v. No critic, no baseline beyond the centered
per-trajectory weight, no replay, no cross-trajectory comparison.

Vector envs: the per-step `info["vector"]` is consumed component-wise. MAD
standardization is fit once from the first batch of rollouts and frozen.
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


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


class MLPPolicy(nn.Module):
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


def obs_to_vec(obs) -> np.ndarray:
    """Flatten any obs to a 1-D float vector."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr


# -----------------------------------------------------------------------------
# Hull contribution
# -----------------------------------------------------------------------------


def hull_contributions(C: np.ndarray) -> np.ndarray:
    """Return per-point leave-one-out hull-shrinkage h_t for the cumulant-trace.

    C has shape (N, k) where N = T+1 (includes c_{-1} = 0 and c_0..c_T).
    The per-step h_t for step t is the L2 distance from c_t to the convex hull
    of the remaining points {c_{-1},...,c_T} \\ {c_t}.

    Returned array has length N (one per point, including the leading c_{-1}).
    The caller drops the first element to get one weight per step.

    Falls back to robust per-axis extremality when the hull is ill-defined
    (k=1, fewer than k+1 unique points, scipy errors). Never returns NaNs.
    """
    N, k = C.shape
    if N <= 1:
        return np.zeros(N, dtype=np.float64)
    # 1-D fallback: hull is the segment [min, max]; LOO shrinkage is non-zero
    # only at the unique min/max points and equals (next-extreme distance).
    if k == 1:
        x = C[:, 0]
        h = np.zeros(N, dtype=np.float64)
        # leave-one-out min/max
        order = np.argsort(x)
        if N >= 2:
            i_min, i_max = order[0], order[-1]
            i_min2, i_max2 = order[1], order[-2]
            h[i_min] = max(0.0, x[i_min2] - x[i_min])
            h[i_max] = max(0.0, x[i_max] - x[i_max2])
        return h

    # k >= 2 — try scipy ConvexHull leave-one-out.
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        # No scipy -> fall back to per-axis extremality measure.
        return _per_axis_extremality(C)

    # Compute reference hull volume.
    try:
        full_hull = ConvexHull(C, qhull_options="QJ")
        full_vol = float(full_hull.volume)
        on_hull = set(int(i) for i in np.unique(full_hull.vertices))
    except (QhullError, ValueError, Exception):
        return _per_axis_extremality(C)

    h = np.zeros(N, dtype=np.float64)
    # For interior points, h_t = 0 by definition. Only loop over hull vertices.
    for i in on_hull:
        idx = np.arange(N) != i
        sub = C[idx]
        try:
            sub_hull = ConvexHull(sub, qhull_options="QJ")
            sub_vol = float(sub_hull.volume)
        except (QhullError, ValueError, Exception):
            sub_vol = full_vol  # treat as no shrinkage
        # Volume-shrinkage proxy (in volume units). Convert to a length-like
        # quantity by taking the (1/k)-th root so different envs are
        # comparable; clip at 0 to avoid numerical negatives.
        delta = max(0.0, full_vol - sub_vol)
        if delta <= 0.0:
            h[i] = 0.0
        else:
            h[i] = delta ** (1.0 / k)
    return h


def _per_axis_extremality(C: np.ndarray) -> np.ndarray:
    """Robust fallback: a step's contribution is the distance by which it
    extends the per-axis bounding box, summed across axes (in L2 sense).

    This preserves the spirit of "hull-extremality": only steps that push the
    cumulant trace into a previously-unattained per-axis extreme get nonzero
    weight; interior steps get zero. It is invariant to per-axis scale once
    inputs are standardized, and it is NOT a linear scalarization — the weight
    is a non-linear order-statistic.
    """
    N, k = C.shape
    h = np.zeros(N, dtype=np.float64)
    for axis in range(k):
        x = C[:, axis]
        order = np.argsort(x)
        if N >= 2:
            i_min, i_max = order[0], order[-1]
            i_min2, i_max2 = order[1], order[-2]
            h[i_min] += max(0.0, x[i_min2] - x[i_min]) ** 2
            h[i_max] += max(0.0, x[i_max] - x[i_max2]) ** 2
    return np.sqrt(h)


# -----------------------------------------------------------------------------
# MAD standardizer (frozen after first batch)
# -----------------------------------------------------------------------------


class MADStandardizer:
    def __init__(self, k: int):
        self.k = k
        self.scale = np.ones(k, dtype=np.float64)
        self.fit_done = False

    def fit(self, vecs: np.ndarray) -> None:
        if self.fit_done or len(vecs) == 0:
            return
        med = np.median(vecs, axis=0)
        mad = np.median(np.abs(vecs - med), axis=0)
        self.scale = np.where(mad > 1e-8, mad, 1.0)
        self.fit_done = True

    def __call__(self, vecs: np.ndarray) -> np.ndarray:
        return vecs / self.scale[None, :]


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # Probe obs / action / vector shape with a reset+step.
    obs, _ = env.reset(seed=int(seed))
    obs_vec = obs_to_vec(obs)
    obs_dim = obs_vec.shape[0]
    if hasattr(env.action_space, "n"):
        n_actions = int(env.action_space.n)
    else:
        env.close()
        rng2 = np.random.default_rng(seed + 99_999)
        action_space = env.action_space

        def policy_fn(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} non-discrete action space; CHX requires discrete actions; "
            f"falling back to random.",
            flush=True,
        )
        return policy_fn

    # Probe vector dimension via one step (already reset above).
    a0 = int(rng.integers(n_actions))
    _, _, _, _, info0 = env.step(a0)
    if is_vector and "vector" in info0:
        k_vec = int(np.asarray(info0["vector"]).shape[0])
    else:
        # Scalar envs: synthesize a 1-D "outcome" from scalar reward; CHX in
        # this regime is a hull on a 1-D cumulant -> degenerate (the
        # hypothesis flags this as a self-disqualifying k_eff=1 case). We run
        # the algorithm faithfully anyway; result should match random.
        k_vec = 1

    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MLPPolicy(obs_dim, n_actions).to(device)
    optim = torch.optim.Adam(policy.parameters(), kwargs := dict(lr=3e-3)) if False else torch.optim.Adam(policy.parameters(), lr=3e-3)

    # Hyper-parameters held constant (no per-env tuning).
    boltz_temperature = 1.0
    max_steps_per_ep = 500
    grad_clip = 5.0

    standardizer = MADStandardizer(k_vec)
    standardizer_fit_buffer: list[np.ndarray] = []
    fit_target = 8  # fit MAD from the first 8 rollouts of per-step vectors
    fit_count = 0

    t0 = time.monotonic()
    deadline = t0 + max(1, time_budget_s - 5)  # leave headroom for eval

    env = harness.make_env(env_id, seed + 1)
    n_episodes = 0
    n_steps = 0
    n_updates = 0
    k_eff_warned = False

    while time.monotonic() < deadline:
        obs, _ = env.reset(seed=int(seed) + 7919 * (n_episodes + 1))
        ep_obs: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_logp: list[torch.Tensor] = []
        ep_vecs: list[np.ndarray] = []

        done = False
        steps = 0
        while not done and steps < max_steps_per_ep and time.monotonic() < deadline:
            obs_v = obs_to_vec(obs)
            obs_t = torch.from_numpy(obs_v).to(device).unsqueeze(0)
            logits = policy(obs_t)[0] / boltz_temperature
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            a_t = dist.sample()
            logp_t = dist.log_prob(a_t)

            a = int(a_t.item())
            obs2, reward, term, trunc, info = env.step(a)
            done = bool(term) or bool(trunc)

            if is_vector and "vector" in info:
                v = np.asarray(info["vector"], dtype=np.float64)
                if v.shape[0] != k_vec:
                    v = np.resize(v, (k_vec,))
            else:
                v = np.array([float(reward)], dtype=np.float64)

            ep_obs.append(obs_v)
            ep_actions.append(a)
            ep_logp.append(logp_t)
            ep_vecs.append(v)

            obs = obs2
            steps += 1
            n_steps += 1

        n_episodes += 1
        T = len(ep_vecs)
        if T == 0:
            continue

        V = np.stack(ep_vecs, axis=0)  # (T, k)

        # Fit MAD standardizer from the first few rollouts, then freeze.
        if not standardizer.fit_done:
            standardizer_fit_buffer.append(V)
            fit_count += 1
            if fit_count >= fit_target:
                allv = np.concatenate(standardizer_fit_buffer, axis=0)
                standardizer.fit(allv)

        # Standardize per-step vectors with frozen MAD scale (1.0 until fit).
        V_std = standardizer(V)

        # Cumulant trace c_t = sum_{s<=t} v_s, with c_{-1} = 0 prepended.
        C = np.vstack(
            [np.zeros((1, k_vec), dtype=np.float64), np.cumsum(V_std, axis=0)]
        )  # (T+1, k)

        # k_eff diagnostic on cumulant trace deviations.
        if not k_eff_warned and n_episodes >= 4:
            X = C - C.mean(axis=0, keepdims=True)
            try:
                s = np.linalg.svd(X, compute_uv=False)
                if s.sum() > 0:
                    k_eff = float((s.sum() ** 2) / (s ** 2).sum())  # participation ratio
                else:
                    k_eff = 0.0
            except Exception:
                k_eff = float(k_vec)
            print(f"[train] k_eff~={k_eff:.2f} (k={k_vec})", flush=True)
            k_eff_warned = True

        # Per-point leave-one-out hull contributions over (T+1) points.
        h_all = hull_contributions(C)  # (T+1,)
        # Drop c_{-1} and keep weights for steps 0..T-1.
        h = h_all[1:]  # (T,)

        s_h = float(h.sum())
        if s_h <= 1e-12:
            # Degenerate trajectory (no extremal step); skip update.
            continue

        h_hat = h / s_h  # (T,) sums to 1
        weights = h_hat - 1.0 / T  # centered, sums to 0
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)

        logp_stack = torch.stack(ep_logp)  # (T,)
        # Surrogate: MAXIMIZE sum_t (h_hat - 1/T) * log pi(a_t | s_t).
        # Optimizer minimizes loss, so loss = - that.
        loss = -(weights_t * logp_stack).sum()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optim.step()
        n_updates += 1

    env.close()

    train_s = time.monotonic() - t0
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_steps} episodes={n_episodes} "
        f"updates={n_updates} train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # Greedy evaluation policy.
    policy.eval()

    def policy_fn(o: np.ndarray):
        with torch.no_grad():
            ov = obs_to_vec(o)
            ot = torch.from_numpy(ov).to(device).unsqueeze(0)
            logits = policy(ot)[0]
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
