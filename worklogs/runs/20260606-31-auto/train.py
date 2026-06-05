"""KSV — Kernel-Smoothed Suffix-Pair Vote.

Realizes hypothesis 20260606-31-auto:
  - Experience object: sliding replay buffer of full trajectories with
    per-step vector signal v_t (from info["vector"] on vector envs;
    terminal-only [reward, -1] on sparse envs as a step-penalty channel).
  - Core primitive: per-(action, channel) sign-vote tensor
        SV[a, m] = sum over pairs (i, j) with a_t^i = a, a_t^j != a of
                   w_ij(o) * sign(delta_m^{ij})
    with w_ij(o) = kappa(e(o), e(o_t^i)) * kappa(e(o), e(o_t^j)),
    kappa = Gaussian on a learned auxiliary forward-model embedding,
    matched step t = step in the trajectory whose embedding is closest
    in cosine to e(o), delta_m^{ij} = (c_T^i - c_t^i)[m] - (c_T^j - c_t^j)[m].
  - Improvement operator: per-decision logit nudge alpha * (D-(a) - D+(a))
    with strict coordinate-wise Pareto dominance.
  - Execution rule: a ~ softmax(logits_theta(o) + alpha * (D- - D+)).
  - Vector feedback: only signs of channel-wise suffix-cumulant differences
    enter; no scalar collapse; cross-channel rule is coordinate-wise partial
    order only.
  - Side information: vector diagnostics + transition geometry (auxiliary
    forward-model embedding, MSE-trained, used as embedding only).
  - Base policy update: REINFORCE-style entropy regularizer ONLY (no scalar
    reward gradient on the policy).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ----- hyperparameters (intentionally fixed; not tuned to chase a number) -----
EMBED_DIM = 32
HIDDEN = 64
LR_POLICY = 3e-4
LR_FWD = 3e-4
ENTROPY_COEF = 0.01
ALPHA_NUDGE = 1.0           # scale of Pareto logit nudge
KERNEL_BANDWIDTH = 1.0      # Gaussian kernel bandwidth on cosine distance
N_PAIRS_M = 128             # M, sub-sampling count
BUFFER_TRAJ_CAP = 64        # sliding window capacity (trajectories)
UPDATE_PERIOD_B = 32        # one policy update per B env steps
FWD_BATCH = 64
MAX_TRAJ_STEPS = 500
LOG_EVERY = 50              # diagnostic log period (decision steps)


def _flatten_obs(obs) -> np.ndarray:
    arr = np.asarray(obs)
    return arr.astype(np.float32).reshape(-1)


def _obs_dim_from_env(env, env_id: str) -> int:
    obs, _ = env.reset()
    return _flatten_obs(obs).shape[0]


def _vector_dim(env_id: str) -> int:
    # k = number of cumulant channels (vector envs use info["vector"];
    # sparse envs use [scalar_reward, -1] -> k=2 (reward + step-penalty);
    # craftax uses [scalar_reward, -1] -> k=2.)
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    return 2


def _vector_signal(env_id: str, reward: float, info: dict) -> np.ndarray:
    if "vector" in info:
        return np.asarray(info["vector"], dtype=np.float64)
    # Sparse / scalar env: reward channel + step-penalty channel.
    return np.array([float(reward), -1.0], dtype=np.float64)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardModel(nn.Module):
    """Self-supervised forward model: predict next obs from (o, a).

    The encoder produces e(o) which is used as the kernel embedding.
    The model is trained by MSE on the predicted next-observation, but
    we only ever use `encode(o)` as an embedding (no planning / value).
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, EMBED_DIM),
        )
        self.predictor = nn.Sequential(
            nn.Linear(EMBED_DIM + n_actions, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, obs_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        h = torch.cat([z, a_onehot], dim=-1)
        return self.predictor(h)


def _strict_pareto_counts(SV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute D-(a) and D+(a) under strict coordinate-wise Pareto.

    Strict dominance: a' >- a iff SV[a', m] >= SV[a, m] for all m AND
    SV[a', m] > SV[a, m] for some m. Then:
      D-(a) = #{a' : SV[a, :] strictly dominates SV[a', :]}
      D+(a) = #{a' : SV[a', :] strictly dominates SV[a, :]}
    """
    n = SV.shape[0]
    D_minus = np.zeros(n, dtype=np.float64)
    D_plus = np.zeros(n, dtype=np.float64)
    for a in range(n):
        diff = SV[a, :][None, :] - SV  # shape (n, k); positive => a beats a' on m
        ge_all = np.all(diff >= 0.0, axis=1)
        gt_any = np.any(diff > 0.0, axis=1)
        a_dominates = ge_all & gt_any
        a_dominates[a] = False
        # a' dominates a:
        ge_all_b = np.all(-diff >= 0.0, axis=1)
        gt_any_b = np.any(-diff > 0.0, axis=1)
        a_dominated = ge_all_b & gt_any_b
        a_dominated[a] = False
        D_minus[a] = a_dominates.sum()
        D_plus[a] = a_dominated.sum()
    return D_minus, D_plus


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = harness.make_env(env_id, seed)
    if not hasattr(env.action_space, "n"):
        # Falls back to random; only discrete-action envs in the panel.
        action_space = env.action_space
        env.close()

        def policy_fn(_obs):
            return action_space.sample()
        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} (non-discrete; random fallback)",
            flush=True,
        )
        return policy_fn

    n_actions = int(env.action_space.n)
    obs0, _ = env.reset(seed=seed)
    obs_dim = _flatten_obs(obs0).shape[0]
    k_dim = _vector_dim(env_id)

    policy = PolicyNet(obs_dim, n_actions).to(device)
    fwd = ForwardModel(obs_dim, n_actions).to(device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR_POLICY)
    opt_fwd = torch.optim.Adam(fwd.parameters(), lr=LR_FWD)

    # Sliding replay buffer of full trajectories.
    # Each entry: dict with 'obs' (np.ndarray T x obs_dim), 'acts' (np.ndarray T),
    # 'cum_suffix' (np.ndarray T x k of c_T - c_t including step t; we precompute
    # at trajectory close), 'embeds' (np.ndarray T x EMBED_DIM).
    buffer: list[dict] = []

    # Diagnostic accumulators.
    diag_pareto_nonzero = 0
    diag_total_decisions = 0
    diag_pair_weight_thresh = 0.0
    diag_mean_active_pairs = 0.0
    diag_chan_sign_counts = np.zeros((k_dim, 3), dtype=np.int64)  # neg/zero/pos

    # ----- helpers using current fwd model -----
    def encode_np(obs_arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            z = fwd.encode(x)
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z.detach().cpu().numpy()

    def add_trajectory(obs_seq, act_seq, vec_seq):
        """Close a trajectory: precompute suffix cumulants and embeddings."""
        T = len(obs_seq)
        if T == 0:
            return
        obs_arr = np.stack(obs_seq, axis=0).astype(np.float32)
        act_arr = np.asarray(act_seq, dtype=np.int64)
        vec_arr = np.asarray(vec_seq, dtype=np.float64).reshape(T, k_dim)
        # Total cumulant c_T = sum_{u=0}^{T-1} v_u.
        total = vec_arr.sum(axis=0)
        # cum_prefix[t] = sum_{u=0}^{t-1} v_u  (cumulant up to but excluding t)
        cum_prefix = np.zeros((T, k_dim), dtype=np.float64)
        cum_prefix[1:] = np.cumsum(vec_arr[:-1], axis=0)
        # Suffix cumulant c_T - c_t (we treat c_t = cum_prefix[t], i.e.
        # contribution of steps t..T-1 inclusive).
        suffix = total[None, :] - cum_prefix  # shape (T, k)
        embeds = encode_np(obs_arr)  # (T, EMBED_DIM), L2-normalized
        buffer.append({
            "obs": obs_arr,
            "acts": act_arr,
            "suffix": suffix,
            "embeds": embeds,
        })
        if len(buffer) > BUFFER_TRAJ_CAP:
            buffer.pop(0)

    def refresh_embeddings():
        """Re-encode buffered trajectories with the current fwd model."""
        for tr in buffer:
            tr["embeds"] = encode_np(tr["obs"])

    def compute_SV(o_flat: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compute SV[a, m] via M-pair sub-sampling.

        Returns SV (n_actions x k_dim), diagnostics dict.
        """
        diag = {"active_pairs": 0, "chan_signs": np.zeros((k_dim, 3), dtype=np.int64)}
        SV = np.zeros((n_actions, k_dim), dtype=np.float64)
        if len(buffer) < 2:
            # Need at least two trajectories to form a pair (i != j).
            return SV, diag
        # Encode current obs with current fwd model.
        e_o = encode_np(o_flat)[0]  # (EMBED_DIM,) L2-normalized
        # Find matched step per trajectory: argmax cosine similarity to e_o.
        # All traj embeds are L2-normalized; matched = argmax dot.
        matched_idx = np.empty(len(buffer), dtype=np.int64)
        matched_emb = np.empty((len(buffer), EMBED_DIM), dtype=np.float32)
        matched_act = np.empty(len(buffer), dtype=np.int64)
        matched_sfx = np.empty((len(buffer), k_dim), dtype=np.float64)
        for ti, tr in enumerate(buffer):
            sims = tr["embeds"] @ e_o.astype(np.float32)
            t_match = int(np.argmax(sims))
            matched_idx[ti] = t_match
            matched_emb[ti] = tr["embeds"][t_match]
            matched_act[ti] = tr["acts"][t_match]
            matched_sfx[ti] = tr["suffix"][t_match]

        n_traj = len(buffer)
        # Sample M unordered pairs (i != j) uniformly with replacement at the
        # pair level. We allow with-replacement subsampling per the M-pair
        # rule; the variance ablation slot covers this.
        i_idx = rng.integers(0, n_traj, size=N_PAIRS_M)
        j_idx = rng.integers(0, n_traj, size=N_PAIRS_M)
        keep = i_idx != j_idx
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
        if len(i_idx) == 0:
            return SV, diag

        # Kernel weights: kappa(e(o), e_i) * kappa(e(o), e_j) on cosine distance.
        # cos_dist = 1 - dot (since both L2-normalized).
        ce_o = e_o.astype(np.float32)
        ei = matched_emb[i_idx]
        ej = matched_emb[j_idx]
        d_i = 1.0 - ei @ ce_o
        d_j = 1.0 - ej @ ce_o
        wi = np.exp(-(d_i ** 2) / (2.0 * KERNEL_BANDWIDTH ** 2))
        wj = np.exp(-(d_j ** 2) / (2.0 * KERNEL_BANDWIDTH ** 2))
        w = wi * wj  # (M_eff,)
        active = int((w > 1e-3).sum())
        diag["active_pairs"] = active

        ai = matched_act[i_idx]
        aj = matched_act[j_idx]
        # delta_m^{ij} = suffix_i - suffix_j (per channel)
        delta = matched_sfx[i_idx] - matched_sfx[j_idx]  # (M_eff, k)
        sign_delta = np.sign(delta)  # in {-1, 0, +1}

        # We want SV[a, m] += w * sign(delta) for pairs where a_i = a, a_j != a.
        # By symmetry of the pair (i, j) vs (j, i): also pairs where a_j = a,
        # a_i != a contribute -sign(delta)·w to SV[a, m] (because we'd swap i<->j,
        # and delta flips sign). To avoid double-counting, we treat the ordered
        # pair (i, j) as the canonical "i picked action a, j picked something
        # else" case: for each pair, we accumulate into SV[a_i] when a_i != a_j
        # using +sign(delta), and into SV[a_j] using -sign(delta). The hypothesis
        # only counts each unordered pair once with the convention a_t^i = a,
        # a_t^j != a; but the unordered-pair contribution to action a vs action
        # a' is exactly w * sign((suffix on a-side) - (suffix on other-side))
        # which is symmetric under our two-sided accumulation. We therefore use
        # the two-sided form, which preserves the sign-of-difference semantic
        # and ensures SV[a, m] is anti-symmetric across the action partition.
        diff_act = ai != aj
        if not np.any(diff_act):
            # No useful pairs.
            for m in range(k_dim):
                vals, counts = np.unique(sign_delta[:, m].astype(np.int64),
                                         return_counts=True)
                for v, c in zip(vals, counts):
                    diag["chan_signs"][m, int(v) + 1] += int(c)
            return SV, diag

        w_e = w[diff_act]
        sd_e = sign_delta[diff_act]
        ai_e = ai[diff_act]
        aj_e = aj[diff_act]

        # Accumulate into SV.
        for m in range(k_dim):
            contrib_i = w_e * sd_e[:, m]   # action_i side
            contrib_j = -w_e * sd_e[:, m]  # action_j side (opposite sign)
            np.add.at(SV[:, m], ai_e, contrib_i)
            np.add.at(SV[:, m], aj_e, contrib_j)
            # diag histogram: count signs (over all sampled pairs, not just diff_act).
            vals, counts = np.unique(sign_delta[:, m].astype(np.int64),
                                     return_counts=True)
            for v, c in zip(vals, counts):
                diag["chan_signs"][m, int(v) + 1] += int(c)
        return SV, diag

    # ----- training loop -----
    obs_seq: list[np.ndarray] = []
    act_seq: list[int] = []
    vec_seq: list[np.ndarray] = []

    obs, _ = env.reset(seed=seed + 1)
    obs_flat = _flatten_obs(obs)
    env_steps = 0
    ep_steps = 0
    log_actions: list[torch.Tensor] = []  # logits at chosen actions for entropy update

    last_pi_step = 0

    while time.monotonic() - t0 < time_budget_s * 0.95:
        # Compute Pareto nudge from buffered trajectories (uses current fwd embeds).
        SV, diag = compute_SV(obs_flat)
        D_minus, D_plus = _strict_pareto_counts(SV)
        nudge = ALPHA_NUDGE * (D_minus - D_plus)  # (n_actions,)
        margin_nonzero = bool(np.any(np.abs(nudge) > 0))

        # Policy logits.
        x = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(x).squeeze(0)
        nudge_t = torch.as_tensor(nudge, dtype=torch.float32, device=device)
        adj_logits = logits + nudge_t
        probs = F.softmax(adj_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = int(dist.sample().item())

        diag_total_decisions += 1
        if margin_nonzero:
            diag_pareto_nonzero += 1
        diag_mean_active_pairs += diag["active_pairs"]
        diag_chan_sign_counts += diag["chan_signs"]

        # Step env.
        next_obs, reward, term, trunc, info = env.step(action)
        v_t = _vector_signal(env_id, reward, info)
        obs_seq.append(obs_flat.copy())
        act_seq.append(action)
        vec_seq.append(v_t)

        # Stash logits at chosen action for entropy regularizer update later.
        log_actions.append(adj_logits.detach())  # not used in grad; only count
        env_steps += 1
        ep_steps += 1

        # Periodic policy + forward-model update.
        if env_steps - last_pi_step >= UPDATE_PERIOD_B:
            last_pi_step = env_steps
            # Entropy regularizer update on policy: maximize entropy of the
            # adjusted-logit distribution at recently visited states. We
            # recompute logits with grad on a small recent batch.
            recent_obs = obs_seq[-UPDATE_PERIOD_B:]
            xb = torch.as_tensor(np.stack(recent_obs, axis=0),
                                 dtype=torch.float32, device=device)
            logits_b = policy(xb)
            log_probs_b = F.log_softmax(logits_b, dim=-1)
            probs_b = log_probs_b.exp()
            entropy_b = -(probs_b * log_probs_b).sum(dim=-1).mean()
            loss_pi = -ENTROPY_COEF * entropy_b  # only entropy term
            opt_pi.zero_grad()
            loss_pi.backward()
            opt_pi.step()

            # Forward-model MSE update on a random batch from buffer + recent
            # transitions.
            if len(buffer) > 0 or len(obs_seq) > 1:
                batch_obs = []
                batch_act = []
                batch_next = []
                # Pull from current trajectory (in-progress).
                if len(obs_seq) >= 2:
                    n_take = min(FWD_BATCH // 2, len(obs_seq) - 1)
                    idxs = rng.integers(0, len(obs_seq) - 1, size=n_take)
                    for ix in idxs:
                        batch_obs.append(obs_seq[ix])
                        batch_act.append(act_seq[ix])
                        batch_next.append(obs_seq[ix + 1])
                # Pull from buffered trajectories.
                if len(buffer) > 0:
                    n_take2 = max(FWD_BATCH - len(batch_obs), 0)
                    for _ in range(n_take2):
                        ti = int(rng.integers(0, len(buffer)))
                        tr = buffer[ti]
                        if tr["obs"].shape[0] < 2:
                            continue
                        ix = int(rng.integers(0, tr["obs"].shape[0] - 1))
                        batch_obs.append(tr["obs"][ix])
                        batch_act.append(int(tr["acts"][ix]))
                        batch_next.append(tr["obs"][ix + 1])
                if len(batch_obs) >= 4:
                    xb_o = torch.as_tensor(np.stack(batch_obs, axis=0),
                                           dtype=torch.float32, device=device)
                    a_oh = F.one_hot(
                        torch.as_tensor(batch_act, dtype=torch.long, device=device),
                        num_classes=n_actions,
                    ).float()
                    xb_n = torch.as_tensor(np.stack(batch_next, axis=0),
                                           dtype=torch.float32, device=device)
                    pred = fwd(xb_o, a_oh)
                    loss_fwd = F.mse_loss(pred, xb_n)
                    opt_fwd.zero_grad()
                    loss_fwd.backward()
                    opt_fwd.step()

        # Diagnostic logging.
        if diag_total_decisions % LOG_EVERY == 0:
            margin_frac = diag_pareto_nonzero / max(1, diag_total_decisions)
            mean_active = diag_mean_active_pairs / max(1, diag_total_decisions)
            chan_sign_frac_zero = (
                diag_chan_sign_counts[:, 1]
                / np.maximum(diag_chan_sign_counts.sum(axis=1), 1)
            )
            print(
                f"[ksv] env={env_id} step={env_steps} "
                f"buffer={len(buffer)} pareto_margin_nonzero_frac={margin_frac:.3f} "
                f"mean_active_pairs={mean_active:.1f} "
                f"chan_zero_frac={np.array2string(chan_sign_frac_zero, precision=3)} "
                f"t={time.monotonic() - t0:.1f}s",
                flush=True,
            )

        done = bool(term) or bool(trunc) or ep_steps >= MAX_TRAJ_STEPS
        if done:
            add_trajectory(obs_seq, act_seq, vec_seq)
            obs_seq, act_seq, vec_seq = [], [], []
            obs, _ = env.reset(seed=seed + 1 + env_steps)
            obs_flat = _flatten_obs(obs)
            ep_steps = 0
            # Refresh embeddings periodically (cheap; buffer is small).
            refresh_embeddings()
        else:
            obs = next_obs
            obs_flat = _flatten_obs(obs)

    # Close last trajectory if any.
    if obs_seq:
        add_trajectory(obs_seq, act_seq, vec_seq)
    refresh_embeddings()
    env.close()

    # Snapshot model state for deterministic deployment policy_fn.
    policy.eval()
    fwd.eval()

    # Snapshot buffer as immutable arrays (shallow copy is fine; we read only).
    final_buffer = list(buffer)

    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} "
        f"train_s={time.monotonic() - t0:.1f} budget_s={time_budget_s} "
        f"buffered_trajs={len(final_buffer)} "
        f"pareto_margin_nonzero_frac="
        f"{diag_pareto_nonzero / max(1, diag_total_decisions):.3f}",
        flush=True,
    )

    def policy_fn(obs_in):
        o_flat = _flatten_obs(obs_in)
        # Recompute SV with the (frozen) buffer + frozen fwd embeds.
        if len(final_buffer) >= 2:
            with torch.no_grad():
                e_o_t = fwd.encode(
                    torch.as_tensor(o_flat, dtype=torch.float32, device=device).unsqueeze(0)
                )
                e_o_t = F.normalize(e_o_t, dim=-1, eps=1e-8)
                e_o = e_o_t.detach().cpu().numpy()[0].astype(np.float32)
            matched_emb = np.empty((len(final_buffer), EMBED_DIM), dtype=np.float32)
            matched_act = np.empty(len(final_buffer), dtype=np.int64)
            matched_sfx = np.empty((len(final_buffer), k_dim), dtype=np.float64)
            for ti, tr in enumerate(final_buffer):
                sims = tr["embeds"] @ e_o
                t_match = int(np.argmax(sims))
                matched_emb[ti] = tr["embeds"][t_match]
                matched_act[ti] = tr["acts"][t_match]
                matched_sfx[ti] = tr["suffix"][t_match]
            n_traj = len(final_buffer)
            i_idx = rng.integers(0, n_traj, size=N_PAIRS_M)
            j_idx = rng.integers(0, n_traj, size=N_PAIRS_M)
            keep = i_idx != j_idx
            i_idx, j_idx = i_idx[keep], j_idx[keep]
            SV = np.zeros((n_actions, k_dim), dtype=np.float64)
            if len(i_idx) > 0:
                d_i = 1.0 - matched_emb[i_idx] @ e_o
                d_j = 1.0 - matched_emb[j_idx] @ e_o
                w = np.exp(-(d_i ** 2) / (2.0 * KERNEL_BANDWIDTH ** 2)) * \
                    np.exp(-(d_j ** 2) / (2.0 * KERNEL_BANDWIDTH ** 2))
                ai = matched_act[i_idx]
                aj = matched_act[j_idx]
                delta = matched_sfx[i_idx] - matched_sfx[j_idx]
                sd = np.sign(delta)
                diff_act = ai != aj
                if np.any(diff_act):
                    w_e = w[diff_act]
                    sd_e = sd[diff_act]
                    ai_e = ai[diff_act]
                    aj_e = aj[diff_act]
                    for m in range(k_dim):
                        np.add.at(SV[:, m], ai_e, w_e * sd_e[:, m])
                        np.add.at(SV[:, m], aj_e, -w_e * sd_e[:, m])
            D_minus, D_plus = _strict_pareto_counts(SV)
            nudge = ALPHA_NUDGE * (D_minus - D_plus)
        else:
            nudge = np.zeros(n_actions, dtype=np.float64)

        with torch.no_grad():
            x = torch.as_tensor(o_flat, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(x).squeeze(0).detach().cpu().numpy()
        adj = logits + nudge
        # Deterministic policy: argmax of adjusted logits.
        return int(np.argmax(adj))

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
