"""TPM (Trajectory-Pair Mass-Transport) candidate.

Substrate contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120

Faithful realization of hypothesis 20260606-14-auto:
- Experience object: completed episodes, each storing the action sequence
  and the terminal vector cumulant `c_T` from `info["vector"]`. On scalar
  envs `c_T` is the scalar return (k=1) -- flagged as scalar limit.
- Core primitive: per-(episode-position-bucket, action) signed per-channel
  median-difference vector M[b, a, :] in R^k. Indexed by episode position
  only; never by any state-hash, observation hash, or cumulant bucket.
  (Position is bucketed log-spaced per the reviewer's storage cap.)
- Improvement operator: logit nudge by Pareto-non-dominance margin
  alpha * (n_a - m_a) where n_a, m_a are counts of actions a' that
  M[b, a, :] dominates / is dominated by, in the coordinate-wise partial
  order on R^k. Coefficient is a count, never a scalar magnitude.
- Execution rule: a_t ~ softmax(logit_theta(s_t, t)) with a small
  position embedding for t.
- Vector feedback rule: per-channel from end to end. No scalarization.
- Rollout-cost discipline: one env interaction per data point; one
  improvement update per N-episode buffer refresh.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ----------------------------- Hyperparameters -----------------------------
BUFFER_CAP = 256
N_REFRESH = 16
ALPHA = 0.5
MIN_COUNT_PER_SIDE = 3       # reviewer's minimum-count guard
POS_BUCKETS = 32             # log-spaced position buckets
EMBED_DIM = 16
HIDDEN = 64
LR = 3e-3
ENTROPY_COEF = 0.01          # entropy floor (failure mode (d))
DITHER_EPS = 0.05            # small action-dither (failure mode (d))
MAX_EPISODE_STEPS = 2000


# ----------------------------- Position bucketing --------------------------
def _bucket_index(t: int) -> int:
    """Log-spaced bucket index for position t (0 .. POS_BUCKETS-1).

    Bucket b corresponds roughly to positions in [2^b, 2^(b+1)). Keeps
    storage sub-linear in horizon, preserving the agent's clock as the
    indexing coordinate (no observation/state-hash dependence).
    """
    if t <= 0:
        return 0
    b = int(np.floor(np.log2(t + 1)))
    if b >= POS_BUCKETS:
        b = POS_BUCKETS - 1
    return b


# ----------------------------- Policy network ------------------------------
class PositionConditionedPolicy(nn.Module):
    """Policy conditioned on (s_t, t-bucket) via a small position embedding."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.pos_emb = nn.Embedding(POS_BUCKETS, EMBED_DIM)
        self.body = nn.Sequential(
            nn.Linear(obs_dim + EMBED_DIM, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
        )
        self.head = nn.Linear(HIDDEN, n_actions)

    def forward(self, obs: torch.Tensor, pos_bucket: torch.Tensor) -> torch.Tensor:
        emb = self.pos_emb(pos_bucket)
        h = self.body(torch.cat([obs, emb], dim=-1))
        return self.head(h)


def _flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs).astype(np.float32).reshape(-1)


# ----------------------------- TPM primitive -------------------------------
def _compute_M(buffer, n_actions: int, k: int):
    """Compute M[b, a, m] = signed median-difference between B^+_{b,a}
    (terminal cumulants of episodes whose action at a position-in-bucket-b
    equaled a) and B^-_{b,a} (terminal cumulants of episodes whose action
    at the same positions differed from a, restricted to episodes whose
    length covers that position).
    """
    pos_plus = [[[] for _ in range(n_actions)] for _ in range(POS_BUCKETS)]
    pos_minus = [[[] for _ in range(n_actions)] for _ in range(POS_BUCKETS)]

    for actions, c_T in buffer:
        for t, a in enumerate(actions):
            b = _bucket_index(int(t))
            a = int(a)
            pos_plus[b][a].append(c_T)
            for a_other in range(n_actions):
                if a_other != a:
                    pos_minus[b][a_other].append(c_T)

    M = np.zeros((POS_BUCKETS, n_actions, k), dtype=np.float64)
    valid = np.zeros((POS_BUCKETS, n_actions), dtype=bool)
    for b in range(POS_BUCKETS):
        for a in range(n_actions):
            plus = pos_plus[b][a]
            minus = pos_minus[b][a]
            if len(plus) < MIN_COUNT_PER_SIDE or len(minus) < MIN_COUNT_PER_SIDE:
                continue
            plus_arr = np.stack(plus)
            minus_arr = np.stack(minus)
            M[b, a, :] = np.median(plus_arr, axis=0) - np.median(minus_arr, axis=0)
            valid[b, a] = True
    return M, valid


def _pareto_margin(M: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """margin[b, a] = (#a' that M[b,a,:] dominates) - (#a' that dominate M[b,a,:]),
    coordinate-wise partial order on R^k.
    """
    POS, A, _ = M.shape
    margin = np.zeros((POS, A), dtype=np.float64)
    for b in range(POS):
        valid_actions = np.where(valid[b])[0]
        if len(valid_actions) < 2:
            continue
        rows = M[b, valid_actions, :]
        ge = np.all(rows[:, None, :] >= rows[None, :, :] - 1e-12, axis=2)
        gt = np.any(rows[:, None, :] > rows[None, :, :] + 1e-12, axis=2)
        dom = ge & gt
        n_dominates = dom.sum(axis=1)
        n_dominated_by = dom.sum(axis=0)
        for idx, a in enumerate(valid_actions):
            margin[b, a] = float(n_dominates[idx] - n_dominated_by[idx])
    return margin


# ----------------------------- Train loop ----------------------------------
def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 12345)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    obs0, _ = env.reset(seed=seed)
    obs_flat = _flatten_obs(obs0)
    obs_dim = int(obs_flat.size)
    if not hasattr(env.action_space, "n"):
        env.close()
        raise RuntimeError("TPM requires a discrete action space.")
    n_actions = int(env.action_space.n)

    # Probe k by stepping once.
    _, _, _, _, info_probe = env.step(0)
    if is_vector:
        k = int(np.asarray(info_probe["vector"]).size)
    else:
        k = 1
    obs1, _ = env.reset(seed=seed)
    obs = _flatten_obs(obs1)

    if not is_vector:
        print(
            f"[train] env={env_id} k=1 (scalar limit): TPM operator degenerates "
            f"to median-shift MC update per failure mode (c). Flagging scalar limit.",
            flush=True,
        )

    policy = PositionConditionedPolicy(obs_dim, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # Rolling buffer: each entry (obs_seq[T, obs_dim], action_seq[T], c_T[k]).
    buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    n_episodes = 0
    n_updates = 0
    nudge_nonzero_cells_total = 0
    nudge_decision_steps = 0
    cell_count_total = 0
    total_decision_steps = 0

    obs_seq: list[np.ndarray] = []
    action_seq: list[int] = []
    c_T_accum = np.zeros(k, dtype=np.float64)
    ep_steps = 0

    t0 = time.monotonic()

    while time.monotonic() - t0 < time_budget_s:
        b = _bucket_index(ep_steps)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            pos_t = torch.tensor([b], dtype=torch.long)
            logits = policy(obs_t, pos_t).squeeze(0).numpy()
        if rng.random() < DITHER_EPS:
            a = int(rng.integers(n_actions))
        else:
            logits_stable = logits - logits.max()
            probs = np.exp(logits_stable)
            probs = probs / max(probs.sum(), 1e-12)
            a = int(rng.choice(n_actions, p=probs))

        obs_seq.append(obs.copy())
        action_seq.append(a)

        next_obs, reward, term, trunc, info = env.step(a)
        if is_vector:
            c_T_accum += np.asarray(info["vector"], dtype=np.float64)
        else:
            c_T_accum[0] += float(reward)
        ep_steps += 1
        total_decision_steps += 1
        done = bool(term) or bool(trunc) or ep_steps >= MAX_EPISODE_STEPS

        if done:
            obs_arr = (
                np.stack(obs_seq) if obs_seq else np.zeros((0, obs_dim), np.float32)
            )
            buffer.append(
                (obs_arr, np.array(action_seq, dtype=np.int64), c_T_accum.copy())
            )
            if len(buffer) > BUFFER_CAP:
                buffer.pop(0)
            n_episodes += 1

            obs_seq = []
            action_seq = []
            c_T_accum = np.zeros(k, dtype=np.float64)
            ep_steps = 0
            obs0, _ = env.reset(seed=seed + 1000 + n_episodes)
            obs = _flatten_obs(obs0)

            # Improvement update every N_REFRESH episodes.
            if (
                n_episodes % N_REFRESH == 0
                and len(buffer) >= MIN_COUNT_PER_SIDE * 2
            ):
                buf_for_M = [(act, c) for (_o, act, c) in buffer]
                M, valid = _compute_M(buf_for_M, n_actions, k)
                margin = _pareto_margin(M, valid)

                nz = int(np.count_nonzero(margin))
                cell_count_total += int(margin.size)
                nudge_nonzero_cells_total += nz

                # Build supervised batch for the position-conditioned logit
                # nudge: for each (episode, t) decision, target_logits =
                # current_logits + alpha * margin[bucket(t), :].
                obs_batch_list = []
                bucket_batch_list = []
                offset_batch_list = []
                taken_action_list = []
                for (obs_ep, act_ep, _c) in buffer:
                    T = len(act_ep)
                    for t in range(T):
                        bb = _bucket_index(t)
                        obs_batch_list.append(obs_ep[t])
                        bucket_batch_list.append(bb)
                        taken_action_list.append(int(act_ep[t]))
                        offset_batch_list.append(margin[bb, :].copy())

                if not obs_batch_list:
                    continue

                obs_batch = torch.from_numpy(
                    np.stack(obs_batch_list).astype(np.float32)
                )
                bucket_batch = torch.tensor(bucket_batch_list, dtype=torch.long)
                offset_batch = torch.from_numpy(
                    np.stack(offset_batch_list).astype(np.float32)
                )
                taken_actions = torch.tensor(taken_action_list, dtype=torch.long)

                taken_margin = offset_batch.gather(
                    1, taken_actions.unsqueeze(1)
                ).squeeze(1)
                nudge_decision_steps += int((taken_margin.abs() > 1e-12).sum().item())

                cur_logits = policy(obs_batch, bucket_batch)
                with torch.no_grad():
                    target_logits = cur_logits + ALPHA * offset_batch
                    target_log_probs = F.log_softmax(target_logits, dim=-1)
                    target_probs = target_log_probs.exp()
                cur_log_probs = F.log_softmax(cur_logits, dim=-1)
                kl = (target_probs * (target_log_probs - cur_log_probs)).sum(dim=-1)
                cur_probs = cur_log_probs.exp()
                ent = -(cur_probs * cur_log_probs).sum(dim=-1)

                loss = kl.mean() - ENTROPY_COEF * ent.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                optimizer.step()
                n_updates += 1
        else:
            obs = _flatten_obs(next_obs)

    env.close()

    nudge_frac = (
        nudge_decision_steps / max(total_decision_steps, 1)
        if total_decision_steps
        else 0.0
    )
    cells_nz_frac = (
        nudge_nonzero_cells_total / max(cell_count_total, 1)
        if cell_count_total
        else 0.0
    )

    print(
        f"[train] env={env_id} seed={seed} episodes={n_episodes} "
        f"updates={n_updates} train_s={time.monotonic() - t0:.1f} "
        f"budget_s={time_budget_s} k={k} is_vector={is_vector} "
        f"nudge_step_frac={nudge_frac:.3f} nz_cell_frac={cells_nz_frac:.3f} "
        f"buffer={len(buffer)}",
        flush=True,
    )

    eval_step_counter = [0]

    def policy_fn(obs_arr):
        flat = _flatten_obs(obs_arr)
        b_eval = _bucket_index(eval_step_counter[0])
        eval_step_counter[0] += 1
        with torch.no_grad():
            obs_t = torch.from_numpy(flat).float().unsqueeze(0)
            pos_t = torch.tensor([b_eval], dtype=torch.long)
            logits = policy(obs_t, pos_t).squeeze(0).numpy()
        return int(np.argmax(logits))

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
