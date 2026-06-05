"""CWTP — Confluence-Witness Trajectory Pairs.

Realizes hypothesis 20260606-03-auto:
- Buffer of completed trajectories with (obs_hash, action, vec_t).
- For each unordered pair, find observation-hash confluences and walk
  back to the latest divergence (latest equal-hash with distinct
  actions). Emit witness (s_div, a, a', Δv) where Δv is per-channel
  segment-cumulant difference between the bracketed segments.
- Sign-vote tensor V[s_div, a, a', m] += sign(Δv[m]).
- Logit nudge at decision time: Δlogit(s,a) = α·(#a' a-dominates - #a' a'-dominates a),
  where dominance is Pareto-non-domination on the row-normalized sign-vote vector.
- Base logits come from a small policy network trained with max-entropy regularizer
  on top of nudges. Softmax sampling with temperature annealed 1.0 -> 0.5.
- No scalar reward weighting in the operator. No critic. No value backup.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness


# ---------- hashing ----------

def obs_hash(obs) -> int:
    """Deterministic compact hash of an observation."""
    a = np.asarray(obs)
    if a.dtype == object:
        a = np.array(a.tolist())
    b = a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


# ---------- policy net ----------

class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def flatten_obs(obs) -> np.ndarray:
    a = np.asarray(obs).astype(np.float32, copy=False)
    return a.reshape(-1)


# ---------- CWTP core ----------

class CWTP:
    """Sign-vote tensor V[s_div, a, a', m]. Stored sparsely.

    V[s][a][a'] is a numpy array of shape (k,) of int sign-vote counts.
    Symmetric: V[s][a'][a] = -V[s][a][a'] is implicit (we only store
    canonical (a, a') with a < a' and use the sign to decode direction).

    Convention: we store V[s][(min_a, max_a)] = sign(Δv) where Δv is the
    cumulant of the trajectory contributing the smaller action minus the
    one contributing the larger action.
    """

    def __init__(self, n_actions: int, k_channels: int):
        self.n_actions = n_actions
        self.k = k_channels
        # V[s][(a_lo, a_hi)] -> int array shape (k,). Sign convention:
        # value = signs of (segcum_lo - segcum_hi).
        self.V: dict[int, dict[tuple[int, int], np.ndarray]] = defaultdict(dict)

    def add_witness(self, s_div: int, a: int, ap: int, dv: np.ndarray) -> None:
        if a == ap:
            return
        if a < ap:
            key = (a, ap)
            sgn = np.sign(dv).astype(np.int32)
        else:
            key = (ap, a)
            sgn = -np.sign(dv).astype(np.int32)
        cell = self.V[s_div].get(key)
        if cell is None:
            self.V[s_div][key] = sgn.copy()
        else:
            cell += sgn

    def logit_nudge(self, s: int, n_actions: int, alpha: float) -> np.ndarray:
        """Compute Δlogit[a] = α * (#a' a dominates - #a' a' dominates a)."""
        cells = self.V.get(s)
        if not cells:
            return np.zeros(n_actions, dtype=np.float32)
        # Build per-action dominance counts.
        dom_count = np.zeros(n_actions, dtype=np.int32)
        dominated_count = np.zeros(n_actions, dtype=np.int32)
        for (a_lo, a_hi), counts in cells.items():
            # row v(a_lo, a_hi) is sign(counts) component-wise after normalization.
            # We need v(a_lo, a_hi) = counts / |counts|_1 (normalize)
            l1 = np.abs(counts).sum()
            if l1 == 0:
                continue
            v = counts.astype(np.float32) / float(l1)
            # a_lo dominates a_hi iff v >= 0 componentwise with at least one > 0.
            ge_zero = np.all(v >= 0)
            any_pos = np.any(v > 0)
            le_zero = np.all(v <= 0)
            any_neg = np.any(v < 0)
            if ge_zero and any_pos:
                dom_count[a_lo] += 1
                dominated_count[a_hi] += 1
            elif le_zero and any_neg:
                dom_count[a_hi] += 1
                dominated_count[a_lo] += 1
            # else: incomparable (mixed signs) -> no nudge
        return alpha * (dom_count - dominated_count).astype(np.float32)

    def stats(self) -> dict:
        n_states = len(self.V)
        n_pairs = sum(len(d) for d in self.V.values())
        return {"n_states": n_states, "n_pairs": n_pairs}


def extract_witnesses(traj_i, traj_j, cwtp: CWTP) -> int:
    """Find all confluence-witness events between two trajectories.

    For each shared obs-hash position pair (t_i, t_j), walk backwards
    simultaneously to find the latest divergence: the largest
    (d_i, d_j) with d_i <= t_i, d_j <= t_j, hashes equal at d, but
    actions[d_i] != actions[d_j]. Then witness = (s_div, a_i, a_j, Δv)
    where Δv = sum_{u=d_i+1..t_i} v_u^i - sum_{u=d_j+1..t_j} v_u^j.

    Hypothesis says we take the *latest* divergence preceding the
    confluence. We iterate confluences (t_i, t_j) and for each find the
    largest d <= t with equal hashes-and-distinct-actions in a backward
    walk. To keep cost in line, we deduplicate witnesses by
    (s_div, a_i, a_j, t_i, t_j) — each confluence contributes at most
    one witness.

    Returns the number of witnesses emitted.
    """
    hashes_i = traj_i["hashes"]
    hashes_j = traj_j["hashes"]
    actions_i = traj_i["actions"]
    actions_j = traj_j["actions"]
    cum_i = traj_i["cum"]  # shape (T_i+1, k); cum[0]=0; cum[t]=sum_{u<t} v_u
    cum_j = traj_j["cum"]
    Ti = len(actions_i)
    Tj = len(actions_j)
    # hashes lists have length T+1 (include terminal hash). For witness
    # positions we need both d and t to point at *decision* indices
    # (where an action was taken), so 0 <= d < t <= T-1; equivalently
    # 0 <= d < t < T. The terminal-hash slot is used only as a
    # confluence anchor that allows t = T-1 to be a legitimate
    # confluence (since the trajectories share the obs at step T).

    # Index hash positions in each trajectory for fast confluence finding.
    # Restrict to decision indices (0..T-1).
    pos_j: dict[int, list[int]] = defaultdict(list)
    for t in range(Tj):
        pos_j[hashes_j[t]].append(t)

    n_witnesses = 0

    for t_i in range(Ti):
        if t_i == 0:
            continue  # need a divergence before t_i, t_i must be >= 1
        h = hashes_i[t_i]
        if h not in pos_j:
            continue
        for t_j in pos_j[h]:
            if t_j == 0:
                continue
            # Find latest divergence preceding (t_i, t_j) by walking
            # backwards. We walk positions d in min(t_i, t_j) to 1 and
            # check if hashes_i[d_i] == hashes_j[d_j] for some pair.
            # Simplification & honesty: the hypothesis says "walk
            # backwards in both trajectories simultaneously". The
            # natural reading is to walk both at the same offset
            # k = 1, 2, ... and check (t_i - k, t_j - k). Diagonal
            # walk. The "latest divergence" is the smallest k such
            # that hashes match at (t_i-k, t_j-k) and actions differ.
            d_i = -1
            d_j = -1
            kmax = min(t_i, t_j)
            for k in range(1, kmax + 1):
                di = t_i - k
                dj = t_j - k
                if hashes_i[di] == hashes_j[dj]:
                    if actions_i[di] != actions_j[dj]:
                        d_i, d_j = di, dj
                        break
                    # equal action at equal hash: continue walking back
                    continue
                else:
                    # hashes diverge before we found an action-distinct
                    # equal-hash position; the segments don't share a
                    # prior "same-state different-action" anchor along
                    # the diagonal walk; skip.
                    break
            if d_i < 0:
                continue
            s_div = hashes_i[d_i]
            a_i = int(actions_i[d_i])
            a_j = int(actions_j[d_j])
            # segment cumulant from d+1..t (inclusive on t, exclusive on d).
            # cum[u] = sum_{r<u} v_r, so sum_{u=d+1..t} v_u = cum[t+1] - cum[d+1].
            seg_i = cum_i[t_i + 1] - cum_i[d_i + 1]
            seg_j = cum_j[t_j + 1] - cum_j[d_j + 1]
            dv = seg_i - seg_j
            cwtp.add_witness(s_div, a_i, a_j, dv)
            n_witnesses += 1
    return n_witnesses


# ---------- training loop ----------

def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t_start = time.monotonic()
    rng = np.random.default_rng(seed + 1234)
    torch.manual_seed(seed + 4321)

    env = harness.make_env(env_id, seed)
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # Determine action space.
    action_space = env.action_space
    if hasattr(action_space, "n"):
        n_actions = int(action_space.n)
    else:
        # Continuous: not supported by CWTP's discrete-action primitive.
        env.close()

        def fallback(_obs):
            return action_space.sample()

        print(
            f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 "
            f"budget_s={time_budget_s} note=continuous-action-not-supported",
            flush=True,
        )
        return fallback

    # Determine vector dimensionality. For scalar envs we use scalar reward
    # as a single-channel vector for the witness primitive (so the operator
    # still has a structural shape) — but per the hypothesis falsifier,
    # vector envs are the load-bearing test. The candidate is allowed to
    # run on sparse envs too; the operator may simply be silent there.
    obs0, _ = env.reset(seed=seed)
    flat0 = flatten_obs(obs0)
    in_dim = int(flat0.shape[0])

    # Probe one step to learn vector dim.
    sample_a = 0
    try:
        _o, _r, _term, _trunc, _info = env.step(sample_a)
        if is_vector and "vector" in _info:
            k_channels = int(np.asarray(_info["vector"]).reshape(-1).shape[0])
        else:
            k_channels = 1
    except Exception:
        k_channels = 1
    env.close()

    # Recreate env for training.
    env = harness.make_env(env_id, seed)
    obs, _ = env.reset(seed=seed)

    cwtp = CWTP(n_actions=n_actions, k_channels=k_channels)
    policy = PolicyNet(in_dim=in_dim, n_actions=n_actions, hidden=64)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Training hyperparameters (from hypothesis):
    #   - softmax temp annealed 1.0 -> 0.5
    #   - max-entropy regularizer on base logits
    #   - alpha scales the logit nudge
    alpha = 1.0
    ent_coef = 0.05  # entropy floor (mitigation against premature collapse)
    max_buffer = 64  # cap on retained completed trajectories (FIFO)
    max_traj_len = 400
    max_witness_seconds_per_update = 1.5  # cap on witness extraction wall-time

    # Buffer of completed trajectories.
    buffer: deque[dict] = deque(maxlen=max_buffer)
    # Replay log for policy gradient (max-entropy regularizer + nudge integration)
    # We use a simple REINFORCE-style update on log-prob × advantage proxy where
    # the "advantage" is the per-step nudge magnitude — this is the
    # "policy network trained with maximum-entropy regularizer on top of the
    # nudges" prescription. The nudge supplies the directional signal; the
    # entropy term keeps the policy from collapsing.

    env_steps = 0
    total_episodes = 0
    total_witnesses = 0

    def temperature_now() -> float:
        # Linear anneal 1.0 -> 0.5 over the budget.
        elapsed = time.monotonic() - t_start
        frac = min(1.0, elapsed / max(1e-6, float(time_budget_s)))
        return float(1.0 - 0.5 * frac)

    def compute_action_distribution(obs_arr, h, temp) -> tuple[np.ndarray, torch.Tensor]:
        """Return (probs, logits_tensor). logits = base + nudge."""
        x = torch.from_numpy(flatten_obs(obs_arr)).float().unsqueeze(0)
        base_logits = policy(x).squeeze(0)  # (n_actions,)
        nudge = cwtp.logit_nudge(h, n_actions, alpha)
        nudge_t = torch.from_numpy(nudge).float()
        combined = (base_logits + nudge_t) / max(temp, 1e-3)
        probs = F.softmax(combined, dim=-1).detach().cpu().numpy()
        return probs, base_logits

    def select_action(obs_arr, h, temp, log: list | None):
        probs, base_logits = compute_action_distribution(obs_arr, h, temp)
        # Numerical safety
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()
        a = int(rng.choice(n_actions, p=probs))
        if log is not None:
            log.append({"obs": flatten_obs(obs_arr), "h": h, "a": a, "probs": probs.copy()})
        return a

    # ---------- main loop ----------
    seeding_phase_s = 30.0
    last_update_t = t_start

    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= time_budget_s - 5.0:  # leave ~5s for cleanup
            break

        # One rollout.
        obs, _ = env.reset(seed=seed + total_episodes + 1)
        traj_hashes: list[int] = []
        traj_actions: list[int] = []
        traj_vecs: list[np.ndarray] = []
        traj_obs: list[np.ndarray] = []
        traj_logprobs: list[torch.Tensor] = []
        traj_entropies: list[torch.Tensor] = []
        traj_nudges: list[float] = []

        in_seeding = (elapsed < seeding_phase_s)
        temp = temperature_now()

        steps_this_ep = 0
        while True:
            h = obs_hash(obs)
            traj_hashes.append(h)
            traj_obs.append(flatten_obs(obs))

            # Action selection.
            x = torch.from_numpy(flatten_obs(obs)).float().unsqueeze(0)
            base_logits = policy(x).squeeze(0)
            if in_seeding:
                # Uniform-random rollouts for the falsifier's seeding phase.
                a = int(rng.integers(n_actions))
                # Also record entropy of the policy net for max-ent regularizer.
                logp = F.log_softmax(base_logits, dim=-1)[a]
                p = F.softmax(base_logits, dim=-1)
                entropy = -(p * F.log_softmax(base_logits, dim=-1)).sum()
                nudge_mag = 0.0
            else:
                nudge = cwtp.logit_nudge(h, n_actions, alpha)
                nudge_t = torch.from_numpy(nudge).float()
                combined_logits = (base_logits + nudge_t) / max(temp, 1e-3)
                logp_all = F.log_softmax(combined_logits, dim=-1)
                p = F.softmax(combined_logits, dim=-1)
                p_np = p.detach().cpu().numpy()
                p_np = np.clip(p_np, 1e-8, 1.0)
                p_np /= p_np.sum()
                a = int(rng.choice(n_actions, p=p_np))
                logp = logp_all[a]
                entropy = -(p * logp_all).sum()
                nudge_mag = float(np.abs(nudge).sum())

            traj_actions.append(a)
            traj_logprobs.append(logp)
            traj_entropies.append(entropy)
            traj_nudges.append(nudge_mag)

            try:
                obs2, reward, term, trunc, info = env.step(a)
            except Exception as e:
                print(f"[train] env.step exception: {e!r}", flush=True)
                term = True
                trunc = False
                info = {}
                reward = 0.0
                obs2 = obs

            # Vector signal.
            if is_vector and "vector" in info:
                v = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
                if v.shape[0] != k_channels:
                    # Pad/truncate to k_channels for safety.
                    vv = np.zeros(k_channels, dtype=np.float64)
                    vv[: min(k_channels, v.shape[0])] = v[: min(k_channels, v.shape[0])]
                    v = vv
            else:
                v = np.array([float(reward)], dtype=np.float64)
                if v.shape[0] != k_channels:
                    vv = np.zeros(k_channels, dtype=np.float64)
                    vv[0] = v[0]
                    v = vv
            traj_vecs.append(v)

            obs = obs2
            env_steps += 1
            steps_this_ep += 1

            done = bool(term) or bool(trunc) or steps_this_ep >= max_traj_len
            if done:
                break
            # Time guard within episode.
            if time.monotonic() - t_start >= time_budget_s - 5.0:
                break

        # End of episode: record terminal hash for confluence detection.
        traj_hashes.append(obs_hash(obs))
        # Build cumulant array: cum[u] = sum_{r<u} v_r, shape (T+1, k).
        T = len(traj_vecs)
        cum = np.zeros((T + 1, k_channels), dtype=np.float64)
        if T > 0:
            cum[1:] = np.cumsum(np.stack(traj_vecs, axis=0), axis=0)

        traj = {
            "hashes": traj_hashes,
            "actions": traj_actions,
            "cum": cum,
            "obs": traj_obs,
            "logprobs": traj_logprobs,
            "entropies": traj_entropies,
            "nudges": traj_nudges,
        }
        buffer.append(traj)
        total_episodes += 1

        # ---------- update phase: extract witnesses, update sign-vote tensor.
        update_t0 = time.monotonic()
        new_witnesses = 0
        n_buf = len(buffer)
        if n_buf >= 2:
            # Compare the new trajectory against existing ones.
            # Cap the number of comparisons by time budget.
            for j_idx in range(n_buf - 1):
                if time.monotonic() - update_t0 > max_witness_seconds_per_update:
                    break
                other = buffer[j_idx]
                w = extract_witnesses(traj, other, cwtp)
                new_witnesses += w
        total_witnesses += new_witnesses

        # ---------- policy network update: REINFORCE with nudge-as-advantage
        # plus max-entropy regularizer.
        # The nudge magnitude at each step serves as the directional signal
        # that the operator considers significant; we reinforce actions that
        # were taken under high-nudge states. The entropy regularizer keeps
        # the policy stochastic (mitigating collapse).
        if traj["logprobs"]:
            logps = torch.stack(traj["logprobs"])
            ents = torch.stack(traj["entropies"])
            nudges_arr = np.asarray(traj["nudges"], dtype=np.float32)
            # Advantage proxy: nudge magnitude, centered.
            if nudges_arr.size > 1 and nudges_arr.std() > 1e-6:
                adv = (nudges_arr - nudges_arr.mean()) / (nudges_arr.std() + 1e-6)
            else:
                adv = np.zeros_like(nudges_arr)
            adv_t = torch.from_numpy(adv).float()
            pg_loss = -(logps * adv_t).mean()
            ent_loss = -ent_coef * ents.mean()
            loss = pg_loss + ent_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

        # Periodic stats.
        if total_episodes % 10 == 0:
            stats = cwtp.stats()
            print(
                f"[train] ep={total_episodes} env_steps={env_steps} "
                f"witnesses_total={total_witnesses} new_w={new_witnesses} "
                f"V.states={stats['n_states']} V.pairs={stats['n_pairs']} "
                f"buf={len(buffer)} temp={temp:.3f} elapsed={elapsed:.1f}s "
                f"seeding={in_seeding}",
                flush=True,
            )

    env.close()

    # Final summary.
    stats = cwtp.stats()
    elapsed = time.monotonic() - t_start
    # Falsifier check: print the per-thousand-step witness rate.
    rate = (total_witnesses * 1000.0 / max(1, env_steps))
    print(
        f"[train] DONE env={env_id} seed={seed} env_steps={env_steps} "
        f"episodes={total_episodes} train_s={elapsed:.1f} "
        f"witnesses={total_witnesses} witness_per_kstep={rate:.3f} "
        f"V.states={stats['n_states']} V.pairs={stats['n_pairs']}",
        flush=True,
    )

    # ---------- frozen policy for evaluation ----------
    # Evaluation policy uses the same softmax-with-nudge but is greedy
    # (argmax of combined logits) for deterministic eval. We sample
    # only via argmax to match a "deterministic policy_fn" expectation.
    # Final temperature held at the annealed end-value (0.5).
    final_temp = 0.5
    policy.eval()

    def policy_fn(obs_arr):
        with torch.no_grad():
            x = torch.from_numpy(flatten_obs(obs_arr)).float().unsqueeze(0)
            base_logits = policy(x).squeeze(0)
            h = obs_hash(obs_arr)
            nudge = cwtp.logit_nudge(h, n_actions, alpha)
            nudge_t = torch.from_numpy(nudge).float()
            combined = (base_logits + nudge_t) / max(final_temp, 1e-3)
            return int(torch.argmax(combined).item())

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
