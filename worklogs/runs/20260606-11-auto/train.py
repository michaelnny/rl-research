"""ACS — Action-Conditional Suffix-Spectrum.

Per-(state-cluster, action) tensor S[s,a,m,f] holds the running mean over
visits of within-band variance of the channel-firing indicator
1_m(t+tau) = [v_{t+tau}[m] != 0] across log-spaced lag bins
band_f = [2^f, 2^{f+1}). At decision time, channel-MAD-standardise rows,
flatten to (k*F)-vector, compute Pareto-non-dominated set with strict-margin
dominance, and nudge logits by +alpha for non-dominated actions, -alpha/(|A|-|P|)
otherwise. Softmax sampling, on-policy. F=3 to control Pareto saturation.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


# ----------------------------- ACS internals ------------------------------

F_BANDS = 3  # bands: [1,2), [2,4), [4,8). Cap F<=3 per failure-mode (d).
ALPHA0 = 1.0  # initial logit-nudge magnitude
DOMINANCE_MARGIN = 1e-3  # strict-margin dominance to combat saturation
EMA_DECAY = 0.05  # running-mean step for spectrum entries
TEMP = 1.0  # softmax temperature

# Lag bands in suffix indices: f in {0..F-1}, band_f = [2^f, 2^{f+1}).
BAND_LO = [1 << f for f in range(F_BANDS)]
BAND_HI = [1 << (f + 1) for f in range(F_BANDS)]


def _flatten_obs(obs) -> np.ndarray:
    a = np.asarray(obs)
    return a.reshape(-1).astype(np.float32, copy=False)


def _hash_state(obs_flat: np.ndarray, n_buckets: int = 4096) -> int:
    """Cheap integer hash of an observation flat-vector (state-cluster proxy)."""
    # bucket into discrete bytes-ish, then xor-hash
    if obs_flat.dtype.kind in "fc":
        # discretise floats to a few bits
        q = np.round(obs_flat * 4.0).astype(np.int32)
    else:
        q = obs_flat.astype(np.int32)
    h = np.uint64(1469598103934665603)
    for v in q.tobytes():
        h ^= np.uint64(v)
        h = (h * np.uint64(1099511628211)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return int(h % n_buckets)


def _band_variance(indicator_suffix: np.ndarray, lo: int, hi: int) -> float:
    """Var of within-band mean-firing-rate.

    indicator_suffix[i] is 1_m(t+1+i) for i=0,1,...
    band_f = [2^f, 2^{f+1}); lag tau in [lo, hi). Returns 0 if band has no samples.
    """
    n = len(indicator_suffix)
    a = max(0, lo - 1)
    b = min(n, hi - 1)
    if b - a <= 0:
        return 0.0
    seg = indicator_suffix[a:b]
    if len(seg) < 2:
        # variance of a single sample is 0; report 0 (will be filtered by counts).
        return 0.0
    return float(seg.var())


def _band_filled(n_suffix: int, lo: int) -> bool:
    """Did the suffix reach into this band at all?"""
    return n_suffix >= lo


class ACS:
    def __init__(self, n_actions: int, n_channels: int, n_buckets: int, seed: int):
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.n_buckets = n_buckets
        self.F = F_BANDS
        self.rng = np.random.default_rng(seed)
        # spectrum tensor: S[s,a,m,f]
        self.S = np.zeros((n_buckets, n_actions, n_channels, self.F), dtype=np.float64)
        # visit counts per (s,a,m,f) — number of band-samples accumulated
        self.N = np.zeros((n_buckets, n_actions, n_channels, self.F), dtype=np.int64)
        # base logits per (s,a)
        self.logits = np.zeros((n_buckets, n_actions), dtype=np.float64)
        # global episode counter for alpha decay
        self.ep = 0

    def alpha(self) -> float:
        # decay alpha across episodes
        return ALPHA0 / (1.0 + 0.01 * self.ep)

    def policy_logits(self, s: int) -> np.ndarray:
        """Read the spectrum at s, compute per-action k*F flattened standardised
        rows, find Pareto-non-dominated set with strict margin, return nudged logits."""
        base = self.logits[s].copy()  # (A,)
        S_s = self.S[s]  # (A, k, F)
        N_s = self.N[s]  # (A, k, F)
        # Mask under-populated cells: treat cells with < 2 samples as 0
        mask = N_s >= 2
        Smasked = np.where(mask, S_s, 0.0)
        A, k, F = Smasked.shape
        # channel-MAD standardisation across actions, per (m,f)
        med = np.median(Smasked, axis=0, keepdims=True)  # (1,k,F)
        mad = np.median(np.abs(Smasked - med), axis=0, keepdims=True) + 1e-9
        Z = (Smasked - med) / mad  # (A,k,F)
        flat = Z.reshape(A, k * F)  # per-action vector

        # Strict-margin Pareto-non-dominance over actions:
        # action a is dominated iff exists a' s.t. flat[a'] >= flat[a] + margin elementwise
        # AND flat[a'] > flat[a] + margin in at least one coordinate.
        margin = DOMINANCE_MARGIN
        non_dominated = np.ones(A, dtype=bool)
        for a in range(A):
            # vectorised over all a' != a
            diff = flat - flat[a]  # (A, k*F): a' minus a
            ge = np.all(diff >= -margin, axis=1)  # a' weakly dominates a within margin
            gt = np.any(diff > margin, axis=1)    # a' strictly exceeds a in at least one coord
            ge[a] = False
            if np.any(ge & gt):
                non_dominated[a] = False

        nP = int(non_dominated.sum())
        if nP == 0 or nP == A:
            # uninformative — no nudge
            return base
        alpha = self.alpha()
        nudge = np.where(non_dominated, alpha, -alpha / (A - nP))
        return base + nudge

    def sample_action(self, s: int) -> int:
        logits = self.policy_logits(s) / TEMP
        logits -= logits.max()
        p = np.exp(logits)
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            return int(self.rng.integers(self.n_actions))
        p /= p_sum
        return int(self.rng.choice(self.n_actions, p=p))

    def update_episode(
        self,
        states: list[int],
        actions: list[int],
        channel_indicators: np.ndarray,  # (T, k) of 0/1
    ) -> None:
        """Accumulate per-(s,a,m,f) running-mean of within-band variance over the
        suffix following each visit."""
        T = len(states)
        if T == 0:
            return
        for t in range(T):
            s = states[t]
            a = actions[t]
            n_suffix = T - 1 - t  # number of indicator entries in suffix
            if n_suffix < 1:
                continue
            suffix = channel_indicators[t + 1 : T]  # (n_suffix, k)
            for m in range(self.n_channels):
                ind_m = suffix[:, m]  # (n_suffix,)
                for f in range(self.F):
                    lo, hi = BAND_LO[f], BAND_HI[f]
                    if not _band_filled(n_suffix, lo):
                        continue
                    v = _band_variance(ind_m, lo, hi)
                    # EMA running mean
                    self.N[s, a, m, f] += 1
                    n = self.N[s, a, m, f]
                    if n == 1:
                        self.S[s, a, m, f] = v
                    else:
                        # blend toward EMA — keep "average over visits" semantics
                        self.S[s, a, m, f] += EMA_DECAY * (v - self.S[s, a, m, f])
        self.ep += 1


# ------------------------------ glue ------------------------------------

def _resolve_action_space(env):
    space = env.action_space
    if hasattr(space, "n"):
        return int(space.n), True
    return None, False


def _channels_from_step(env_id: str, info: dict, reward: float, k: int) -> np.ndarray:
    """Return per-step vector v_t in R^k. For vector envs use info['vector'];
    for scalar envs synthesise a single channel from the scalar reward."""
    if "vector" in info:
        v = np.asarray(info["vector"], dtype=np.float64).reshape(-1)
        if v.shape[0] != k:
            # pad / truncate defensively
            out = np.zeros(k, dtype=np.float64)
            out[: min(k, v.shape[0])] = v[: min(k, v.shape[0])]
            return out
        return v
    # scalar env: 1-channel: reward
    return np.array([reward], dtype=np.float64)


def _channels_arity(env_id: str) -> int:
    if env_id == "deep-sea-treasure-concave-v0":
        return 2
    if env_id == "resource-gathering-v0":
        return 3
    return 1  # sparse / craftax — single-channel scalar reward


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    n_actions, is_discrete = _resolve_action_space(env)
    if not is_discrete:
        env.close()
        rng = np.random.default_rng(seed + 99_999)

        def fallback(_obs):
            return env.action_space.sample()

        print(
            f"[train] env={env_id} seed={seed} non-discrete: random fallback",
            flush=True,
        )
        return fallback

    k = _channels_arity(env_id)
    n_buckets = 4096
    acs = ACS(n_actions=n_actions, n_channels=k, n_buckets=n_buckets, seed=seed)

    t0 = time.monotonic()
    deadline = t0 + max(1, time_budget_s - 5)
    env_steps = 0
    n_episodes = 0
    # Diagnostics: per-channel cell fill count (#cells with >= 5 samples)
    cells_with_5 = np.zeros(k, dtype=np.int64)

    while time.monotonic() < deadline:
        try:
            obs, _ = env.reset(seed=seed + 7919 * n_episodes + 17)
        except Exception:
            obs, _ = env.reset()
        states: list[int] = []
        actions: list[int] = []
        ch_list: list[np.ndarray] = []
        steps = 0
        done = False
        while not done and steps < harness.MAX_EPISODE_STEPS and time.monotonic() < deadline:
            obs_flat = _flatten_obs(obs)
            s = _hash_state(obs_flat, n_buckets=n_buckets)
            a = acs.sample_action(s)
            try:
                obs2, reward, term, trunc, info = env.step(a)
            except Exception:
                break
            v = _channels_from_step(env_id, info, float(reward), k)
            ind = (v != 0.0).astype(np.float64)
            states.append(s)
            actions.append(a)
            ch_list.append(ind)
            obs = obs2
            steps += 1
            env_steps += 1
            done = bool(term) or bool(trunc)
        if states:
            ch_arr = np.stack(ch_list, axis=0)  # (T, k)
            acs.update_episode(states, actions, ch_arr)
        n_episodes += 1

    env.close()

    # Diagnostics: per-channel fill rate at end of training
    for m in range(k):
        cells_with_5[m] = int((acs.N[..., m, :] >= 5).sum())
    total_cells = acs.N[..., 0, :].size
    fill_rates = (cells_with_5 / max(1, total_cells)).tolist()

    # Snapshot parameters into a closure for greedy evaluation
    S_final = acs.S.copy()
    N_final = acs.N.copy()
    base_logits = acs.logits.copy()
    F = acs.F
    alpha_eval = acs.alpha()

    def policy_fn(obs):
        obs_flat = _flatten_obs(obs)
        s = _hash_state(obs_flat, n_buckets=n_buckets)
        S_s = S_final[s]  # (A,k,F)
        N_s = N_final[s]
        mask = N_s >= 2
        Smasked = np.where(mask, S_s, 0.0)
        A = Smasked.shape[0]
        med = np.median(Smasked, axis=0, keepdims=True)
        mad = np.median(np.abs(Smasked - med), axis=0, keepdims=True) + 1e-9
        Z = (Smasked - med) / mad
        flat = Z.reshape(A, -1)
        margin = DOMINANCE_MARGIN
        non_dominated = np.ones(A, dtype=bool)
        for a in range(A):
            diff = flat - flat[a]
            ge = np.all(diff >= -margin, axis=1)
            gt = np.any(diff > margin, axis=1)
            ge[a] = False
            if np.any(ge & gt):
                non_dominated[a] = False
        nP = int(non_dominated.sum())
        logits = base_logits[s].copy()
        if 0 < nP < A:
            nudge = np.where(non_dominated, alpha_eval, -alpha_eval / (A - nP))
            logits = logits + nudge
        return int(np.argmax(logits))

    train_s = time.monotonic() - t0
    print(
        f"[train] env={env_id} seed={seed} env_steps={env_steps} episodes={n_episodes} "
        f"train_s={train_s:.1f} budget_s={time_budget_s} k={k} F={F} "
        f"fill_rates={fill_rates}",
        flush=True,
    )
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
