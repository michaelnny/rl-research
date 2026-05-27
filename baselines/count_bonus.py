"""Count-based exploration bonus baseline.

Tabular Q-learning with an exploration bonus `β / sqrt(N(s,a))` added to
the reward. The strongest "novelty" baseline a candidate's mechanism must
structurally differ from on prerequisite-structure tasks.

Same obs-hashing and continuous-action discretization scheme as
`eps_greedy_q.py`.
"""

from __future__ import annotations

import time

import numpy as np

import harness


def _obs_key(obs: np.ndarray) -> bytes:
    arr = np.asarray(obs)
    if arr.dtype.kind == "f":
        arr = np.clip(arr * 8.0, -127, 127).astype(np.int8)
    return arr.tobytes()


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    rng = np.random.default_rng(seed)
    env = harness.make_env(env_id, seed=seed)
    action_space = env.action_space

    if hasattr(action_space, "n"):
        n_actions = int(action_space.n)
        action_decoder = lambda a: a  # noqa: E731
    else:
        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        d = min(low.shape[0], 4)
        bins = 5
        n_actions = bins**d
        grid = np.linspace(0, 1, bins)

        def action_decoder(a: int):
            idxs = []
            x = int(a)
            for _ in range(d):
                idxs.append(x % bins)
                x //= bins
            cont = np.asarray(
                [low[i] + grid[idx] * (high[i] - low[i]) for i, idx in enumerate(idxs)],
                dtype=np.float32,
            )
            full = np.zeros_like(low, dtype=np.float32)
            full[:d] = cont
            return full

    Q: dict[bytes, np.ndarray] = {}
    N: dict[bytes, np.ndarray] = {}
    eps = 0.05
    alpha = 0.3
    gamma = 0.99
    beta = 1.0  # exploration bonus weight

    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs)
    t0 = time.monotonic()
    while time.monotonic() - t0 < time_budget_s:
        s = _obs_key(obs)
        if s not in Q:
            Q[s] = np.zeros(n_actions, dtype=np.float64)
            N[s] = np.zeros(n_actions, dtype=np.float64)
        bonus = beta / np.sqrt(N[s] + 1.0)
        a = int(rng.integers(n_actions)) if rng.random() < eps else int(np.argmax(Q[s] + bonus))
        next_obs, r, term, trunc, _info = env.step(action_decoder(a))
        next_obs = np.asarray(next_obs)
        s2 = _obs_key(next_obs)
        if s2 not in Q:
            Q[s2] = np.zeros(n_actions, dtype=np.float64)
            N[s2] = np.zeros(n_actions, dtype=np.float64)
        N[s][a] += 1
        target = float(r) + (0.0 if term else gamma * float(Q[s2].max()))
        Q[s][a] += alpha * (target - Q[s][a])
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
            obs = np.asarray(obs)
    env.close()

    Q_frozen = {k: v.copy() for k, v in Q.items()}

    def policy_fn(obs):
        s = _obs_key(np.asarray(obs))
        if s not in Q_frozen:
            return action_decoder(0)
        a = int(np.argmax(Q_frozen[s]))
        return action_decoder(a)

    return policy_fn
