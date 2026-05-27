"""ε-greedy tabular Q-learning baseline.

Hashes the observation to a discrete state key and runs vanilla Q-learning
with ε=0.1 exploration. Works on any env with a discrete action space —
including continuous-obs envs (the obs hash is lossy but adequate as a
floor baseline). For continuous-action envs (mo-reacher) we
discretize each action dim into 5 bins.

This is the strong "tabular" baseline a candidate must beat to claim
prerequisite-structure exploration is happening.
"""

from __future__ import annotations

import time

import numpy as np

import harness


def _obs_key(obs: np.ndarray) -> bytes:
    """Lossy hash of obs to a tabular key. Use 8-bit quantization on floats."""
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
        # Continuous action space → 5-bin discretization per dim, capped at 5^4 actions.
        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        d = min(low.shape[0], 4)  # cap dim
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
    eps = 0.1
    alpha = 0.3
    gamma = 0.99

    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs)
    t0 = time.monotonic()
    while time.monotonic() - t0 < time_budget_s:
        s = _obs_key(obs)
        if s not in Q:
            Q[s] = np.zeros(n_actions, dtype=np.float64)
        a = int(rng.integers(n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
        next_obs, r, term, trunc, _info = env.step(action_decoder(a))
        next_obs = np.asarray(next_obs)
        s2 = _obs_key(next_obs)
        if s2 not in Q:
            Q[s2] = np.zeros(n_actions, dtype=np.float64)
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
