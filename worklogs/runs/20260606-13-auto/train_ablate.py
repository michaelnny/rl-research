"""LYRA ablation: frozen random 2-frame instead of QR-tracked Oseledec frame.

Per the hypothesis ## Ablation plan: replace the online QR update of the
2-frame Q with a fixed random orthonormal 2-frame Q resampled at the start
of training and never updated. The score-function gradient computation
proceeds identically using this random frame in place of (u_1, u_2):
    c_i = gamma * exp(beta * r_t) * (q_i^T phi_s) * (phi_sp^T q_i)

If the random-frame ablation matches LYRA, the Lyapunov primitive is
decorative and the algorithm collapses to "REINFORCE projected onto a
random 2-frame." Everything else in the training loop (encoder, policy
parameterization, hyperparameters, episode termination) is held fixed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running this file from arbitrary cwd: ensure repo root (which holds
# harness.py) is on sys.path. Repo root = grandparent of this file's parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402


# --- hyperparameters --------------------------------------------------------
GAMMA = 0.97
BETA = 1.0
ALPHA = 0.05
PHI_DIM = 64
ENTROPY_COEF = 0.01
GRAD_CLIP = 5.0


# --- math helpers (copied verbatim from train.py) ---------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((d, k)).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    return Q


# --- feature encoder (copied verbatim) --------------------------------------
class RandomEncoder:
    def __init__(self, input_dim: int, d: int, rng: np.random.Generator):
        self.input_dim = input_dim
        self.d = d
        self.W = rng.standard_normal((input_dim, d)).astype(np.float64) / np.sqrt(input_dim)
        self.b = rng.standard_normal(d).astype(np.float64) * 0.01

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(-1) / 255.0
        z = x @ self.W + self.b
        z = np.tanh(z)
        n = np.linalg.norm(z)
        if n < 1e-8:
            z = np.full_like(z, 1.0 / np.sqrt(self.d))
        else:
            z = z / n
        return z


# --- policy (copied verbatim) -----------------------------------------------
class LinearSoftmaxPolicy:
    def __init__(self, d: int, n_actions: int, rng: np.random.Generator):
        self.d = d
        self.n_actions = n_actions
        self.W = rng.standard_normal((n_actions, d)).astype(np.float64) * 0.01
        self.b = np.zeros(n_actions, dtype=np.float64)

    def logits(self, phi: np.ndarray) -> np.ndarray:
        return self.W @ phi + self.b

    def probs(self, phi: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(phi))

    def sample(self, phi: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        p = self.probs(phi)
        a = int(rng.choice(self.n_actions, p=p))
        return a, p

    def grad_log_pi(
        self, phi: np.ndarray, action: int, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        e_a = np.zeros_like(p)
        e_a[action] = 1.0
        diff = e_a - p
        dW = np.outer(diff, phi)
        db = diff
        return dW, db

    def grad_entropy(self, phi: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_p = np.log(np.clip(p, 1e-12, 1.0))
        avg_log_p = np.dot(p, log_p)
        dlogits = -p * (log_p - avg_log_p)
        dW = np.outer(dlogits, phi)
        db = dlogits
        return dW, db


def _flat_input_dim(env) -> int:
    space = env.observation_space
    return int(np.prod(space.shape))


def _ensure_action_n(env) -> int:
    if not hasattr(env.action_space, "n"):
        raise RuntimeError(f"LYRA expects discrete actions; got {env.action_space}")
    return int(env.action_space.n)


def _step_frozen_frame(
    encoder: RandomEncoder,
    Q: np.ndarray,
    obs: np.ndarray,
    next_obs: np.ndarray,
    reward: float,
    *,
    gamma: float,
    beta: float,
) -> tuple[float, float]:
    """Ablation: c_i with frozen random Q instead of QR-tracked frame.

    Q is the frozen random orthonormal (d,2) frame; we never update it,
    we never compute QR, and we never accumulate Lyapunov logs (there is
    no growing leading direction to track because Q is not aligned to
    the cocycle's Oseledec splitting).
    """
    phi_s = encoder(obs)
    phi_sp = encoder(next_obs)
    tilt = gamma * float(np.exp(np.clip(beta * reward, -50.0, 50.0)))
    q1 = Q[:, 0]
    q2 = Q[:, 1]
    c1 = tilt * float(q1 @ phi_s) * float(phi_sp @ q1)
    c2 = tilt * float(q2 @ phi_s) * float(phi_sp @ q2)
    return c1, c2


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    rng = np.random.default_rng(seed + 99_999)

    env = harness.make_env(env_id, seed)
    n_actions = _ensure_action_n(env)
    input_dim = _flat_input_dim(env)

    encoder = RandomEncoder(input_dim, PHI_DIM, rng)
    policy = LinearSoftmaxPolicy(PHI_DIM, n_actions, rng)

    # Frozen random 2-frame; never updated.
    Q_frozen = _random_orthonormal(PHI_DIM, 2, rng)
    T_steps = 0

    obs, _ = env.reset(seed=seed)

    ep_return = 0.0
    ep_returns: list[float] = []
    ep_steps = 0

    deadline = t0 + max(1.0, float(time_budget_s) - 1.0)
    while time.monotonic() < deadline:
        phi_s = encoder(obs)
        action, p = policy.sample(phi_s, rng)
        next_obs, reward, term, trunc, _info = env.step(action)
        ep_return += float(reward)
        ep_steps += 1

        c1, c2 = _step_frozen_frame(
            encoder=encoder,
            Q=Q_frozen,
            obs=obs,
            next_obs=next_obs,
            reward=float(reward),
            gamma=GAMMA,
            beta=BETA,
        )

        dW_log, db_log = policy.grad_log_pi(phi_s, action, p)
        dW_ent, db_ent = policy.grad_entropy(phi_s, p)

        coef = c1 - c2
        coef = float(np.clip(coef, -GRAD_CLIP, GRAD_CLIP))
        policy.W += ALPHA * (coef * dW_log + ENTROPY_COEF * dW_ent)
        policy.b += ALPHA * (coef * db_log + ENTROPY_COEF * db_ent)

        T_steps += 1
        if T_steps % 2000 == 0:
            mean_recent = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
            print(
                f"[lyra-ablate] env={env_id} steps={T_steps} "
                f"recent_return={mean_recent:.4f} train_s={time.monotonic()-t0:.1f}",
                flush=True,
            )

        if term or trunc or ep_steps >= harness.MAX_EPISODE_STEPS:
            ep_returns.append(ep_return)
            ep_return = 0.0
            ep_steps = 0
            obs, _ = env.reset(seed=seed + 1 + len(ep_returns))
        else:
            obs = next_obs

    env.close()
    print(
        f"[train-ablate] env={env_id} seed={seed} env_steps={T_steps} "
        f"train_s={time.monotonic()-t0:.1f} budget_s={time_budget_s} "
        f"episodes={len(ep_returns)}",
        flush=True,
    )

    W = policy.W.copy()
    b = policy.b.copy()

    def policy_fn(obs_in: np.ndarray):
        phi = encoder(obs_in)
        logits = W @ phi + b
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
