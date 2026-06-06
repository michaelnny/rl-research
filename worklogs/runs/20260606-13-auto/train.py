"""LYRA (Lyapunov Reward Asymmetry) candidate train.

Implements the probe described in hypothesis.md: gradient ascent on the
Lyapunov-exponent gap Delta = lambda_1 - lambda_2 of the reward-tilted
policy cocycle M_t = gamma * exp(beta * r_t) * P^pi, with the leading
2-frame tracked online via Benettin-Galgani-Strelcyn QR.

Update (per step):
    phi_s, phi_sp = Phi(s_t), Phi(s_{t+1})
    tilt          = gamma * exp(beta * r_t)
    Z             = tilt * outer(phi_s, phi_sp @ Q)        # d x 2
    Q_new, R      = qr(Z)
    L            += log|diag(R)|
    c_i           = tilt * (u_i^T phi_s) * (phi_sp^T u_i)
    theta        += alpha * (c1 - c2) * grad_theta log pi(a|s)

Sparse stage: MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0.
Phi is a fixed random linear encoder of the flattened observation,
fixed by the seed.
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
ALPHA = 0.05               # policy step size
PHI_DIM = 64               # state-feature dim d
POLICY_HIDDEN = 64
ENTROPY_COEF = 0.01        # tiny exploration regularizer on policy
GRAD_CLIP = 5.0            # per-step clip on (c1 - c2)*score and entropy term


# --- math helpers -----------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _qr_2(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduced QR of a (d, 2) matrix; returns (Q (d,2), R (2,2)).

    Falls back to a tiny random orthonormal frame if Z is rank-deficient.
    """
    # numpy's reduced QR
    Q, R = np.linalg.qr(Z, mode="reduced")
    # ensure R has positive diagonal so logs are well-defined; flip signs as needed
    diag = np.diag(R)
    sign = np.where(diag < 0.0, -1.0, 1.0)
    Q = Q * sign[None, :]
    R = R * sign[:, None]
    return Q, R


def _random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal (d, k) frame via QR of a Gaussian matrix."""
    A = rng.standard_normal((d, k)).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    return Q


# --- feature encoder --------------------------------------------------------
class RandomEncoder:
    """Fixed random linear encoder Phi: obs -> R^d.

    For MiniGrid (H, W, 3) uint8 observations: flatten, scale to [0,1],
    multiply by a fixed random projection of shape (input_dim, d).
    Output is L2-normalized so ||phi||=1, which keeps the cocycle from
    blowing up under exp(beta * r).
    """

    def __init__(self, input_dim: int, d: int, rng: np.random.Generator):
        self.input_dim = input_dim
        self.d = d
        # scaled Gaussian projection
        self.W = rng.standard_normal((input_dim, d)).astype(np.float64) / np.sqrt(input_dim)
        # bias keeps phi nonzero on all-zero obs
        self.b = rng.standard_normal(d).astype(np.float64) * 0.01

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(-1) / 255.0
        z = x @ self.W + self.b
        # tanh squashes, then L2-normalize
        z = np.tanh(z)
        n = np.linalg.norm(z)
        if n < 1e-8:
            z = np.full_like(z, 1.0 / np.sqrt(self.d))
        else:
            z = z / n
        return z


# --- policy: linear-then-softmax over Phi(s) --------------------------------
class LinearSoftmaxPolicy:
    """pi(a|s) = softmax( W @ phi(s) + b )_a; theta = (W, b).

    Score function: grad_W log pi(a|s) = (e_a - pi(.|s)) phi(s)^T
                    grad_b log pi(a|s) =  e_a - pi(.|s)
    """

    def __init__(self, d: int, n_actions: int, rng: np.random.Generator):
        self.d = d
        self.n_actions = n_actions
        # small init keeps initial pi close to uniform
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
        """Returns (dW, db) of grad_theta log pi(a|s)."""
        e_a = np.zeros_like(p)
        e_a[action] = 1.0
        diff = e_a - p           # (n_actions,)
        dW = np.outer(diff, phi)  # (n_actions, d)
        db = diff
        return dW, db

    def grad_entropy(self, phi: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """grad_theta H(pi(.|s)) where H = -sum p log p.

        d H / d logits_a = -p_a * (log p_a + H + ... )  -> use standard
        identity: dH/dlogits = -p*(log p) + p*sum(p log p) = -p*(log p - <log p>_p).
        """
        log_p = np.log(np.clip(p, 1e-12, 1.0))
        avg_log_p = np.dot(p, log_p)
        dlogits = -p * (log_p - avg_log_p)
        dW = np.outer(dlogits, phi)
        db = dlogits
        return dW, db


# --- training loop ----------------------------------------------------------
def _flat_input_dim(env) -> int:
    space = env.observation_space
    return int(np.prod(space.shape))


def _ensure_action_n(env) -> int:
    if not hasattr(env.action_space, "n"):
        raise RuntimeError(f"LYRA expects discrete actions; got {env.action_space}")
    return int(env.action_space.n)


def _step_qr_update(
    encoder: RandomEncoder,
    policy: LinearSoftmaxPolicy,
    Q: np.ndarray,
    L: np.ndarray,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: int,
    p: np.ndarray,
    reward: float,
    *,
    gamma: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Apply the LYRA per-step update (returns updated Q, L, c1, c2).

    Computes the rank-1 cocycle action on the current 2-frame, runs QR,
    accumulates log singular values and returns the gradient coefficients
    c_i = tilt * (u_i^T phi_s) * (phi_sp^T u_i) for i in {1,2}.
    """
    phi_s = encoder(obs)            # (d,)
    phi_sp = encoder(next_obs)      # (d,)
    # bound the tilt to avoid overflow on rare large rewards
    tilt = gamma * float(np.exp(np.clip(beta * reward, -50.0, 50.0)))
    # Z[:, i] = tilt * phi_s * (phi_sp^T Q[:, i])  =>  Z = tilt * phi_s outer (phi_sp^T Q)
    proj = phi_sp @ Q                         # (2,)
    Z = tilt * np.outer(phi_s, proj)          # (d, 2)

    # If Z is essentially zero (rare; when phi_sp is orthogonal to Q),
    # keep Q unchanged and contribute zero coefficients.
    if np.linalg.norm(Z) < 1e-12:
        return Q, L, 0.0, 0.0

    Q_new, R = _qr_2(Z)
    diag = np.diag(R)
    # accumulate log of |diag|; clip away near-zero to avoid -inf
    safe = np.clip(np.abs(diag), 1e-30, None)
    L = L + np.log(safe)

    u1 = Q_new[:, 0]
    u2 = Q_new[:, 1]
    c1 = tilt * float(u1 @ phi_s) * float(phi_sp @ u1)
    c2 = tilt * float(u2 @ phi_s) * float(phi_sp @ u2)
    return Q_new, L, c1, c2


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    rng = np.random.default_rng(seed + 99_999)

    env = harness.make_env(env_id, seed)
    n_actions = _ensure_action_n(env)
    input_dim = _flat_input_dim(env)

    encoder = RandomEncoder(input_dim, PHI_DIM, rng)
    policy = LinearSoftmaxPolicy(PHI_DIM, n_actions, rng)

    # Initial orthonormal 2-frame on R^d
    Q = _random_orthonormal(PHI_DIM, 2, rng)
    L = np.zeros(2, dtype=np.float64)
    T_steps = 0

    obs, _ = env.reset(seed=seed)

    initial_gap = None
    final_gap = None
    last_log_t = t0
    ep_return = 0.0
    ep_returns: list[float] = []
    ep_steps = 0

    # Train loop until time budget.
    # We leave eval_grace seconds (default 30) for final evaluation; subtract a buffer.
    deadline = t0 + max(1.0, float(time_budget_s) - 1.0)
    while time.monotonic() < deadline:
        phi_s = encoder(obs)
        action, p = policy.sample(phi_s, rng)
        next_obs, reward, term, trunc, _info = env.step(action)
        ep_return += float(reward)
        ep_steps += 1

        Q, L, c1, c2 = _step_qr_update(
            encoder=encoder,
            policy=policy,
            Q=Q,
            L=L,
            obs=obs,
            next_obs=next_obs,
            action=action,
            p=p,
            reward=float(reward),
            gamma=GAMMA,
            beta=BETA,
        )

        # Score function
        dW_log, db_log = policy.grad_log_pi(phi_s, action, p)
        # Entropy regularizer to keep some exploration in sparse-reward envs
        dW_ent, db_ent = policy.grad_entropy(phi_s, p)

        coef = c1 - c2
        # clip the gap-gradient coefficient to stabilize updates
        coef = float(np.clip(coef, -GRAD_CLIP, GRAD_CLIP))
        policy.W += ALPHA * (coef * dW_log + ENTROPY_COEF * dW_ent)
        policy.b += ALPHA * (coef * db_log + ENTROPY_COEF * db_ent)

        T_steps += 1
        if T_steps == 1:
            initial_gap = (L[0] - L[1])  # one-step proxy
        if T_steps % 2000 == 0:
            gap = (L[0] - L[1]) / max(1, T_steps)
            now = time.monotonic()
            mean_recent = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
            print(
                f"[lyra] env={env_id} steps={T_steps} gap={gap:.5f} "
                f"lambda1={L[0]/T_steps:.5f} lambda2={L[1]/T_steps:.5f} "
                f"recent_return={mean_recent:.4f} train_s={now-t0:.1f}",
                flush=True,
            )
            last_log_t = now

        if term or trunc or ep_steps >= harness.MAX_EPISODE_STEPS:
            ep_returns.append(ep_return)
            ep_return = 0.0
            ep_steps = 0
            obs, _ = env.reset(seed=seed + 1 + len(ep_returns))
        else:
            obs = next_obs

    env.close()

    if T_steps > 0:
        final_gap = (L[0] - L[1]) / T_steps
    else:
        final_gap = 0.0
    print(
        f"[train] env={env_id} seed={seed} env_steps={T_steps} train_s={time.monotonic()-t0:.1f} "
        f"budget_s={time_budget_s} final_gap={final_gap:.5f} initial_gap_proxy={initial_gap} "
        f"episodes={len(ep_returns)}",
        flush=True,
    )

    # Greedy deterministic policy for evaluation: argmax over softmax logits.
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
