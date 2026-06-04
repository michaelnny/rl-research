"""Build baselines.json from fresh local baseline implementations.

Local baselines are intentionally small and transparent:
- random
- tabular_q for scalar discrete envs
- count_bonus_q for scalar discrete envs
- scalarized_q_sweep for vector discrete envs (disqualifier/control baseline)

Craftax reported SOTA is stored separately and uses the same metric as
harness.evaluate: reward as percent of maximum return 226.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402

BaselinePolicy = Callable[[str, int, int], harness.PolicyFn]

REPORTED_SOTA = {
    "Craftax-Symbolic-v1": {
        "metric": "reward_percent_of_max_226",
        "source": "https://github.com/michaeltmatthews/craftax",
        "verified_baselines_1b": {
            "PPO-RNN": 15.3,
            "RND": 12.0,
            "PPO": 11.9,
            "ICM": 11.9,
            "E3B": 11.0,
        },
        "reported_unverified_1b": {
            "PPO-GTrXL": 18.3,
            "PQN-RNN": 16.0,
        },
        "verified_baselines_1m": {
            "PPO-RNN": 2.3,
            "RND": 2.2,
            "PPO": 2.2,
            "ICM": 2.2,
            "E3B": 2.2,
        },
        "note": "Craftax scoreboard reports reward as percent of max return 226; external entries are reported but not verified upstream.",
    }
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=sorted(harness.STAGES), default="all")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=harness.N_EVAL_EPISODES)
    p.add_argument("--train-budget-s", type=int, default=10)
    p.add_argument(
        "--baselines",
        default="random,tabular_q,count_bonus_q,scalarized_q_sweep",
        help="Comma-separated subset of random,tabular_q,count_bonus_q,scalarized_q_sweep",
    )
    return p.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    values = [int(s.strip()) for s in text.split(",") if s.strip()]
    if not values:
        raise SystemExit("expected at least one seed")
    return values


def obs_key(obs: np.ndarray) -> bytes:
    arr = np.asarray(obs)
    if arr.dtype.kind == "f":
        arr = np.clip(arr * 8.0, -127, 127).astype(np.int8)
    return arr.tobytes()


def random_policy(env_id: str, seed: int, _budget_s: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    env.close()
    rng = np.random.default_rng(seed + 99_999)

    def policy_fn(_obs):
        if hasattr(action_space, "n"):
            return int(rng.integers(int(action_space.n)))
        return action_space.sample()

    return policy_fn


def q_policy(
    env_id: str,
    seed: int,
    budget_s: int,
    *,
    reward_fn: Callable[[float, dict], float],
    count_bonus: bool = False,
) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        env.close()
        return random_policy(env_id, seed, budget_s)
    n_actions = int(action_space.n)
    rng = np.random.default_rng(seed)
    q: dict[bytes, np.ndarray] = {}
    counts: dict[bytes, np.ndarray] = {}
    alpha = 0.3
    gamma = 0.99
    eps = 0.1
    beta = 1.0

    obs, _ = env.reset(seed=seed)
    deadline = time.monotonic() + max(0, budget_s)
    while time.monotonic() < deadline:
        s = obs_key(np.asarray(obs))
        if s not in q:
            q[s] = np.zeros(n_actions, dtype=np.float64)
            counts[s] = np.zeros(n_actions, dtype=np.float64)
        values = q[s]
        if count_bonus:
            values = values + beta / np.sqrt(counts[s] + 1.0)
        action = int(rng.integers(n_actions)) if rng.random() < eps else int(np.argmax(values))
        next_obs, reward, term, trunc, info = env.step(action)
        signal = reward_fn(float(reward), info)
        s2 = obs_key(np.asarray(next_obs))
        if s2 not in q:
            q[s2] = np.zeros(n_actions, dtype=np.float64)
            counts[s2] = np.zeros(n_actions, dtype=np.float64)
        counts[s][action] += 1.0
        target = signal if term else signal + gamma * float(np.max(q[s2]))
        q[s][action] += alpha * (target - q[s][action])
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    frozen = {k: v.copy() for k, v in q.items()}

    def policy_fn(obs):
        s = obs_key(np.asarray(obs))
        if s not in frozen:
            return 0
        return int(np.argmax(frozen[s]))

    return policy_fn


def scalar_reward(_env_id: str) -> Callable[[float, dict], float]:
    return lambda reward, _info: reward


def vector_scalarization(weights: np.ndarray) -> Callable[[float, dict], float]:
    def fn(_reward: float, info: dict) -> float:
        return float(np.dot(weights, np.asarray(info["vector"], dtype=np.float64)))

    return fn


def tabular_q_policy(env_id: str, seed: int, budget_s: int) -> harness.PolicyFn:
    return q_policy(env_id, seed, budget_s, reward_fn=scalar_reward(env_id), count_bonus=False)


def count_bonus_q_policy(env_id: str, seed: int, budget_s: int) -> harness.PolicyFn:
    return q_policy(env_id, seed, budget_s, reward_fn=scalar_reward(env_id), count_bonus=True)


def scalarized_q_sweep_score(env_id: str, seed: int, budget_s: int, episodes: int) -> float:
    k = len(harness.HV_REF[env_id])
    weights = [np.eye(k)[i] for i in range(k)] + [np.ones(k) / k]
    scores = []
    for i, w in enumerate(weights):
        policy = q_policy(
            env_id,
            seed + 1000 * (i + 1),
            budget_s,
            reward_fn=vector_scalarization(w),
            count_bonus=False,
        )
        scores.append(float(harness.evaluate(policy, env_id, seed=seed, n_episodes=episodes)))
    return float(max(scores))


def applicable_baselines(env_id: str, requested: list[str]) -> list[str]:
    out = []
    for name in requested:
        if (
            name == "random"
            or (name in {"tabular_q", "count_bonus_q"} and harness.ENV_TYPE[env_id] == "scalar")
            or (name == "scalarized_q_sweep" and harness.ENV_TYPE[env_id] == "vector")
        ):
            out.append(name)
    return out


def score_baseline(env_id: str, name: str, seed: int, budget_s: int, episodes: int) -> float:
    if name == "random":
        policy = random_policy(env_id, seed, budget_s)
        return float(harness.evaluate(policy, env_id, seed=seed, n_episodes=episodes))
    if name == "tabular_q":
        policy = tabular_q_policy(env_id, seed, budget_s)
        return float(harness.evaluate(policy, env_id, seed=seed, n_episodes=episodes))
    if name == "count_bonus_q":
        policy = count_bonus_q_policy(env_id, seed, budget_s)
        return float(harness.evaluate(policy, env_id, seed=seed, n_episodes=episodes))
    if name == "scalarized_q_sweep":
        return scalarized_q_sweep_score(env_id, seed, budget_s, episodes)
    raise ValueError(name)


def summarize(scores: list[float]) -> dict:
    return {
        "threshold": float(max(scores)),
        "mean": float(statistics.fmean(scores)),
        "std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        "min": float(min(scores)),
        "max": float(max(scores)),
        "scores": [float(x) for x in scores],
    }


def score_env(
    env_id: str, seeds: list[int], budget_s: int, episodes: int, requested: list[str]
) -> dict:
    names = applicable_baselines(env_id, requested)
    per = {}
    for name in names:
        scores = [score_baseline(env_id, name, seed, budget_s, episodes) for seed in seeds]
        per[name] = summarize(scores)
        print(
            f"  {name:20s} threshold={per[name]['threshold']:.6f} "
            f"mean={per[name]['mean']:.6f} std={per[name]['std']:.6f}",
            flush=True,
        )
    random_threshold = per.get("random", {"threshold": 0.0})["threshold"]
    strong_local = max((v["threshold"] for v in per.values()), default=random_threshold)
    block = {
        "random": float(random_threshold),
        "strong": float(max(random_threshold, strong_local)),
        "strong_local": float(strong_local),
        "per_baseline": per,
        "seeds": seeds,
        "episodes": episodes,
        "train_budget_s": budget_s,
    }
    if env_id in REPORTED_SOTA:
        block["reported_sota"] = REPORTED_SOTA[env_id]
    return block


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    requested = [x.strip() for x in args.baselines.split(",") if x.strip()]
    known = {"random", "tabular_q", "count_bonus_q", "scalarized_q_sweep"}
    unknown = [x for x in requested if x not in known]
    if unknown:
        raise SystemExit(f"unknown baseline(s): {unknown}")

    out = {}
    for env_id in harness.STAGES[args.stage]:
        print(env_id, flush=True)
        out[env_id] = score_env(env_id, seeds, args.train_budget_s, args.episodes, requested)
    harness.BASELINES_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {harness.BASELINES_PATH}", flush=True)


if __name__ == "__main__":
    main()
