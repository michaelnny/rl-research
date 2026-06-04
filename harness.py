"""Lean RL research harness.

Active benchmark:
- sparse: MiniGrid DoorKey + KeyCorridor
- vector: Deep-Sea Treasure concave + Minecart

The harness intentionally does not include hard-tier or MuJoCo/JAX/NetHack
scaffolding. Historical research memory stays in prior_attempts.md and
worklogs/attempts/; it is not part of the hot path.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*Overriding environment.*")
warnings.filterwarnings("ignore", message=r".*reward returned by `step\(\)`.*")
warnings.filterwarnings("ignore", message=".*Box .*precision lowered.*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

ROOT = Path(__file__).parent.resolve()
BASELINES_PATH = ROOT / "baselines.json"

SPARSE_ENVS = ["MiniGrid-DoorKey-8x8-v0", "MiniGrid-KeyCorridorS3R3-v0"]
VECTOR_ENVS = ["deep-sea-treasure-concave-v0", "minecart-v0"]
ENVS = SPARSE_ENVS + VECTOR_ENVS
PANEL = ENVS

STAGES = {
    "quick": ["deep-sea-treasure-concave-v0"],
    "sparse": SPARSE_ENVS,
    "vector": VECTOR_ENVS,
    "all": ENVS,
}

ENV_TYPE = {env: "scalar" for env in SPARSE_ENVS} | {env: "vector" for env in VECTOR_ENVS}
PANEL_TYPE = ENV_TYPE  # compatibility for train.py/tests

HV_REF = {
    "deep-sea-treasure-concave-v0": np.array([0.0, -100.0], dtype=np.float64),
    "minecart-v0": np.array([-1.0, -1.0, -200.0], dtype=np.float64),
}

TIME_BUDGET_S = 120
TIME_BUDGET_PER_ENV = TIME_BUDGET_S
N_EVAL_EPISODES = 8
MAX_EPISODE_STEPS = 2000

PolicyFn = Callable[[np.ndarray], object]


def make_env(env_id: str, seed: int = 0):
    if env_id not in ENV_TYPE:
        raise ValueError(f"unknown env_id: {env_id!r}; choices={ENVS}")
    if env_id.startswith("MiniGrid-"):
        return _make_minigrid(env_id, seed)
    return _make_vector_env(env_id, seed)


def _make_minigrid(env_id: str, seed: int):
    import gymnasium as gym
    import minigrid
    from minigrid.wrappers import ImgObsWrapper

    minigrid.register_minigrid_envs()
    env = ImgObsWrapper(gym.make(env_id))
    env.reset(seed=seed)
    return env


def _make_vector_env(env_id: str, seed: int):
    import gymnasium as gym
    import mo_gymnasium  # noqa: F401

    env = _VectorRewardWrapper(gym.make(env_id))
    env.reset(seed=seed)
    return env


class _VectorRewardWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        vec = np.asarray(reward, dtype=np.float64)
        info = dict(info)
        info["vector"] = vec
        return obs, float(vec.sum()), term, trunc, info

    def close(self):
        self.env.close()


def evaluate(
    policy_fn: PolicyFn,
    env_id: str,
    *,
    seed: int = 0,
    n_episodes: int = N_EVAL_EPISODES,
) -> float:
    env = make_env(env_id, seed + 10_000)
    is_vector = ENV_TYPE[env_id] == "vector"
    scalar_returns: list[float] = []
    vector_returns: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 5_000 + ep)
        done = False
        steps = 0
        ret = 0.0
        vret = np.zeros_like(HV_REF[env_id]) if is_vector else None
        while not done and steps < MAX_EPISODE_STEPS:
            action = policy_fn(np.asarray(obs))
            obs, reward, term, trunc, info = env.step(action)
            ret += float(reward)
            if is_vector:
                vret += np.asarray(info["vector"], dtype=np.float64)
            done = bool(term) or bool(trunc)
            steps += 1
        scalar_returns.append(ret)
        if is_vector:
            vector_returns.append(vret.copy())

    env.close()
    if is_vector:
        return float(hypervolume(np.stack(vector_returns), HV_REF[env_id]))
    return float(np.mean(scalar_returns))


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    points = points[np.all(points > ref, axis=1)]
    if len(points) == 0:
        return 0.0
    points = pareto_front(points)
    if ref.shape[0] == 2:
        order = np.argsort(-points[:, 0])
        total = 0.0
        prev_y = ref[1]
        for x, y in points[order]:
            if y > prev_y:
                total += (x - ref[0]) * (y - prev_y)
                prev_y = y
        return float(total)
    if ref.shape[0] == 3:
        order = np.argsort(points[:, 2])
        total = 0.0
        prev_z = ref[2]
        for i in order:
            z = points[i, 2]
            if z <= prev_z:
                continue
            total += hypervolume(points[points[:, 2] >= z, :2], ref[:2]) * (z - prev_z)
            prev_z = z
        return float(total)
    raise ValueError("only 2-D and 3-D vector rewards are supported")


def pareto_front(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if keep[i]:
            dominates = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
            keep[i] = not dominates.any()
    return points[keep]


def load_baselines() -> dict[str, dict[str, float]]:
    out = {env: {"random": 0.0, "strong": 0.0} for env in ENVS}
    if not BASELINES_PATH.exists():
        return out
    raw = json.loads(BASELINES_PATH.read_text())
    for env in ENVS:
        block = raw.get(env, {})
        random_score = float(block.get("random", 0.0))
        strong_score = float(block.get("strong", random_score))
        out[env] = {"random": random_score, "strong": max(random_score, strong_score)}
    return out


__all__ = [
    "BASELINES_PATH",
    "ENVS",
    "ENV_TYPE",
    "HV_REF",
    "N_EVAL_EPISODES",
    "PANEL",
    "PANEL_TYPE",
    "SPARSE_ENVS",
    "STAGES",
    "TIME_BUDGET_PER_ENV",
    "TIME_BUDGET_S",
    "VECTOR_ENVS",
    "evaluate",
    "hypervolume",
    "load_baselines",
    "make_env",
    "pareto_front",
]
