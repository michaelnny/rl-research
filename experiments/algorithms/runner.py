"""Algorithm contract and evaluation runner for novel RL attempts.

`Algorithm` is the thin protocol every candidate plugs into. `evaluate_algorithm`
runs a fixed protocol against a registered RLH-Bench env ID and returns a JSON-
serializable record so different attempts produce comparable artifacts.

This module is NOT substrate: it lives under `experiments/` and only consumes
the `rlh_bench` public surface. New algorithms should subclass `Algorithm` (or
duck-type it) and run themselves through `evaluate_algorithm`.

Quick usage::

    from experiments.algorithms.runner import evaluate_algorithm
    record = evaluate_algorithm(
        algorithm=MyAlgo(...),
        env_id="RecoverableKeyFuelMaze-Small-v0",
        train_seed=0,
        eval_seeds=range(20),
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np

from rlh_bench import make_env, pareto_non_dominated, rollout


Policy = Callable[[np.ndarray], np.ndarray]


@runtime_checkable
class Algorithm(Protocol):
    """The contract every candidate algorithm must satisfy.

    An algorithm is anything that, given an env factory and a training budget,
    returns a callable policy. ``train`` is allowed to spend wall-clock and
    rollouts as it likes; the harness records the cost.

    Implementations must respect the substrate boundary:
      * may read ``info["reward_vector"]`` / ``info["reward_names"]`` /
        ``info["is_success"]``;
      * may build any internal model, buffer, optimizer, or wrapper;
      * may NOT edit `src/rlh_bench/`;
      * may NOT introduce per-step shaping rewards into the env;
      * may NOT pull in baseline RL libraries (stable-baselines3, RLlib, etc.).
    """

    name: str

    def train(self, env_factory: Callable[[], Any], *, seed: int) -> Policy:
        """Produce a policy ready for evaluation. ``env_factory`` returns a fresh
        env each call. ``seed`` is the training seed."""


# ----- evaluation harness --------------------------------------------------- #


@dataclass
class EvalRecord:
    """JSON-friendly summary written under experiments/results/."""

    algorithm: str
    env_id: str
    reward_mode: str
    train_seed: int
    eval_seeds: list[int]
    train_seconds: float
    success_rate: float
    mean_return: float
    std_return: float
    mean_reward_vector: list[float]
    reward_names: list[str]
    mean_length: float
    first_success_episode: int | None
    pareto_non_dominated_count: int
    per_episode: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


def evaluate_algorithm(
    algorithm: Algorithm,
    env_id: str,
    *,
    train_seed: int = 0,
    eval_seeds: Iterable[int] = range(20),
    reward_mode: str = "scalar",
    notes: str = "",
    save_to: str | Path | None = None,
) -> EvalRecord:
    """Train once, evaluate over many seeds, return a comparable record.

    The training env factory always uses ``reward_mode``; algorithms that want
    vector rewards from the env should request ``reward_mode="vector"`` (the
    scalarization-vs-vector distinction is enforced socially, not in code — see
    ``CLAUDE.md`` and ``docs/AGENT_GUIDE.md``).
    """

    env_factory = lambda: make_env(env_id, reward_mode=reward_mode)  # noqa: E731

    t0 = time.perf_counter()
    policy = algorithm.train(env_factory, seed=train_seed)
    train_seconds = time.perf_counter() - t0

    eval_seeds = list(eval_seeds)
    returns: list[float] = []
    vectors: list[np.ndarray] = []
    successes: list[int] = []
    lengths: list[int] = []
    per_episode: list[dict[str, Any]] = []

    for seed in eval_seeds:
        env = env_factory()
        result = rollout(env, policy, seed=seed)
        returns.append(result.scalar_return)
        vectors.append(result.reward_vector)
        successes.append(int(bool(result.info.get("is_success", False))))
        lengths.append(result.length)
        per_episode.append(
            {
                "seed": seed,
                "scalar_return": float(result.scalar_return),
                "reward_vector": result.reward_vector.tolist(),
                "success": successes[-1],
                "length": result.length,
            }
        )

    vec_arr = np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
    first_success = next((i + 1 for i, s in enumerate(successes) if s), None)
    pareto_mask = pareto_non_dominated(vec_arr) if vec_arr.size else np.zeros(0, dtype=bool)
    reward_names = list(make_env(env_id).reward_spec.names)

    record = EvalRecord(
        algorithm=algorithm.name,
        env_id=env_id,
        reward_mode=reward_mode,
        train_seed=train_seed,
        eval_seeds=eval_seeds,
        train_seconds=train_seconds,
        success_rate=float(np.mean(successes)) if successes else 0.0,
        mean_return=float(np.mean(returns)) if returns else 0.0,
        std_return=float(np.std(returns)) if returns else 0.0,
        mean_reward_vector=(vec_arr.mean(axis=0).tolist() if vec_arr.size else []),
        reward_names=reward_names,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        first_success_episode=first_success,
        pareto_non_dominated_count=int(pareto_mask.sum()),
        per_episode=per_episode,
        notes=notes,
    )

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(asdict(record), indent=2, default=float))

    return record


# ----- reference algorithm wrappers ---------------------------------------- #


@dataclass
class CemAlgorithm:
    """Wrap the bundled CEM trainer as an ``Algorithm`` reference baseline."""

    iterations: int = 8
    population: int = 32
    elite_frac: float = 0.20
    eval_episodes: int = 1
    name: str = "cem"

    def train(self, env_factory: Callable[[], Any], *, seed: int) -> Policy:
        from rlh_bench.baselines import LinearPolicy, train_cem

        cem = train_cem(
            env_factory=env_factory,
            iterations=self.iterations,
            population=self.population,
            elite_frac=self.elite_frac,
            eval_episodes=self.eval_episodes,
            seed=seed,
        )
        probe = env_factory()
        obs_dim = int(np.prod(probe.observation_space.shape))
        return LinearPolicy(
            params=cem.best_params,
            obs_dim=obs_dim,
            action_space=probe.action_space,
        )
