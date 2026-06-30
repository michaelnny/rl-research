"""Toy example of a new algorithm using the Algorithm runner.

Random-shooting / random search at the policy-parameter level — not novel,
not strong, just here to show the shape a candidate plugs in at. Real
attempts go under ``experiments/algorithms/<name>.py`` and use the same
``Algorithm`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rlh_bench import rollout
from rlh_bench.baselines import LinearPolicy
from rlh_bench.baselines.cem import policy_num_params

from experiments.algorithms.runner import Algorithm, evaluate_algorithm


@dataclass
class RandomShootingAlgorithm:
    """Sample N linear-policy parameter vectors; pick the best on a single
    deterministic rollout. Worse than CEM by design — useful for the example."""

    samples: int = 64
    init_std: float = 0.7
    name: str = "random-shooting"

    def train(self, env_factory: Callable[[], Any], *, seed: int):
        rng = np.random.default_rng(seed)
        probe = env_factory()
        dim = policy_num_params(probe)
        obs_dim = int(np.prod(probe.observation_space.shape))

        best_score = -np.inf
        best_params = np.zeros(dim, dtype=np.float32)
        for i in range(self.samples):
            params = rng.normal(scale=self.init_std, size=dim).astype(np.float32)
            env = env_factory()
            policy = LinearPolicy(params=params, obs_dim=obs_dim, action_space=env.action_space)
            score = rollout(env, policy, seed=seed + i).scalar_return
            if score > best_score:
                best_score = score
                best_params = params

        env = env_factory()
        return LinearPolicy(params=best_params, obs_dim=obs_dim, action_space=env.action_space)


if __name__ == "__main__":
    record = evaluate_algorithm(
        algorithm=RandomShootingAlgorithm(samples=64),
        env_id="RecoverablePointMaze-Small-v0",
        eval_seeds=range(10),
        notes="example algorithm, not a research contender",
        save_to="experiments/results/example_random_shooting.json",
    )
    print(
        f"success_rate={record.success_rate:.2f} "
        f"mean_return={record.mean_return:.3f} "
        f"train_seconds={record.train_seconds:.2f}"
    )
