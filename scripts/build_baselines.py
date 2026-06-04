"""Build baselines.json for the lean benchmark.

The baseline is intentionally just a random policy floor. `strong` is set to
random until a deliberate, small comparison baseline is added.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=sorted(harness.STAGES), default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=harness.N_EVAL_EPISODES)
    return p.parse_args()


def random_policy(env_id: str, seed: int):
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    env.close()
    rng = np.random.default_rng(seed + 99_999)

    def policy_fn(_obs):
        if hasattr(action_space, "n"):
            return int(rng.integers(int(action_space.n)))
        return action_space.sample()

    return policy_fn


def main() -> None:
    args = parse_args()
    out = {}
    for env_id in harness.STAGES[args.stage]:
        score = harness.evaluate(
            random_policy(env_id, args.seed), env_id, seed=args.seed, n_episodes=args.episodes
        )
        out[env_id] = {"random": float(score), "strong": float(score)}
        print(f"{env_id:36s} random={score:.6f}", flush=True)
    harness.BASELINES_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {harness.BASELINES_PATH}", flush=True)


if __name__ == "__main__":
    main()
