"""Agent-editable algorithm entry point.

Contract:
    uv run train.py --env ENV --seed 0 --time-budget-s 120

Implement `train()` and return a deterministic policy_fn(obs) -> action.
For vector envs, training code must read info["vector"] from env.step().
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


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    """Default candidate: deterministic random-policy floor.

    Replace this function during a research attempt. Keep the CLI and final
    score output unchanged.
    """
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    env.close()
    rng = np.random.default_rng(seed + 99_999)

    def policy_fn(_obs: np.ndarray):
        if hasattr(action_space, "n"):
            return int(rng.integers(int(action_space.n)))
        return action_space.sample()

    print(
        f"[train] env={env_id} seed={seed} env_steps=0 train_s=0.0 budget_s={time_budget_s}",
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
