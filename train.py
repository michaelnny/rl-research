"""Agent-editable training script.

This is the ONLY file the agent edits. The harness (`harness.py`) provides
the env, eval, and panel; the agent provides the algorithm.

Default content: a random-policy baseline that exercises the harness
end-to-end. Replace the body of `train(env_id, seed, time_budget_s)` with
your algorithm. The contract is:

  - Read `info['vector']` from `env.step()` if `env_id` is a vector-reward
    env (minecart, deep-sea-treasure, mo-reacher). The scalar `reward`
    returned alongside is the env's default scalarization; using it as
    your training signal on a vector env is a scalarized-vector-reward
    rebadge — the disqualifier list rejects this. Consume the vector.

  - Stop training when `time.monotonic() - t0 >= time_budget_s`.

  - Build a deterministic `policy_fn(obs) -> action` and pass it to
    `harness.evaluate(...)`. Print the summary block at the end. The
    final line MUST be `final_score: <float>` so `run_panel.py` can
    grep it.

Anything else inside `train()` is fair game: the network, the loss, the
update rule, the replay buffer (or none), the exploration scheme, the
eval policy. The whole RL primitive lives in one file.

CLI:
    uv run train.py --env <env_id> --seed <int> --time-budget-s <int>
                    [--logdir <path>]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import harness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env",
        required=True,
        choices=list(harness.PANEL_SMOKE) + list(harness.PANEL_HARD),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--time-budget-s",
        type=int,
        default=harness.TIME_BUDGET_SMOKE,
        help="Hard wallclock cap for training. Eval happens after this.",
    )
    p.add_argument("--logdir", type=str, default=None)
    return p.parse_args()


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    """Train against `env_id` for at most `time_budget_s` seconds, return a
    deterministic policy_fn(obs) -> action.

    Default implementation: random policy. Trains nothing. The agent
    overwrites this with their algorithm.
    """
    env = harness.make_env(env_id, seed=seed)
    action_space = env.action_space
    env.close()

    t0 = time.monotonic()
    n_env_steps = 0

    # ---- AGENT: replace everything between these markers with your algorithm.
    # The default is a no-op training loop that just walks the env to confirm
    # the harness pipeline runs end-to-end.
    train_env = harness.make_env(env_id, seed=seed)
    train_env.reset(seed=seed)
    while time.monotonic() - t0 < time_budget_s:
        a = action_space.sample()
        _obs, _r, term, trunc, _info = train_env.step(a)
        n_env_steps += 1
        if term or trunc:
            train_env.reset()
    train_env.close()
    # ---- END AGENT region.

    train_seconds = time.monotonic() - t0
    print(
        f"[train] env={env_id} seed={seed} env_steps={n_env_steps} train_s={train_seconds:.1f}",
        flush=True,
    )

    # Deterministic eval policy. Default: uniform random with eval-only RNG
    # so the eval is at least reproducible across runs.
    eval_rng = np.random.default_rng(seed + 99_999)

    def policy_fn(obs: np.ndarray) -> int:
        return (
            int(eval_rng.integers(action_space.n))
            if hasattr(action_space, "n")
            else action_space.sample()
        )

    return policy_fn


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()
    policy_fn = train(args.env, args.seed, args.time_budget_s)
    score = harness.evaluate(policy_fn, args.env, seed=args.seed)
    wallclock_s = time.monotonic() - t0

    # Summary block — exactly the format `run_panel.py` grep's for.
    # Final line must be `final_score: ...`.
    print("---", flush=True)
    print(f"env:           {args.env}", flush=True)
    print(f"seed:          {args.seed}", flush=True)
    print(f"env_type:      {harness.PANEL_TYPE[args.env]}", flush=True)
    print(f"wallclock_s:   {wallclock_s:.1f}", flush=True)
    print(f"final_score:   {score:.6f}", flush=True)


if __name__ == "__main__":
    main()
