"""Skeleton train.py for a candidate algorithm.

Copy this file to ``lab/runs/<run_id>/train.py`` and fill the four ``TODO(algorithm)``
holes. Every other line is contract boilerplate that the framework primitives
already handle — you do not re-derive it.

This skeleton is **algorithm-agnostic on purpose**. It does not assume a neural
network, gradient descent, or a particular learner shape. The eval helper takes
a ``policy_fn(obs) -> action`` callable, not a network, so non-network candidates
(evolution strategies, random search, energy-based sampling, etc.) wire in the
same way as gradient-based ones.

What this file does for you (do not re-implement):
  * Parse the contract CLI (``--env / --seed / --total-env-steps / --logdir / --max-wallclock-s``).
  * Seed torch / numpy / stdlib / CUDA.
  * Build the right vector env adapter for the env id (gym-classic / atari /
    minecart / dm-control) — same code path on Stage A sanity envs and Stage B
    primary benchmark.
  * Enforce the wallclock budget cleanly (no SIGKILL surprise).
  * Emit every TensorBoard scalar the contract requires, on the cadence the
    contract requires (≥20 evals over the run).
  * Write a per-seed summary at ``<logdir>/result.json`` on exit (success or
    failure). The Engineer aggregates per-seed summaries into the run-level
    ``result.json`` via ``rl_research.contract.write_result``.
  * Capture exceptions, log a traceback, and still produce a summary so the
    failure becomes evidence in the corpus instead of a silent crash.

What you fill in (the four ``TODO(algorithm)`` holes):
  1. Build your learner state.
  2. Define ``policy_fn`` for deterministic eval.
  3. One inner update step inside the training loop.
  4. (Optional) param checksum, if your learner has parameters that should
     visibly move during training.

See ``docs/contract.md`` §train.py contract for the contract surface, and
``docs/roles/engineer.md`` §"Use the framework — do not re-derive" for the
primitives table. ``src/rl_research/baselines/ppo.py`` is the reference example
of how the framework wires together end-to-end (do not clone its algorithm —
it is the frozen yardstick, not a candidate template).
"""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_research.envs import make_vec
from rl_research.evaluate import evaluate as eval_policy
from rl_research.runtime import (
    WallclockBudget,
    parse_train_cli,
    seed_everything,
    write_config_json,
)
from rl_research.tb import EvalCadence, RunLogger


def main() -> None:
    args = parse_train_cli(description="<your algorithm>")
    seed_everything(args.seed)

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    write_config_json(logdir, args)

    logger = RunLogger(logdir)
    cadence = EvalCadence(total_env_steps=args.total_env_steps, n_evals=20)
    budget = WallclockBudget(args.max_wallclock_s)

    n_envs = 1  # tune for your algorithm; PPO uses 4-8 on classic, 8 on Atari
    adapter = make_vec(args.env, n_envs=n_envs, seed=args.seed)

    # ------------------------------------------------------------------ #
    # TODO(algorithm) #1: build learner state.                           #
    # ------------------------------------------------------------------ #
    # Examples — pick one or invent your own. The framework is indifferent. #
    #                                                                    #
    #   (a) Gradient-based network learner:                              #
    #         net = MyNetwork(obs_dim, act_dim).cuda()                   #
    #         opt = torch.optim.Adam(net.parameters(), lr=3e-4)          #
    #         learner = (net, opt)                                       #
    #                                                                    #
    #   (b) Evolution-strategy population:                               #
    #         learner = init_population(size=64, dim=...)                #
    #                                                                    #
    #   (c) Random-search / score-based / population-based:              #
    #         learner = init_state(...)                                  #
    learner: Any = None  # replace
    del learner  # silence linter on the placeholder; remove once you assign it
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # TODO(algorithm) #2: deterministic eval policy_fn.                  #
    # ------------------------------------------------------------------ #
    # Called by ``eval_policy`` with one un-batched observation; must    #
    # return one action. Typical patterns:                               #
    #   * discrete + network:    int(logits.argmax())                    #
    #   * continuous + network:  action_mean.numpy()                     #
    #   * non-network (ES/RS):   apply best params to obs                #
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("define a deterministic eval policy_fn")

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # TODO(algorithm) #4 (optional): param checksum.                     #
    # ------------------------------------------------------------------ #
    # If your learner has parameters that should visibly change during   #
    # training, surface a 16-hex digest so the Stage A "params actually  #
    # moved" gate has telemetry to look at:                              #
    #     from rl_research.runtime import param_checksum                 #
    #     return param_checksum(net)                                     #
    # If your learner is parameter-free (e.g. tabular), return None.     #
    def current_param_checksum() -> str | None:
        return None

    # ------------------------------------------------------------------ #

    started_at = datetime.now(UTC)
    eval_history: list[tuple[int, float, float]] = []
    final_per_channel: np.ndarray | None = None
    initial_checksum = current_param_checksum()
    final_checksum = initial_checksum
    status = "completed"
    error_traceback: str | None = None
    env_steps = 0

    try:
        adapter.reset(seed=args.seed)

        # Initial step-0 eval — the random-init baseline. Without it the
        # Stage A "moved above random" check has nothing to compare against.
        m, s, pc = eval_policy(args.env, policy_fn, seed=args.seed, n_episodes=10)
        cadence.force(0)
        logger.log_eval(0, m, s, pc)
        logger.log_progress(0, env_steps=0, wallclock_s=0.0, param_checksum=initial_checksum)
        eval_history.append((0, m, s))
        if pc is not None:
            final_per_channel = pc

        while env_steps < args.total_env_steps and not budget.expired():
            # ---------------------------------------------------------- #
            # TODO(algorithm) #3: one rollout / update step.             #
            # ---------------------------------------------------------- #
            # Step the adapter, collect transitions, and update your     #
            # learner. The framework does not constrain the structure —  #
            # you can do on-policy rollouts, off-policy buffer updates,  #
            # population evals, gradient-free perturbations, anything.   #
            #                                                            #
            #   actions = ...                                            #
            #   next_obs, rewards, term, trunc, final_obs = adapter.step(actions)
            #   ... your update ...                                      #
            #   train_loss = ...                                         #
            #   logger.log_train(env_steps, train_loss)                  #
            steps_taken = n_envs  # how many env steps the inner step consumed
            env_steps += steps_taken
            # ---------------------------------------------------------- #

            if cadence.maybe_eval(env_steps):
                m, s, pc = eval_policy(args.env, policy_fn, seed=args.seed, n_episodes=10)
                final_checksum = current_param_checksum()
                logger.log_eval(env_steps, m, s, pc)
                logger.log_progress(
                    env_steps,
                    env_steps=env_steps,
                    wallclock_s=budget.elapsed_s(),
                    param_checksum=final_checksum,
                )
                eval_history.append((env_steps, m, s))
                if pc is not None:
                    final_per_channel = pc

        if budget.expired() and env_steps < args.total_env_steps:
            status = "killed-budget"

        # Final eval pinned to the very end of training, regardless of cadence
        # rounding — so the per-seed summary's ``final_return`` is the actual
        # last-step return, not an arbitrary mid-cadence sample.
        m, s, pc = eval_policy(args.env, policy_fn, seed=args.seed, n_episodes=10)
        final_checksum = current_param_checksum()
        logger.log_eval(env_steps, m, s, pc)
        logger.log_progress(
            env_steps,
            env_steps=env_steps,
            wallclock_s=budget.elapsed_s(),
            param_checksum=final_checksum,
        )
        eval_history.append((env_steps, m, s))
        if pc is not None:
            final_per_channel = pc

    except Exception:
        status = "killed-error"
        error_traceback = traceback.format_exc()
        traceback.print_exc()

    logger.close()
    adapter.close()
    ended_at = datetime.now(UTC)

    if final_per_channel is not None:
        # Multi-signal: report per-channel arrays. ``best`` is the channel-wise
        # max across the eval history; the Engineer / Curator will scalarize
        # downstream as their analysis demands.
        best_return: float | list[float] = final_per_channel.tolist()
        final_return: float | list[float] = final_per_channel.tolist()
    else:
        scalar_returns = [m for _, m, _ in eval_history]
        best_return = max(scalar_returns) if scalar_returns else float("nan")
        final_return = scalar_returns[-1] if scalar_returns else float("nan")

    summary: dict[str, Any] = {
        "env_id": args.env,
        "seed": args.seed,
        "env_steps": int(env_steps),
        "wallclock_s": float(budget.elapsed_s()),
        "best_return": best_return,
        "final_return": final_return,
        "status": status,
        "initial_param_checksum": initial_checksum,
        "final_param_checksum": final_checksum,
        "params_changed": (
            None if initial_checksum is None else initial_checksum != final_checksum
        ),
        "n_evals": len(eval_history),
    }
    if error_traceback:
        summary["error"] = error_traceback

    # Per-seed summary; the Engineer aggregates these into a run-level
    # result.json via rl_research.contract.write_result. Do not call
    # write_result here — multiple seeds writing concurrently would race.
    (logdir / "result.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
