"""Baseline sweep across the registered envs (v2 substrate redesign).

Runs each family's baseline portfolio (from
``rlh_bench.baselines.scheduling.SCHEDULING_BASELINES`` and
``rlh_bench.baselines.maze.MAZE_BASELINES``) against the currently
registered envs. v0/Large tiers were removed pending validation;
they will be included automatically if they are re-registered.

Writes:

  experiments/results/baselines.json  -- per-env per-policy summary
  docs/baseline_report.md              -- human-readable markdown

Usage::

    PYTHONPATH=src .venv/bin/python experiments/run_baselines.py

Optional CLI flags:

    --episodes N         episodes per (env, policy) cell (default 20)
    --include-large      retained for compatibility; no effect unless Large tiers are registered
    --seed N             base seed (default 0)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlh_bench import (
    make_env,
    pareto_non_dominated,
    registered_envs,
    rollout,
)
from rlh_bench.baselines import (
    RandomPolicy,
    ZeroPolicy,
)
from rlh_bench.baselines.maze import MAZE_BASELINES, MAZE_ORACLE_DIAGNOSTICS
from rlh_bench.baselines.scheduling import SCHEDULING_BASELINES
from rlh_bench.seed_bands import seed_band_for


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "experiments" / "results" / "baselines.json"
REPORT_PATH = REPO_ROOT / "docs" / "baseline_report.md"


def _summarize(env_id: str, policy_factory, seeds: list[int]) -> dict[str, Any]:
    """Roll out `policy_factory(env)` against each seed in ``seeds`` and aggregate.

    ``policy_factory`` is a per-rollout, no-training baseline factory. Do not
    put trainable algorithms here: they need a train-on-train-seeds/evaluate-on-
    held-out harness (see ``experiments/algorithms/runner.py``), not repeated
    construction inside the evaluation loop.
    """

    if not seeds:
        raise ValueError("_summarize requires at least one seed")

    returns: list[float] = []
    vectors: list[np.ndarray] = []
    successes: list[int] = []
    lengths: list[int] = []
    durations: list[float] = []

    for seed in seeds:
        env = make_env(env_id)
        policy = policy_factory(env)
        t0 = time.perf_counter()
        result = rollout(env, policy, seed=seed)
        durations.append(time.perf_counter() - t0)
        returns.append(result.scalar_return)
        vectors.append(result.reward_vector)
        successes.append(int(bool(result.info.get("is_success", False))))
        lengths.append(result.length)

    vec = np.stack(vectors)
    first_success = next((i + 1 for i, s in enumerate(successes) if s), None)
    pareto_mask = pareto_non_dominated(vec) if vec.shape[0] > 0 else np.zeros(0, dtype=bool)

    return {
        "episodes": len(seeds),
        "seeds": list(seeds),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_reward_vector": vec.mean(axis=0).round(4).tolist(),
        "reward_names": list(make_env(env_id).reward_spec.names),
        "success_rate": float(np.mean(successes)),
        "first_success_episode": first_success,
        "mean_length": float(np.mean(lengths)),
        "mean_sec_per_episode": float(np.mean(durations)),
        "pareto_non_dominated_count": int(pareto_mask.sum()),
    }


def _select_envs(include_large: bool) -> list[str]:
    """Return registered envs, optionally excluding registered Large tiers."""
    all_ids = list(registered_envs())
    if include_large:
        return all_ids
    return [eid for eid in all_ids if "Large" not in eid]


def _baselines_for(env_id: str) -> list[tuple[str, Any]]:
    """Return (name, policy_factory) pairs for the given env ID.

    Returns the honest learner-facing portfolio. Policies read the
    observation plus the public model API (env.actuator_matrix,
    env.seed) — they do NOT touch private underscore-prefixed env
    attributes. Oracle diagnostics are available via
    :func:`_oracle_diagnostics_for` and reported separately.
    """
    base_factories: list[tuple[str, Any]] = [
        ("zero", lambda env: ZeroPolicy(env.action_space)),
        ("random", lambda env: RandomPolicy(env.action_space, seed=0)),
    ]
    if "Scheduling" in env_id:
        portfolio = SCHEDULING_BASELINES
    elif "KeyFuelMaze" in env_id:
        portfolio = MAZE_BASELINES
    else:
        portfolio = []
    factories = base_factories[:]
    for PolicyCls in portfolio:
        # Skip the trivial zero policy if already covered.
        if "Zero" in PolicyCls.__name__:
            continue
        factories.append((PolicyCls.name, lambda env, cls=PolicyCls: cls(env)))
    return factories


def _oracle_diagnostics_for(env_id: str) -> list[tuple[str, Any]]:
    """Return (name, factory) pairs for oracle/planner diagnostics.

    Oracle diagnostics read privileged env-internal state (waypoint
    coordinates, gate phases, etc.) and are NOT comparable to the
    learner-facing portfolio. They exist to verify feasibility:
    if no oracle succeeds, the env is broken; if a learner matches
    or beats the oracle, that's surprising and informative.
    """
    if "KeyFuelMaze" in env_id:
        portfolio = MAZE_ORACLE_DIAGNOSTICS
    else:
        portfolio = []
    return [(cls.name, lambda env, cls=cls: cls(env)) for cls in portfolio]


def _fmt_vec(vec: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"


def _render_markdown(records: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Baseline report",
        "",
        f"Generated by `experiments/run_baselines.py`. Seed base = {records['seed']}.",
        "",
        "Baseline portfolios per family, evaluated on the currently registered "
        "envs. v0/Large tiers were removed pending validation. All values use scalar "
        "reward mode. Vector means are reported component-wise in reward-spec "
        "order; every component is larger-is-better.",
        "",
        "Each row is one policy on one env. The policy portfolios are defined in "
        "`rlh_bench.baselines.scheduling` and `rlh_bench.baselines.maze`. The "
        "*decomposition diagnostic* (`short_horizon_*`) is the load-bearing test "
        "for whether the long-horizon claim holds: if it solves an env, the env "
        "is short-horizon in disguise.",
        "",
    ]

    for env_id, by_policy in records["envs"].items():
        lines.append(f"## {env_id}")
        lines.append("")
        names = next(iter(by_policy.values()))["reward_names"]
        lines.append(f"Reward names: `{names}`")
        lines.append("")
        held_out_by_policy = (records.get("held_out") or {}).get(env_id) or {}
        if held_out_by_policy:
            lines.append(
                "| policy | episodes | train succ | held-out succ | gap | train return | held-out return | sec/ep |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for name, row in by_policy.items():
                ho_row = held_out_by_policy.get(name, {})
                train_succ = row["success_rate"]
                ho_succ = ho_row.get("success_rate", float("nan"))
                gap = train_succ - ho_succ
                lines.append(
                    "| {name} | {episodes} | {train_s:.2f} | {ho_s:.2f} | {gap:+.2f} | {train_r:.3f} | {ho_r:.3f} | {sec:.4f} |".format(
                        name=name,
                        episodes=row["episodes"],
                        train_s=train_succ,
                        ho_s=ho_succ,
                        gap=gap,
                        train_r=row["mean_return"],
                        ho_r=ho_row.get("mean_return", float("nan")),
                        sec=row["mean_sec_per_episode"],
                    )
                )
        else:
            lines.append(
                "| policy | episodes | success | mean return | first success ep | mean length | sec/ep | mean reward vector |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for name, row in by_policy.items():
                lines.append(
                    "| {name} | {episodes} | {success:.2f} | {ret:.3f} | {fs} | {length:.1f} | {sec:.4f} | {vec} |".format(
                        name=name,
                        episodes=row["episodes"],
                        success=row["success_rate"],
                        ret=row["mean_return"],
                        fs=row["first_success_episode"] if row["first_success_episode"] is not None else "—",
                        length=row["mean_length"],
                        sec=row["mean_sec_per_episode"],
                        vec=_fmt_vec(row["mean_reward_vector"]),
                    )
                )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The portfolio shape is the difficulty signal, not any single policy's "
        "success rate. A useful novel algorithm should beat multiple of these "
        "along multiple terminal-vector components."
    )
    lines.append(
        "- The `short_horizon_*` policies are decomposition diagnostics. If they "
        "match the strongest hand-coded policy on a tier, the env is solvable "
        "without long-horizon credit assignment — flag it."
    )
    lines.append(
        "- `capacity_push` on CapacityScheduling is a high-cost feasibility/stress "
        "diagnostic, not a normal heuristic. It buys success/fill by running "
        "all-out production with no maintenance or setup retargeting, paying "
        "heavily in `neg_wear`, `neg_inventory_waste`, `neg_energy`, and "
        "`resilience_margin`. Candidates should match its success while improving "
        "those cost components."
    )
    lines.append(
        "- v0/Large tiers are not currently registered. Once a tier is "
        "re-registered after validation, this script will include it by default "
        "unless it is a Large tier and `--include-large` is omitted."
    )

    # Oracle diagnostics section
    if records.get("oracle_diagnostics"):
        lines.append("")
        lines.append("## Oracle diagnostics (NOT comparable to baselines)")
        lines.append("")
        lines.append(
            "These policies read privileged env-internal state "
            "(waypoint coordinates, gate phases, etc.) that the public "
            "observation does not expose. They exist to verify feasibility "
            "— if no oracle succeeds, the env is broken. They are NOT "
            "comparable to the learner-facing portfolio above and should "
            "never be cited as 'baseline beaten'."
        )
        lines.append("")
        for env_id, by_oracle in records["oracle_diagnostics"].items():
            lines.append(f"### {env_id}")
            lines.append("")
            names = next(iter(by_oracle.values()))["reward_names"]
            lines.append(f"Reward names: `{names}`")
            lines.append("")
            lines.append(
                "| oracle | episodes | success | mean return | mean reward vector |"
            )
            lines.append("| --- | --- | --- | --- | --- |")
            for name, row in by_oracle.items():
                lines.append(
                    "| {name} | {episodes} | {success:.2f} | {ret:.3f} | {vec} |".format(
                        name=name,
                        episodes=row["episodes"],
                        success=row["success_rate"],
                        ret=row["mean_return"],
                        vec=_fmt_vec(row["mean_reward_vector"]),
                    )
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-held-out",
        action="store_true",
        help=(
            "Evaluate on both training and held-out seed bands from "
            "rlh_bench.seed_bands.seed_band_for(env_id). Reports the "
            "success gap, which is the canonical gate-9 signal."
        ),
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")

    records: dict[str, Any] = {
        "seed": args.seed,
        "envs": {},
        "oracle_diagnostics": {},
        "held_out": {} if args.use_held_out else None,
    }
    env_ids = _select_envs(args.include_large)

    for env_id in env_ids:
        print(f"\n=== {env_id} ===")
        # Default seed list: contiguous range starting at args.seed
        # (preserves the previous behavior when --use-held-out is off).
        train_seeds = list(range(args.seed, args.seed + args.episodes))
        if args.use_held_out:
            bands = seed_band_for(env_id)
            if args.episodes > len(bands.train) or args.episodes > len(bands.held_out):
                parser.error(
                    f"--episodes {args.episodes} exceeds seed band length for {env_id} "
                    f"(train={len(bands.train)}, held_out={len(bands.held_out)}). "
                    "Use a smaller value so train and held-out summaries use "
                    "the same number of episodes."
                )
            # Use the documented train band (first args.episodes
            # samples for runtime); the held-out band is sampled
            # similarly from its own range.
            train_seeds = list(bands.train)[: args.episodes]
            held_out_seeds = list(bands.held_out)[: args.episodes]
            print(f"  [train band: {train_seeds[0]}..{train_seeds[-1]}, "
                  f"held-out: {held_out_seeds[0]}..{held_out_seeds[-1]}]")

        by_policy: dict[str, Any] = {}
        held_out_by_policy: dict[str, Any] = {}
        for name, factory in _baselines_for(env_id):
            print(f"  {name:30s} ... ", end="", flush=True)
            summary = _summarize(env_id, factory, train_seeds)
            print(
                f"succ={summary['success_rate']:.2f} return={summary['mean_return']:.3f} "
                f"sec/ep={summary['mean_sec_per_episode']:.3f}"
            )
            by_policy[name] = summary
            if args.use_held_out:
                print(f"  {name + ' (held-out)':30s} ... ", end="", flush=True)
                ho_summary = _summarize(env_id, factory, held_out_seeds)
                print(
                    f"succ={ho_summary['success_rate']:.2f} return={ho_summary['mean_return']:.3f}"
                )
                held_out_by_policy[name] = ho_summary
        records["envs"][env_id] = by_policy
        if args.use_held_out:
            records["held_out"][env_id] = held_out_by_policy

        # Oracle diagnostics (reported separately; not comparable to baselines)
        oracle_factories = _oracle_diagnostics_for(env_id)
        if oracle_factories:
            by_oracle: dict[str, Any] = {}
            for name, factory in oracle_factories:
                print(f"  [oracle] {name:22s} ... ", end="", flush=True)
                summary = _summarize(env_id, factory, train_seeds)
                print(
                    f"succ={summary['success_rate']:.2f} return={summary['mean_return']:.3f}"
                )
                by_oracle[name] = summary
            records["oracle_diagnostics"][env_id] = by_oracle

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(records, indent=2, default=float))
    REPORT_PATH.write_text(_render_markdown(records))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
