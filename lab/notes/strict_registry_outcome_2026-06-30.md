# Sched-v0 / Maze-v0 / Large tiers — strict-validation outcome

Per the proposer's strict criterion ("only registered envs that
actually work as a testbed, free of bugs/flaws"), all four
non-validated tiers were removed from the registry.

## Why each was removed

### CapacityScheduling-v0 — REMOVED

Codex was asked to build a bundle-aware Scheduling baseline to
close gate 6 (myopic-gap) at v0. The policy was implemented
(SchedulingBundleAwarePolicy in src/rlh_bench/baselines/scheduling.py)
and works well on Small (succ=0.70, mandatory_fill=0.75, NOT brute
force). But on v0 it reached only succ=0.0, mandatory_fill=0.16.

A threshold sensitivity check on v0:

  - With tight thresholds (current v0 setting): uniform=30%,
    bundle_aware=0%, capacity_push=80%. Gate 6 violated.
  - With relaxed thresholds (Small defaults): uniform=100%,
    bundle_aware=30%, capacity_push=100%. Same gate-6 violation,
    different direction.

There is no threshold setting where (a) uniform fails AND
(b) bundle_aware succeeds AND (c) capacity_push isn't brute-force
dominant. The v0 design has K=48 projects competing for capacity
that uniform can spread thin enough to satisfy modest bundles; a
bundle-aware policy can't focus enough capacity on the bundle
members to beat that.

This is a structural calibration issue with the v0 tier as
designed. Fixable in principle (smaller K, different bundle
distribution, sharper compatibility), but redesigning v0 is a
substrate change that needs its own design pass. **Removed from
the registry** until that redesign happens.

### KeyFuelMaze-v0 — REMOVED

The only policy that succeeds at v0 is the oracle (succ=0.80).
No honest observation-only baseline reaches >0%. Codex's earlier
review estimated that an observation-only memory planner would be
~200-500 LOC of memory + FSA logic — a research project, not a
baseline.

Per the strict criterion: an env where we have no honest baseline
demonstrating learnability is not a validated testbed. **Removed
from the registry** until a memory planner is implemented and
demonstrates honest learnability.

### CapacityScheduling-Large + KeyFuelMaze-Large — REMOVED

Built but baselines never run. Unvalidated at K=128 / M=16 / H=10k
scale. Latent bugs (normalization, generator collisions, accumulator
overflow at long horizons) cannot be ruled out. **Removed from the
registry** pending a Large baseline sweep. Re-validation requires
running the sweep with the existing baseline portfolio and
confirming the gates hold at scale.

## What stays in the codebase

- The env classes (`RecoverableCapacitySchedulingEnv`,
  `RecoverableKeyFuelMazeEnv`) and their dataclasses remain. Small
  uses them.

- `SchedulingBundleAwarePolicy` stays in `SCHEDULING_BASELINES` —
  it's a useful non-brute baseline on Sched-Small (succ=0.70,
  mandatory_fill=0.75) and earns its place in the portfolio there.
  It will be useful again if v0 is re-registered after redesign.

- `MazeOracleRoutePlannerPolicy` stays in `MAZE_ORACLE_DIAGNOSTICS`.
  It only runs on Small now (Maze-v0 / Large are not in the
  registry), where it correctly serves as the feasibility upper
  bound.

- All probe scripts, gate tests, world_gen helpers, and seed_bands
  infrastructure remain unchanged. They were designed to work
  across all tiers; only Small actually exercises them now, but
  the code is ready for re-validation work.

## What got deleted from tests

  - `test_recoverability_v0_no_catastrophic_collapse` (referenced
    Sched-v0).
  - `test_reward_components_stay_within_3x_cross_tier` (compared
    Small vs Large — moot with one tier).
  - `test_keyfuel_reward_components_stay_within_3x_cross_tier`
    (same).
  - `test_idle_tail_maze_v0` (Maze-v0 deleted).
  - `RecoverableCapacityScheduling-v0` parameterization on
    `test_tail_affects_terminal_for_scheduling`.
  - `test_registered_tiers_have_no_trailing_action_dims` had its
    tier list shrunk to [Small].

Two cross-family invariants (determinism + terminal-only + info
contract, parametrized over registered_envs()) still pass on the
trimmed registry; they automatically follow registry changes.

Test count: 94 → 77 (17 tests for deleted envs removed; no failures
on the trimmed registry).

## Re-registration policy

To bring any env back into the registry, the change requires:

  1. A real, honest demonstration that the env works as a testbed:
     - trivial policies fail
     - smart policies succeed
     - brute-force policies aren't dominant (or are explicitly
       labeled as stress diagnostics, not baselines)
  2. All 12 acceptance gates hold at the tier being added (or are
     explicitly waived in the audit with rationale)
  3. Codex peer review of the validation evidence
  4. A clear commit message documenting the validation work

This is the same bar that Small passed. Anything less leaks
unreliable testbed surface into the lab loop's evaluation.

## Tasks closed (moved to "deleted")

  - task 48: bundle-aware Scheduling baseline (the policy lives;
    Sched-v0 doesn't)
  - task 49: memory planner for Maze-v0 (deferred — first real
    algorithm work, not a baseline)
  - task 50: validate Sched-Large (deferred — same standard as v0)
  - task 51: validate Maze-Large (deferred — same)
