# v0 calibration — 2026-06-30

## Summary

This is the calibration note for the v0 tier of both env families,
after two Codex passes (one calibration, one adversarial self-review)
plus follow-up cleanup by the proposer.

## What was wrong (initial state)

Initial v0 baselines (commit f301b08):

- **CapacityScheduling-v0**: `SchedulingUniformPolicy` (intensity 0.5)
  reached `succ=1.00` and dominated all other policies. Violated
  acceptance gate 6 (myopic-gap).
- **KeyFuelMaze-v0**: every policy in the portfolio reached
  `succ=0.00`. Violated acceptance gate 3 (feasibility).

## What I changed (Codex's first pass, then cleanup)

Files modified:

- `src/rlh_bench/envs/registration.py` — tightened v0 success
  thresholds for CapacityScheduling:
  - `success_fill_threshold`: 0.55 → 0.85
  - `success_mandatory_threshold`: 0.50 → 0.85
  - `quality_required`: 0.55 → 0.75

- `src/rlh_bench/baselines/scheduling.py` — added
  `SchedulingCapacityPushPolicy` (a brute-force high-capacity
  policy that ignores wear / setup / inventory and pays heavily
  in those terminal-vector components). Reaches succ=0.80 on v0
  with high cost-component penalties.

- `src/rlh_bench/baselines/maze.py` — added
  `MazeOracleRoutePlannerPolicy` (originally called
  `MazeRoutePlannerPolicy` until the adversarial review caught it).
  This uses privileged env internals (waypoint coordinates, gate
  phases, etc.) so it is classified as an **oracle diagnostic**,
  not a baseline. Reaches succ=0.80-0.90 on Maze v0/Small as a
  feasibility check.

- `src/rlh_bench/baselines/__init__.py` — exports
  `MAZE_ORACLE_DIAGNOSTICS` separately from `MAZE_BASELINES`.

- `experiments/run_baselines.py` — oracle diagnostics are now run
  separately and reported in a clearly labeled
  "Oracle diagnostics (NOT comparable to baselines)" section of
  `docs/baseline_report.md`.

- `src/rlh_bench/envs/keyfuel_maze.py` — added public `seed`
  property so baselines can detect world changes without
  underscore-prefixed access.

- `tests/test_keyfuel_maze_env.py` — added
  `test_seed_property_tracks_reset` and
  `test_oracle_route_planner_not_in_baseline_portfolio` to encode
  the honest-vs-oracle separation as a test contract.

## Before / after numbers (20 seeds each)

### CapacityScheduling-v0 (H=2000, K=48, M=8, P=8, action_dim=96)

| policy                | before  | after  |
| --------------------- | ------- | ------ |
| zero                  | 0.00    | 0.00   |
| random                | 0.00    | 0.00   |
| uniform               | 1.00    | 0.30   |
| capacity_push (new)   | —       | 0.80   |
| backlog_priority      | 0.00    | 0.00   |
| earliest_deadline     | 0.05    | 0.00   |
| maintenance_aware     | 0.00    | 0.00   |
| setup_aware           | 0.00    | 0.00   |
| short_horizon_rollout | 0.00    | 0.00   |

Acceptance:
- uniform ≤ 0.30 ✓
- at least one heuristic ≥ 0.40 ✓ (capacity_push at 0.80)
- zero / random ≤ 0.05 ✓

Caveat: capacity_push is a "high-cost feasibility/stress
diagnostic", not a clever heuristic. The original myopic policies
(backlog_priority, EDD, etc.) still fail at v0. The threshold
sensitivity probe in
`lab/notes/codex_v0_calibration_review_2026-06-30.md` shows that
even relaxing thresholds back to Small values would only get
earliest_deadline to 5% — the structural issue is bundle coverage,
not threshold height. A genuinely bundle-aware non-brute heuristic
would be needed to satisfy gate 6 more cleanly.

### KeyFuelMaze-v0 (H=2000, D=32, world 48×48)

| policy                  | before | after  |
| ----------------------- | ------ | ------ |
| zero                    | 0.00   | 0.00   |
| random                  | 0.00   | 0.00   |
| random_constant         | 0.00   | 0.00   |
| greedy_landmark         | 0.00   | 0.00   |
| fuel_aware_greedy       | 0.00   | 0.00   |
| efficient_actuator      | 0.00   | 0.00   |
| short_horizon_lookahead | 0.00   | 0.00   |
| **oracle_route_planner**| —      | **0.80** (oracle; not a baseline) |

Acceptance:
- ✗ No honest baseline reaches `succ ≥ 0.20`. The original
  acceptance criterion was satisfied by an oracle masquerading
  as a baseline; the adversarial review caught and reclassified
  it.

The honest reading: KeyFuelMaze-v0 is **feasible** (the oracle
proves it) but the cheap observation-only baselines in the
portfolio aren't competent enough to solve it. This is actually
the right shape for the lab's mission: the env admits a
hand-coded solution under privileged information, but solving it
with observation-only policies is a research problem.

What would close gate 3 honestly: an observation-only memory
planner (~200-500 LOC per Codex's review) that maintains a
discovered-landmark map across timesteps. That is plausibly the
first candidate algorithm the lab tries, not a baseline.

### CapacityScheduling-Small (regression check)

| policy                | before | after  |
| --------------------- | ------ | ------ |
| zero                  | 0.00   | 0.00   |
| uniform               | 0.75   | 0.75   |
| capacity_push (new)   | —      | 1.00   |
| backlog_priority      | 0.40   | 0.40   |
| earliest_deadline     | 0.40   | 0.40   |
| maintenance_aware     | 0.05   | 0.05   |
| setup_aware           | 0.15   | 0.15   |
| short_horizon_rollout | 0.35   | 0.35   |

No regression on existing policies; capacity_push hits 100% on
Small (consistent with its role as a high-cost feasibility floor).

### KeyFuelMaze-Small (regression check)

| policy                  | before | after  |
| ----------------------- | ------ | ------ |
| zero                    | 0.00   | 0.00   |
| greedy_landmark         | 0.05   | 0.05   |
| fuel_aware_greedy       | 0.05   | 0.05   |
| short_horizon_lookahead | 0.10   | 0.10   |
| oracle_route_planner    | —      | 0.90 (oracle, separate) |

No regression.

## Codex's adversarial review of its own fixes

Full review at `lab/notes/codex_v0_calibration_review_2026-06-30.md`.

Five concerns examined; Codex's verdicts:

1. **Oracle in baseline clothing**: REAL. Fix applied: separated
   `MAZE_ORACLE_DIAGNOSTICS` from `MAZE_BASELINES`.
2. **capacity_push tautology**: REAL but acceptable. The reward
   vector shows it pays heavily on cost components, so a candidate
   algorithm has clear room for Pareto improvement.
3. **Original heuristics still fail at v0**: Structural myopia,
   not a bug. Their weighted_fill rates are 36-56%; they just
   miss bundle all-of-N coverage.
4. **Small regression from capacity_push=100%**: Acceptable. Don't
   hide it; label it as a stress diagnostic in reports.
5. **v0 thresholds over-tightened**: Contested. Relaxing them
   makes uniform too strong unless dynamics change too.

Additional items Codex flagged:
- ⚠ `RecoverableKeyFuelMazeEnv.reset()` "samples gate_phases twice":
  **FALSE ALARM** on inspection. There's one `__init__` zero and
  one `reset()` assignment, not a double sample within reset.
- ✓ Maze baselines used `getattr(env, "_seed", None)`: cleaned up
  by adding a public `seed` property.

## Outstanding work after this pass

- **KeyFuelMaze-v0 honest baseline gap (gate 3)**: needs a memory-
  based observation-only planner. Documented; deferred to first
  algorithm work.
- **Bundle-aware Scheduling heuristic (gate 6)**: would let the
  original heuristics succeed on v0 without uniform/capacity_push
  brute force. Deferred.
- **Reward normalization audit (gate 10)**: not done in this pass.
- **Held-out seed protocol (gate 9)**: infrastructure ready
  (`seed_band_for`), runtime not enforced.

These get tracked for the next iteration.
