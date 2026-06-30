# Acceptance gates audit — v2 substrate redesign

Each of the 12 acceptance gates from the v2 plan
(`lab/notes/PLAN_substrate_redesign_v2_2026-06-30.md`), with current
status as of commit 3526fbc on branch `lab/substrate-redesign`.

Status legend: ✓ PASS, ⚠ PARTIAL, ✗ FAIL, — DEFERRED.

---

## 1. Determinism gate ✓

Same seed + same action sequence → same terminal vector exactly
(mod float epsilon).

Verified:
- `tests/test_capacity_scheduling_env.py::test_same_seed_same_action_same_terminal_vector`
- `tests/test_keyfuel_maze_env.py::test_same_seed_same_action_same_terminal_vector`
- Manual: all 6 registered envs `reset(seed=42)` → identical
  observations across separate env instances.

Different seeds → different worlds:
- `tests/test_capacity_scheduling_env.py::test_different_seeds_different_worlds`
- `tests/test_keyfuel_maze_env.py::test_different_seeds_different_worlds`

## 2. Terminal-only reward gate ✓

`reward == 0` for every non-terminal step; non-zero on terminal
step only.

Verified:
- `tests/test_capacity_scheduling_env.py::test_reward_is_terminal_only`
  and the scalar-mode variant
- `tests/test_keyfuel_maze_env.py::test_reward_is_terminal_only_*`
- Manual on both Small tiers: exactly 1 non-zero reward per episode.

## 3. Feasibility gate ⚠

CapacityScheduling Small: ✓ — uniform=75% and EDD=40% show
the env is feasible.

CapacityScheduling v0: ✓ feasible (uniform=100%, indicating the
env is too easy at v0 — see gate 6).

KeyFuelMaze Small: ✓ — short_horizon_lookahead=10%,
greedy_landmark=5%.

KeyFuelMaze v0: ⚠ — no policy in the portfolio reaches success.
Open question: is the v0 maze actually feasible? See
`lab/notes/codex_v0_calibration_oneshot.md` for the calibration
TODO.

## 4. No-idle-tail gate (partial)

For baseline + oracle trajectories, meaningful state changes occur
throughout the episode. The final 25% of the horizon must affect
terminal-vector components on most worlds.

Not formally measured. Anecdotally:
- CapacityScheduling: demand_calendar covers the full horizon
  (Gaussian peaks spread across H); backlog grows and shrinks
  through the episode. Wear / heat / setup churn accumulate
  throughout. The "no idle tail" property holds by construction.
- KeyFuelMaze: agents that finish early reach the extraction
  zone; agents that don't have seals still to collect. The
  "no idle tail" property is *not enforced* — a policy that
  completes early can idle until horizon. But the route_efficiency
  and seal_completion components capture this.

Status: ⚠ — passes by construction for Scheduling; for Maze
the "idle after extraction" failure mode is possible but the
terminal vector still scores correctly.

## 5. Lookahead-depth gate (informal)

Short receding-horizon policies should underperform longer-horizon
diagnostics.

The decomposition diagnostics in both portfolios:
- `SchedulingShortHorizonRolloutPolicy` (CapacityScheduling)
- `MazeShortHorizonLookaheadPolicy` (KeyFuelMaze)

Current baseline numbers:
- Scheduling-Small: short_horizon=35% vs uniform=75% — uniform
  dominates, the short-horizon diagnostic is *not* the best,
  which is what we want (it leaves headroom for a true
  long-horizon learner).
- Scheduling-v0: short_horizon=0% vs uniform=100% — calibration
  issue at v0 (uniform shouldn't be that strong).
- KeyFuelMaze-Small: short_horizon=10% vs greedy=5% — the
  lookahead policy slightly beats greedy, which is consistent
  with the env rewarding lookahead.

Status: ⚠ — gate signal is present at Small but the v0
calibration issue muddies interpretation.

## 6. Myopic-gap gate ✗ (at v0)

Greedy / earliest-deadline policies must leave significant
headroom in *multiple* terminal-vector components.

Small tiers ✓:
- Scheduling-Small: backlog_priority and earliest_deadline both
  hit 40% success; mean_return < uniform; both leave clear
  headroom on multiple components.
- KeyFuelMaze-Small: greedy_landmark=5% success; route_efficiency
  and fuel_margin both have headroom.

v0 tier ✗:
- Scheduling-v0: uniform=100% success, dominates all heuristic
  policies. This violates the gate. Calibration TODO in
  `lab/notes/codex_v0_calibration_oneshot.md`.
- KeyFuelMaze-v0: no policy >0% success — the inverse problem
  (env may be too hard for any of the cheap baselines).

## 7. Recoverability gate ✓ (for Scheduling)

Injected perturbations at early, middle, late times should produce
graded terminal-vector degradation.

Verified:
- `tests/test_capacity_scheduling_env.py::test_recoverability_is_graded`
  injects 20 steps of negative action at t=50 / t=250 / t=450 and
  checks the terminal fill degrades vs baseline but doesn't collapse.
- KeyFuelMaze: `test_fuel_exhaustion_is_recoverable_not_terminal`
  shows fuel-out doesn't terminate the episode.

A formal recoverability *curve* across multiple injection times
hasn't been published; the gate is partial.

## 8. Action-complexity gate ✓

Top-1/top-2 action sparsification should not preserve most return
in v0/Large when dense action use is claimed.

Verified:
- `tests/test_capacity_scheduling_env.py::test_top_k_sparsification_loses_return`
  shows top-1 sparsification of a random positive action drops fill
  by > 0.05.
- KeyFuelMaze: `test_actuator_matrix_is_redundant` confirms that
  higher D maps through the actuator matrix to 2-D force (not
  D-dimensional force), so action complexity is behavioral
  (per-actuator cost differences) rather than syntactic.

## 9. Seed-generalization gate (infrastructure ready) ⚠

Held-out seeds must change outcomes; train→held-out gap is a
reportable warning.

Infrastructure ready:
- `rlh_bench.seed_bands.seed_band_for(env_id)` returns published
  train / validation / held-out / debug ranges per tier.
- Different seeds generate different worlds (gate 1).

Not yet enforced: candidate algorithms must report against
held-out band. The current `experiments/run_baselines.py` uses
seed=0 base and runs 20 episodes; effectively training-set
evaluation. To pass this gate strictly, the sweep should be
re-run twice: once on train, once on held-out. Deferred.

Status: ⚠ — infrastructure ready, runtime use pending.

## 10. Reward-normalization gate ⚠

Component magnitudes should be comparable across Small/v0/Large
after normalization.

Looking at the v0 baseline report numbers:
- Scheduling Small uniform: mean reward vector includes wear=−0.16,
  fill=0.83. Scales reasonable.
- Scheduling v0 uniform: reward_vector mean shows fill=1.00 but
  the lateness and energy components differ in scale. Not all
  components are bounded across tiers.

This gate is partially satisfied but a full normalization audit
hasn't been done. The biggest gap is that `neg_setup_churn`,
`neg_inventory_waste`, and `neg_energy` are raw cost accumulators
divided only by `horizon * action_dim` — at Large their absolute
magnitudes can dominate the scalarized return.

Status: ⚠ — partial; needs a normalization pass.

## 11. Baseline-portfolio gate ✓

Each family must ship with ≥6 cheap policies plus ≥1 decomposition
diagnostic.

CapacityScheduling: 7 policies (zero, uniform, backlog_priority,
earliest_deadline, maintenance_aware, setup_aware,
short_horizon_rollout). 1 diagnostic ✓.

KeyFuelMaze: 6 policies (zero, random_constant, greedy_landmark,
fuel_aware_greedy, efficient_actuator, short_horizon_lookahead).
1 diagnostic ✓.

## 12. Runtime gate ✓

Small episodes ~30s upper bound; Large episodes ~5min upper bound.

Measured:
- Scheduling Small: 0.09s/ep (gate: <30s) ✓
- Scheduling v0: 0.54s/ep ✓
- Scheduling Large: 7.64s/ep (gate: <300s) ✓
- KeyFuelMaze Small: 0.03s/ep ✓
- KeyFuelMaze v0: 0.18s/ep ✓
- KeyFuelMaze Large: 1.20s/ep ✓

All well under their gates.

---

## Summary

After the strict-validation pass (`lab/notes/strict_registry_outcome_2026-06-30.md`),
the registry contains only the two Small-tier envs. v0 and Large
tiers were removed because they couldn't be honestly validated as
testbeds. The audit below covers only the registered envs.

| Gate                          | CapSched-Small | KeyFuelMaze-Small | Notes                          |
| ----------------------------- | -------------- | ----------------- | ------------------------------ |
| 1.  Determinism               | ✓              | ✓                 | Cross-family parametrized test |
| 2.  Terminal-only             | ✓              | ✓                 | Cross-family parametrized test |
| 3.  Feasibility               | ✓              | ✓                 | Healthy portfolio spread       |
| 4.  No-idle-tail              | ✓              | ✓                 | Tail-zero probe + tests        |
| 5.  Lookahead-depth           | ⚠              | ⚠                 | Decomposition diagnostics present; varied-depth probe deferred |
| 6.  Myopic-gap                | ✓              | ✓                 | Multiple non-brute baselines succeed at varying rates |
| 7.  Recoverability            | ✓              | ✓                 | Graded-curve tests + probe     |
| 8.  Action-complexity         | ✓              | ✓                 | No trailing no-op dims         |
| 9.  Seed-generalization       | ✓              | ✓                 | `--use-held-out` flag + tests  |
| 10. Reward-normalization      | n/a            | n/a               | Cross-tier check requires ≥2 tiers; deferred until v0/Large re-validated |
| 11. Baseline portfolio        | ✓              | ✓                 | Oracle / portfolio cleanly separated |
| 12. Runtime                   | ✓              | ✓                 | <0.2s/ep                       |

**Headline**: 10/12 gates ✓ on both registered envs. Gate 5
(lookahead-depth) is partial — the `short_horizon_*` decomposition
diagnostics are present in the portfolio but no varied-depth probe
formally compares short vs long lookahead horizons. Gate 10
(reward normalization) is moot with a single tier per family;
it returns when v0/Large are re-validated.

## What changed in this strict-validation pass

- `RecoverableCapacityScheduling-v0` removed — no calibration
  setting passes gate 6 cleanly.
- `RecoverableCapacityScheduling-Large-v0` removed — never had
  a baseline sweep.
- `RecoverableKeyFuelMaze-v0` removed — only oracle succeeds; no
  honest baseline reaches success.
- `RecoverableKeyFuelMaze-Large-v0` removed — never had a baseline
  sweep.

See `lab/notes/strict_registry_outcome_2026-06-30.md` for the
per-env reasoning and re-registration policy.

The two Small-tier envs are the validated testbed. Algorithm work
should target them; results on the deleted tiers cannot be cited
without first re-registering through validation.

## v0 calibration pass (2026-06-30 post-merge of 1bbd notes)

Codex ran two passes:
1. Calibration pass — tightened Sched-v0 thresholds, added
   capacity_push (Sched) and route_planner (Maze).
2. Adversarial self-review — caught that route_planner was reading
   privileged env internals; separated `MAZE_ORACLE_DIAGNOSTICS`
   from `MAZE_BASELINES`.

Documents:
- `lab/notes/codex_v0_calibration_oneshot.md` — brief used.
- `lab/notes/codex_v0_calibration_2026-06-30.md` — proposer-written
  calibration note (Codex's brief timed out before writing its own).
- `lab/notes/codex_v0_calibration_review_oneshot.md` — review brief.
- `lab/notes/codex_v0_calibration_review_2026-06-30.md` — Codex's
  adversarial self-review (280 lines).

## Next pass

The substrate is in a *defensible* state at Small + v0. The
remaining gates that warrant pre-merge attention:

- **Gate 10 (reward normalization)**: some `neg_*` components scale
  with horizon at Large; would benefit from an audit pass.
- **Gate 9 (held-out seed protocol)**: `run_baselines.py` should
  optionally evaluate on held-out seeds and report train→held-out
  gap.
- **Maze-v0 honest baseline (gate 3)**: 200-500 LOC memory planner;
  best done during real algorithm work rather than as another
  baseline.
- **Bundle-aware Scheduling heuristic (gate 6 cleanup)**: would let
  original myopic policies succeed at v0 without brute force.
