# Acceptance gates audit — current substrate

Status as of the strict-validation pass
(`lab/notes/strict_registry_outcome_2026-06-30.md`). After v0 and
Large tiers were deleted, the registry contains only the two
Small-tier envs; this audit covers only those. The 12 gates are
those defined in
`lab/notes/planning/PLAN_substrate_redesign_v2_2026-06-30.md`.

Status legend: ✓ PASS, ⚠ PARTIAL, n/a NOT APPLICABLE.

| Gate                          | CapSched-Small | KeyFuelMaze-Small | Notes                          |
| ----------------------------- | -------------- | ----------------- | ------------------------------ |
| 1.  Determinism               | ✓              | ✓                 | Cross-family parametrized test |
| 2.  Terminal-only             | ✓              | ✓                 | Cross-family parametrized test |
| 3.  Feasibility               | ✓              | ✓                 | Multiple non-trivial policies succeed |
| 4.  No-idle-tail              | ✓              | ✓                 | Tail-zero probe + 2 tests      |
| 5.  Lookahead-depth           | ⚠              | ⚠                 | Decomposition diagnostics present; varied-depth probe deferred |
| 6.  Myopic-gap                | ✓              | ✓                 | Healthy spread across the portfolio |
| 7.  Recoverability            | ✓              | ✓                 | Graded-curve tests + recoverability probe |
| 8.  Action-complexity         | ✓              | ✓                 | No trailing no-op dims         |
| 9.  Seed-generalization       | ✓              | ✓                 | `--use-held-out` flag + tests  |
| 10. Reward-normalization      | n/a            | n/a               | Cross-tier check requires ≥2 tiers; deferred until v0/Large re-validated |
| 11. Baseline portfolio        | ✓              | ✓                 | Oracle / portfolio cleanly separated |
| 12. Runtime                   | ✓              | ✓                 | <0.2s/ep                       |

**10/12 gates ✓ on both registered envs.** Two partials:

- **Gate 5 (lookahead-depth)**: the `short_horizon_*` decomposition
  diagnostics are in the portfolio (`SchedulingShortHorizonRolloutPolicy`,
  `MazeShortHorizonRolloutPolicy`), but no varied-depth probe formally
  compares short vs long lookahead horizons. Deferred — a varied-depth
  probe is straightforward to add when first algorithm work needs it.

- **Gate 10 (reward-normalization)**: cross-tier check is moot with
  one tier per family. The check returns when v0/Large are
  re-registered.

## Per-gate evidence

### Gate 1 — Determinism
- `tests/test_capacity_scheduling_env.py::test_same_seed_same_action_same_terminal_vector`
- `tests/test_keyfuel_maze_env.py::test_same_seed_same_action_same_terminal_vector`
- `tests/test_cross_family_invariants.py::test_determinism_all_envs` (parametrized over the registry)
- Plus: different seeds produce different worlds, tested per-family.

### Gate 2 — Terminal-only reward
- `tests/test_capacity_scheduling_env.py::test_reward_is_terminal_only` (scalar + vector variants)
- `tests/test_keyfuel_maze_env.py::test_reward_is_terminal_only_*`
- `tests/test_cross_family_invariants.py::test_reward_is_terminal_only_all_envs` (parametrized)
- Manual: each env returns exactly 1 non-zero reward per episode.

### Gate 3 — Feasibility
- Sched-Small: `uniform`=75%, `capacity_push`=100% (cost-heavy), `bundle_aware`=70%, `backlog_priority`=`earliest_deadline`=40%.
- Maze-Small: `oracle_route_planner`=90% (diagnostic), `short_horizon_lookahead`=10%, `greedy_landmark`=`fuel_aware_greedy`=5%.

### Gate 4 — No-idle-tail
- `experiments/probes/idle_tail.py` — measures terminal-vector shift when last 25% of trajectory is zeroed.
- `tests/test_idle_tail_gate.py::test_tail_affects_terminal_for_scheduling` (≥3 components shift by >0.05)
- `tests/test_idle_tail_gate.py::test_tail_affects_terminal_for_maze` (≥2 components shift by >0.01)

### Gate 5 — Lookahead-depth ⚠
- Portfolios include decomposition diagnostics:
  `SchedulingShortHorizonRolloutPolicy`, `MazeShortHorizonRolloutPolicy`.
- On Sched-Small: short_horizon=35% vs uniform=75% — short-horizon is *not* the best, leaving headroom for a long-horizon learner.
- On Maze-Small: short_horizon_lookahead=10% vs greedy_landmark=5% — slight win, consistent with the env rewarding lookahead.
- No varied-depth probe; deferred.

### Gate 6 — Myopic-gap
- Sched-Small portfolio spread (10 policies): success rates range from 0% (zero/random) through 5–15% (maintenance_aware, setup_aware) through 35–40% (short_horizon_rollout, backlog_priority, earliest_deadline) through 70–75% (bundle_aware, uniform) to 100% (capacity_push, the labeled stress diagnostic).
- Maze-Small: similar spread across 7 honest policies.

### Gate 7 — Recoverability
- `experiments/probes/recoverability.py` — bursts at 0.05H / 0.50H / 0.85H.
- `tests/test_recoverability_gate.py::test_scheduling_recoverability_graded_curve` (≥2 components shift; worst-burst fill stays >0.3)
- `tests/test_recoverability_gate.py::test_maze_recoverability_responds_to_perturbation` (≥2 maze components shift)
- `tests/test_keyfuel_maze_env.py::test_fuel_exhaustion_is_recoverable_not_terminal` (fuel-out doesn't end the episode)

### Gate 8 — Action-complexity
- `tests/test_capacity_scheduling_env.py::test_top_k_sparsification_loses_return` — top-1 sparsification of a random positive action drops fill by >0.05.
- `tests/test_capacity_scheduling_env.py::test_registered_tiers_have_no_trailing_action_dims` — action_dim exactly equals K+3M+P at every registered tier.
- `tests/test_keyfuel_maze_env.py::test_actuator_matrix_is_redundant` — higher action_dim is per-world redundant actuators, not extra force dimensions.

### Gate 9 — Seed-generalization
- `rlh_bench.seed_bands.seed_band_for(env_id)` returns published train / validation / held-out / debug ranges per tier.
- `tests/test_seed_bands.py` — all four bands pairwise disjoint; representative seeds produce different worlds.
- `experiments/run_baselines.py --use-held-out` evaluates the portfolio on both train and held-out seeds and reports the gap.

### Gate 11 — Baseline portfolio
- `rlh_bench.baselines.scheduling.SCHEDULING_BASELINES` — 9 learner-facing policies, including 1 decomposition diagnostic.
- `rlh_bench.baselines.maze.MAZE_BASELINES` — 6 learner-facing policies, including 1 decomposition diagnostic.
- `rlh_bench.baselines.maze.MAZE_ORACLE_DIAGNOSTICS` — privileged-info diagnostics kept strictly separate.

### Gate 12 — Runtime
- Sched-Small: ~0.09s/episode (gate: <30s) ✓
- Maze-Small: ~0.03s/episode ✓

## Known deferrals

- **Gate 5 varied-depth probe**: cheap to add; defer until first
  algorithm work needs the signal.
- **Gate 10 cross-tier**: returns when v0 / Large are re-registered.
- **v0 / Large re-registration**: see
  `lab/notes/strict_registry_outcome_2026-06-30.md` for the per-env
  paths back into the registry.
