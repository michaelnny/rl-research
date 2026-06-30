# One-shot brief — reward normalization audit (gate 10)

You previously calibrated the v0 tier and reviewed it adversarially.
Now: audit reward-vector component scaling across tiers.

Read this brief as your full identity for this single invocation.

## Acceptance gate 10 (from the v2 plan)

> Component magnitudes should be comparable across Small/v0/Large
> after normalization.
> No cost component should grow linearly with horizon unless
> explicitly normalized.
> Scalar reports should not be dominated by an accidental scale
> mismatch.

## What I measured

Mean reward vector for random policy, 3 seeds per tier:

### CapacityScheduling

```
component             Small      v0       Large
success               0.000     0.000    0.000
weighted_fill_rate    0.470     0.347    0.470
mandatory_fill_rate   0.000     0.000    0.028
neg_lateness         -2.118    -8.344  -19.290   <-- 9x scaling
neg_shortfall_tail   -0.126    -0.402   -0.122
neg_wear             -0.993    -1.000   -1.000
neg_heat_violation   -0.074    -0.630   -1.518   <-- 20x scaling
neg_setup_churn      -0.027    -0.056   -0.108
neg_inventory_waste   0.000     0.000   -0.316
neg_energy           -0.331    -0.333   -0.333
resilience_margin     0.000     0.000    0.000
```

Problems:
- `neg_lateness` grows linearly with horizon (or worse — 9x for 4x
  horizon ratio Small→v0, 18x for 20x horizon ratio Small→Large).
- `neg_heat_violation` grows fast across tiers.
- These contaminate the scalarized scalar return at higher tiers.

### KeyFuelMaze

```
component             Small      v0       Large
success               0.000     0.000    0.000
seal_completion       0.000     0.000    0.000
key_coverage          0.000     0.000    0.000
fuel_margin           0.836     0.000    0.000
neg_damage           -0.424    -1.000   -0.920
neg_lateness         -0.375    -0.375   -0.375
neg_energy           -0.400    -0.375   -0.369
neg_collision         0.000     0.000    0.000
route_efficiency      1.000     1.000    1.000
```

Maze is mostly OK — components stay in similar ranges. `neg_damage`
saturates at -1.0 by design.

## What you should do

Audit each scheduling reward component and propose minimal
normalization fixes. The bar:

  - No component should systematically grow with horizon for a
    constant-quality policy.
  - Components should sit in roughly [−2, 1] across all tiers for
    typical policies (zero / random / capacity_push). Extreme values
    on bad policies are fine.
  - Scalarization weights in `DEFAULT_SCHEDULING_REWARD_SPEC` shouldn't
    need re-tuning per tier — if they do, the components are doing
    work the weights should be doing.

Look at the source in `src/rlh_bench/envs/capacity_scheduling.py`,
function `_terminal_reward_vector`. The candidates:

  - `neg_lateness`: currently `self._total_lateness / max(horizon, 1)`.
    But `_total_lateness` accumulates per-step (`past_deadline.float * backlog`)
    across the horizon — so dividing by horizon doesn't actually
    cancel the growth. Should it be divided by `horizon * K` instead?
    Or normalized against expected total demand?
  - `neg_heat_violation`: currently `total_heat_violation / horizon`.
    The accumulator grows because heat tends to plateau at 1.0
    once policies are aggressive. Maybe cap the accumulator OR
    normalize by `(horizon * M)`.
  - `neg_inventory_waste`: currently `total_inventory_waste / P`.
    Doesn't scale with horizon explicitly but at Large the
    accumulator is much larger because there's more time to
    overflow inventory caps.

Don't go nuclear: minimal targeted changes that bring the random
policy's mean reward vectors within a factor of 2-3 of each other
across tiers. Acceptance:

  - On random policy with 3 seeds per tier, no scheduling reward
    component differs by more than 3x between Small and Large.

## Rules

- Modify only `src/rlh_bench/envs/capacity_scheduling.py` (the
  `_terminal_reward_vector` method specifically) and possibly the
  default weights in `DEFAULT_SCHEDULING_REWARD_SPEC`.
- Do NOT change action layout, observation layout, or the names/
  count of reward components.
- Do NOT introduce per-step reward shaping.
- Re-run pytest after each change: must still show 61 passing.
- Re-run the baseline sweep:
  `PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20`
  Check that the Small / v0 baseline numbers stay in their
  calibrated shape (no policy that was at 0% now reaches 50%, etc.).
  Specifically, capacity_push v0 should remain 60-90%.

## Deliverable

Write the audit + fixes to
`lab/notes/codex_reward_normalization_audit_2026-06-30.md`:

```markdown
# Reward normalization audit — 2026-06-30

## What was wrong
<one paragraph per offending component>

## What I changed
<diff summary; specific lines>

## Before / after random-policy means
<table per component across tiers>

## Sanity: baseline shape on Small + v0 unchanged?
<table>
```

## How a session ends

When all scheduling components fit the 3x cross-tier bound, the
baseline shape is preserved, and the audit note is written. Do not
commit; the proposer will commit after sanity-checking.
