# CapacityScheduling calibration diagnosis — 2026-06-30

## Summary

The Small tier was degenerate for two structural reasons: capacity was effectively a demand flood, and the state couplings were charged on *served backlog* after clipping instead of on *committed operation*. Any positive allocation therefore cleared the tiny normalized demand stream, stored excess as inventory, and accrued almost no wear/heat. I fixed this in `src/rlh_bench/envs/capacity_scheduling.py` without changing the action layout or terminal reward-vector shape.

Files changed:

- `src/rlh_bench/envs/capacity_scheduling.py`
- Added/kept probe: `experiments/probes/capacity_scheduling_panel.py`

No `world_gen.py` changes were needed.

## What was structurally wrong

1. **Capacity/demand scale was wrong.**
   - Config region: `capacity_scheduling.py` lines ~120-155.
   - The demand calendar normalizes each project to total demand ≈ 1.0 over the full horizon, so Small has only ≈16 units of demand. The old `max_capacity_per_mode=0.3` left far too much usable supply once integrated over 500 steps, especially because excess capacity could become inventory.
   - Also, the `mode_alloc` softmax means the implementation splits an aggregate mode-operation budget across modes; this made the comment/design math misleading. The important calibration quantity is integrated effective supply after setup/compatibility, not the raw `M * cap * H` number alone.

2. **Wear/heat were charged on clipped demand service, not on operating effort.**
   - Production/use region: currently lines ~451-548; the old bug was in the old `mode_used -> mode_util` calculation.
   - Production was first capped by backlog. Because per-step backlog is tiny under normalized demand, `mode_used` was tiny even when the policy commanded nontrivial capacity every step. Then `mode_util = mode_used / max_capacity_per_mode`, so wear and heat stayed near zero.
   - Result: an over-capacity policy could run all 500 steps and get `mean_wear≈0`, `mean_heat≈0`.

3. **Setup churn was mostly bookkeeping.**
   - Setup region: currently lines ~430-468 and ~550-562.
   - The old dynamics moved the setup mixture slowly and accumulated terminal setup churn, but retargeting setup did not materially reduce same-step capacity and barely affected wear/heat. `rand_pos` therefore paid a visible `churn` number but still filled everything.

4. **Inventory was a free second demand channel.**
   - Inventory region: currently lines ~564-575.
   - The old excess calculation mixed the wear proxy with actual unused capacity, and most unserved excess could be packed into buffers. With the capacity glut, this let positive policies preload perishable inventory without a meaningful operational trade-off.

5. **Bundles were not inherently easy; they were made easy by the flood.**
   - `make_bundles` samples random project subsets. I did not find evidence that bundles were biased toward easy projects. They scored 1.0 because all projects were being served to ≈1.0.

## Fix applied

One coherent calibration change: make committed operation scarce and costly, while keeping recoverability.

- Tuned Small/default knobs:
  - `max_capacity_per_mode: 0.30 -> 0.45` after adding real setup/wear losses. The raw cap is higher than the first attempted scarcity value because setup pressure and wear now actually bite.
  - `wear_rate: 0.08 -> 0.008`
  - `wear_recovery_rate: 0.04 -> 0.001`
  - `heat_buildup_rate: 0.08 -> 0.008`
  - `heat_decay_rate: 0.03 -> 0.001`
  - `quality_required: 0.85 -> 0.55`
  - `success_fill_threshold: 0.80 -> 0.55`
  - added `setup_capacity_penalty: 0.45`

- Structural changes:
  1. Compute setup target/pressure before capacity. Setup pressure now reduces same-step productive capacity.
  2. Charge wear/heat on committed operating effort plus setup pressure, not on post-backlog-clipped served demand.
  3. Inventory now only receives a small fraction (`0.20`) of actual unused productive capacity (`mode_capacity - mode_used`), so it is a weak smoothing buffer rather than a free solver.

I kept the strongest couplings: **wear ↔ capacity**, **setup inertia/pressure**, and **bundle/prioritized service**. Inventory remains present but deliberately weaker because in the first cut it was hurting calibration more than helping.

## Baseline panel

Probe command:

```bash
PYTHONPATH=src .venv/bin/python experiments/probes/capacity_scheduling_panel.py
```

### Before

Numbers from the failing calibration report:

| policy | succ | fill | mand | wear | heat | churn | energy | late |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0 | 2222 |
| ones | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 3.6 | 16000 | 0 |
| half_pos | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 3.6 | 4000 | 0 |
| neg_half | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 4000 | 2222 |
| rand_pm | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 13.7 | 5335 | 0.01 |
| rand_pos | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 12.8 | 5336 | 0.00 |

### After

20 seeds, Small config from the brief:

| policy | succ | fill | mand | wear | heat | churn | energy | late |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0 | 2222.34 |
| ones | 0.55 | 0.66 | 0.35 | 0.68 | 0.68 | 3.6 | 16000 | 722.85 |
| half_pos | 0.05 | 0.54 | 0.08 | 0.45 | 0.32 | 3.6 | 4000 | 1042.69 |
| rand_pm | 0.00 | 0.47 | 0.00 | 1.00 | 0.99 | 13.7 | 5335 | 1061.90 |
| rand_pos | 0.00 | 0.46 | 0.00 | 1.00 | 1.00 | 12.8 | 5336 | 1094.91 |

This now has the intended shape:

- `zero`: no accidental success.
- `ones`: partial success (`0.55`) and fill (`0.66`), with real wear/heat.
- `rand_pm`: no success and partial fill (`0.47`), with high churn-driven wear/heat.
- mandatory bundles no longer saturate for arbitrary positive actions.

Determinism spot-check also passed for same seed + same action stream.

## Caveat / next calibration check

A hand-written competent policy should now be implemented as a separate baseline. The env has room for it through compatible-mode selection, setup-aware retargeting, and selective bundle completion, but I did not add that baseline here. The next acceptance check should verify that a focused setup-aware policy reaches roughly `succ≈0.6-0.9` without simply raising uniform/random policies back to saturation.
