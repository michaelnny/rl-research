# Reward normalization audit — 2026-06-30

## What was wrong

`neg_lateness` was divided only by `horizon`, but `_total_lateness` is a per-step sum over project backlog after each project's deadline. That made the component scale with the number of projects and left large-tier scalar returns dominated by a bookkeeping scale mismatch rather than worse scheduling quality.

`neg_heat_violation` was divided only by `horizon`, but `_total_heat_violation` is a per-step sum over modes. Longer horizons also spend a larger fraction of the episode after heat has reached its plateau, so random/aggressive policies were punished much more on v0/Large than on Small even when the per-mode dynamics were qualitatively the same.

`neg_setup_churn` was already horizon-normalized, but the accumulator is a sum over modes. Small→Large has 4× as many modes, and the random-policy component grew almost exactly 4×.

`neg_inventory_waste` was divided only by product count. The accumulator is an integrated overflow/perishability flow, so longer tiers have more chances to accumulate waste even when per-step waste is small.

Other scheduling components were already dimensionless or averaged appropriately: fill rates, shortfall tail, mean wear, energy per action dimension per step, and terminal resilience.

## What I changed

Only `src/rlh_bench/envs/capacity_scheduling.py` was changed, inside `_terminal_reward_vector`:

- Lines 692-701: introduced shared `horizon`, `num_projects`, `num_modes`, and `num_products` denominators; changed `neg_lateness` normalization from `total_lateness / horizon` to `total_lateness / (horizon * num_projects)`.
- Lines 705-712: changed heat to a per-mode per-step average, `total_heat_violation / (horizon * num_modes)`, with a saturation cap of `0.5 * (max_heat - 0.9)` so persistent plateau heat cannot dominate longer tiers.
- Line 713: changed setup churn from `total_setup_churn / horizon` to `total_setup_churn / (horizon * num_modes)`.
- Lines 714-718: changed inventory waste from `total_inventory_waste / num_products` to `total_inventory_waste / (horizon * num_products)`.
- No reward component names/counts, action/observation layouts, per-step rewards, or scalarization weights were changed.

## Before / after random-policy means

3 seeds per tier, `RandomPolicy(action_space, seed=0)`, env seeds 0-2.

| component | Small before | v0 before | Large before | Small after | v0 after | Large after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| success | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| weighted_fill_rate | 0.492 | 0.348 | 0.469 | 0.492 | 0.348 | 0.469 |
| mandatory_fill_rate | 0.167 | 0.000 | 0.028 | 0.167 | 0.000 | 0.028 |
| neg_lateness | -2.039 | -8.345 | -19.331 | -0.127 | -0.174 | -0.151 |
| neg_shortfall_tail | -0.115 | -0.401 | -0.122 | -0.115 | -0.401 | -0.122 |
| neg_wear | -0.995 | -1.000 | -1.000 | -0.995 | -1.000 | -1.000 |
| neg_heat_violation | -0.072 | -0.629 | -1.519 | -0.018 | -0.050 | -0.050 |
| neg_setup_churn | -0.027 | -0.056 | -0.108 | -0.007 | -0.007 | -0.007 |
| neg_inventory_waste | 0.000 | 0.000 | -0.320 | 0.000 | 0.000 | -0.000 |
| neg_energy | -0.333 | -0.332 | -0.333 | -0.333 | -0.332 | -0.333 |
| resilience_margin | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Post-fix nonzero Small↔Large component ratios are within 3× for the cost components that had accidental tier scaling: lateness 1.19×, heat 2.79×, setup churn 1.02×, energy 1.00×, wear 1.01×. Inventory waste is now a per-step/product average and is effectively zero under random policy on all tiers (`-3.2e-05` on Large).

## Sanity: baseline shape on Small + v0 unchanged?

Command: `PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20`.

`PYTHONPATH=src .venv/bin/python -m pytest` also passed: 61 passed.

| env | policy | success before | success after | return before | return after |
| --- | --- | ---: | ---: | ---: | ---: |
| Small | zero | 0.00 | 0.00 | -0.216 | -0.008 |
| Small | random | 0.00 | 0.00 | 0.044 | 0.146 |
| Small | uniform | 0.75 | 0.75 | 1.099 | 1.164 |
| Small | capacity_push | 1.00 | 1.00 | 1.547 | 1.630 |
| Small | backlog_priority | 0.40 | 0.40 | 0.609 | 0.686 |
| Small | earliest_deadline | 0.40 | 0.40 | 0.673 | 0.730 |
| Small | maintenance_aware | 0.05 | 0.05 | 0.163 | 0.251 |
| Small | setup_aware | 0.15 | 0.15 | 0.254 | 0.347 |
| Small | short_horizon_rollout | 0.35 | 0.35 | 0.555 | 0.636 |
| v0 | zero | 0.00 | 0.00 | -0.676 | -0.024 |
| v0 | random | 0.00 | 0.00 | -0.378 | 0.055 |
| v0 | uniform | 0.30 | 0.30 | 0.817 | 0.880 |
| v0 | capacity_push | 0.80 | 0.80 | 1.208 | 1.410 |
| v0 | backlog_priority | 0.00 | 0.00 | -0.213 | 0.119 |
| v0 | earliest_deadline | 0.00 | 0.00 | -0.112 | 0.159 |
| v0 | maintenance_aware | 0.00 | 0.00 | -0.166 | 0.170 |
| v0 | setup_aware | 0.00 | 0.00 | -0.342 | 0.066 |
| v0 | short_horizon_rollout | 0.00 | 0.00 | -0.111 | 0.164 |

Success-rate shape is unchanged in this 20-episode sweep. No policy that was at 0% success moved to 50%, and `capacity_push` on v0 remains in the requested 60-90% band at 80%.
