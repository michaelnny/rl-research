---
id: 26
slug: trac-transition-refractive-action-channels
status: failed
sprint: 2026-06-06
verdict_in_one_line: "JSD between channel-partitioned successor histograms; silent on all 4 core envs due to cluster-revisitation bootstrap wall"
side_information: [vector diagnostics, transition geometry]
nearest_prior: "15-fed-frontier-expanding-dispersion (bootstrap-wall family)"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: f9f6a683f1107eba91de9087b88834e3c19534ef
---

# 26 — TRAC (Transition-Refractive Action Channels)

## One-sentence idea

For each (state-cluster, action, channel) cell maintain two empirical histograms of successor clusters — one where channel `m` fired in a post-action window, one where it did not — and use the Jensen-Shannon divergence between them as a partial-order signal to nudge policy logits without ever scalarizing vector feedback.

## Core primitive

`R[c, a, m] = JSD(H_fire[c,a,m] || H_nofire[c,a,m])` where `H_fire[c,a,m]` collects successor-cluster `c'` values from transitions where channel `m` fired anywhere in the window `[t+1, t+W]` after taking action `a` at cluster `c`, and `H_nofire[c,a,m]` collects `c'` values where it did not. The result is a non-negative `k`-vector per (c, a) cell — a divergence between two conditional empirical distributions over discrete future state, never a discounted expectation, never a TD-bootstrapped target.

## Improvement operator

At decision step with cluster `c`, for each candidate action `a` compute the row `R[c, a, :] ∈ R^k`. Compute `n_a = Σ_{a'≠a} 1[R[c,a,:] ≻ R[c,a',:]]` (strict Pareto-dominance count) and `m_a` (actions that dominate `a`). Logit update: `logit(a | c) ← logit(a | c) + α(n_a − m_a)`. No scalar reward appears; no critic; no scalarization of vector channels.

## Why it looked promising

- JSD fires on any channel-firing event within the window, not only on terminal reward — argued to avoid the terminal-reward-only bootstrap collapse that killed FED/CEC/PICAV/CHX.
- Step-penalty channels fire on every step, so H_fire and H_nofire were predicted to accumulate non-empty support from trajectory 1.
- Structurally distinct from successor features/GVFs (no discounted cumulant expectation) and from CWAI (no learned forward-model Jacobian — an empirical integral over realized transitions vs. a local differential).
- Reviewer approved as novel-direction with no rebadge concerns.
- Clean primitive + improvement operator shape with explicit monotonic improvement claim in infinite-sample limit.

## What was tested

Stage: core (MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0, deep-sea-treasure-concave-v0, resource-gathering-v0), 120 s/env, 4 workers.
Commit: f9f6a683f1107eba91de9087b88834e3c19534ef

Scores:
- DoorKey-8x8: 0.0 (random=0.137, strong=0.137)
- KeyCorridorS3R3: 0.0 (random=0.0, strong=0.0)
- DST: 98.0 (random=194.0, strong=285.0) — below random
- RG: 0.011 (random=1.331, strong=1.331) — below random

beat_random=0, beat_strong=0. Ran without crash or contract violation.

## Why it failed

The hypothesis's own falsifier (a) was confirmed: fewer than 20% of (c, a, m) cells accumulated the minimum sample mass (≥ 8 fired-vs-not-fired splits with nonempty successor support) within the 120 s budget.

The core issue: even though step-penalty channels seed H_fire from trajectory 1, the JSD primitive requires the same (cluster c, action a) pair to be revisited enough times under the current stochastic policy to fill both histogram sides. On long-horizon sparse navigation envs (DoorKey-8x8, KeyCorridor), under uniform/near-uniform exploration, the agent traverses wide regions of state space without revisiting the same cluster-action combination — the cluster-indexed cell-collision bottleneck is structurally equivalent to the observation-hash bucket coverage failure that killed FED/CEC/TPP.

The step-penalty argument (H_fire populates from step 1) does not rescue TRAC because the bottleneck is revisitation of the same (c, a) pair, not the firing frequency of a single channel across singleton visits. This is a second-order bootstrap wall: not "wait for terminal reward" but "wait for (state-cluster, action) pair revisitation under a policy that hasn't yet discovered reward."

On DST and RG, the terminal-only reward channel compounds this: H_fire for the treasure/pickup channels stays empty until the first successful trajectory, which doesn't arrive within budget.

## Lesson / constraint added

Cluster-indexed conditional-distribution primitives face the same bootstrap wall as hash-indexed primitives: any primitive requiring per-(cluster, action) histogram coverage on sparse long-horizon envs will be silent within a 120 s budget. The step-penalty-seeding argument does not help because cell-collision frequency, not channel-firing frequency, is the binding constraint. Future candidates must either pair with an explicit exploration/coverage primitive, or use a primitive that fires from singleton state visits without cross-visit aggregation.

## Nearest neighbors in the literature

- **Successor features / GVFs** (Barreto et al. 2017): predict expected discounted cumulant vector under policy; extract behavior by inner product with task-weight vector. TRAC uses a divergence between two empirical distributions, not a cumulant expectation.
- **CWAI** (alive-promising candidate): learned forward-model Jacobian column-norm. TRAC uses an empirical histogram divergence — no forward model, no gradient, no embedding.
- **Conditional mutual information estimation** (MINE, DEMI, etc.): TRAC's JSD is a consistent estimator of I(c'; fired_m | c, a) but is computed via plug-in histograms without neural approximation.
- **FED / CEC / TPP / PCR** (attempts 15, 18, 21, 24): all failed for the bootstrap-wall family reason; TRAC extends that ruling to the cluster-indexed divergence sub-family.

## Artifacts

- `worklogs/runs/20260606-13-auto/hypothesis.md`
- `worklogs/runs/20260606-13-auto/review.md`
- `worklogs/runs/20260606-13-auto/result.json`
- `worklogs/runs/20260606-13-auto/panel.txt`
- `worklogs/runs/20260606-13-auto/curator.md`
- Commit: f9f6a683f1107eba91de9087b88834e3c19534ef
