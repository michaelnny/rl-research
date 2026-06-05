---
id: 37
slug: hrc-horizon-recursive-concordance
status: failed
sprint: 2026-06-06
verdict_in_one_line: "cluster-revisitation bottleneck silenced the K-vector; product-with-floor fallback is additive zero, not an alternative signal; below-random on all envs"
side_information: [transition geometry, vector diagnostics]
nearest_prior: "26-trac, 32-csd, 24-pcr"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: b963ec6c3e93728e86eb579c73f64bf12dd5aa49
---

# 37 — HRC: Horizon-Recursive Concordance

## One-sentence idea

Maintain a per-(state-cluster, action) paired statistic of snapshot-policy argmax self-concordance at L exponentially-spaced horizons (K-vector) and sign-conditional channel propensities with a variance gate (P-vector); nudge logits by the product of the two Pareto-non-dominance margins with a floor-1 channel multiplier so the horizon side fires even when channels are silent.

## Core primitive

`(K[s,a,:] ∈ [0,1]^L, P[s,a,:] ∈ [-1,1]^k)` where `K[s,a,l]` is the empirical fraction of buffer transitions `(o_t = s, a_t = a)` for which the current snapshot policy's argmax at `o_{t+h_l}` matches the actually-taken `a_{t+h_l}` (horizons `h_l ∈ {1,4,16,64,256}`), and `P[s,a,m]` is the difference between empirical fractions of trajectories whose terminal channel-m cumulant exceeds vs. falls below the running per-channel median — hard-clipped to zero when within-channel cumulant variance is below a self-tuned threshold.

## Improvement operator

At each decision step compute Pareto-non-dominance margins `Δ_K` and `Δ_P` over the action set independently on the K-vector and P-vector; nudge logits by `α · Δ_K · max(Δ_P, 1)`. The floor-1 on the channel side means the operator degrades to horizon-concordance-only when channels are silent, rather than to zero. Logit update is additive; no Bellman backup, no critic, no scalar reward weight.

## Why it looked promising

- Genuine structural gap from PCR (#24): the product-with-floor composition was supposed to eliminate the AND-gate that killed PCR by letting the horizon side fire unilaterally.
- The K-vector is reward-independent in a structurally stronger sense than PCR's recovery-latency scalar — it would fire on a zero-reward environment.
- Multi-scale concordance (L horizons) was intended to expose short-vs-long disagreement structure that PCR's 1-D scalar collapsed.
- The floor-1 multiplier was designed as an explicit no-AND-gate fallback, not as a zero-operation.
- Reviewer rated `novel-direction` with detailed structural-distinction analysis confirming the mechanism is not a rebadge.

## What was tested

Panel stage: `core` (4 envs: DoorKey-8x8, KeyCorridor-S3R3, DST, RG), 120 s budget per env. Commit `b963ec6c3e93728e86eb579c73f64bf12dd5aa49`. No retries.

Scores vs. random / strong:
- DoorKey-8x8: 0.0 vs 0.137 / 0.137 — below random
- KeyCorridor-S3R3: 0.0 vs 0.0 / 0.0 — neutral (all at floor)
- DST: 99.0 vs 194.0 / 285.0 — below random (nearby-treasure preference)
- RG: 0.011 vs 1.331 / 1.331 — far below random

beat_random: 0, beat_strong: 0.

## Why it failed

The cluster-revisitation bottleneck (predicted failure mode c in the hypothesis) was the binding constraint. The K-vector cells `K[s,a,:]` require the same (cluster, action) pair to be visited multiple times — identical requirement to TRAC (#26) and CSD (#32). Under uniform exploration on long-horizon sparse envs within 120 s, cluster-action cells do not accumulate sufficient mass. When K cells are sparse, Pareto margins `Δ_K ≈ 0` for all actions, and the product `Δ_K · max(Δ_P, 1)` is near-zero regardless of the floor-1 safeguard (because `0 · 1 = 0`). The floor-1 fallback degrades to an additive zero nudge on the logits — a no-op, not an alternative signal source.

On DST the below-random score (99 vs random 194) matches the ATP/PRAR failure: when K cells are uniformly sparse, whatever residual gradient exists tends to encode shortest-path-to-any-terminal state (nearby low-value treasure), actively harming performance.

The P-vector bootstrap collapse (predicted mode c+d) also fired: on DST/RG the reward channel has near-zero cross-trajectory variance until after the first rewarded trajectory, so P cells are degenerate during the bootstrap window, contributing only step-penalty signal (k_eff = 1 on the channel side).

## Lesson / constraint added

The product-with-floor composition does not rescue a per-(cluster, action) primitive from the cluster-revisitation bottleneck; the floor fallback yields zero logit nudge, not an alternative firing path. Any candidate whose main primitive is indexed by (cluster, action) pairs must include an explicit cell-seeding or revisitation-guaranteeing mechanism that fires before the main operator is expected to discriminate. Rules out the "multi-scale snapshot-concordance + channel-propensity product-of-Pareto-margins" family without such a mechanism.

## Nearest neighbors in the literature

- **TRAC (#26)** / **CSD (#32)**: same cluster-revisitation bottleneck, different primitive (JSD / TV over successor cluster distributions vs. snapshot argmax concordance fractions).
- **PCR (#24)**: same snapshot-policy self-concordance idea, 1-D scalar instead of L-vector, AND-gate operator instead of product-with-floor.
- **Ensemble-curiosity / disagreement methods** (e.g., DISAGREEMENT, EMaQ): use disagreement between policy snapshots or ensemble members to drive exploration; the K-vector is structurally close to snapshot-policy disagreement, which does not provide reward-correlated signal without explicit reward coupling.

## Artifacts

- `worklogs/runs/20260606-29-auto/train.py`
- `worklogs/runs/20260606-29-auto/hypothesis.md`
- `worklogs/runs/20260606-29-auto/review.md`
- `worklogs/runs/20260606-29-auto/result.json`
- `worklogs/runs/20260606-29-auto/panel.txt`
- commit `b963ec6c3e93728e86eb579c73f64bf12dd5aa49`
