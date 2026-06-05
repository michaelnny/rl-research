---
id: 32
slug: csd-channel-conditional-successor-disagreement
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Cell-revisitation bottleneck (not per-cell sample-size) killed the TV-over-conditional-next-cluster primitive — same failure as TRAC (#26)."
side_information: [vector diagnostics, transition geometry]
nearest_prior: "26"
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 4090695072eba7a987e2ad179d7b73e2e68b9cee
---

# 32 — CSD: Channel-Conditional Successor Disagreement

## One-sentence idea

For each (cluster, action, channel) triple, measure the total-variation distance between the empirical next-cluster distribution on trajectories where channel m fired in the K-step post-action window vs. those where it did not; nudge policy logits toward Pareto-non-dominated actions in the k-dimensional TV row.

## Core primitive

`S[c,a,m] = TV(p(c'|c,a, m fires in [t+1..t+K]), p(c'|c,a, m does not fire in [t+1..t+K]))` where `p(c'|c,a,·)` are Laplace-smoothed empirical distributions over next-cluster identities. The primitive is a (|C|×|A|×k) tensor computed by periodic buffer scan; TV is bounded in [0,1] and well-defined with one sample per side.

## Improvement operator

At cluster c, compute `n_a = #{a' ≠ a : S[c,a,:] ≻ S[c,a',:]}` and `m_a = #{a' ≠ a : S[c,a',:] ≻ S[c,a,:]}` in the strict coordinate-wise partial order on R_+^k. Logit update: `Δlogit[c,a] = α(n_a − m_a)`. When all actions are mutually incomparable the nudge is zero.

## Why it looked promising

- TV-vs-JSD: Laplace-smoothed TV is well-defined with one sample per side; plug-in JSD is undefined there. This directly targeted TRAC's stated small-sample failure.
- K-step window: relaxed TRAC's immediate post-action window requirement; a channel firing anywhere in K steps contributes, lowering the effective firing-rate threshold.
- Step-penalty channel seeds the operator from episode 1 (unlike terminal-only channels), so the primitive is non-degenerate before first reward.
- Clean single-primitive / single-operator structure with no critic, no Bellman, no scalar collapse.
- Structural distinction from CWAI (alive): empirical distribution vs. Jacobian column-norm; TV bounded in [0,1] vs. gradient norm that shrinks under stochasticity.

## What was tested

Stage: core (MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0, deep-sea-treasure-concave-v0, resource-gathering-v0). Time budget: 120 s/env. Results: 0.0 / 0.0 / 0.0 / 0.011 vs random 0.137 / 0.0 / 194.0 / 1.331. beat_random=0, beat_strong=0. Wallclock: 115.5 s. Commit: 4090695072eba7a987e2ad179d7b73e2e68b9cee.

## Why it failed

Same (cluster, action) cell-revisitation bottleneck as TRAC (#26). TRAC's ruling named the failure as "revisitation frequency of (cluster, action) pairs under uniform exploration on long-horizon sparse envs" — not the sample-size-per-cell once the cell is revisited. CSD's improvements (TV vs JSD, K-step window) reduce the number of samples needed per cell once a pair is revisited but do not change how often the pair is revisited. On DoorKey-8x8 and KeyCorridor the agent navigates a large state space; the same (cluster, action) pair is rarely encountered twice in 120 s, so the conditional distributions have near-degenerate support and the TV primitive stays near-zero throughout. Score pattern is identical to TRAC.

## Lesson / constraint added

Reducing the per-cell sample-size requirement is insufficient for the cluster-indexed conditional-distribution family. The next candidate in this family must either abandon the cluster-indexed cell structure (so the primitive aggregates across all states, not per cell) or pair with an explicit exploration primitive that actively increases (cluster, action) revisitation frequency before the TV comparison is trusted.

## Nearest neighbors in the literature

- **TRAC (#26)** — JSD over post-action successor-cluster histograms partitioned by channel firing; same Pareto-non-dominance operator; same revisitation bottleneck.
- **Successor features / GVFs** — compute expected cumulants conditioned on state; CSD computes a distributional divergence conditioned on next-cluster identity, not an expectation.
- **Conditional mutual information probes** — CSD's TV lower-bounds `I(C_{t+1}; F_m | c, a)` (Pinsker-type); literature probes use labels and gradient-based estimation rather than empirical histogram TV.
- **CWAI (alive-promising)** — Jacobian column-norm of a learned forward model per (action, channel); shares the "coupling between action and channel-firing" intuition but uses parameter-space gradient norms rather than empirical distributional divergence.

## Artifacts

`worklogs/runs/20260606-21-auto/train.py`, `worklogs/runs/20260606-21-auto/hypothesis.md`, `worklogs/runs/20260606-21-auto/result.json`. Commit: 4090695072eba7a987e2ad179d7b73e2e68b9cee.
