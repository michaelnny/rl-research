---
id: 36
slug: prar-policy-regime-antisymmetric-residual
status: failed
sprint: 2026-06-06
verdict_in_one_line: "k_eff collapses to 1 on terminal-only vector channels; Pareto vote becomes scalar frequency-biased comparison favoring nearby low-value treasure — same as ATP"
side_information: [vector diagnostics, policy-distribution-shape]
nearest_prior: "31 (atp-action-tangent-persistence), 17 (chx-cumulant-hull-extremality)"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: cb806fd815d815cb93acd38d4f60f3f9550a2438
---

# 36 — PRAR (Policy-Regime Antisymmetric Residual)

## One-sentence idea

Index a position-weighted within-episode firing residual by the policy's own action-distribution shape (entropy/top-prob/top-2-gap regime bins) rather than by state-hash or k-means cluster, then nudge logits via Pareto-non-dominance count over the k-vector F[r,a,:] at each decision step.

## Core primitive

`F[r, a, m] = (1/N[r,a]) Σ_E Σ_{t: regime(π_E, o_t)=r, a_t=a} w(t/T_E) · (v_t[m] − μ_E[m])`
where `μ_E[m]` is the per-episode mean of channel m, `w(s) = 2s − 1` is a fixed antisymmetric position weight, and the regime tag `regime(π, o) = (H_bin, p_top1_bin, gap_bin)` (27 regimes) is computed from the policy's own output distribution — no observation hash, no state-feature k-means. Updated as a running mean per completed episode; cell-sample gate: α=0 for N[r,a] < 5.

## Improvement operator

At each decision step, look up F[r_t, a, :] for each action a; compute Pareto-non-dominance margins D_+(a) (actions coordinate-wise dominated by a) and D_-(a) (actions dominating a); nudge raw logits by α(D_+(a) − D_-(a)). Sample from softmax(logits + α·nudge). No critic, no Bellman, no scalar collapse.

## Why it looked promising

- The regime axis is populated on every decision step regardless of what the environment returns — eliminates the observation-hash-collision bootstrap wall that killed FED/CEC/TPP/PCR/TRAC.
- The antisymmetric position weight w(s) = 2s-1 is specifically designed to assign negative weight to early steps and positive weight to late steps, meaning on terminal-only-reward envs F[r,a,terminal] encodes whether action a in regime r co-occurs with late-episode terminal firings.
- The Pareto-non-dominance count is invariant to per-channel monotone rescaling — structurally not a w^T r scalarization.
- Policy-distribution-shape as the indexing axis is genuinely novel: populated densely from the first step, no environment instrumentation required.
- The structural distinction from KTAC (k-means clusters + Kemeny rank-aggregation) and DPC (trajectory-pair divergence-state hashes + sign-votes) was correctly articulated.

## What was tested

Stage: vector (DST, RG), 120 s/env each, single run.
- deep-sea-treasure-concave-v0: score=189.0, random=194.0, strong=285.0 — below random
- resource-gathering-v0: score=0.011, random=1.331, strong=1.331 — below random
n_beat_random=0, n_beat_strong=0.
Commit: cb806fd815d815cb93acd38d4f60f3f9550a2438.

## Why it failed

The hypothesis correctly predicted failure mode (c): when the step-penalty channel fires at constant magnitude every step within an episode, `v_t[m] − μ_E[m] = 0`, zeroing that channel's contribution to F. This leaves k_eff=1: only the terminal-only reward channel survives the antisymmetric weighting.

With a single surviving channel, the Pareto-non-dominance vote degenerates to a scalar comparison on F[r,a,terminal]. On DST-style envs, exploration under a base policy visits nearby low-value treasure far more often than distant high-value treasure, accumulating higher N[r,a] counts and more stable F estimates for nearby-treasure actions. The operator steers the policy toward nearby termination — the same mechanism as ATP (#31), which preferred "shorter persistence horizon to any terminal state." The active harm (189 vs random 194) confirms the operator is working but targeting the wrong terminal states.

This is the CHX/CRP/ATP family collapse: any primitive that reduces to k_eff=1 on the substrate's terminal-only reward channels produces scalar frequency-biased comparison. The regime indexing provides no rescue: even with 27 regimes populated, each regime's F slice is still k_eff=1 and carries the same exploration-frequency bias.

## Lesson / constraint added

Any primitive that relies on antisymmetric position weighting (or hull geometry, rank position, or persistence horizons) to extract signal from terminal-only vector channels will face k_eff=1 collapse on the current substrate; the Pareto vote then reduces to scalar comparison biased by exploration frequency toward easily-reachable (low-value) terminal states. A viable candidate must maintain k_eff ≥ 2 non-degenerate channels before the first rewarded trajectory, or pair an explicit exploration primitive that ensures high-value terminal states are visited before the F/Pareto signal locks in.

## Nearest neighbors in the literature

- **ATP (#31) / Action-Tangent Persistence:** same failure — forward-model persistence horizons on terminal-only channels collapsed to fastest-termination preference; both are "scalar comparison on single surviving channel biased by exploration frequency."
- **CHX (#17):** within-trajectory hull geometry collapsed to episode start/end with k_eff=1; same root cause.
- **Successor Features / GVFs:** PRAR's F tensor is conceptually related to per-channel cumulant predictors conditioned on a regime context; the antisymmetric weighting adds a within-episode temporal structure GVFs don't have, but the collapse pattern is analogous.

## Artifacts

- Hypothesis: worklogs/runs/20260606-28-auto/hypothesis.md
- Review: worklogs/runs/20260606-28-auto/review.md
- Train script: worklogs/runs/20260606-28-auto/train.py
- Panel output: worklogs/runs/20260606-28-auto/panel.txt
- Result: worklogs/runs/20260606-28-auto/result.json
- Commit: cb806fd815d815cb93acd38d4f60f3f9550a2438
