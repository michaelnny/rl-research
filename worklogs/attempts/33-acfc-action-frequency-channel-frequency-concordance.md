---
id: 33
slug: acfc-action-frequency-channel-frequency-concordance
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Cross-episode sign-concordance C[a,m] silently degenerates to step-penalty-only before first reward; bootstrap wall confirmed."
side_information: [vector diagnostics, transition geometry]
nearest_prior: "24-pcr / bootstrap-wall family"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 9946a31bdd3227928b13499c771249e126dabf1e
---

# 33 — ACFC: Action-Frequency / Channel-Frequency Concordance

## One-sentence idea

Maintain a cross-episode sign-concordance matrix C[a,m] by correlating action-frequency differences with channel-firing-count differences across trajectory pairs, then bias policy logits by the Pareto-non-dominance count over C rows — no state hash, no clustering, no critic.

## Core primitive

`C[a, m] = mean_{i<j in buffer} sign(f^i_a − f^j_a) · sign(g^i_m − g^j_m)` where `f^i_a` is the normalized frequency of action `a` in episode `i` and `g^i_m` is the total channel-m firing count in episode `i`. A ring buffer of B=64 completed episodes is maintained; C is recomputed after each episode batch. The matrix lies in [-1,1]^{|A|×k} and indexes cross-episode co-variation between the action profile and the channel-firing profile of the whole trajectory — no per-state bucketing, no observation hash, no reward label.

## Improvement operator

At every decision step, add state-independent logit bias `b[a] = α · (n_a^{dom} − n_a^{sub})` to policy logits, where `n_a^{dom}` counts actions `a'` whose concordance row `C[a',:]` is coordinate-wise Pareto-dominated by `C[a,:]` and `n_a^{sub}` counts the reverse. The policy network is trained via supervised imitation on `softmax(ℓ_θ(o_t) + sg(b))` — cross-entropy on collected steps, no reward weighting, no Bellman, no critic.

## Why it looked promising

- Drops the state-hash index entirely, removing the hash-collision bootstrap wall that killed DPC/FED/CEC/TPP/PCR.
- Whole-episode frequency aggregation concentrates at long horizon (longer episodes yield better frequency estimates).
- Claims to provide non-trivial signal from episode 1 because step-penalty channel fires on every step and contributes sign differences.
- Pareto-non-dominance over C rows is genuine multi-channel structure — never collapses to scalar without all channels being collinear.
- Structural distinction from DPC is genuine: index is action-frequency side (not state-hash), sign is channel-firing-count (not terminal cumulant).

## What was tested

Stage: core (4 envs), 120 s/env budget, 0 retries. Commit 9946a31bdd3227928b13499c771249e126dabf1e.

| Env | Score | Random | Strong | Beat random | Beat strong |
|-----|-------|--------|--------|-------------|-------------|
| MiniGrid-DoorKey-8x8-v0 | 0.0 | 0.137 | 0.137 | No | No |
| MiniGrid-KeyCorridorS3R3-v0 | 0.0 | 0.0 | 0.0 | No | No |
| deep-sea-treasure-concave-v0 | 99.0 | 194.0 | 285.0 | No | No |
| resource-gathering-v0 | 0.011 | 1.331 | 1.331 | No | No |

Zero envs beat random, zero beat strong. Below or at random on all four envs.

## Why it failed

The "dense from episode 1" claim holds only for the step-penalty channel. On Deep Sea Treasure and Resource Gathering, goal/treasure/gem channels fire zero times in failed episodes; before any rewarded trajectory is collected, g^i_m = 0 for all i for those channels. This means C[a, goal-channels] = 0 for all actions throughout the bootstrap window, and C[a, step-penalty] is the only non-zero column. The Pareto-non-dominance operator over a rank-1 (effectively scalar) concordance matrix reduces to scalar step-penalty minimization: prefer actions that are associated with shorter episodes. On sparse envs this is a fastest-termination preference — the hypothesis's own failure mode (a) confirmed. On DoorKey and KeyCorridor, the identical-frequency degeneracy (failure mode b) also applies: a near-random policy produces similar action frequencies across all episodes, making Δfreq_a ≈ 0 for every pair and C ≈ 0; the operator is silent.

This is the same bootstrap-wall pattern as PCR (#24): a reward-independent primitive (C is well-defined from step 1) paired with an operator whose useful multi-channel structure requires at least one rewarded trajectory to populate the goal-channel columns of C. Before that, the operator behaves either as step-penalty scalar minimization (harmful on DST, confirmed by 99.0 vs 194.0 random) or as zero bias (silent on DoorKey/KeyCorridor).

Applicable cross-attempt failure modes: "The primitive needs reward correlation to bootstrap."

## Lesson / constraint added

Eliminating the state-hash index does not cure the pre-reward bootstrap collapse: any concordance or co-variation primitive over k-dimensional channel counts degenerates to the step-penalty channel alone before the first rewarded trajectory, because terminal-only reward channels contribute zero count differences. Future candidates must either (a) use a primitive that is structurally informative on the step-penalty channel only in a way that is not a step-penalty rebadge, or (b) use an explicit exploration primitive that seeds the buffer with rewarded trajectories before the concordance operator is applied.

## Nearest neighbors in the literature

- Rank correlation / Kendall's τ between action profiles and outcome profiles (macro-level behavioral science).
- Multi-objective evolutionary strategy: Pareto front in fitness-landscape over action archetypes.
- DPC (divergent-prefix concordance, attempt on this substrate): same sign-concordance structure but indexed by state-of-first-divergence.
- KTAC (cluster-conditioned trajectory pair rankings): per-cluster pairwise channel rankings — ACFC is unclustered version with frequency-domain index.

## Artifacts

- `worklogs/runs/20260606-23-auto/train.py`
- `worklogs/runs/20260606-23-auto/result.json`
- `worklogs/runs/20260606-23-auto/hypothesis.md`
- `worklogs/runs/20260606-23-auto/review.md`
- `worklogs/runs/20260606-23-auto/curator.md`
