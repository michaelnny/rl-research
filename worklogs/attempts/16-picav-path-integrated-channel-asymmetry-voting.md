---
id: 16
slug: picav-path-integrated-channel-asymmetry-voting
status: failed
sprint: 2026-06-05
verdict_in_one_line: "Signed pair-contribution vectors collapse to zero on terminal-only vector channels; same bootstrap wall as FED despite different math object."
side_information: [vector diagnostics, transition geometry]
nearest_prior: "15"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 4d57c6acd364641febdb0e53d8d007b2d179c209
---

# 16 — PICAV: Path-Integrated Channel-Asymmetry Voting

## One-sentence idea

For each (observation-hash bucket, action) pair, accumulate the signed antisymmetric pair-contribution vector `δ_{jk,t} = v_t[j]·Δn_t[k] − v_t[k]·Δn_t[j]` across trajectories, then nudge policy logits toward actions that upper-orthant Pareto-dominate the other available actions in the same bucket.

## Core primitive

For a trajectory with per-step vector feedback `v_t ∈ ℝ^k` and running future-cumulant `n_t[k] = Σ_{s>t} v_s[k]`, the per-step pair-contribution is `δ_{jk,t} = v_t[j]·(n_{t-1}[k] − n_t[k]) − v_t[k]·(n_{t-1}[j] − n_t[j])`. This is antisymmetric in (j,k) and measures whether channel j tended to fire before channel k along the trajectory. The primitive is the per-(obs-hash-bucket b, action a) running mean vector `μ(b,a) ∈ ℝ^{k(k-1)/2}` over all trajectory visits.

## Improvement operator

For each obs-hash bucket b with multiple actions sampled, compute the upper-orthant Pareto frontier F(b) = {a : μ(b,a) is not coordinate-wise ≤ μ(b,a') for any a' ≠ a}. For non-frontier action a, find the frontier action a* that coordinate-wise dominates a on the most coordinates D(a*,a); add logit-shift `α·D(a*,a)·(log π(a*|s) − log π(a|s))` to the policy loss. No reward weight, no Bellman backup, no critic.

## Why it looked promising

- Claimed to bypass FED's bootstrap wall: pair-contributions are defined and nonzero on every step where any two channels have nonzero increments — not just reward-bearing steps.
- Structurally distinct from FED: primitive is signed temporal-ordering moment over channel pairs, not extension of an attainment set of outcome cumulants.
- Explicit predicted failure modes enumerated including terminal-only channels and k=2 degenerate cases.
- Monotonic improvement claim for `J(π) = E_π[Σ_{j<k} |A_{jk}|]` was plausible.
- Reviewer verdict: `novel-direction` — mathematical object not found in prior attempts or disqualifier families.

## What was tested

Stage: `vector`, envs: `deep-sea-treasure-concave-v0` and `resource-gathering-v0`, time budget 120s/env. No retries. Run ID: 20260605-06-auto, commit: 4d57c6acd364641febdb0e53d8d007b2d179c209.

Results: DST score 0.0 vs random 194.0 / strong 285.0; Resource Gathering score 0.011 vs random 1.331 / strong 1.331. beat_random=0, beat_strong=0.

## Why it failed

The hypothesis correctly identified the risk: on Deep Sea Treasure, the treasure channel fires only at the terminal step. Therefore `Δn_t[treasure] = 0` on every non-terminal step, making `δ_{j,treasure,t} = 0` for the entire episode except the final step. The pair-contribution vectors are flat throughout — the claimed bypass of FED's bootstrap wall does not hold when one or more vector channels is terminal-only. The primitive reduces to a near-single-step indicator on DST, producing the same zero-signal outcome as FED.

On Resource Gathering the score of 0.011 (vs random 1.331) suggests the obs-hash bucket mechanism also failed to accumulate sufficient sample mass within the 120s budget to generate useful orthant-dominance signal — possibly due to bucket collision or inadequate warm-up.

Cross-attempt failure mode: "The primitive needs reward correlation to bootstrap, but reward correlation does not exist on long-horizon sparse tasks until a deep unrewarded path is traversed" applies here in a new form: the pair-contribution primitive needs *non-terminal per-step increments on all channels* to fire, which is a structural prerequisite that DST's terminal treasure channel violates.

## Lesson / constraint added

Any primitive that computes cross-channel temporal-ordering moments (or pair-contributions) requires that all channels in the pair carry non-terminal per-step increments — on envs where any channel is terminal-only, the pair-contribution collapses to zero throughout the episode, reinstating the bootstrap wall.

## Nearest neighbors in the literature

- **FED (#15):** same bootstrap failure mode on terminal-reward envs despite different mathematical object (attainment-set extension vs. signed pair-ordering moment).
- **GVFs / successor features:** PICAV does not perform Bellman recursion on cumulants; however the per-(state, action) cumulant accumulation is adjacent to multi-objective successor features.
- **TOP (#11):** TOP uses per-channel first-hit time profiles; PICAV uses antisymmetric pairwise temporal integrals — different objects, same env failure surface (terminal-only channels).
- **Antisymmetric / skew-symmetric tensor methods in MORL:** cross-channel covariance structure is explored in some MORL literature, but the specific bootstrap-from-trajectory accumulation scheme is not a named method.

## Artifacts

- `worklogs/runs/20260605-06-auto/hypothesis.md`
- `worklogs/runs/20260605-06-auto/review.md`
- `worklogs/runs/20260605-06-auto/result.json`
- `worklogs/runs/20260605-06-auto/train.py`
- `worklogs/runs/20260605-06-auto/panel.txt`
- Commit: 4d57c6acd364641febdb0e53d8d007b2d179c209
