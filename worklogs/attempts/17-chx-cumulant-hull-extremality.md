---
id: 17
slug: chx-cumulant-hull-extremality
status: failed
sprint: 2026-06-05
verdict_in_one_line: "Hull-extremality weights collapse to return-to-go when vector channels are terminal-only; scored below random on both vector panel envs."
side_information: [vector diagnostics]
nearest_prior: "16-picav / return-to-go disqualifier family"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: 24ff559f735f5c555aa68377f6f5e590127c4a0e
---

# 17 — CHX (Cumulant-Hull Extremality)

## One-sentence idea

For a single rolled-out trajectory, weight each (state, action) step's log-prob update by how much removing that step's cumulant point would shrink the convex hull of the trajectory's vector-cumulant trace in `R^k` — rewarding steps that push the policy into previously-unattained outcome directions.

## Core primitive

Cumulant trace `c_t = Σ_{s≤t} v_s ∈ R^k` (where `v_t` is the per-step `info["vector"]` signal). Hull contribution: `h_t = max(0, dist(c_t, ConvHull({c_{-1}, …, c_T} \ {c_t})))` — the L2 distance by which the hull shrinks if step `t` is removed. Normalized: `\hat h_t = h_t / (Σ_t h_t + ε)`. This is a within-trajectory geometric quantity — no value, no advantage, no reward, no cross-trajectory comparison.

## Improvement operator

One policy update per trajectory: `θ ← θ + α · Σ_t (\hat h_t − 1/T) · ∇_θ log π_θ(a_t | s_t)`. Hull-exterior steps (positive `\hat h_t − 1/T`) get positive weight; interior steps get negative weight. No critic, no Bellman recursion, no scalarized reward. The centered weight `(\hat h_t − 1/T)` has zero mean within the trajectory.

## Why it looked promising

- The hull-contribution weight is provably non-scalarizable: no fixed `w` makes `w^T v_t` reproduce convex-hull membership, ruling out the "scalarized vector-reward" disqualifier family.
- Structurally distinct from VCC (no cross-trajectory Pareto/buckets) and FED (no obs-hash buckets, no bootstrap threshold).
- Within-trajectory computation: `O(T log T)` via qhull, no replay, no donor trajectories.
- Monotonic improvement claim is well-stated: under the condition that `E[h_t | a_t]` is non-uniform across actions, the update increases expected per-trajectory hull-volume in `R^k`.
- Hypothesis explicitly identified the k_eff=1 self-disqualification condition.

## What was tested

Stage: `vector` (Deep Sea Treasure concave, Resource Gathering). 120s/env, 2 workers. Run ID: 20260605-08-auto. Commit: 24ff559f735f5c555aa68377f6f5e590127c4a0e.

Results:
- deep-sea-treasure-concave-v0: score=99.0, random=194.0, strong=285.0 — BELOW RANDOM
- resource-gathering-v0: score=0.011, random=1.331, strong=1.331 — BELOW RANDOM
- beat_random=0, beat_strong=0

## Why it failed

The hypothesis itself identified the degenerate case: "on envs with k_eff=1 vector signal, the hull degenerates to a 1-D segment whose extremes are just the min and max steps — this IS a return-to-go rebadge." Both panel vector envs triggered this condition. Deep Sea Treasure's treasure channel fires only at the terminal step; Resource Gathering's reward channel is similarly sparse/terminal. The cumulant trace is a near-line in the step-penalty dimension throughout each episode, with the reward dimension contributing only the final point. Under these conditions, the hull's extreme points are the trajectory start (c_{-1}=0) and end (c_T), and hull contribution `h_t` correlates with episode position / accumulated reward rather than action-specific vector-outcome novelty. CHX thus collapses to a return-to-go variant — exactly the self-disqualifying scenario the hypothesis warned about.

This is the same terminal-only vector channel collapse as PICAV (#16): both primitives require dense per-step vector signal to work as intended; the substrate's vector envs do not provide it.

## Lesson / constraint added

Any within-trajectory signal-geometry primitive (hull extremality, Pareto moment, cross-channel asymmetry, trace curvature) requires the per-step `info["vector"]` to have non-degenerate multi-channel variance *throughout* each episode. The panel's vector envs violate this because primary value channels are terminal-only. Future candidates in this family must either (a) target environments with dense per-step vector feedback, or (b) pair the geometry primitive with an exploration mechanism that generates dense intra-episode vector signal before the update fires.

## Nearest neighbors in the literature

- **Return-weighted REINFORCE**: CHX reduces to this when k_eff=1; the hull's extreme point on a 1-D trace is just the max-return step.
- **REINFORCE with per-step baseline**: The centered weight `(\hat h_t − 1/T)` is structurally analogous to a within-trajectory baseline, but CHX's baseline arises from hull geometry rather than value estimation.
- **Convex hull multi-objective optimization** (e.g., Convex Coverage Set methods): CHX adapts the CHCS concept from policy space to within-trajectory outcome space, but applies it as a policy-gradient weight rather than a set-covering objective.
- **PICAV (#16)**: Closest sibling — same terminal-only collapse failure; different primitive (cross-channel antisymmetric moments vs hull geometry).

## Artifacts

- Hypothesis: `worklogs/runs/20260605-08-auto/hypothesis.md`
- Review: `worklogs/runs/20260605-08-auto/review.md`
- Result: `worklogs/runs/20260605-08-auto/result.json`
- Curator: `worklogs/runs/20260605-08-auto/curator.md`
- Commit: 24ff559f735f5c555aa68377f6f5e590127c4a0e
