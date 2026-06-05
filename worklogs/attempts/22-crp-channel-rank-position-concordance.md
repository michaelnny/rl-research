---
id: 22
slug: crp-channel-rank-position-concordance
status: failed
sprint: 2026-06-06
verdict_in_one_line: "CRP rank-percentile degenerates to constant 1.0 on terminal-only channels, same failure as CHX/PICAV; 0.0/0.011 vs random 194.0/1.331"
side_information: [vector diagnostics, transition geometry]
nearest_prior: "17 (CHX), 16 (PICAV), 20 (LRA) — terminal-only channel collapse family"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: b5933419d4e1d1d049432854d621cc0a541be05f
---

# 22 — CRP: Channel Rank-Position Concordance

## One-sentence idea

For each (state-cluster, action, channel), track the within-trajectory rank-percentile
of the channel's firing magnitude (restricted to steps where the channel fires), and
nudge policy logits toward actions whose trend-corrected rank vector is
Pareto-non-dominated across channels.

## Core primitive

`R[s, a, m]` = running mean of within-trajectory rank-percentile of channel m's
firing magnitude at action a in cluster s, computed over the channel's own firing-set
`S_m(τ) = {t : v_t[m] ≠ 0}` per trajectory τ. The rank-percentile `ρ_t[m]` is
unitless and magnitude-invariant: scaling channel m by any constant leaves R unchanged.
Two actions with identical successor features (same total channel mass) can have
orthogonal R tensors if one fires the channel early-ranked vs late-ranked.

## Improvement operator

At each gradient step, compute channel trend vector `σ ∈ {-1,+1}^k` from
buffer-wide per-channel mean magnitude. Form `R̃[s,a,m] = R[s,a,m]` if `σ_m = +1`
else `1 - R[s,a,m]`. Mark action `a` preferred at cluster `s` iff its R̃ row is
Pareto-non-dominated by every other visited action at that cluster. Apply fixed-α
logit nudge toward all preferred actions. No critic, no Bellman backup, no scalar
advantage.

## Why it looked promising

- Rank-percentile is formally distinct from magnitude integrals (GVFs/SF): two
  trajectories with identical successor features can disagree on every R cell.
- Unitless and channel-scale-invariant by construction — avoids magnitude-based
  collapse that killed CHX (#17).
- Pareto comparison law treats all k channels independently, no scalarization.
- Fires from a single completed trajectory (no cross-trajectory hash collisions
  required), claimed to escape the FED/CEC bootstrap wall.
- Hypothesis included a well-specified falsifier (a) for degenerate rank distributions.

## What was tested

Stage: vector (deep-sea-treasure-concave-v0, resource-gathering-v0), 120s budget.
DST: score = 0.0 vs random = 194.0, strong = 285.0.
RG: score = 0.011 vs random = 1.331, strong = 1.331.
beat_random = 0, beat_strong = 0.
Commit: b5933419d4e1d1d049432854d621cc0a541be05f. No retries.

## Why it failed

On both panel vector envs, the primary reward/signal channel fires only at the
terminal step. Within a single trajectory, a channel that fires exactly once has
rank-percentile = 1.0 (trivially, rank-of-1-within-1-element = 1.0). This makes
every R[s,a,m] cell = 1.0 for that channel across all actions and clusters — no
Pareto comparison can distinguish actions. The magnitude-invariance claim of CRP
is a non-advantage when the firing-set is a singleton: rank carries no temporal
position information when there is only one firing event. This is precisely the
hypothesis's own falsifier (a) ("more than 80% of cells degenerate to constant
rank-percentile"), which was confirmed by the scores. Extends the cross-attempt
ruling from CHX (#17) and PICAV (#16) to rank-based within-trajectory primitives:
the fundamental problem is not magnitude vs rank, but that terminal-only channels
provide exactly one data point per trajectory regardless of which statistic is
computed on it.

## Lesson / constraint added

Any within-trajectory rank-position or temporal-position statistic on a channel
degenerates to a constant when that channel fires only once per episode — ruling
out the entire family of "per-channel within-trajectory rank/order statistics" on
the substrate's terminal-only vector envs; a future candidate must use a channel
that produces intra-episode variation, or must not rely on within-trajectory
ordering as its distinguishing primitive.

## Nearest neighbors in the literature

- GVFs / successor features: magnitude integrals; CRP is rank-percentile, but
  both are equally degenerate on terminal-only channels.
- CHX (#17): hull-extremality of within-trajectory cumulant trace — also collapses
  on terminal-only channels (same family).
- PICAV (#16): signed cross-channel temporal-ordering moments — terminal-only
  collapse (same family).
- Ranking-based policy gradient (e.g. PGPE rank-based): uses episode-level return
  rank, not per-channel within-trajectory rank; different level of analysis.

## Artifacts

- `worklogs/runs/20260606-07-auto/train.py`
- `worklogs/runs/20260606-07-auto/result.json`
- `worklogs/runs/20260606-07-auto/panel.txt`
- `worklogs/runs/20260606-07-auto/hypothesis.md`
- `worklogs/runs/20260606-07-auto/review.md`
