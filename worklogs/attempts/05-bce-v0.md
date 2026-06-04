---
id: 05
slug: bce-v0
status: failed
sprint: 2026-05-26
verdict_in_one_line: "Branch certificates over local successor support and outcome cones; ablation showed all the lift came from successor novelty — collapses to count-based exploration."
side_information: [transition geometry, vector diagnostics]
nearest_prior: "count-based exploration with renamed counts"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 05 — BCE-v0 (Branch-Certificate Editing)

## One-sentence idea

Learn local branch certificates from trajectory experience and use them
to expand promising/controllable branches while preserving nondominated
vector outcomes.

## Core primitive

A **branch certificate** at history `h`, action `a`:

\[
B_{h,a} = (\mu_{h,a},\, \kappa_{h,a},\, E_{h,a},\, C_{h,a}),
\]

where `μ` is empirical successor support, `κ` is controllability /
concentration, `E` is a branch-expansion score, and `C` is the
nondominated downstream vector-outcome cone.

## Improvement operator

Maximum-entropy behavior projection that increases probability of
controllable / expanding branches and preserves nondominated outcome
cones.

## Why it looked promising

- Native vector consumption via the outcome cone `C`.
- Branch-local: cheap statistics over rollouts.
- Combines exploration (`E`, `κ`) with vector-aware exploitation (`C`).

## What was tested

DeepSea-style cases. Some passed. Ablation isolated the contribution of
each cone.

## Why it failed

Ablation showed reward discovery came almost entirely from `E` —
branch expansion / successor novelty. The outcome cone `C` helped *after*
discovery but did not *solve* discovery. The discovery mechanism
collapses to count-based exploration / episodic novelty / Go-Explore
frontier logic. This hits the disqualifier-family rule for "count-based
exploration with renamed counts" and "Go-Explore with renamed cells."

## Lesson / constraint added

If the discovery mechanism is "expand successor support" or "visit less-
seen branches," the candidate is a count-based / novelty rebadge. The
mechanism that produces *new informative trajectories* must not be a
counting head under a different name.

## Nearest neighbors in the literature

- Count-based exploration (Bellemare et al. 2016, Tang et al. 2017).
- Episodic novelty / EC-style exploration.
- Go-Explore (Ecoffet et al. 2019).
- RND (Burda et al. 2018) — random-network distillation as a novelty
  bonus.

## Artifacts

_n/a_ — prototype exists in pre-substrate notebooks; not committed to this
repo.
