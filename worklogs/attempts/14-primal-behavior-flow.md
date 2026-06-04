---
id: 14
slug: primal-behavior-flow
status: failed
sprint: 2026-05-29
verdict_in_one_line: "Mathematically central but does not yet expose a side-information advantage for sparse long-horizon problems; risks collapse to occupancy-measure LPs / max-ent RL / GFlowNets / mirror-descent PI."
side_information: []
nearest_prior: "Policy mirror descent / occupancy-measure LP"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 14 — Primal Behavior Flow Pivot

## One-sentence idea

Revisit RL's primal form: policies induce occupancy / behavior flows.
Maybe behavior flow, not value, should be the central object.

## Core primitive

The **occupancy-like behavior flow**:

\[
\mu_\pi(s, a) = \sum_{t \ge 0} \gamma^t P_\pi(s_t = s,\, a_t = a)
\]

with flow conservation:

\[
\sum_a \mu(s, a) = d_0(s) + \gamma \sum_{s', a'} P(s | s', a')\, \mu(s', a').
\]

## Improvement operator

Correct / improve outcome-indexed behavior flow and recover the policy by
normalization:

\[
\pi(a | s) = \frac{\mu(s, a)}{\sum_b \mu(s, b)}.
\]

## Why it looked promising

- The dual to value-centric Bellman backup; mathematically primary in the
  LP formulation of MDPs.
- Occupancy is policy-relative but composes linearly under flow
  conservation.
- Avoids scalar value head entirely.

## What was tested / why it failed

Conceptual analysis. The path was not pursued further because it does
not yet expose a side-information advantage for sparse long-horizon
problems — there is no story for *how* the flow gets discovered when
reward correlations don't exist. Without that story it collapses to
existing families:

- occupancy-measure LPs (the textbook primal form);
- max-ent RL / soft Q-learning;
- GAIL / occupancy matching (Ho & Ermon 2016);
- GFlowNets (Bengio et al. 2021);
- flow matching (Lipman et al. 2023);
- mirror-descent policy iteration.

This is a new failure mode: **abstract mathematical pivot without an
exposed side-information advantage**. The math is centerable, but
without an answer to "what informs the flow update before reward
correlation exists?" it is a notational shift.

## Lesson / constraint added

A pivot from one mathematical center (value, flow, distribution, etc.)
to another is not by itself a new RL family. The candidate must say
*what new side information* the new center makes usable, and *how* that
side information drives the discovery of new informative trajectories
before reward correlation exists.

## Nearest neighbors in the literature

- Occupancy-measure LP / linear-programming MDPs.
- Max-entropy RL (Ziebart et al. 2010, Haarnoja et al. 2018).
- GAIL / occupancy matching (Ho & Ermon 2016).
- GFlowNets (Bengio et al. 2021); flow matching (Lipman et al. 2023).
- Mirror-descent policy iteration (Geist et al. 2019).

## Artifacts

_n/a — conceptual only._
