---
id: 13
slug: etb-hpc
status: failed
sprint: 2026-05-29
verdict_in_one_line: "ETB worked on custom prerequisite tasks but is goal-conditioned hindsight + event-options; HPC improved transfer but is supervised hindsight policy compression / GCSL-like imitation."
side_information: [event traces, object state]
nearest_prior: "Hindsight Experience Replay / options-and-hierarchical-RL"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 13 — ETB / HPC (Behavioral Coordinate Discovery)

## One-sentence idea

Learn reusable behavior coordinates from ordinary rollouts instead of
hand-designing them. Two complementary primitives:
**ETB** (Event-Time Behavioral Basis) and
**HPC** (Hindsight Policy Compression).

## Core primitive

**ETB.** For an event `g` and a context `c`:

\[
B_k(g, c) = (a^*_{g,c},\, d^*_{g,c})
\]

where `a*` is the first action on the shortest observed suffix from
context `c` to event `g`, and `d*` is the observed first-hit depth.

**HPC.** Compress event-reaching suffixes into a conditional policy
program by minimum-description-length (MDL) compression:

\[
C_k = \operatorname{MDLCompress}\{ (\phi(h_t), g, a_t, T_g - t) \}.
\]

## Improvement operator

ETB: increase the probability of `a*_{g,c}` in context `c` when the agent
intends event `g`. HPC: distill the compressed program back into the
policy.

## Why it looked promising

- Replaces the hand-event-lens of #06 / #11 with learned coordinates.
- Composes naturally via the `(g, c)` index.
- Native consumption of trajectory data.

## What was tested

Custom prerequisite tasks: ETB worked. Transfer experiments: HPC improved
transfer somewhat.

## Why it failed

Audit against existing families:

- **ETB** is structurally goal-conditioned hindsight + event-options. The
  `(g, c) → a*` table is an option index; the first-action-on-shortest-
  suffix rule is hindsight goal relabeling.
- **HPC** is supervised hindsight policy compression — GCSL-like
  imitation under another name.

These may be useful **components** (a torch network, a replay buffer, a
sequence model — per the disqualifier-families "components vs
explanation" rule), but neither is a new foundational RL primitive.

## Lesson / constraint added

A method that learns event coordinates from ordinary rollouts is at
strong risk of being a hindsight-relabeling / options rebadge. The next
candidate must show a coordinate-discovery principle that is not
hindsight + options under another name.

## Nearest neighbors in the literature

- Hindsight Experience Replay (Andrychowicz et al. 2017).
- Goal-conditioned supervised learning / GCSL (Ghosh et al. 2020).
- Options framework (Sutton, Precup, Singh 1999) and modern option
  discovery (Bagaria et al., Eysenbach et al.).
- HER-conditioned policies; goal-conditioned RL writ large.

## Artifacts

_n/a — explored offline; not implemented in this substrate._
