---
id: 06
slug: t-ctbp
status: failed
sprint: 2026-05-26
verdict_in_one_line: "Strong DeepSea / slip / vector-toy results, but the math is a mechanism stack of 6+ components rather than a single primitive."
side_information: [event traces, transition geometry]
nearest_prior: "05"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 06 — T-CTBP (Transported Causal/Event-Transform Boundary Projection)

## One-sentence idea

Learn local event-transform separators and transport them across context
classes so that useful local action effects generalize to deeper / unseen
states.

## Core primitive

A separator over an event lens `e` and a transport class `ρ(h)=u`:

\[
\Lambda_u(a,b) = 1
\iff
\mathbb E[\Delta e \mid \rho(h)=u, a] \succ
\mathbb E[\Delta e \mid \rho(h)=u, b].
\]

## Improvement operator

In every context assigned to the same transport class `u`, increase log-
odds of `a` over `b`:

\[
\log\frac{\pi_{k+1}(a|h)}{\pi_{k+1}(b|h)}
\ge
\log\frac{\pi_k(a|h)}{\pi_k(b|h)} + \eta.
\]

## Why it looked promising

- Strong DeepSea scaling.
- Slip-resistant on stochastic-transition diagnostics.
- Native vector consumption via the event-lens choice.

## What was tested

DeepSea, alternating-sea, vector toy. All passed.

## Why it failed

The math is a mechanism stack:

```
event transforms
+ transport classes
+ support gates
+ vector cones
+ local logits
+ transported logits
```

That is six named components without a single composition law that
stitches them together. This is the canonical example of cross-attempt
failure mode "the mechanism is a stack of named components, not a
primitive." A real candidate should be one mathematical object plus one
composition law, like value plus Bellman backup.

## Lesson / constraint added

A candidate's primitive and improvement operator must fit on one page.
More than 3 named components without a single composition law = stack,
not primitive.

## Nearest neighbors in the literature

- Transfer learning across context classes; transportable causal
  structure (Pearl & Bareinboim).
- Successor-feature transfer (Barreto et al. 2017).
- Hierarchical transport / domain randomization.

## Artifacts

_n/a_ — prototype only in pre-substrate notebooks.
