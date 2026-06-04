---
id: 07
slug: order-projection
status: failed
sprint: 2026-05-26
verdict_in_one_line: "Elegant on DeepSea-style diagnostics; collapses on MiniGrid DoorKey because passive action correlation is not causal prerequisite structure."
side_information: [event traces]
nearest_prior: "Scalar-weighted log-prob update (REINFORCE family)"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 07 — OPP / Order Projection

## One-sentence idea

Learn a partial order over local behavior choices from trial-and-error
outcomes, then KL-project the policy onto that order.

## Core primitive

A relation `R_k ⊆ (U×A)×(U×A)` where

\[
(u, a) \succ_k (u, b)
\]

means action `a` empirically dominates action `b` in context class `u`.

## Improvement operator

Minimal KL policy change satisfying learned log-odds inequalities:

\[
\pi_{k+1} = \arg\min_\pi \mathbb E_{u\sim D_k} D_{KL}(\pi(\cdot|u)\,\|\,\pi_k(\cdot|u))
\]

subject to

\[
(u,a)\succ_k(u,b)
\Rightarrow
\log\frac{\pi_{k+1}(a|u)}{\pi_{k+1}(b|u)}
\ge
\log\frac{\pi_k(a|u)}{\pi_k(b|u)} + \eta.
\]

## Why it looked promising

- One primitive (partial order) + one operator (KL projection).
- No critic, no scalar value head.
- Empirically beat baselines on DeepSea, alternating-sea, vector toy.

## What was tested

DeepSea, alternating-sea, vector toy: looked good. **MiniGrid DoorKey**
(the official benchmark): failed.

## Why it failed

It confused passive correlation with causal improvement.

> Action `a` appeared more often in better trajectories
> ⇏ increasing `a` improves the policy.

DoorKey requires a causal prerequisite chain:
`get key → open door → reach goal`. A successful trajectory contains many
actions; some are necessary, some are incidental, some are actively bad
but compensated for later. Local action preference cannot represent the
prerequisite chain. This is the canonical example of cross-attempt mode
"passive correlation ≠ causal prerequisite structure."

## Lesson / constraint added

Passive action-order projection is dead as a central primitive. A
candidate that picks actions because they "appeared in good rollouts"
is structurally a scalar-weighted log-prob update under another name.

## Nearest neighbors in the literature

- REINFORCE / advantage-weighted log-prob (the canonical form of
  "passive correlation → policy update").
- Preference-based RL — DPO, IPO, RLHF (Christiano et al. 2017,
  Rafailov et al. 2023) — when the preference signal is passive.
- Decision Transformer reward conditioning (Chen et al. 2021).

## Artifacts

_n/a_ — prototype only in pre-substrate notebooks.
