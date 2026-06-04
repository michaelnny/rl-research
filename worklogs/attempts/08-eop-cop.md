---
id: 08
slug: eop-cop
status: failed
sprint: 2026-05-26
verdict_in_one_line: "Manufactured thousands of plausible-looking symbolic certificates; none of them solved DoorKey. Volume ≠ understanding."
side_information: [event traces, object state]
nearest_prior: "07"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 08 — EOP / COP (Event-Effect / Clause Order Projection on MiniGrid)

## One-sentence idea

Repair Order Projection (#07) by learning more structured symbolic
clauses: if predicate `p(o)` holds, prefer action `a` over `b`.

## Core primitive

Local symbolic certificates over predicates `p(o)`:

\[
(p(o), a) \succ (p(o), b)
\]

with predicates including carrying state, front-object identity, relative
direction, distance bucket, time bin, etc.

## Improvement operator

Same KL projection as Order Projection, now onto clause-induced action
constraints.

## Why it looked promising

- Adds symbolic structure on top of OPP — should plausibly capture
  prerequisite-style clauses ("if carrying key, prefer open-door").
- Empty-5x5 worked.

## What was tested

MiniGrid Empty-5x5: passed. MiniGrid DoorKey: failed despite the system
producing thousands of certificates.

## Why it failed

Certificate volume was not understanding. Without a composition law for
prerequisites, symbolic local preferences remain brittle and spurious;
many plausible-looking clauses are just artifacts of trajectory frequency,
not prerequisite structure. This is the cross-attempt failure mode
"avoid systems that generate lots of symbolic constraints without a
strong composition law."

## Lesson / constraint added

Symbolic certificate enumeration is not a substitute for a composition
law. If the candidate's evidence is "the system produced lots of
certificates" rather than "the certificates compose to encode the right
prerequisite chain," it will fail the same way.

## Nearest neighbors in the literature

- Reward-machine learning (Toro Icarte et al. 2022).
- Symbolic option discovery / temporal-logic skill learning.
- Inductive logic programming for control.

## Artifacts

_n/a_ — prototype only in pre-substrate notebooks.
