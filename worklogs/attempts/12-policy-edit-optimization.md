---
id: 12
slug: policy-edit-optimization
status: failed
sprint: 2026-05-29
verdict_in_one_line: "Policy-edit response optimization works on a hand-designed semantic edit basis, but a scalar edit-ES with the same basis matches it — the basis was carrying the lift, not the optimizer."
side_information: [environment instrumentation]
nearest_prior: "CEM / ES / CMA-ES elite refitting"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 12 — Policy Response / Policy-Edit Optimization

## One-sentence idea

Make policy primary: instead of learning value, estimate how policy edits
change the outcome law, and apply edits whose estimated response points
into a desirable vector-outcome cone.

## Core primitive

For a policy edit `e` and an outcome law `M(π) = L_π(Y)`, the **policy
response** at the current policy is

\[
R_\pi(e) \;=\; \left.\frac{d}{d\epsilon} M(\pi_{\theta+\epsilon e})\right|_{\epsilon=0}.
\]

## Improvement operator

Apply edits whose estimated response points into a desirable vector-
outcome cone. The policy itself is the object being refined; no critic
supplies a scalar weight.

## Why it looked promising

- The deployed object is the policy — operating directly on it is
  conceptually clean.
- Native vector consumption via the outcome cone.
- Avoids both Bellman backup and scalar-weighted log-prob updates.

## What was tested

Compared against a **scalar edit-ES** (evolution strategies) baseline that
used the *same semantic edit basis*. Once the edit basis was held fixed
between methods, the scalar edit-ES matched the policy-response method.

## Why it failed

Once tested against scalar edit-ES on the same semantic edit basis, the
benefit came from the **hand-designed behavior coordinates**, not from the
optimizer. Without a new coordinate-discovery primitive, this collapses
toward ES / NES / direct policy search / policy-gradient-like
perturbation methods — i.e. the disqualifier-family rule "CEM / ES /
CMA-ES elite refitting." This is also a clean instance of cross-attempt
failure mode "hand-engineered coordinates are side information."

## Lesson / constraint added

If a method optimizes over a hand-designed edit / coordinate basis, the
lift attributed to the optimizer must be ablated against a scalar
optimizer on the same basis. If they match, the basis is the algorithm
and the candidate is an ES rebadge.

## Nearest neighbors in the literature

- Evolution Strategies for RL (Salimans et al. 2017).
- Augmented random search (Mania et al. 2018).
- Natural evolution strategies (Wierstra et al. 2014).
- Direct policy search / CMA-ES.

## Artifacts

_n/a — explored offline; not implemented in this substrate._
