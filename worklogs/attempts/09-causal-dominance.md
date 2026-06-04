---
id: 09
slug: causal-dominance
status: failed
sprint: 2026-05-26
verdict_in_one_line: "Conceptually correct fix for OPP's passive-correlation flaw — but bureaucratic, expensive in interventions, not a clean primitive."
side_information: [reachability/reset structure, environment instrumentation]
nearest_prior: "02"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 09 — Causal Dominance Certificates

## One-sentence idea

Replace passive action preference (#07/#08) with local interventional
evidence: compare outcomes under `do(c, ι⁺)` versus `do(c, ι⁻)`.

## Core primitive

A **causal dominance certificate**:

\[
C = (c,\, \iota^+,\, \iota^-,\, \delta,\, \rho)
\]

such that, with confidence `1−ρ`, intervention `ι⁺` stochastically
dominates `ι⁻` over vector-outcome upper sets:

\[
P(Y \in U \mid \mathrm{do}(c, \iota^+)) \ge P(Y \in U \mid \mathrm{do}(c, \iota^-)) + \delta_U.
\]

## Improvement operator

Project policy toward certified locally dominant interventions.

## Why it looked promising

- Cleanly fixes the passive-correlation flaw of OPP / EOP-COP.
- First-principles correct: causal evidence implies policy improvement.
- Native vector consumption via the upper-set quantifier.

## Why it failed

In principle it is right; in practice it is bureaucratic. It needs many
controlled interventions (BRIC's expensive-rollout flaw, #02), it risks
becoming a causal-testing / planning framework rather than an RL
primitive, and the math (intervention pairs, upper sets, lower
confidence bounds, vector dominance) is not simple enough to look like a
new RL family. It was not carried to implementation.

## Lesson / constraint added

A causal-correctness fix that costs many counterfactual environment
trials per accepted clause is a BRIC rebadge. The next candidate must
extract causal evidence passively or near-passively.

## Nearest neighbors in the literature

- Causal RL (Bareinboim, Lattimore, Sutton et al.).
- Counterfactual policy evaluation (Thomas & Brunskill 2016).
- Interventional credit assignment.

## Artifacts

_n/a_ — conceptual only; not implemented.
