---
verdict: failed-implementation
nearest_prior_or_disqualifier: CHX-17 / bootstrap-wall family (terminal-only channel constraint)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** SDS's per-channel empirical CDF compared via strict first-order stochastic dominance is genuinely distinct from all named disqualifiers. It does not bootstrap a return distribution (no Bellman), does not extract a scalar from the distribution (no mean/quantile criterion), and does not reduce to Pareto-mean (VCC uses point-valued mean vectors; SDS uses distributional CDFs). The partial order on CDFs is structurally incomparable to Pareto-mean under variable renaming.
- **Primitive vs stack:** One primitive (per-(cluster, action, channel) quantile sketch) + one improvement operator (multi-channel SD conjunction logit nudge). Clean shape.
- **Implementation blockage — not mechanism failure:** The "≥ 2 non-degenerate channels" gate is the correct structural guard — removing it would collapse the operator to single-channel scalar comparison (a named disqualifier). With the gate in place on a substrate where reward channels are terminal-only, the gate never fires within the 120 s budget, leaving the operator permanently silent. Scores (DST 99.0 vs random 194.0; RG 0.011 vs random 1.331) match the silent-operator pattern, not a mechanism-induced collapse. This is a substrate incompatibility, not a rebadge.
- **Evidence quality:** beat_random=0, beat_strong=0 on both vector envs. No positive evidence, but the failure mode is the gate blocking activation rather than the operator misfiring when active.
- **Failure-mode informativeness:** Rules out this specific implementation on the current substrate's terminal-only reward structure. The mechanism survives on envs where ≥ 2 channels fire intra-episode (e.g. dense multi-objective envs). The idea is worth retrying with a modified gate: allow single-channel SD during the bootstrap window, or use a softer non-degeneracy criterion that distinguishes "not yet populated" from "populated but degenerate."

## Lesson for the next iteration

SDS's multi-channel SD conjunction requires a gate redesign that allows the operator to fire with one non-degenerate channel during the bootstrap window — the "≥ 2 channels" guard is the correct structural constraint but too restrictive for the current substrate's terminal-only reward channels; a graduated gate (1-channel OK with lower alpha, 2+ channels full alpha) would let the mechanism activate early while preserving the anti-rebadge property.
