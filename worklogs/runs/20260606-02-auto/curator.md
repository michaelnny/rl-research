---
verdict: alive-weak
nearest_prior_or_disqualifier: rsd-reconvergent-segment-dominance
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction.** DPC's improvement operator fires only at pairwise first-divergence steps (not all steps), uses per-channel sign-votes plus a Pareto-non-dominance gate, and never emits a scalar return. Substituting any fixed `wᵀv` for the channel vector demonstrably changes the gate's satisfaction, which is the decisive test against the MC-advantage disqualifier. Distinct from RSD (which requires shared end-hash; DPC drops that condition) and SIT (which grafts behavior with extra rollouts; DPC only nudges logits). Structural distinction is articulated and not trivially reducible.
- **Primitive count.** One primitive (divergence-event sign tensor `V[s_div, a, a', m]`) plus one improvement operator (per-channel majority-margin logit nudge with Pareto-non-dominance gate). Not a stack.
- **Side-information channel.** Explicitly named as vector diagnostics (per-step `info["vector"]`) and transition geometry (action-prefix divergence indexing). Both canonical channels.
- **Evidence quality.** Beat strong on Deep Sea Treasure (289.0 vs strong 285.0, random 194.0) — a genuine above-strong score on a vector env. But Resource Gathering scored exactly at random/strong (1.331 for all three), showing zero signal on the harder multi-objective env. The hypothesis predicted DST might degrade toward pairwise-signed-MC-advantage due to collinear channels; this appears to be happening. RG is the cleaner test of whether per-channel structure actually helps, and it shows no improvement. Only 1 of 2 vector envs showed signal; the one with genuine multi-channel structure showed none.
- **Failure-mode informativeness.** The RG zero-signal is informative: either (a) the operator fires but the sign votes are noise-dominated by stochastic transitions, or (b) the bootstrap problem from RSD persists in a different form — divergence events form, but the cumulant differences across pairs are too noisy to accumulate a consistent Pareto-non-dominated margin. The hypothesis's own predicted failure mode #1 (stochastic transitions dominating sign votes) is the leading candidate. The diagnostic `invocation rate` and `pairs per unique (s_div, a, a')` should be checked in the run logs.

## Lesson for the next iteration

DPC achieved a genuine above-strong score on DST but zero improvement on RG; the next iteration should examine whether the Pareto-non-dominance gate actually fires on RG and whether stochastic transitions in RG are swamping the per-channel sign votes, and consider whether a suffix-length or visit-count gate can filter noisy divergence events from rare high-signal ones.
