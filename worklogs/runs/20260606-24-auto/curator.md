---
verdict: inconclusive
nearest_prior_or_disqualifier: attempt-16-PICAV (signed cross-channel temporal-ordering moment family)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** IXC's primitive (`max_{ℓ>0} Λ − max_{ℓ<0} Λ`, the lag-asymmetric cross-channel firing-coincidence tensor) is structurally the same family as PICAV (#16): both are antisymmetric statistics capturing whether channel m tends to precede or follow channel m' under action conditioning. The lag-spacing and variance-filter refinements do not change the fundamental identity — when any vector channel is terminal-only, cross-channel entries involving it are zero for all lag values, and the primitive collapses to step-penalty self-coupling, exactly as PICAV did.
- **Primitive vs stack:** The candidate is structured as one primitive + one improvement operator (Pareto-non-dominance on filtered coupling-margin). The stack (variance-filter top-B + Pareto vote + logit-anchor + cumulant-prediction head) is more complex than a single composition law, but the core is recognizably one primitive. This is not the reason for abandonment.
- **Evidence quality:** No panel run was attempted. The Reviewer's known-rebadge verdict was accepted and the Engineer was skipped. Zero empirical evidence, but the structural argument is strong: the hypothesis's own failure modes (a) and (b) predict exactly the PICAV collapse on terminal-only reward channels.

## Lesson for the next iteration

IXC confirms that any antisymmetric cross-channel lag statistic (whether instantaneous as in PICAV, or lag-max as here) is structurally ruled out whenever any vector channel fires only at the terminal step — the "signed cross-channel temporal-ordering moment" family is exhausted at the structural level, not just empirically.
