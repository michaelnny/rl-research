---
verdict: null-result
nearest_dead_family: none
---

## Verdict reasoning

- PARGRAD (per-step bivariate weak-dominance-with-at-least-one-strict count as score-function gradient weight) is a structurally distinct, coherent probe that passed Reviewer triage cleanly; the mechanism is not in any dead family.
- Candidate score 99 is below the random floor of 194, failing the tertiary falsifier. PARGRAD regressed from its predecessor COPDEV (194) rather than improving it; directional sharpening did not help.
- The ablation (random-uniform per-step weight) fully collapsed to score=0 via degenerate 1-step episodes, meaning the +99-point candidate-vs-ablation delta reflects ablation instability rather than a meaningful structural contrast. Beating a collapsed ablation is not evidence that the claimed primitive is load-bearing.
- The gradnorm_var and mean_pt_trend logs (ablation arm, overwritten by both train scripts using the same filenames) show: gradnorm_var non-zero only for the first ~600 steps then zero; mean_pt drops from 0.5 (empty-buffer prior) to 0.0 immediately and stays there. These are signatures of the ablation collapsing rather than the candidate training; the candidate's corresponding observables were not captured.
- The Pareto-direction property (upward-drifting mean_pt_trend) was the primary discriminator claimed in the hypothesis; without the candidate's logs and with the candidate below random floor, there is no positive signal to carry forward from this iteration.

## Lesson for the next Researcher

Both COPDEV and PARGRAD score-function approaches on DST-concave using per-step bivariate empirical rank weights have now failed to beat the random floor; the next probe should either test the bivariate dominance mechanism on a non-degenerate two-channel substrate (where channel 2 is not deterministic given survival) or move to a structurally different mechanism family entirely.
