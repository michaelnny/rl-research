---
verdict: failed-structural
nearest_prior_or_disqualifier: actor-critic disqualifier family (scalar-weighted log-prob)
side_information: [learned dynamics, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction attempt:** PCGA avoids consuming the auxiliary head's output magnitude as a policy weight — only the cosine direction in parameter space enters, compared via coordinate-wise Pareto dominance across actions. The Reviewer correctly identified it as novel-direction on paper. However, the panel shows below-random scores on both vector envs (DST 99.0 vs random 194.0; RG 0.011 vs random 1.331), consistent with the hypothesis's own predicted degenerate modes: trunk-gradient near-uniformity (failure mode a) and/or channel collinearity on DST's terminal-only treasure dimension (failure mode c).

- **Primitive count:** One primitive (parameter-space cosine alignment tensor A[s,a,m]) + one improvement operator (Pareto-dominance logit nudge). Clean single-primitive shape. The structural failure is not a stack problem.

- **Evidence quality:** Ran to completion (status=completed, 0 retries) on the vector stage. Both envs scored below random. No env showed above-random signal. The bootstrap argument (step-penalty channel always dense → head gradient non-degenerate from step 1) was plausible but insufficient: the shared trunk means ∇_θ log π(a|s) and ∇_θ ĉ_m(s) both derive from the same trunk parameters, making A[s,a,m] nearly action-invariant (dominated by trunk-layer activations common to all action outputs), so the Pareto vote sees near-uniform rows and produces near-symmetric nudges — effectively random perturbation.

- **Failure-mode informativeness:** This rules out the "parameter-space gradient cosine alignment via shared trunk + Pareto vote" family. A separated trunk (distinct parameters per action) would defeat the purpose; a decorrelation mechanism between the policy head and auxiliary head gradients is needed but constitutes a new primitive. The family is structurally dead without that mechanism.

## Lesson for the next iteration

Parameter-space gradient alignment between a shared-trunk policy and a shared-trunk auxiliary head collapses to near-uniform cosines because both gradient vectors are dominated by shared trunk parameters — the action-discrimination information is confined to the final output layer which contributes negligibly to the full-parameter cosine; any future candidate using parameter-space alignment must operate on action-specific parameter subsets or use a gradient-projection step that zeroes the shared-trunk component before computing cosines.
