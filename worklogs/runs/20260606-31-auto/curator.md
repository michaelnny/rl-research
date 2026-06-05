---
verdict: alive-weak
nearest_prior_or_disqualifier: dpc-divergent-prefix-concordance (alive-weak, 20260606-02-auto)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** KSV replaces DPC's first-divergence-state hash index with a continuous Gaussian kernel on a learned observation embedding, so every trajectory pair contributes non-zero weight to every decision step. This is a genuine structural break from the entire FED/CEC/CSD/TRAC/TPP/CWTP family (all hash-collision-gated) and from KTAC (k-means cluster identity). The kernel ensures the primitive fires without discrete coincidence.
- **Primitive vs stack:** one primitive (kernel-weighted per-(action, channel) signed pair-vote tensor SV[a,m]) plus one operator (strict Pareto-non-dominance margin to logit nudge). The auxiliary forward model is a component supplying the embedding, not the explanation. Passes the one-primitive rule.
- **Evidence quality:** DST score 471.0 beats strong 285.0 convincingly (margin +186), confirming the kernel fires on DST and produces directional signal that surpasses the strong baseline. RG score 1.331 exactly ties random and strong — no signal. Only one vector env beat strong; the second vector env (RG) produced the collinearity-collapse / stochastic-sign-cancellation failure predicted by the hypothesis's own failure modes (a) and (c). Evidence is real but thin — one env, no sparse-stage results.

## Lesson for the next iteration

KSV's kernel overcomes the discrete-hash bootstrap wall on DST but falls on RG via stochastic sign cancellation in the SV tensor; the next iteration should focus on whether a variance-weighted or sign-confidence gate on the pair contributions can suppress zero-signal pairs on stochastic envs before the Pareto vote fires.
