---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-26 (TRAC)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** CSD's structural claim over TRAC (#26) is that Laplace-smoothed TV (well-defined with one sample per side) over a K-step firing window reduces the *sample-size requirement per (cluster, action) cell*. This is genuine and not a notational shift. However, TRAC's ruling explicitly named the (cluster, action) *revisitation frequency* as the kill condition — not the sample-size-per-cell once the cell is visited. CSD does not change how often the agent revisits the same (cluster, action) pair under uniform exploration on long-horizon sparse envs, so the bottleneck is intact. Scored 0.0 / 0.0 / 0.0 / 0.011 vs random 0.137 / 0.0 / 194.0 / 1.331 — same failure signature as TRAC.
- **Primitive vs stack:** One primitive (TV between two conditional next-cluster distributions), one improvement operator (Pareto-non-dominance logit nudge). Primitive count is clean. The failure is not architectural.
- **Evidence quality:** beat_random=0, beat_strong=0 on all four core envs. The only non-zero score (RG 0.011) is below random (1.331) by 2 orders of magnitude. No env showed the operator firing usefully; the panel result is consistent with the operator staying silent throughout the 120 s budget.

## Lesson for the next iteration

CSD confirms that reducing the per-cell sample-size requirement (TV vs JSD, K-step window) is insufficient — the (cluster, action) revisitation-frequency bottleneck is the binding constraint for any cluster-indexed conditional-distribution primitive on long-horizon sparse envs, and the next candidate must either abandon the cluster-indexed cell structure or pair it with an explicit exploration primitive that actively increases cell-revisitation frequency before the operator is trusted.
