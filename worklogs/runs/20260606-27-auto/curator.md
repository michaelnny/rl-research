---
verdict: failed-structural
nearest_prior_or_disqualifier: PCR (#24) / FED-family bootstrap-wall
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction claimed but not sufficient:** PHI's Lévy-area primitive in observation-embedding space is genuinely non-zero from step 1 (the geometry fires without reward), but the channel-projection layer (k linear regressors `P_m` fit against terminal cumulants) re-introduces the bootstrap dependency. On DoorKey and KeyCorridor, no rewarded trajectories arrive within 120 s so `P_m` coefficients are pure noise — the operator fires random nudges. On DST and RG, the treasure/resource channel is terminal-only, so `P_m` for that channel sees near-zero targets throughout training; the only non-degenerate regressor is for the step-penalty channel, whose geometry-projection reduces to "prefer shorter paths" — equivalent to scalar step-penalty minimization. This is structurally the same bootstrap-wall as PCR (#24): a reward-independent primitive paired with a reward-gated projection layer inherits the full FED-family collapse.
- **Primitive vs stack:** The core primitive (Lévy-area matrix + k independent linear regressors + Pareto-non-dominance count) is close to the one-primitive-one-operator requirement, but the two-stage structure (geometric signal → regression → vote) means the second stage is silently degenerate whenever the first stage's output lacks correlation with reward channels — which is exactly the bootstrap-wall condition.
- **Evidence quality:** 0/4 envs beat random. DST scored 99 vs random 194 (below random by 48%), RG scored 0.011 vs random 1.331 (below random by 99%). The below-random harm on vector envs confirms the operator is not just silent but actively harmful once the step-penalty channel dominates the regressor fit.

## Lesson for the next iteration

Any algorithm that requires a learned mapping from a reward-independent geometric feature to per-channel cumulant values inherits the FED-family bootstrap-wall unless the mapping itself can produce a useful signal before the first rewarded trajectory arrives — and a linear regression fit against terminal cumulants cannot.
