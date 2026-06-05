---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-15 (FED — Frontier-Expanding Dispersion)
side_information: [reachability/reset structure, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction vs FED:** CEC's exit-hash bucketing (one bucket key per episode, from the terminal observation) is architecturally distinct from FED's visited-state-hash bucketing (one bucket key per mid-trajectory observation). The hypothesis correctly identified this as the structural fix to FED's sparse-bucket failure. But the scores are indistinguishable: 0.0 on Deep Sea Treasure and 0.011 on Resource Gathering vs random baselines of 194.0 and 1.331. The concordance signal never fired, which is exactly what CEC's own falsifier predicted would happen if median bucket size stayed below 2 distinct exit-hash buckets with ≥ 2 samples each. Exit-hash aggregation did not provide enough sample mass within the 120 s budget.
- **Primitive count:** One primitive (per-(state-hash, action) Pareto-bucket multiset indexed by exit-hash) plus one improvement operator (signed concordance count). Structurally clean — not a stack failure.
- **Side information channel:** Declared correctly — reachability/reset structure (exit-observation hash) and vector diagnostics (info["vector"]). Both are on the canonical list. Not a channel declaration failure.
- **Evidence quality:** Beat random on 0 of 2 vector envs; beat strong on 0 of 2. No signal above chance. The hypothesis's own stated falsifier triggered: concordance was structurally zero because the seeding phase could not accumulate ≥ 2 samples per (state, action, exit-hash) bucket within the time budget on Deep Sea Treasure, the env most predicted to succeed.
- **Failure-mode informativeness:** This failure extends the FED family ruling. FED showed that mid-trajectory obs-hash bucketing cannot bootstrap. CEC shows that exit-hash bucketing also cannot bootstrap within 120 s on these envs. Together they rule out the entire "empirical Pareto-front / cumulant-multiset indexed by any observation hash" family unless paired with an explicit sample-efficient exploration primitive that achieves repeated state-action-exit coverage before the concordance comparisons are made.

## Lesson for the next iteration

The "empirical Pareto-front / cumulant-multiset indexed by observation-hash" family is ruled out in both the mid-trajectory (FED) and terminal-exit (CEC) variants; any future candidate using a hash-indexed multiset as its core primitive must solve the bootstrap/coverage problem first with an explicit exploration mechanism, not just choose a cheaper bucket key.
