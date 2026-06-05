---
verdict: failed-structural
nearest_prior_or_disqualifier: scalar-weighted-log-prob (step-penalty channel domination)
side_information: [vector diagnostics]
---

## Verdict reasoning

- **Structural distinction (notional):** CPR's Chebyshev sup-norm projection `min_θ sup_m KL(π̂_m || π_θ)` is mathematically distinct from `wᵀr`-weighted log-prob — no fixed `w` replicates the per-step argmax that switches `m*`. The reviewer correctly flagged this as novel-direction.
- **Structural collapse at runtime:** On both vector envs (DST, RG), the treasure/reward channel has a degenerate posterior throughout the bootstrap window (it never fires before the first rewarded trajectory). With k-1 channels degenerate, `sup_m KL(π̂_m || π_θ)` reduces to the single non-degenerate channel — the step-penalty channel — making every gradient step a single-channel weighted cloning update. This is the hypothesis's own stated falsifier (§8a: single-channel domination) confirmed. The Chebyshev operator cannot distinguish channels when only one has non-degenerate posterior mass.
- **Primitive count:** One primitive (k-tuple of channel posteriors) + one operator (Chebyshev KL projection) — not a stack. Primitive count is correct; the failure is runtime degeneration.
- **Evidence quality:** DST scored 99.0 vs random 194.0 (below random); RG scored 0.011 vs random 1.331 (below random). beat_random=0, beat_strong=0 on both vector envs. The DST non-zero score reflects step-penalty avoidance (short-episode preference) with no treasure-channel contribution — confirming single-channel reduction, not Chebyshev behavior.
- **Failure-mode informativeness:** High. This failure rules out the entire "per-channel empirical posterior + Chebyshev/sup-norm aggregation" family whenever the substrate has terminal-only reward channels. The bootstrap window will always produce degenerate posteriors for reward channels, forcing the sup to reduce to the step-penalty channel. This extends the FED/CEC/PCR/TRAC bootstrap-wall ruling to the "replay-reweighted posterior construction" sub-family: the aggregation level (sup-norm vs Pareto front) does not rescue the collapse when posteriors are degenerate.

## Lesson for the next iteration

The Chebyshev-center family requires all k channel posteriors to be simultaneously non-degenerate; on terminal-only-reward substrates this condition cannot be met during the bootstrap window, so any sup-norm or max-over-channels aggregation collapses to single-channel behavior — the next candidate must either provide a bootstrap signal that populates reward-channel posteriors before terminal reward appears, or abandon per-channel posterior construction entirely.
