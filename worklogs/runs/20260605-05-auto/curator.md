---
verdict: alive-weak
nearest_prior_or_disqualifier: attempt-15-FED
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction.** VCC's indexing object is the running vector cumulant `c_t = Σ_{s≤t} v_s`, not observation-hash. This is a genuine geometric difference from FED (#15), SIT, and RSD: cumulant-bucket partitions vector-trajectory-space, which is dense by construction (every step contributes), while obs-hash partitions state-space and suffers the bootstrap-sparsity wall that killed FED. The distinction survives scrutiny and is not a renaming.

- **Primitive count.** One primitive (continuation-cumulant Pareto front indexed by cumulant bucket) + one improvement operator (logit nudge by cross-action dominance margin). The composition law is a single rule. Not a stack. Passes the primitive-count filter.

- **Evidence quality.** Beat random on 0/2 vector envs. Deep Sea Treasure scored 99.0 vs random baseline 194.0 — *below* random. Resource Gathering 0.011 vs random 1.331 — also below random. The method ran to completion with no retries or crashes, so this is a learning failure, not an implementation crash. Evidence is thin to negative on the target envs.

- **Failure-mode informativeness.** The below-random performance on both vector envs is most consistent with: (a) cumulant bucket explosion or collapse within the 120s budget — buckets either all unique (no continuation comparisons possible) or all identical (operator uninformative), or (b) the dominance margin nudge producing unhelpful or destabilizing logit perturbations before the Pareto fronts accumulate meaningful signal. Neither failure mode rules out the structural idea — they point to implementation tuning needs (quantization granularity, nudge magnitude α, warm-up period before nudges activate). The theory (dense cumulant partition → informative dominance margins → policy improvement) has not been structurally disproven; the experiment simply didn't produce learning within budget.

- **Side-information channel.** Vector diagnostics (env's `info["vector"]`) + transition geometry (running cumulant). Both named from the canonical list. Clean declaration.

## Lesson for the next iteration

VCC's cumulant-bucket density argument is structurally sound but the implementation needs diagnostic reporting on bucket-occupancy histograms and dominance-margin statistics before further panel runs — without that signal, the engineer is flying blind on quantization and α tuning.
