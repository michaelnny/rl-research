---
verdict: alive-weak
nearest_prior_or_disqualifier: dpc-divergent-prefix-concordance (alive-weak candidate)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** TPM's index is episode-position `t` alone — every episode of length ≥ t contributes to `M[t,a,:]` with no hash-collision requirement. This is structurally distinct from DPC (first-divergence-state hash), the entire FED/CEC/CWTP/TPP family (obs-hash or exit-hash gating), and TRAC (cluster revisitation). The distinction is articulated and non-trivial. On the disqualifier list, the closest is scalar-weighted log-prob (REINFORCE/MC-advantage), but the update coefficient is a count of coordinate-wise Pareto dominations — invariant to per-channel monotone reparameterization — not a scalar return magnitude. In the k=1 degenerate case it collapses toward median-shift MC advantage, which the hypothesis correctly flags, but on k>1 envs the partial order is genuinely distinct.

- **Primitive count:** One primitive (signed per-channel median-difference Wasserstein vector M[t,a,:] keyed by episode-position) + one improvement operator (logit nudge by Pareto non-domination margin). Within the required shape: single composition law, no stacked components.

- **Evidence quality:** Weak and negative — DST scored 99.0 vs random 194.0 (below random), RG scored 0.011 vs random 1.331 (below random). beat_random=0, beat_strong=0. The run completed without errors. The failure is consistent with predicted failure mode (e) from the hypothesis: cold-start with terminal-only channels means M is populated only by step-penalty signal until the first successful episode, and within 120s the buffer may not accumulate enough mass at individual (t, a) cells for the Pareto margin to fire reliably. However, no diagnostic logging was captured to confirm whether the Pareto gate actually fired at all.

- **Failure mode informativeness:** The below-random scores do not rule out the episode-position-indexed family — they are consistent with cold-start and inadequate buffer density within the 120s budget. No obs-hash bootstrap wall applies here; the failure is a different class. The hypothesis's own falsifier (Pareto gate firing rate < 5%) was not measured, so the run does not confirm or deny it.

## Lesson for the next iteration

TPM's episode-position primitive avoids the hash-indexed bootstrap wall but introduces a cell-density cold-start: the Pareto margin cannot fire until each (t, a) cell accumulates ≥ 3 samples on both sides, and the next attempt should add diagnostic logging of Pareto-gate firing rate and consider a warm-start strategy (entropy floor or position-bucket coarsening) to accelerate cell population within the 120s budget.
