---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-17-CHX (terminal-only channel collapse family); attempt-16-PICAV
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction vs disqualifier:** CRP's rank-percentile primitive is
  formally distinct from GVFs / successor features (magnitude integrals vs rank
  position). The Reviewer confirmed this. However, on envs where a vector channel
  fires exactly once (at the terminal step), within-trajectory rank-percentile is
  constant 1.0 — invariant across all actions. The claimed structural advantage
  (magnitude-invariant rank) provides zero discriminating information when the
  channel fires only once per trajectory, which is the defining characteristic of
  both panel vector envs (DST treasure channel = terminal only; RG channels have
  similar sparsity). The hypothesis's own falsifier (a) — "more than 80% of cells
  degenerate to constant rank-percentile" — was apparently confirmed by the 0.0 /
  0.011 scores. This reduces the effective mechanism to "do nothing useful on
  terminal-only envs," which is the same failure mode that killed CHX (#17),
  PICAV (#16), and LRA (#20). Rank-position does not escape the terminal-only
  channel trap.

- **Primitive vs stack:** The mechanism is one primitive (R[s,a,m] running mean of
  within-trajectory rank-percentile) plus one improvement operator (Pareto logit
  nudge on trend-corrected R̃). Not a stack. This criterion was clean.

- **Evidence quality:** 0 envs beat random, 0 beat strong. Both vector envs scored
  below random (DST: 0.0 vs 194.0 random; RG: 0.011 vs 1.331 random). The
  falsifier predicted by the hypothesis itself was confirmed: rank-position carries
  no distinguishing signal when channels are terminal-only, making CRP equivalent
  in effect to having no operator on the substrate's actual panel envs.

## Lesson for the next iteration

Any per-trajectory rank-position or within-trajectory temporal-position statistic
collapses to a degenerate constant on envs where vector channels fire only at
episode termination — extending the CHX/PICAV ruling to cover rank-based as well
as magnitude-based within-trajectory signal-geometry primitives; a future candidate
must identify a side-information channel whose signal is non-degenerate throughout
the episode, not only at the terminal step.
