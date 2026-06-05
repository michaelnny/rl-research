---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-18 (CEC) / FED-CEC bootstrap-wall family
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** TPP's postfix-divergence anchor (walking backward from matched terminal observations) is not a trivial rename of DPC (prefix-divergence) or CEC (exit-hash bucketed cumulants). However, the mechanism still requires terminal-observation-hash collisions before any nudge fires — the same gate that killed CEC (#18). The structural novelty of the backward walk does not escape the bootstrap dependency on hash collisions.
- **Primitive vs stack:** One primitive (postfix-divergence-anchored Pareto-vote count W[s,a]) + one improvement operator (logit nudge toward W-plurality action). Clean shape. The failure is not about stack complexity.
- **Evidence quality:** Both vector envs below random (0.0 vs 194.0 on DST; 0.011 vs 1.331 on RG). The operator almost certainly never fired — no diagnostic output in panel.txt to confirm terminal-hash collision rates, but identical score pattern to CEC/CWTP/PICAV strongly implies W stayed empty throughout the 120 s budget. The hypothesis's own falsifier ("below 5% terminal-hash collision rate → dead") was almost certainly confirmed, though not logged explicitly.
- **Family membership:** Extends the FED/CEC ruling to the "terminal-observation-matched trajectory pair + backward lockstep walk" variant. The bootstrap wall is the root cause regardless of whether the collision point is mid-trajectory (CEC), at the terminal (TPP-variant), or requires both (CWTP). Any primitive that is silent until hash-collision coverage exceeds a threshold fails on long-horizon sparse envs within a 120 s budget.

## Lesson for the next iteration

TPP closes off the "terminal-observation-hash collision as gating condition" sub-family of the FED/CEC bootstrap-wall family; no hash-collision-gated pair primitive escapes this ruling without a paired explicit exploration primitive that drives coverage before the gate fires.
