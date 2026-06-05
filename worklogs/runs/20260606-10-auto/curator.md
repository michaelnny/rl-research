---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-15 (FED family — bootstrap-wall / terminal-outcome-gated operator)
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction:** The recovery primitive R[c,a] ∈ R^L (step-lag until the policy's modal action is re-confirmed at L thresholds) is genuinely novel — reward-independent and computed for free from existing logits. However, the improvement operator is gated entirely by terminal-outcome Pareto-dominance sign. When no rewarded trajectory has been collected, the gate is universally silent and the operator produces zero updates. The structural novelty of the primitive is real, but the update path still hits the bootstrap wall that defines the FED/CEC/TPP/CWTP family.
- **Primitive vs stack:** The hypothesis is cleanly one primitive (R[c,a]) + one improvement operator (Pareto-meet of recovery-non-dominance and outcome-non-dominance). Not a stack. The composition law is well-stated.
- **Side information:** Both channels are named and legitimate. Transition geometry (action-logit sequences along realized trajectories) is a new channel; vector diagnostics (terminal outcome as binary sign gate) is established. The channel declaration is honest.
- **Evidence quality:** beat_random = 0, beat_strong = 0 across all four core envs. DST scored 99.0 vs random 194.0 (below random). DoorKey, KeyCorridor scored 0.0. Resource Gathering 0.011 vs random 1.331. The gate silence is confirmed: falsifier condition (b) — operator fires on < 5% of decision steps — was almost certainly met, consistent with the hypothesis's own predicted failure mode (b).
- **Failure-mode informativeness:** This failure rules out a broad sub-family: any algorithm whose improvement operator requires a terminal-outcome sign gate, regardless of how novel and reward-independent the primitive is. The lesson generalizes beyond PCR: a reward-independent primitive paired with a terminal-outcome-gated operator inherits the full bootstrap wall of the outcome-gated family.

## Lesson for the next iteration

A reward-independent primitive (e.g., policy self-consistency measurement) cannot escape the bootstrap wall if the improvement operator still requires a terminal-outcome sign gate to assign direction — the gate must be replaced with a direction signal that fires before any reward is observed, such as a pure geometry criterion derived from the primitive itself.
