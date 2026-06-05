---
verdict: alive-promising
nearest_prior_or_disqualifier: RND/curiosity (disqualifier family), #14 Primal Behavior Flow Pivot
side_information: [learned dynamics, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction**: CWAI uses input-Jacobian column-norms of the forward model at each operating point — a property of the local linearization that is nonzero even when prediction error is zero. RND/curiosity use prediction *error magnitude* as a scalar novelty signal. These are not variable renamings: a perfectly-fit forward model with zero prediction error can still have a non-trivial action-differentiated Jacobian. The Pareto-non-dominance operator over rows of G is coordinate-wise over channels, never collapsing to a weighted sum, so the vector disqualifier (scalarized vector-reward maximization) does not apply.
- **Primitive count**: One primitive (the Jacobian column-norm matrix G ∈ R^{|A|×k}) and one improvement operator (Pareto-non-dominated logit nudge). The forward model is a component, not an additional operator — it exposes the primitive but is not itself the improvement.
- **Side-information channel**: Declared cleanly as "learned dynamics" (forward model trained on transitions) and "vector diagnostics" (info["vector"] as predicted output dimensions, never as reward).
- **Evidence quality**: Beat strong on Deep Sea Treasure (1419.0 vs strong 285.0 — a 4.97x margin), which is a vector env. Failed on Resource Gathering (0.011 vs random 1.331), consistent with the hypothesis's own predicted failure mode (stochastic transitions shrink the Jacobian toward the noise floor). Only 2 envs tested (vector stage). Additional evidence from core/all stages is needed to confirm generalization.
- **Failure mode informativeness**: The RG failure is informative and predicted — it does not kill the family, it confirms the stochastic-transition weakness the hypothesis named. The DST result is strong enough to warrant continued investigation.

## Lesson for the next iteration

Advance CWAI to the core stage (sparse + vector envs) to test whether DST performance generalizes to non-vector envs, and verify the rank-1 collapse falsifier by logging per-channel Jacobian column-norm distributions on DST.
