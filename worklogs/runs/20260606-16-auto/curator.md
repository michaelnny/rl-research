---
verdict: failed-implementation
nearest_prior_or_disqualifier: cwai-channel-wise-action-influence
side_information: [vector diagnostics, learned dynamics]
---

## Verdict reasoning

- **Structural distinction holds.** CID's central primitive is a one-step log-likelihood-ratio of binary firing probability (`log q_m(ô_{t+1}) − log q_m(o_t)`) evaluated in probability space, not a Jacobian column norm or gradient magnitude. This distinction from CWAI/JFP is well-articulated and survives variable renaming: probability-ratio objects do not shrink under stochastic transitions the way gradient-magnitude objects do. The Reviewer correctly passed this as `novel-direction`. No disqualifier family match.

- **Primitive count is acceptable.** The candidate has two declared components (binary classifier `q_m` + forward model `f_φ`) united by a single composition law (Pareto-non-dominance vote on the k-dimensional LR vector). The forward model is a support component, not a second independent primitive. This is within the "one primitive + one improvement operator" guideline.

- **Evidence quality is insufficient — but failure is mechanical, not structural.** Both vector envs scored below random (DST: 99.0 vs random 194.0; RG: 0.001 vs random 1.331; beat_random=0, beat_strong=0). However, the hypothesis explicitly predicted that if `f_φ` cannot distinguish actions early in training, LR rows are nearly action-invariant and the Pareto nudge goes to zero — leaving the policy to drift with a weak base signal. DST at 99.0 (not zero) is consistent with a mostly-silent Pareto nudge combined with mild base-policy learning. The hypothesis also described the "canonical version" that uses *actually observed* `o_{t+1}` to update LR statistics offline rather than querying `f_φ` at decision time — this alternative path avoids the forward-model bottleneck entirely and was not what ran. The implementation apparently chose the `f_φ`-at-decision-time path, hitting the action-invariant-LR-rows failure mode the hypothesis itself predicted.

- **Failure-mode informativeness.** The failure rules out `f_φ`-at-decision-time CID specifically, not the probability-space LR primitive in general. The offline canonical version (compute LR from actually-observed `o_{t+1}` in the replay buffer, no decision-time forward model query) remains untested and addresses the mechanical bottleneck directly.

## Lesson for the next iteration

Re-implement CID using the offline canonical path (`LR_m` computed from actually-observed `o_{t+1}` post-hoc on buffer samples, no `f_φ` forward-model query at decision time) to eliminate the action-invariant-LR failure mode that silenced the Pareto nudge in this run.
