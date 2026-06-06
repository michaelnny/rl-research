---
verdict: ablation-failure
nearest_dead_family: none
---

## Verdict reasoning

- **Principle and schema**: TEAR proposes a per-trajectory backward-linear adjoint co-state (trajectory-empirical Jacobian + Hamiltonian-weighted score-function gradient) targeting the vector stage. Schema is valid; Reviewer approved as `probe`; claimed stage and feedback signal are coherent and correctly matched.
- **Candidate scores**: DST 99.0 (random=194.0, so -95 below random), RG 0.011 (random=1.331, so -1.32 below random). Neither env beats the random baseline. beat_random=0, beat_strong=0.
- **Ablation scores**: Exactly identical — DST 99.0, RG 0.011, delta=0.0 on both envs. The trajectory-empirical backward adjoint and the i.i.d. Gaussian lambda replacement produce indistinguishable results.
- **Verdict**: The candidate does not beat random on either env, and the ablation matches the candidate exactly (delta=0.0 on both). The backward adjoint primitive carries no information load; the algorithm reduces to REINFORCE with an arbitrary weight scalar in this regime. This is a clean ablation-failure, not an infrastructure failure — the run completed in 260s with both claim and ablation ladders reaching `completed`.
- **What this teaches**: The rank-1 trajectory-empirical Jacobian and the backward linear adjoint recursion do not produce meaningful credit-assignment signal in the 120s budget on DST/RG. The identical ablation scores suggest the learning signal is dominated by the score-function entropy noise regardless of the weight (Hamiltonian vs. random). The H_t weight does not stabilize or guide learning when the adjoint propagation path through the rank-1 Jacobian is not selective.

## Lesson for the next Researcher

Do not retry Pontryagin/adjoint co-state variants with rank-1 or trajectory-empirical Jacobians as the credit-assignment primitive — the ablation shows the backward recursion contributes nothing over a random weight, and the method fails to beat random on either DST or RG.
