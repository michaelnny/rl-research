---
verdict: null-result
nearest_dead_family: none
---

## Verdict reasoning

- **Principle and schema**: LYRA proposes gradient ascent on the Lyapunov-exponent gap `Δ = λ_1 − λ_2` of a reward-tilted policy cocycle, estimated online via QR factorization (Benettin-Galgani-Strelcyn), with a score-function policy gradient weighted by the per-frame-index tilt coefficients `(c_1 − c_2)`. Schema is valid; Reviewer verdict was `probe`; both prior `revise` fixes were addressed (sparse stage consistency, pseudocode broadcasting).
- **Panel results**: LYRA scored 0.0 on both sparse envs (MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0) at the 120s claim budget. Beat-random=0, beat-strong=0. The empirical Lyapunov gap could not be evaluated for the falsifier check because episodes never terminated with reward.
- **Ablation**: The random-frame ablation also scored 0.0 on both envs. Candidate and ablation are tied at the floor — the Oseledec primitive is untestable because neither version learned to obtain any reward in the time budget.
- **Diagnosis**: The failure is not a false ablation match — both are stuck at zero reward. The rank-1 sample-path gradient estimator `(c_1 − c_2) · ∇_θ log π_θ` produces zero net signal when all rewards are zero (multiplicative tilt `exp(β · 0) = 1` is uniform, removing the coherence distinction between frame directions). LYRA's signal generation depends entirely on reward variability, but the sparse-stage envs were not solved at all within the budget; the gradient was near-zero throughout.
- **Lesson**: The multiplicative tilt `exp(β r)` only distinguishes frame directions when the policy already obtains some non-zero rewards; in a fully-zero-reward regime the gap gradient vanishes and the algorithm behaves identically to random exploration. A warm-start curriculum or a weaker initial baseline (e.g., first stage `quick` on CartPole) would be needed to test whether the Lyapunov primitive helps once reward is non-zero.

## Lesson for the next Researcher

LYRA-class gap-ascent algorithms have a cold-start failure: they require at least intermittent positive reward to generate non-trivial gradient signal, so attempting them first on hard-sparse envs without a curriculum reveals nothing about the primitive — use a solvable warm-up stage first.
