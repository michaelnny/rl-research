---
verdict: null-result
nearest_dead_family: none
---

## Verdict reasoning

- GRADCOMP proposed rotating the REINFORCE update direction toward the top eigenvector of the empirical per-rollout Fisher (the "compass" primitive), with a cold-start magnitude floor to escape the zero-gradient regime on sparse-reward envs. The Reviewer passed the probe as structurally sound and well-targeted at the sparse stage.
- The sparse-stage panel (DoorKey-8x8 + KeyCorridorS3R3, 120s) produced 0.0 / 0.0 for both candidate and ablation on both envs; `ablation_delta` is 0.0 everywhere. Neither arm produced any non-zero return within the 120s budget.
- This is floor-clamping at the sparse stage: the cold-start Fisher walk never produced a reward-bearing episode within budget, so the "warm" REINFORCE phase (where the Fisher-vs-random distinction would become visible) never activated. The discriminating observables (alignment drift, first-reward-episode index) were never reachable.
- This mirrors the vector-stage floor pattern from runs 15-18, now manifesting at the sparse stage. The 120s budget on DoorKey-8x8 / KeyCorridorS3R3 is insufficient for a policy-gradient method (even with a non-zero cold-start update) to first stumble into the goal under these conditions, making the ablation comparison vacuous.
- The principle is not falsified (the comparison was never reached), but the run teaches that sparse-stage envs under 120s are as floor-clamping as the vector stage was for this class of probe. Future probes whose novelty fires only after first reward need either a shorter-horizon warm-start env or longer compute budget to be meaningfully evaluated.

## Lesson for the next Researcher

Sparse-stage MiniGrid envs (DoorKey-8x8, KeyCorridorS3R3) floor-clamp policy-gradient probes at 120s just as the vector-stage envs do: any mechanism whose novelty requires surviving the cold phase to reach first reward cannot be distinguished from its ablation within budget, so target an env where the policy reaches non-zero return within budget (e.g., quick-stage envs) or build a fundamentally non-gradient-based cold-start mechanism.
