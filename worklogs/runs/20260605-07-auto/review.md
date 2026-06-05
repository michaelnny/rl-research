---
verdict: novel-direction
reviewer_run: 20260605-07-auto
---

## Reasoning

PTW's central primitive — the prefix-witness floor `w(prefix) = coord-min { v(τ′) : τ′_{0:k} = prefix }` — is structurally distinct from every disqualifier family. The update weight `‖Δ(τ,k)‖_{1,+}` is the L1 of strictly-positive coordinates of a coordinate-wise minimum difference between two sets of observed terminal vectors matched by action-sequence prefix; this is not return-to-go (which is a single trajectory's tail sum), not advantage (which requires a learned or bootstrapped baseline scalar), not a Pareto-dominance margin (VCC's functional), and not a count or curiosity bonus. The indexing key is the action-sequence prefix trie, not an observation hash, which structurally avoids the bootstrap-wall failure that killed FED (#15) and PICAV (#16). The hypothesis articulates the distinction from the nearest live candidate (VCC) and from the nearest disqualifier family (advantage / return-to-go) in precise functional terms, not just vocabulary avoidance.

All ten required-candidate-shape slots are filled with substantive answers: the experience object, the core primitive with its construction-time non-negativity proof, the improvement operator and its update rule, the execution rule, the vector feedback rule with an honest k=1 degenerate-case audit, rollout-cost discipline with O(T·B) accounting, a nearest-neighbor novelty audit, four predicted failure modes (including the pre-first-success silence on single-channel sparse envs and the magnitude-domination collapse risk), the side-information channel declaration, and a formal monotonic improvement claim. The falsifier is concrete and scoped to the vector envs where the bootstrap problem is structurally absent.

## Risks the Engineer should be aware of

- On DoorKey and KeyCorridor the staircase floor is zero until the first success trajectory enters the buffer, so the operator provides no improvement signal before that point. The run will likely show no improvement on those envs until a success is reached by exploration alone; the Engineer should verify that exploration is sufficient to reach first success in the allocated time budget and that the absence of signal is not mistaken for a bug.
- The monotonic improvement claim holds "in the limit of small step size and large buffer" — at small buffer sizes (early training) the floor is dominated by single-trajectory outcomes and the operator degenerates toward vanilla policy gradient on the few successful prefixes. The Engineer should log the average prefix-sharing depth and the fraction of (τ, k) pairs with non-zero Δ as diagnostic channels, as the hypothesis prescribes.
- Channel-wise standardization (running median / MAD per channel) is required before computing L1 to prevent magnitude-dominated collapse to effective scalar optimization. This must be applied uniformly across all envs without per-env tuning or it reintroduces the scalarization rebadge risk on the vector envs.
