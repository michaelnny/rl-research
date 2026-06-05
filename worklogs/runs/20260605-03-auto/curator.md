---
verdict: alive-weak
nearest_prior_or_disqualifier: sit-suffix-inheritance-trie (alive-weak)
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction:** RSD is genuinely distinct from SIT and from all disqualifier families. SIT uses *divergent suffixes* from a shared prefix (downstream futures differ, requiring verification rollouts); RSD uses *convergent segments* between two shared-endpoint observation nodes (downstream future distribution is identically shared by construction, making verification unnecessary). The improvement operator — behavior-cloning toward a Pareto-dominant segment weighted by its dominance count, with no scalar projection — does not reduce to top-k cloning (Pareto-dominance on k-dim vector is structurally different from scalar ranking), REINFORCE/PPO (no reward-magnitude weight), HER (no virtual goal relabeling), or scalarization (no wᵀr).

- **Primitive vs stack:** One primitive (convergent-segment edge multiset `E(u,v)`) + one improvement operator (Pareto-dominant behavior cloning weighted by dominance count). The candidate shape is clean.

- **Evidence quality:** Zero signal across all 4 envs (beat_random=0, beat_strong=0). The failure mode matches the hypothesis's own "bootstrap collapse" prediction: on sparse-reward envs without a seeding phase, no reward-bearing edge pairs exist for the operator to compare — all segments have near-zero accumulated vector, and the step-cost channel only becomes meaningful once goal-reaching trajectories exist. This is a predictable cold-start failure, not a structural refutation. The multigraph primitive cannot demonstrate itself without at least one reward-bearing closed witness pair.

- **Failure-mode informativeness:** The failure rules out naive warm-start-free RSD on long-horizon sparse envs. It does NOT rule out RSD with an explicit seeding phase (e.g., random rollouts loaded into the multigraph before dominance updates begin). The fix is the same as what SIT still needs: evidence that once reward-bearing segments exist, the operator actually improves outcomes and that observation-hash collision rates are manageable.

## Lesson for the next iteration

RSD and SIT share the same bootstrap gap — the next iteration should implement an explicit random-exploration seeding phase that loads the multigraph with diverse trajectories before the dominance operator activates, then verify that Pareto-dominant edge pairs actually form and that policy improvement follows.
