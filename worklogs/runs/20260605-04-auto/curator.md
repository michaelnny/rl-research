---
verdict: failed-structural
nearest_prior_or_disqualifier: sit-suffix-inheritance-trie (alive-weak candidate)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction from disqualifiers:** FED's improvement operator — a binary indicator over Pareto-front set-extension gating a logit nudge — is not a scalar-weighted update, not a Bellman backup, and not expressible as `wᵀr` for any `w`. The Reviewer correctly found it novel. However, the bootstrap structural wall it hits is identical to SIT's: on all panel envs, the observation-hash buckets never accumulate sufficient sample mass for the empirical Pareto front to form, so the indicator never fires and the policy reduces to a uniform random walk. Scored 0.0 on DoorKey, KeyCorridor, and Deep-Sea-Treasure; 0.011 vs random 1.331 on resource-gathering. Even on the vector envs where FED was predicted to be strong, performance is catastrophically below random. The failure is structural: no amount of logit nudging fixes an empty Pareto front.

- **Primitive count:** One primitive (per-bucket empirical Pareto front over vector-outcome signatures) + one improvement operator (set-extension indicator logit nudge). Shape is correct. The problem is not the count.

- **Evidence quality:** beat_random=0, beat_strong=0 across all 4 envs. The hypothesis explicitly predicted DoorKey weakness but expected vector-env strength — Deep Sea Treasure and Resource Gathering both failed badly. The bootstrap problem surfaces before any channel-wise signal is usable. Same failure mode as SIT (alive-weak, run 20260605-02-auto), which also scored beat_random=0 for the same reason.

- **Failure-mode informativeness:** The failure rules out the broader family of "empirical Pareto front accumulation" methods that rely on observation-hash bucket collisions on long-horizon sparse tasks. The mechanism requires hash collision rates >1% within a rollout to accumulate meaningful sample mass; this condition is not met on panel envs under uniform random exploration, and FED provides no exploration primitive to drive collisions. This is a variant of the canonical cross-attempt failure mode: "primitive needs reward correlation to bootstrap."

## Lesson for the next iteration

Any method that indexes future-compression objects (Pareto fronts, outcome multisets, suffix tries) by observation-hash buckets will reduce to random walk on long-horizon sparse envs until the exploration policy generates hash-bucket collisions, and providing no exploration primitive means the method never bootstraps — the next candidate must either include an explicit exploration primitive or use a primitive that is informative before reward-bearing trajectories exist.
