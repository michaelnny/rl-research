---
verdict: alive-weak
nearest_prior_or_disqualifier: attempt-02-bric
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction holds.** The reviewer's `novel-direction` assessment is correct: SIT's graft points arise from observation-equivalence discovered within the trie itself (on-distribution by construction), whereas BRIC's swap points came from a hand-specified edit grammar (off-distribution). ETB/HPC is goal-conditioned hindsight with an event lens; SIT is unconditional on which states a suffix visits. No disqualifier family match is found under inspection.
- **Primitive count.** One primitive (observation-keyed action-suffix trie) plus one improvement operator (Pareto-dominant suffix inheritance with verification rollout). Clean.
- **Evidence quality is zero.** Beat-random: 0, beat-strong: 0 across all 4 envs. On DoorKey-8x8 the score was literally 0.0 — the trie never found a reward-bearing path. On the vector envs (deep-sea-treasure, resource-gathering) the scores matched random exactly. The exploration policy (softmax over uniformly zero outcomes) is indistinguishable from a random walk when no rewarding suffix exists yet. This maps precisely to cross-attempt failure mode #1: "The primitive needs reward correlation to bootstrap, but reward correlation does not exist on long-horizon sparse tasks until a deep unrewarded path is traversed."
- **Why not failed-structural.** The failure is not conceptual rebadging — the mechanism is novel. The failure is that the trie exploration policy provides no discovery advantage over random before the first reward-bearing suffix is found. The graft operator is correct but has nothing to graft. A dedicated discovery phase (e.g. explicit random rollout seeding into the trie, or a separate novelty-driven path harvester to seed initial reward-bearing nodes) could unlock the mechanism without changing its structural identity.

## Lesson for the next iteration

SIT's graft operator is a sound primitive but needs an explicit bootstrap mechanism to seed at least one reward-bearing suffix into the trie before observation-equivalence grafting can operate — on long-horizon sparse envs the default softmax-over-zero-outcomes policy is random walk, not exploration.
