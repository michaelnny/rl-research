---
verdict: failed-structural
nearest_prior_or_disqualifier: 15-FED
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** PICAV's primitive (signed antisymmetric pair-contribution vector per (obs-hash, action)) is mathematically distinct from FED's attainment-set extension indicator. The Reviewer correctly called `novel-direction`. However, the practical failure is structurally identical to FED: on Deep Sea Treasure, the treasure channel fires only at the terminal step, making `δ_{j,treasure,t} = 0` throughout every non-terminal step. The pair-contribution vectors are flat except at episode termination — reinstating the bootstrap wall the candidate claimed to have eliminated. On Resource Gathering (0.011 vs random 1.331) the primitive also failed to generate useful signal.
- **Primitive vs stack:** One primitive + one improvement operator — well-formed candidate shape. The failure is not a complexity issue.
- **Side information declared cleanly:** `info["vector"]` + observation-hash bucket. Both legitimate canonical channels. Not the source of failure.
- **Evidence quality:** 0 beat random on both vector envs. DST scored 0.0; Resource Gathering scored 0.011 vs random 1.331. No useful signal on either target env. The hypothesis's own falsifier ("panel score must exceed panel_n_beat_strong on at least one vector env") was not met.
- **Failure-mode informativeness:** The failure rules out a family: any primitive that relies on cross-channel pair-contributions (or temporal-ordering moments between channels) will encounter the same collapse whenever any channel in the vector is terminal-only. This is a new structural constraint that narrows the design space for future vector-env candidates.

## Lesson for the next iteration

Any primitive that requires multiple vector channels to carry non-terminal per-step increments will silently collapse on envs like Deep Sea Treasure where one channel (treasure value) fires only at the terminal step — a structural constraint that must be explicitly addressed (e.g., by using return-to-go proxies for terminal channels, or restricting the primitive to continuously-firing channel pairs).
