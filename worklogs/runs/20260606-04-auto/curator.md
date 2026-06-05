---
verdict: failed-structural
nearest_prior_or_disqualifier: count-based-exploration (disqualifier family)
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction collapses on panel envs.** The Pareto-vs-zero improvement operator is structurally novel when some loops can be net-zero or positive across all channels. But both vector panel envs (DST and RG) have a universal step-penalty channel that is strictly negative on every step. This means every intra-trajectory loop accumulates a negative step-penalty contribution, so `L[s,a]` is Pareto-dominated by zero for every `(s,a)` that closes a loop — the suppression mask fires on all looping actions unconditionally. Under variable renaming this is "suppress actions that cause repeated observations," which is count-based exploration suppression, a named disqualifier family. The hypothesis's own declared falsifier confirmed this: "if every vector env requires excluding the step-penalty channel, the family is dead." Both vector envs require it.
- **Primitive count is clean (one primitive, one operator)** — the `L[s,a]` aggregator plus the Pareto-vs-zero logit suppression. Not a stack failure.
- **Loop-collision rate on DoorKey.** The operator's second failure mode also fired: DoorKey-8x8 has a partial-observable state that changes on nearly every step (carrying the key changes the observation), so intra-trajectory hash collisions are rare and the operator almost never fires. Combined with the step-penalty collapse on vector envs, the mechanism is structurally inert on all four panel envs.
- **Evidence is clear: beat_random=0, beat_strong=0 on all 4 envs.** DST scored 99.0 vs random=194.0; RG scored 0.121 vs random=1.331. The scores are below random, consistent with active suppression of productive looping actions rather than neutral behavior.

## Lesson for the next iteration

Any intra-trajectory loop primitive that uses Pareto-vs-zero suppression is a count-suppression rebadge on the panel's vector envs (DST and RG) because the universal step-penalty channel ensures every loop is trivially dominated by zero; the family requires a non-universal step-penalty structure to be structurally distinct.
