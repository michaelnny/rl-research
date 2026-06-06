---
verdict: reviewer-rejected
nearest_dead_family: none
---

## Verdict reasoning

- CHANBI proposed using the angular bisector of per-channel unit-normalized score-function gradients `d* = (u_1 + u_2)/||u_1 + u_2||` as the update direction for vector-reward policy gradient, claiming rescaling-invariance as the load-bearing novelty.
- Schema was internally consistent and the principle was coherent; the rejection was purely a literature match, not a schema or implementability failure.
- The Reviewer identified that CHANBI's `d*` is algebraically identical (for K=2) to IMTL-G (Liu et al., ICLR 2021): the equal-projection constraint `d . u_1 = d . u_2` has a unique solution direction `proportional to (u_1 + u_2)`, which is exactly CHANBI's direction. The paper CHANBI's novelty argument was effectively reproducing was not cited.
- The claimed structural distinction from MGDA, Nash-MTL, and CAGrad is precisely the property IMTL-G was designed and named for; applying the aggregator to RL per-channel score-function gradients is a substrate choice, not a novel mechanism.
- This is the second run in a row (after PARGRAD, 20260606-22-auto) landing in the per-channel gradient aggregation region; adding a family-level corpus entry is warranted.

## Lesson for the next Researcher

The "per-channel unit-normalized gradient aggregation" shape is dead as IMTL-G; any proposal whose load-bearing novelty is "equal projection / rescaling-invariance across vector-reward channels at the parameter-gradient level" is a rebadge of published multi-task learning machinery and will be rejected before compute.
