---
verdict: failed-structural
nearest_prior_or_disqualifier: 18-cec-continuation-endpoint-concordance
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction:** CWTP is genuinely distinct from CEC/FED/RSD/DPC on paper — the "latest divergence before confluence" key means each witness compares only the bracketed segment, not terminal cumulants, and does not require endpoint-pair bucket density. However, the structural distinctness did not survive contact with the substrate. On Deep Sea Treasure, the treasure channel fires only at the terminal step; for a confluence at the terminal observation-hash the bracketed segment `[d_i+1..t_i]` vs `[d_j+1..t_j]` is non-terminal only if there is a non-terminal shared hash — but in a sparse long-horizon env those are rare. The operator required both (1) cross-trajectory observation-hash collisions at non-terminal states AND (2) non-trivial per-step vector signal in the bracketed segment. Neither condition is reliably met on the substrate. Scored 0.0 / 0.011 vs random 194.0 / 1.331 — below random on both envs.
- **Primitive vs stack:** One clean primitive (sign-vote tensor from confluence witnesses) plus one improvement operator (Pareto-non-dominance count logit nudge). Primitive count is acceptable.
- **Evidence quality:** Zero envs beat random (beat_random=0, beat_strong=0). The stated falsifier — confluence pair rate below 1 per 1000 buffer-steps means operator is silent — almost certainly triggered; the 0.0 on DST is consistent with the operator never firing rather than firing and failing to improve. No diagnostic instrumentation was included, so we cannot distinguish "operator silent" from "operator fired but unhelpful." Either way the evidence is negative.
- **Failure-mode informativeness:** This failure extends the sprint-4 ruling. The "latest-divergence-before-confluence" framing does not escape the bootstrap wall — it reduces it to two simultaneously required sparsity conditions rather than one, making it strictly harder to satisfy on sparse long-horizon envs than FED's single obs-hash-bucket condition. Rules out the "pairwise trajectory comparison indexed by intermediate shared state" sub-family as long as confluence pairs are sparse.

## Lesson for the next iteration

Any candidate that requires two simultaneous sparsity conditions to produce a non-trivial signal (cross-trajectory state revisits AND non-terminal vector channel activity between divergence and confluence) inherits the FED bootstrap wall and then adds a second one; the Researcher should look for primitives whose signal fires on every step from the first episode, not only when rare geometric coincidences occur across the trajectory buffer.
