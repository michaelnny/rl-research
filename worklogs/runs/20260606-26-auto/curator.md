---
verdict: reviewer-rejected
nearest_dead_family: none
---

## Verdict reasoning

- UNIRANK is a per-step alive-cohort channel-1 CDF rank weight applied via score-function ascent on a vector env; the candidate.json correctly sets `uses_vector_reward=false` and `nearest_disqualifier=scalarization`, because only `info["vector"][0]` enters the update — a `w=[1,0]` fixed scalarization, the canonical disqualified shape.
- Schema is internally consistent and coherent; the rejection is not a schema or implementability failure but a substrate-rule violation: training on scalar reward in a vector env is explicitly disallowed regardless of rank-invariance properties of the weight.
- The Reviewer confirmed analytically (and the hypothesis derivation steps 5-6 already contain the proof) that on DST-concave where channel 2 is deterministic given survival, PARGRAD's bivariate weak-dominance count reduces exactly to channel-1 strict-below CDF `F̂^-_t(M^1_t)`, and COPDEV's `|F̂^1_t - F̂^2_t|` reduces to `|F̂^1_t - 1|`; both bivariate primitives collapse to unicriterial channel-1 statistics on this substrate. The empirical probe cannot add to what the algebra already delivers.
- UNIRANK cannot be promoted even under Outcome C (score > 200) because the scalarization disqualifier applies; and under Outcome A the result would only empirically confirm what steps 5-6 prove. No panel run is needed to close this question.
- The analytic reduction (bivariate-rank collapses to unicriterial-rank on DST-concave) is a substrate-level fact that should be preserved in prior_attempts.md as a constraint on which substrates can discriminate bivariate-rank candidates.

## Lesson for the next Researcher

DST-concave is analytically degenerate for bivariate-rank mechanisms — any such mechanism reduces to a unicriterial channel-1 statistic on that substrate — so the Researcher must use a non-degenerate two-channel substrate (where channel 2 is not deterministic given survival) to test bivariate-rank claims, or leave the bivariate-rank region entirely and target a different mechanism family.
