---
verdict: ablation-failure
nearest_dead_family: none
---

## Verdict reasoning

- PRISM is a structurally novel probe: rolling Pareto-frontier KDE on vector-return space used as a non-linear, non-scalarizing per-trajectory score-function weight. The reviewer approved it. The primitive is genuinely typed and the ablation plan is load-bearing on paper.
- Both candidate and ablation finished at DST=99.0, RG=0.011 — the exact nearest-treasure-collapse floor. `ablation_delta` is 0.0 on both envs. This is the fourth consecutive occurrence of this pattern (runs 15, 16, 17, 18 all hit DST=99.0/RG=0.011 for both arms).
- The failure is not implementation noise or a wrong ablation design. It is that neither the candidate nor the ablation produces any learning within the 120s vector-stage budget. When both arms are at the random floor, the ablation comparison is vacuous: the test cannot distinguish a novel mechanism from its ablation because neither mechanism fires at all.
- The claimed discriminating observable (`coverage_n >= 3`) cannot be evaluated from the panel output — the panel only reports hypervolume. This means even if coverage was degenerate, the Curator cannot confirm or deny the secondary falsifier from the available evidence.
- This run teaches a substrate-level constraint: the 120s vector-stage budget on DST-concave and RG is insufficient for any score-function / policy-gradient method to depart the random floor, making ablation comparisons at this stage structurally uninformative. Novel mechanism discrimination requires either a longer budget, a simpler warm-start, or a stage where some learning already occurs within budget.

## Lesson for the next Researcher

Vector-stage score-function probes that rely on the policy first departing the random floor before their mechanism can fire will always produce ablation ties at DST=99.0/RG=0.011 within the 120s budget; probe at a shorter/simpler stage first (quick or sparse) to confirm the policy can learn anything, then escalate to vector once some lift is established.
