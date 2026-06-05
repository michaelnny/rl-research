---
verdict: failed-structural
nearest_prior_or_disqualifier: return-to-go (disqualifier family) / attempt-16-picav
side_information: [vector diagnostics]
---

## Verdict reasoning

- **Structural distinction collapsed at execution**: CHX's convex-hull extremality primitive is genuinely novel in its mechanism — per-step L2 distance-to-hull of within-trajectory cumulant trace, non-linear, non-scalarizable. However, the hypothesis itself stated the self-disqualifying condition: "on envs with effectively k_eff=1 vector signal, the hull degenerates to a 1-D segment whose extremes are just the min and max steps — this IS a return-to-go rebadge." Both panel vector envs (Deep Sea Treasure, Resource Gathering) have terminal-only reward channels — the treasure/reward signal is zero throughout each episode and fires only at termination. This makes the cumulant trace a near-line in the cost/step-penalty direction with the reward dimension contributing only at the final point, producing k_eff=1 dynamics. Under this condition the hull's extreme points are the first and last cumulant points, and hull contribution reduces to proximity to the episode start/end — structurally equivalent to return-to-go weighting.

- **Primitive count is clean (one primitive + one operator)**: No stack; the mechanism is well-specified as a single geometric primitive (hull contribution) + one centered log-prob update. The structural collapse is not due to complexity but due to environment incompatibility.

- **Evidence quality is poor, matching the self-predicted failure mode**: Deep Sea Treasure scored 99.0 vs random 194.0 (below random), Resource Gathering scored 0.011 vs random 1.331 (below random). beat_random=0, beat_strong=0. The failure is in the exact envs where CHX was predicted to be strong, triggered by the exact degenerate case the hypothesis flagged.

- **Cross-attempt failure mode match**: Same terminal-only vector channel collapse as PICAV (#16) — "Rules out the signed cross-channel temporal-ordering moment family whenever any vector channel is terminal-only." CHX extends the same ruling: any within-trajectory geometric primitive operating on cumulant traces degenerates when the vector signal is sparse/terminal-only, because the trace is uninformative for all but the final step.

## Lesson for the next iteration

Any within-trajectory signal-geometry primitive (hull extremality, Pareto moment, cross-channel asymmetry) requires the per-step vector signal to have non-degenerate multi-channel variance *throughout the trajectory* — the panel's Deep Sea Treasure and Resource Gathering envs violate this because the primary value channel is terminal-only, so all such primitives collapse to return-to-go variants on the substrate as currently configured.
