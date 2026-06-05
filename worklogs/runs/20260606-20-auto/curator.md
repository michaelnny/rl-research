---
verdict: failed-structural
nearest_prior_or_disqualifier: model-based planning (disqualifier family); CHX (#17, within-trajectory signal-geometry collapse)
side_information: [learned dynamics, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction failed in execution.** The persistence-horizon primitive is theoretically distinct from GVFs/successor features, but on the substrate it collapses to a model-based shortest-path planner. The step-penalty channel fires every step (h* ≈ step-count-to-terminal for all actions), making the Pareto comparison equivalent to "reach termination fastest." DST scored 99.0 vs random 194.0 — active harm, not silence — consistent with the agent preferring low-value near-treasure over high-value far-treasure because short-horizon persistence wins the Pareto vote. This is model-based planning with renamed states.
- **Terminal-only reward channel defeat.** The treasure channel on DST never fires during early episodes, so p_m(ô_h) ≈ 0 for all h, h*[o,a,treasure] = H_max for all actions — that channel contributes nothing to the Pareto vote. The only discriminating channel is step-penalty, and "shorter step-penalty horizon" = "faster path to any terminal event" = greedy shortest-path to termination, which is structurally the CHX/CRP/PICAV collapse pattern extended to forward-model-predicted horizons.
- **Primitive count acceptable** — one primitive (level-set crossing horizon vector) + one improvement operator (Pareto-non-dominance logit nudge). No stack.
- **Evidence quality: negative.** beat_random=0, beat_strong=0 across both vector envs. DST at 99.0 vs random 194.0 is below-random active harm, not just silence. RG at 0.011 vs random 1.331 is consistent with action-uniform forward model predictions.
- **Failure is family-informative.** This extends the CHX/CRP ruling: any primitive based on predicted time-to-first-channel-event collapses to a shortest-path-to-terminal planner on substrates with a dense universal step-penalty channel, because the step-penalty dominates the Pareto comparison and short-horizon = fast termination.

## Lesson for the next iteration

Any forward-model primitive that inherits "faster channel onset is better" on a per-channel basis will collapse to shortest-path planning on envs with a universal step-penalty channel — the step-penalty channel's h* encodes time-to-terminal, not quality-of-terminal, and will dominate the Pareto vote over all sparse reward channels.
