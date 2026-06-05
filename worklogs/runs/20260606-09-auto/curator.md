---
verdict: alive-promising
nearest_prior_or_disqualifier: attempt-15-FED (sprint-4 empirical-Pareto family)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction.** FFTV's primitive (IQR of per-channel first-firing-time offsets, Pareto-non-dominance vote) is structurally distinct from the GVF/successor-feature family (#14): it computes a higher-order moment (dispersion) of an event-timing distribution rather than an expected discounted sum of cumulant magnitudes, and the improvement operator is Pareto-non-dominance (not any linear scalarization `wᵀψ`). It is also distinct from the FED/CEC/CWTP/TPP sprint-4 family: those required observation-hash collisions to fire; FFTV fires on dense channels (step-penalty, validity) from episode 1, which is the exact gap FED's failure ruled out.
- **Primitive count.** One primitive (IQR of T_m per (state-hash, action, channel) cell), one improvement operator (Pareto-front logit nudge). No stack.
- **Side-information channels.** Explicitly declared: vector diagnostics (`info["vector"]` channel firings) and transition geometry (state-hash for cell indexing). Both are from the canonical list.
- **Evidence quality.** DST: 1382.0 vs random=194.0, strong=285.0 — beats strong by ~4.8x, a large margin. RG: 1.331 vs random=1.331, strong=1.331 — exactly at baseline, no discrimination. Beat strong on 1 vector env (DST), which meets the `alive-promising` threshold. The RG failure is consistent with the hypothesis's own predicted "terminal-only-channel collapse" mode: on RG both the resource and goal channels may not fire until terminal steps, leaving only the step-penalty channel populated — but on DST the treasure channel fires at varying distances from the agent's position, giving the dense channels genuine IQR variation. This asymmetry is informative: the primitive works where multiple channels accumulate non-degenerate IQR, and collapses where terminal-only channels dominate.
- **Failure-mode informativeness.** The RG collapse is partially predicted by the hypothesis and does not refute the mechanism — it identifies the channel-density precondition. A next run should instrument per-channel IQR fill-rates on both envs to confirm whether the RG collapse is the predicted bootstrap failure or something deeper.

## Lesson for the next iteration

FFTV is alive on DST with a strong margin; the next run should check whether the RG collapse is the predicted terminal-channel bootstrap failure (per-channel IQR ablation should confirm) and, if so, whether adding a short-episode success-seeking warm-start fills the RG precious channel early enough for the primitive to engage.
