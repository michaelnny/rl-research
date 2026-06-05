---
verdict: failed-structural
nearest_prior_or_disqualifier: PICAV (#16) / CRP (#22) — terminal-only-channel collapse family
side_information: [vector diagnostics, learned dynamics]
---

## Verdict reasoning

- **Structural distinction claim holds in principle, but collapses on the substrate.** PFA's signed cross-product `A_m = p_m(o_t)·q_m(o_{t+1}) − p_m(o_{t+1})·q_m(o_t)` is genuinely bilinear and rotational — not reachable by scalar renaming of CID-canonical. However, on the substrate's terminal-only-reward vector envs the primitive is identically near-zero: for always-firing channels (step-penalty, `p_m ≈ q_m ≈ 1`) the area is `1·1 − 1·1 ≈ 0`; for terminal-only channels (treasure/goal reward, `p_m, q_m ≈ 0` throughout) the area is also ≈ 0. The rotational invariant requires the two-horizon imminence vector to rotate (curve) rather than stay pinned near a fixed point, which is impossible for channels that are either always-on or always-off.
- **One primitive + one operator — structure is clean.** The hypothesis is not a stack and the Pareto vote is a single well-defined operator. The failure is not architectural complexity.
- **Evidence: scores identical to FED-family bootstrap-wall pattern.** DST: 0.0 vs random 194.0; RG: 0.011 vs random 1.331. This is the same signature as FED (#15), PICAV (#16), CEC (#18), ARP (#28), and CRP (#22) — all of which scored 0.0 / 0.011 on the same two envs. The hypothesis's own failure mode (c) partially predicted this: "terminal-only channels make `p_m, q_m` indistinguishable." The additional failure for always-firing channels (`p_m ≈ q_m`) was listed as failure mode (b) but not fully resolved — the step-penalty rescue argument was optimistic; the signed area is zero precisely because `p ≈ q ≈ 1` in that regime.
- **Failure mode rules out a family.** Any two-horizon signed-area primitive with per-channel probability heads trained on binary firing indicators will produce near-zero signed area on any substrate where each vector channel is either always-firing (p ≈ q ≈ 1) or never-firing-until-terminal (p ≈ q ≈ 0). The rotational primitive requires intermediate and diverging horizon regimes to produce non-trivial curvature. This extends the CRP (#22) ruling ("temporal rank position is degenerate for terminal-only channels") to two-horizon probability heads.

## Lesson for the next iteration

Any primitive that depends on non-trivial divergence between short-horizon and long-horizon statistics of a channel is structurally inert on substrates where channels are either always-firing or terminal-only — the two most common channel types on the current panel envs.
