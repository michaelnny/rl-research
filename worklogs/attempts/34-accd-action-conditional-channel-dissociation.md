---
id: 34
slug: accd-action-conditional-channel-dissociation
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Probabilistic-shift classifier primitive silenced by bootstrap wall (saturated + terminal-only channels) on both vector envs; 0.0/0.001 vs random 194.0/1.331."
side_information: [vector diagnostics, learned dynamics]
nearest_prior: "29-pfa / FED-family bootstrap wall"
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: 818e993eb9a5aab8c8f5f7278ae1508ff092fdcc
---

# 34 — ACCD (Action-Conditional Channel Dissociation)

## One-sentence idea

Train two small offline classifiers on the replay buffer — one action-marginal, one action-conditional — and use the signed per-channel probability shift `S_m = ψ_a − ψ_full` as the primitive for a Pareto-non-dominance logit nudge with no scalar collapse.

## Core primitive

For each (observation, action, channel) triple, compute `S_m[o_t, a] = ψ_a(o_t)[m] − ψ_full(o_t)[m]` where `ψ_full` predicts `P(v_m fires in next H steps | o_t)` and `ψ_a` predicts the same conditioned on taking action `a`. Both heads are supervised by binary cross-entropy on observed H-step channel-firing indicators from the replay buffer. The primitive is a signed probability difference — not a gradient norm, not a value, not a return.

## Improvement operator

For each observation, compute the k-vector `S[o_t, a, :]` for each action. Apply a k_eff gate: exclude channels whose `max_a S_m − min_a S_m` is below a noise floor (prevents Pareto saturation when channels are uninformative). Then compute `n_a^{dom} − n_a^{sub}` (Pareto dominance score) and nudge logits by `α · (n_a^{dom} − n_a^{sub}) / |A|`. Update the base policy by behavioral cloning toward the nudged-logit target via cross-entropy; update the ψ heads by BCE on observed labels.

## Why it looked promising

- Structurally distinct from CWAI: couples to categorical posterior shift, not local linearization of a deterministic predictor; sign of shift is invariant under proportional attenuation under stochastic transitions.
- Reviewer gave `novel-direction` with no structural objections.
- Hypothesis correctly predicted failure mode (b) for the substrate but argued graceful degradation to base exploration rather than wrong-direction harm.
- Clean one-primitive + one-operator design; no component stacking.
- Side-information channels (vector diagnostics, learned dynamics) are both cleanly declared and non-trivially used.

## What was tested

Stage: `vector` (DST and RG), 120 s budget, 2 workers.
- `deep-sea-treasure-concave-v0`: score = 0.0 vs random = 194.0, strong = 285.0
- `resource-gathering-v0`: score = 0.001 vs random = 1.331, strong = 1.331
- beat_random = 0, beat_strong = 0
Commit: `818e993eb9a5aab8c8f5f7278ae1508ff092fdcc`, run_id: `20260606-26-auto`.

## Why it failed

The failure mode (b) predicted in the hypothesis confirmed in full: on DST, the step-penalty channel has `P_a ≈ P_full ≈ 1` (always fires, S ≈ 0), and the treasure channel has `P_a ≈ P_full ≈ 0` before the first rewarded trajectory (terminal-only, never seeded in classifier). On RG, same structure. With k_eff = 0 throughout the bootstrap window, the Pareto operator is silent and the policy remains in base-exploration mode. The "probability-shift survives stochasticity better than Jacobian norm" claim could not be tested because the substrate's channel structure (saturated + terminal-only) was structurally hostile before any stochasticity comparison could register. This is the FED-family bootstrap wall, extended to the classifier-based probability-shift primitive.

## Lesson / constraint added

Offline-classifier probability-shift primitives inherit the FED bootstrap wall when all vector channels are either always-saturated (S ≈ 0) or terminal-only (S ≈ 0 before first reward); the constraint is substrate-side-information starvation, not implementation quality. Future candidates must either fire on every step from non-reward channels that are neither saturated nor terminal-only, or include an explicit exploration primitive that seeds channel-firing classifiers before reward appears.

## Nearest neighbors in the literature

- **CWAI (attempt 27, alive-promising)**: nearest structural neighbor; CWAI uses Jacobian column-norm on a deterministic forward model, ACCD uses posterior-shift between two classifiers. Distinct objects, same substrate failure mode.
- **PFA (attempt 29)**: two-horizon probability heads with a rotational primitive; also collapsed to zero signal on saturated + terminal-only channels.
- **FED (attempt 15)**: canonical bootstrap-wall entry; ACCD is the classifier-probability-shift variant of the same failure.

## Artifacts

- Hypothesis: `worklogs/runs/20260606-26-auto/hypothesis.md`
- Review: `worklogs/runs/20260606-26-auto/review.md`
- Result: `worklogs/runs/20260606-26-auto/result.json`
- Panel: `worklogs/runs/20260606-26-auto/panel.txt`
- Curator: `worklogs/runs/20260606-26-auto/curator.md`
- Commit: `818e993eb9a5aab8c8f5f7278ae1508ff092fdcc`
