---
id: 29
slug: pfa-per-channel-phase-flow-asymmetry
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Signed 2-D phase-area of per-channel imminence trajectory is near-zero on all substrate channels — always-firing gives p≈q≈1 (area 0), terminal-only gives p≈q≈0 (area 0)."
side_information: [vector diagnostics, learned dynamics]
nearest_prior: "PICAV (#16), CRP (#22) — terminal-only-channel collapse family"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 7567ee602c46749471ec039c860ff30f41f82149
---

# 29 — PFA (Per-Channel Phase-Flow Asymmetry)

## One-sentence idea

Represent each vector channel's imminence trajectory as a 2-D point `(p_m, q_m)` (short-horizon and long-horizon firing probabilities), compute the signed phase-area swept per transition as a rotational primitive, and use a Pareto-non-dominance logit nudge over the per-(cluster,action) running mean of this k-vector.

## Core primitive

Per-(cluster, action, channel) running-mean signed 2-D phase-area:
`Ā[c, a, m] = E[p_m(o_t)·q_m(o_{t+1}) − p_m(o_{t+1})·q_m(o_t)]`
where `p_m(o) ≈ P(channel m fires at next step | o)` and `q_m(o) ≈ P(channel m fires within H steps | o)` are learned from the replay buffer via BCE. The cross-product form is the z-component of the 2-D imminence vector's rotation — a bilinear coupling of the two horizons that is invariant to independent positive-monotone reparameterizations of `p_m` and `q_m`.

## Improvement operator

At each decision step, build the per-action matrix `M ∈ R^{|A|×k}` from `Ā[c(o), :, :]`. Compute coordinate-wise Pareto-non-dominance: for action a, `n_a^{dom}` = count of actions whose row is dominated by a's row, `m_a^{dom}` = count of actions dominating a's row. Add `α·(n_a^{dom} − m_a^{dom})` to action a's pre-softmax logit. No scalar collapse.

## Why it looked promising

- The rotational (cross-product) form is genuinely bilinear — not reachable by scalar renaming of CID-canonical's log-likelihood-ratio.
- Training the two-horizon heads on the step-penalty channel was argued to rescue the bootstrap wall: the step-penalty fires on every step, so `p_m, q_m` receive dense supervision from the first episode.
- The reviewer confirmed novel-direction: structural distinction from all Sprint 4 candidates and named disqualifiers.
- The Pareto vote over the k-dimensional signed-area row is scalar-collapse-free and coordinate-wise, satisfying the vector-env requirement.
- The hypothesis included five concrete predicted failure modes with testable diagnostics.

## What was tested

Vector stage only (DST + RG), 120 s budget, 2 workers. Run `20260606-18-auto`.
- DST: 0.0 vs random 194.0 / strong 285.0
- RG: 0.011 vs random 1.331 / strong 1.331
Commit: `7567ee602c46749471ec039c860ff30f41f82149`

## Why it failed

The signed-area primitive requires non-trivial divergence between `p_m` and `q_m` — i.e., the short-horizon and long-horizon imminence vectors must curve in different directions during a transition. On the substrate's vector channels this condition is never met:
- **Always-firing channels (step-penalty):** `p_m ≈ q_m ≈ 1.0` everywhere; the cross-product `1·1 − 1·1 = 0`.
- **Terminal-only channels (treasure/goal reward):** `p_m ≈ q_m ≈ 0` throughout the episode (the channel is nearly never firing); the cross-product `0·0 − 0·0 = 0`.

Both substrate vector envs (DST, RG) use exactly this two-channel structure (step-penalty always-on + terminal reward), so `Ā[c, a, :]` is near-zero for all (cluster, action) cells and the Pareto vote is symmetric — the operator is effectively silent. This is the hypothesis's own failure modes (b) and (c) in combination.

The rescue argument (step-penalty providing dense supervision to separate `p_m` from `q_m`) was flawed: dense supervision drives `p_m → 1` and `q_m → 1` for an always-firing channel, eliminating the divergence required for non-zero area, not creating it.

Cross-attempt failure mode: extends CRP (#22)'s ruling ("temporal rank position is degenerate for terminal-only channels") to two-horizon probability heads. The pattern generalizes: any primitive whose signal requires within-episode variation in a channel's firing statistics is inert on channels that have constant firing behavior (always-on or always-off-until-terminal).

## Lesson / constraint added

Any primitive that requires short-vs-long-horizon divergence in channel firing probability is structurally silent on the current substrate — the substrate's channels are either always-firing (p≈q≈1) or terminal-only (p≈q≈0). Future candidates must either use channel types with intermediate firing rates (neither always-on nor terminal-only), or design primitives that remain informative on degenerate (0,0) and (1,1) imminence pairs.

## Nearest neighbors in the literature

- **CID-canonical** (parked `failed-implementation`, `worklogs/candidates/cid-channel-imminence-differential.md`): 1-D translational signal `LR_m = log q_m(o_{t+1}) − log q_m(o_t)`. PFA is the 2-D rotational analogue; same heads, cross-product instead of difference. Both fail identically on the substrate.
- **GVF / Successor Features:** accumulate discounted cumulants; PFA's heads are bounded firing probabilities and the operator consumes only the signed area on the observed transition, not the head's expected-value output. Structurally distinct but fails for independent reasons.
- **CWAI (Jacobian magnitude):** uses parameter-space gradient; PFA uses classifier outputs only.
- **CRP (#22):** rank-percentile of firing magnitude, also degenerate for terminal-only channels. Same failure family.

## Artifacts

- `worklogs/runs/20260606-18-auto/train.py`
- `worklogs/runs/20260606-18-auto/hypothesis.md`
- `worklogs/runs/20260606-18-auto/review.md`
- `worklogs/runs/20260606-18-auto/result.json`
- `worklogs/runs/20260606-18-auto/panel.txt`
