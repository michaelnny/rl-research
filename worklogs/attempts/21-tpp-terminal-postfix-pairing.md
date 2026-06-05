---
id: 21
slug: tpp-terminal-postfix-pairing
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Bootstrap wall: terminal-observation-hash collisions never accumulate within 120 s; W stayed empty, operator never fired; 0.0 / 0.011 vs random 194.0 / 1.331."
side_information: [vector diagnostics, transition geometry]
nearest_prior: "18 (CEC) / FED-CEC bootstrap-wall family"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 04f0c6d3e9e01db8835dca71f84afff82e8f53ab
---

# 21 — TPP (Terminal-Postfix Pairing)

## One-sentence idea

For every pair of completed trajectories whose terminal observation-hashes match, walk both backward in lockstep to find the "postfix-divergence anchor" (last state before the backward-walked sequences first disagreed), accumulate a Pareto-vote count W[s,a], and nudge policy logits toward the W-plurality action at each anchor using terminal vector outcomes.

## Core primitive

A trajectory `τ = (s_0, a_0, ..., s_T)` is paired with `τ'` iff `hash(s_T) = hash(s'_{T'})`. Walking both backward from termination, the **postfix-shared depth** `Δ` is the largest integer such that `hash(s_{T-δ}) = hash(s'_{T'-δ})` for all `0 ≤ δ ≤ Δ`. The **postfix-divergence anchor** is `(s_{T-Δ-1}, a_{T-Δ-1})` — the last action before the backward-walked sequences first disagreed. For each qualifying pair, one action (from the Pareto-non-dominated side via terminal vector `c_T ∈ ℝ^k`) earns a vote: `W[s, a] += 1`. Pairs with Pareto-incomparable terminal vectors abstain.

## Improvement operator

At every observation `s` with non-zero `W[s, ·]`, nudge `logit(a | s) += α · (W[s,a] − max_{a'≠a} W[s,a'])`. Equivalently, KL-project the policy at `s` toward a softmax of `W[s, ·]`. No scalar reward weighting; no Bellman backup; no critic.

## Why it looked promising

- Structurally inverted from DPC: anchors at the terminal and walks backward, so the post-anchor walk is identical in length on both sides by construction — removing DPC's unbounded-suffix variance.
- Terminal observations in gridworld envs are shared across many trajectories (goal cell is the same), so collision rates should be higher than mid-trajectory rates — reasonable falsifier threshold was 5%.
- Vector feedback consumed natively (Pareto comparison of `c_T`, no scalarization).
- Zero additional env interactions per update (pair-comparison is compute-only on the existing buffer).
- The hypothesis correctly articulated its own falsifier: "below 5% terminal-hash collision rate → dead."

## What was tested

Vector stage only: deep-sea-treasure-concave-v0 and resource-gathering-v0, 120 s budget each. Run 20260606-05-auto, commit 04f0c6d3e9e01db8835dca71f84afff82e8f53ab. Scores: DST 0.0 (random 194.0, strong 285.0); RG 0.011 (random 1.331, strong 1.331). Beat random: 0/2. Beat strong: 0/2. No diagnostic logging of terminal-hash collision rates was present in panel.txt.

## Why it failed

The primitive is silent until terminal-observation-hash collisions accumulate in the pair buffer. On long-horizon sparse envs within a 120 s budget, this threshold is never reached: DST requires reaching a treasure cell (multiple steps), and partial-obs trajectory tails vary by facing direction / local context; RG has randomized gem placement making terminal observations unique per episode. The operator W stayed empty, producing no policy nudges. Score pattern is identical to CEC (#18), CWTP (#19), and the FED family. This is the bootstrap-wall cross-attempt failure mode from `prior_attempts.md §cross-attempt failure modes`.

The structural distinction from CEC (exit-hash bucketed cumulant multisets) is genuine — TPP requires matching terminal observations AND a backward lockstep walk — but both require the same prerequisite: sufficient hash-collision coverage before any signal fires.

## Lesson / constraint added

Any hash-collision-gated pair primitive — whether gated on mid-trajectory states (FED/CWTP), exit-hash buckets (CEC), or terminal-observation hashes (TPP) — fails on the substrate's long-horizon sparse envs without a paired exploration primitive that drives coverage before the gate fires. The "terminal-observation as postfix anchor" sub-family is now closed.

## Nearest neighbors in the literature

- **HER (Hindsight Experience Replay):** uses terminal state as a virtual goal; TPP uses it as a collision key for pair selection — different mechanism, same data structure risk.
- **Goal-conditioned BC / GCSL:** imitates trajectories that reached a goal; TPP's anchor selection is similar to identifying "trajectories that reached the same terminal" but uses the backward walk for divergence, not imitation.
- **Successor Representations / GVFs:** index future from a current state; TPP indexes backward from a terminal — different direction, same collision/coverage dependency.
- **CEC (#18):** nearest failed prior; exits-hash bucketing vs terminal-observation matching — structurally distinct but same failure mode.

## Artifacts

- Hypothesis: `worklogs/runs/20260606-05-auto/hypothesis.md`
- Review: `worklogs/runs/20260606-05-auto/review.md`
- Result: `worklogs/runs/20260606-05-auto/result.json`
- Train script: `worklogs/runs/20260606-05-auto/train.py`
- Commit: 04f0c6d3e9e01db8835dca71f84afff82e8f53ab
