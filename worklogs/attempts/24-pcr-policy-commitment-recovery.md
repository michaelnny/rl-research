---
id: 24
slug: pcr-policy-commitment-recovery
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Novel reward-independent primitive (step-lag recovery vector) but terminal-outcome gate silenced operator before any reward — bootstrap wall confirmed."
side_information: [transition geometry, vector diagnostics]
nearest_prior: "15 (FED family — terminal-outcome-gated operator)"
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random: 0
  hard_n_beat_strong: 0
  commit: 87567995096d38cce2d6eb898faea556ddaf8ecd
---

# 24 — PCR (Policy Commitment Recovery)

## One-sentence idea

Measure how many steps it takes the policy's own modal action to be re-confirmed after each realized action (a reward-independent "self-disruption" vector), then nudge logits toward actions whose recovery vectors are Pareto-non-dominated, gated by whether those actions also yield Pareto-better terminal vector outcomes.

## Core primitive

Per-(context-cluster, action) commitment-recovery vector `R[c,a] ∈ R^L`, where component `R[c,a]_ℓ` is the expected step-lag until the snapshot policy's modal action at the observed state is re-confirmed at alignment threshold ℓ, measured along realized trajectories after acting at cluster c with action a. Clusters are coarse logit-sign-pattern buckets; L thresholds include top-1 match, top-2 set match, top-K set match, and logit-cosine threshold. The primitive is reward-independent: it is computed entirely from the policy's own action-logit sequences, which are free at training time.

## Improvement operator

At each cluster c, compute the recovery-Pareto-non-dominated set P(c) of actions (actions whose R[c,a] row is not dominated by any other action's recovery vector). Apply a small fixed logit nudge toward actions in P(c) and away from complement actions — but **only** at clusters where the terminal vector outcomes of trajectories that visited c and took a ∈ P(c) are themselves Pareto-non-dominated relative to trajectories that took a' ∉ P(c). The terminal outcome enters exclusively as a binary sign gate (Pareto-better → nudge up, Pareto-worse → nudge down, incomparable → no update); it is never used as a magnitude.

## Why it looked promising

- The recovery primitive R fires on every step regardless of reward, sidestepping the bootstrap wall's sample-mass requirement.
- Transition geometry (action-logit sequences) is a genuinely new side-information channel, free at training time, distinct from all prior sprint-4 primitives.
- The binary Pareto-meet sign gate avoids scalar weighting and critic-supplied magnitudes, keeping the candidate structurally outside the actor-critic disqualifier family.
- The hypothesis explicitly identified the bootstrap-wall risk for the gate (falsifier condition b) and argued it was avoided because R fires without reward — the argument was logically sound but practically incorrect.

## What was tested

Stage: core (MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0, deep-sea-treasure-concave-v0, resource-gathering-v0). Budget 120 s/env. Single run, 0 retries. Commit 87567995096d38cce2d6eb898faea556ddaf8ecd.

Scores: DoorKey 0.0 (random 0.137), KeyCorridor 0.0 (random 0.0), DST 99.0 (random 194.0, strong 285.0), RG 0.011 (random 1.331). beat_random = 0, beat_strong = 0.

## Why it failed

The improvement operator requires the terminal-outcome Pareto-dominance sign gate to fire before any logit update can occur. On all four envs, the gate was silent throughout the 120 s budget: DoorKey and KeyCorridor never yielded rewarded trajectories, so no cluster accumulated two action-distinct trajectories with differing terminal Pareto outcomes; DST performed below random (99.0 vs 194.0), suggesting the same silence. The falsifier condition (b) — operator fires on < 5% of decision steps — was confirmed. The recovery primitive R accumulated values, but without the gate, the accumulated R signal had no path to influence the policy. This is structurally the same failure as FED (#15), CEC (#18), and TPP (#21): the improvement operator is silent until terminal outcomes arrive, which is precisely the bootstrap wall. The key distinction the hypothesis claimed — that R is reward-independent — was factually correct but irrelevant: R being non-trivial does not help if the gate controlling the operator's direction is outcome-gated.

This failure also rules out the general sub-family: reward-independent primitive + terminal-outcome-gated operator. The gate must be replaced with a direction signal that fires before reward appears.

## Lesson / constraint added

A reward-independent primitive cannot escape the bootstrap wall if the improvement operator's direction assignment is still gated by terminal outcomes — the gate is the bottleneck, not the primitive's sample requirements.

## Nearest neighbors in the literature

- **PEO (#12):** PCR shares the policy-centric framing but differs in using passive step-lag measurement rather than counterfactual edits; failure was different (PEO reduced to scalar ES, PCR hit the gate bootstrap wall).
- **FED (#15) / CEC (#18) / TPP (#21):** Same bootstrap wall — gate/comparator requires terminal outcomes that do not accumulate within 120 s on sparse envs.
- **Intrinsic motivation / empowerment:** Step-lag self-disruption loosely resembles empowerment (capacity to influence future state) but is computed from the policy's own logit trajectory, not from mutual information or reachability.
- **CWAI (alive candidate):** Both use policy-internal signals; CWAI uses a learned transition model's Jacobian, PCR uses logit self-comparisons. Different channel, different failure mode.

## Artifacts

- Hypothesis: `worklogs/runs/20260606-10-auto/hypothesis.md`
- Review: `worklogs/runs/20260606-10-auto/review.md`
- Result: `worklogs/runs/20260606-10-auto/result.json`
- Panel: `worklogs/runs/20260606-10-auto/panel.txt`
- Commit: `87567995096d38cce2d6eb898faea556ddaf8ecd`
