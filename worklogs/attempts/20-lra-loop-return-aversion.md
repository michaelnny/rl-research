---
id: 20
slug: lra-loop-return-aversion
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Pareto-vs-zero loop suppression collapses to count-suppression on panel vector envs because universal step-penalty channel makes every loop trivially dominated by zero."
side_information: [transition geometry, vector diagnostics]
nearest_prior: count-based-exploration
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: dd3a8340af2238556c0522c5e4e6e875cbc0ad37
---

# 20 — LRA: Loop-Return Aversion

## One-sentence idea

Track the per-channel cumulant delta accumulated along within-episode closed loops (the agent returns to the same observation hash), and suppress any action whose loop-signature mean is Pareto-dominated by the zero vector — a "certified wasted-motion" detector requiring no critic, no cross-trajectory comparison, and no scalar collapse.

## Core primitive

`L[s, a] ∈ ℝ^k` — the empirical running mean of per-channel cumulant deltas `Δc = c_{t'} − c_t` over all closed within-episode loops entered by action `a` from observation-hash state `s`. A loop is defined as: the agent takes action `a` at `s`, follows some path, and later in the same episode reaches state `s'` with `obs_hash(s') = obs_hash(s)`. The segment cumulant `Δc` is the per-channel integral of `info["vector"]` over the bracketed steps. Count `n[s,a]` tracks loop witnesses per bucket; suppression requires `n[s,a] ≥ n_min`.

## Improvement operator

Single logit-suppression rule: for each action `a` at state `s`, if `n[s,a] ≥ n_min` and `L[s,a]` is Pareto-dominated by the zero vector (all channels `≤ 0`, at least one `< 0`), subtract `α` from `logit(a)`; otherwise do nothing. The dominance test is a sign-of-coordinate check against the fixed reference zero — no scalar weight `w`, no cross-trajectory matching, no bootstrapped target.

## Why it looked promising

- The intra-trajectory loop is a self-contained natural experiment: start and end states have identical observation hash by construction, so `Δc` is a provable measure of what the detour cost across all vector channels.
- Pareto-vs-zero is a conservative filter: actions that lose one channel but gain another are NOT suppressed, preserving exploration of tradeoffs.
- Requires only one loop witness per `(s,a)` to potentially fire (vs. FED/CEC which needed many samples per bucket for Pareto-front estimation).
- No cross-trajectory matching (unlike RSD) and no novelty bonus (unlike BCE-v0) — the loop event itself is the dominance precondition.
- The hypothesis correctly identified its own failure modes and falsifiers, including the step-penalty collapse as a first-class risk.

## What was tested

Stage: core (sparse + vector envs). Budget: 120 s/env. Envs: MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0, deep-sea-treasure-concave-v0, resource-gathering-v0. Commit: dd3a8340af2238556c0522c5e4e6e875cbc0ad37. n_retries: 0.

Scores:
- DoorKey-8x8: 0.000 vs random=0.137 (below random)
- KeyCorridorS3R3: 0.000 vs random=0.000 (tied)
- deep-sea-treasure-concave: 99.000 vs random=194.000 (below random)
- resource-gathering: 0.121 vs random=1.331 (below random)

beat_random=0, beat_strong=0.

## Why it failed

Two structural failure modes, both predicted by the hypothesis, both confirmed:

1. **Step-penalty channel collapse (vector envs).** Both DST and RG have a step-penalty channel that is strictly negative on every step. For any intra-trajectory loop, `Δc[step_channel] < 0` always. This means `L[s,a]` has at least one strictly negative channel for every `(s,a)` that closes a loop. If no other channel is strictly positive (which is the typical case before terminal reward fires), `L[s,a]` is Pareto-dominated by zero for all looping actions — the suppression mask fires unconditionally on all looping actions. This is structurally identical to "suppress actions that cause repeated observations," which is count-based exploration suppression (a named disqualifier). The hypothesis declared "if every vector env requires excluding the step-penalty channel, the family is dead" — and both vector envs required it.

2. **Hash-collision sparsity (MiniGrid).** On DoorKey-8x8, picking up the key changes the observation (partial observability), so the agent's observation hash changes on nearly every step after key pickup. Intra-trajectory hash collisions are rare, meaning `n[s,a]` almost never reaches `n_min` for most `(s,a)` buckets and the operator almost never fires. The agent effectively runs as a baseline with no LRA signal. This is consistent with the predicted failure mode #1 from the hypothesis.

The combined result is an operator that either fires too aggressively (collapsing to count-suppression) or almost never fires, on all four panel envs.

## Lesson / constraint added

Any intra-trajectory loop primitive that uses Pareto-vs-zero dominance to suppress actions is a count-suppression rebadge on the panel's vector envs (DST and RG) because the universal step-penalty channel makes every within-episode loop trivially dominated by zero. The family requires either: (a) panel envs with no universal-step-penalty channel (structural invariant of the env, not a tuned hyperparameter), or (b) a different dominance reference that accounts for the expected per-step cost of traversal (e.g., Pareto-dominance vs. the expected per-step background rather than vs. zero).

## Nearest neighbors in the literature

- **Count-based exploration** (the collapse target): suppresses repeated observations; LRA reduces to this when step-penalty channel dominates.
- **05-bce-v0**: used returnability as a novelty/frontier bonus (ablated to count); LRA used returnability as a dominance precondition, not a bonus — but the distinction did not survive the step-penalty collapse.
- **19-cwtp-confluence-witness-trajectory-pairs**: cross-trajectory variant that also required hash collisions to bootstrap; LRA avoided the cross-trajectory requirement but hit a different structural wall.
- **Potential-based reward shaping**: can remove step penalties from consideration by defining a potential function — the analog in LRA would be subtracting the expected per-step background from `Δc` before the Pareto test.

## Artifacts

- `worklogs/runs/20260606-04-auto/hypothesis.md`
- `worklogs/runs/20260606-04-auto/review.md`
- `worklogs/runs/20260606-04-auto/result.json`
- `worklogs/runs/20260606-04-auto/panel.txt`
- commit: dd3a8340af2238556c0522c5e4e6e875cbc0ad37
