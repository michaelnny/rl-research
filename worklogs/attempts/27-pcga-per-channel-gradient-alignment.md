---
id: 27
slug: pcga-per-channel-gradient-alignment
status: failed
sprint: 2026-06-05
verdict_in_one_line: "Parameter-space gradient cosine alignment collapsed to action-invariant rows on a shared trunk; Pareto vote produced symmetric nudges, scoring below random on both vector envs."
side_information: [learned dynamics, vector diagnostics]
nearest_prior: actor-critic disqualifier family
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: e082a87d46e58738c959b46b7e61e57884f5f098
---

# 27 — PCGA: Per-Channel Gradient Alignment

## One-sentence idea

At each decision step, compute the cosine similarity in parameter space between `∇_θ log π(a|s)` and each auxiliary cumulant-head gradient `∇_θ ĉ_m(s)` to form an `|A|×k` alignment matrix, then nudge action logits by the coordinate-wise Pareto-dominance vote over that matrix — no scalar collapse, no use of the head's output value.

## Core primitive

`A[s, a, m] = cos(∇_θ log π(a|s), ∇_θ ĉ_m(s))` — the cosine similarity in full parameter space between the direction the policy moves when it commits to action `a` at state `s`, and the direction the auxiliary cumulant-prediction head moves when predicting channel `m`'s cumulant at `s`. This produces a real-valued `|A|×k` matrix per state, computed by `|A|·k` backward passes through the shared trunk.

## Improvement operator

For action `a`, count `n_a = #{a' ≠ a : A[s,a,:] ≻ A[s,a',:]}` (actions `a` Pareto-dominates) and `m_a = #{a' ≠ a : A[s,a',:] ≻ A[s,a,:]}` (actions that dominate `a`). Logit nudge: `Δlogit(a|s) = α·(n_a − m_a)`. The auxiliary head is trained by MSE regression on observed cumulant suffixes from a replay buffer, in a separate optimizer step.

## Why it looked promising

- Claimed to bypass the bootstrap wall: the step-penalty channel fires on every step, so the auxiliary head always has a loss and its gradient is non-degenerate from step 1.
- The head's output magnitude is never consumed by the policy update — only gradient direction, making it structurally distinct from actor-critic.
- Pareto vote over `R^k` avoids scalarization; channels are never summed or weighted.
- Predicted failure modes were concrete and testable (variance of A across actions < 0.1 → degenerate family).
- Parameter-space alignment avoids the forward-model rollout variance that killed CWAI on stochastic envs.

## What was tested

Vector stage only: `deep-sea-treasure-concave-v0` and `resource-gathering-v0`, 120 s per env, 0 retries. Scores: DST=99.0 (random=194.0, strong=285.0), RG=0.011 (random=1.331, strong=1.331). Both envs below random. Commit: `e082a87d46e58738c959b46b7e61e57884f5f098`.

## Why it failed

The shared trunk between the policy network and auxiliary cumulant head causes both `∇_θ log π(a|s)` and `∇_θ ĉ_m(s)` to be dominated by the same large set of trunk parameters. The action-discriminating information lives almost entirely in the final output layer (a small fraction of total parameters), while the trunk layers (large shared weight matrices) contribute nearly identical gradient components to every action's log-prob gradient. As a result, `A[s,a,m]` is nearly action-invariant — the alignment reflects "how much the trunk is moving in the direction of channel `m`'s prediction" uniformly across actions, not "which action's commitment direction aligns best with channel `m`." The Pareto vote over nearly-uniform rows produces near-symmetric `(n_a − m_a)` scores, equivalent to random perturbation of logits. This is the hypothesis's own falsifier (a): per-step variance of A across actions < 0.1. The below-random scores confirm the saturation.

An additional contributor on DST: the step-penalty and treasure gradients are nearly collinear in the terminal-only reward setting (failure mode c), so even if the alignment were action-discriminating, the Pareto order would collapse to a total order on one channel (scalarized vector reward disqualifier).

## Lesson / constraint added

Parameter-space gradient alignment via a shared-trunk architecture is structurally degenerate: the shared trunk dominates both gradient vectors, washing out the action-level discrimination that the primitive requires. Future candidates using gradient-direction alignment must either (a) operate on action-specific parameter subsets (final output layer only) or (b) apply a projection step that removes the shared trunk component from both gradients before computing cosines. Option (a) reduces to output-space Jacobian alignment (similar to CWAI); option (b) is a new mechanism that needs its own structural justification.

## Nearest neighbors in the literature

- **Actor-critic (disqualifier):** The auxiliary head gradient direction is conceptually analogous to a critic's gradient — both are "what direction should the network move to improve some objective at this state." PCGA avoids using the head's *value* as a weight, but the gradient direction plays the role of "advantage signal direction."
- **PCGrad / gradient surgery (multi-task learning):** Projects conflicting gradients to resolve task interference; PCGA uses cosine between task-specific gradients as the alignment signal rather than conflict resolution.
- **Successor features / GVFs:** The auxiliary cumulant head is functionally a GVF head; PCGA's use of its gradient direction rather than its value output is the structural distinction, but the head itself is within the GVF family.
- **CWAI (alive candidate):** CWAI uses output-space Jacobian; PCGA uses parameter-space cosine. CWAI's failure on stochastic envs (Jacobian noise floor) does not transfer, but PCGA's failure mode (trunk uniformity) is more fundamental.

## Artifacts

- `worklogs/runs/20260606-15-auto/train.py`
- `worklogs/runs/20260606-15-auto/result.json`
- `worklogs/runs/20260606-15-auto/panel.txt`
- `worklogs/runs/20260606-15-auto/hypothesis.md`
- `worklogs/runs/20260606-15-auto/review.md`
- Commit: `e082a87d46e58738c959b46b7e61e57884f5f098`
