---
id: 39
slug: blic-block-lookback-imminence-concordance
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Offline cluster-conditioned imminence-shift tensor reduces to scalar step-penalty minimization on terminal-only-reward substrates; scored 0.0/0.011 vs random 194.0/1.331."
side_information: [vector diagnostics, learned dynamics]
nearest_prior: cid-channel-imminence-differential
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: e98b0be56f51b7542bdc90d8cf64f2165e10e1a3
---

# 39 — BLIC: Block-Lookback Imminence Concordance

## One-sentence idea

Accumulate per-(cluster, action, channel) running empirical means of one-step
imminence-shift δ_m = q_m(o_{t+1}) − q_m(o_t) on replay transitions, then
apply a Pareto-non-dominance count logit nudge over the k-vector IC[s,a,:] at
each decision step — no decision-time forward model, no critic, no Bellman.

## Core primitive

The per-(cluster, action, channel) running empirical mean
`IC[s, a, m] = E[ q_m(o_{t+1}) − q_m(o_t) | cluster(o_t)=s, a_t=a ]`,
where q_m is a small supervised binary classifier trained on windowed labels
`y_m(t) = 1 iff channel m fires somewhere in {t+1, …, t+H}`. Cluster identity
comes from online k-means on the policy's penultimate-layer activation. The
primitive is the IC tensor itself, not q_m. Unlike CID (the nearest prior),
δ_m is computed on actually-observed o_{t+1} from replay, not on
forward-model counterfactuals.

## Improvement operator

At each decision step in cluster s, compute for each action a the signed
Pareto-non-dominance margin `m_a = n_a^dom − n_a^sub` over the k-vector
IC[s, a, :] vs. IC[s, a', :] for all a' ≠ a; apply logit nudge
`Δlogit(a) = α · m_a`. Then sample from softmax(logits + Δlogit).

## Why it looked promising

- Structurally distinct from CID: eliminates the decision-time forward model
  that caused CID's action-invariant LR rows.
- Step-penalty channel fires every step, providing non-trivial IC entries from
  episode 1 — predicted to keep the Pareto operator active before any rare
  reward channel fires.
- Probability-difference (not log-ratio) is numerically stable when q_m ≈ 0
  throughout the bootstrap window (terminal-only channels).
- Reviewer passed as novel-direction; falsifiers were explicitly stated and
  testable.
- Hypothesis explicitly noted the rebadge boundary for k=1 (scalar collapse)
  and argued k≥2 substrate would avoid it.

## What was tested

Stage: vector. Envs: deep-sea-treasure-concave-v0, resource-gathering-v0.
Time budget: 120 s. Seeds: default (run once, n_retries=0).
DST: 0.0 vs random 194.0 vs strong 285.0 — beat_random=0, beat_strong=0.
RG: 0.011 vs random 1.331 vs strong 1.331 — beat_random=0, beat_strong=0.
Commit: e98b0be56f51b7542bdc90d8cf64f2165e10e1a3.

## Why it failed

The step-penalty channel dominates the Pareto vote because terminal-only
reward channels (DST treasure, RG reward) yield δ_m ≈ 0 throughout the
bootstrap window. The k-vector IC[s,a,:] has only the step-penalty dimension
carrying non-trivial signal, reducing the effective rank to 1. A single-channel
Pareto comparison degenerates to scalar step-penalty minimization — the same
mechanism that killed ATP (#31), PRAR (#36), and TCP (#23). The hypothesis
predicted this as falsifier (a); it was confirmed. Scores were indistinguishable
from the bootstrap-wall floor (0.0 / 0.011) shared by FED, PICAV, ACCD, and
most prior sprint-4 candidates.

The key insight: the prediction that "step-penalty channel provides a non-trivial
nudge from episode 1" is correct, but a non-trivial single-channel nudge IS
scalar step-penalty minimization. The Pareto operator is not multi-dimensional
if only one dimension is populated.

## Lesson / constraint added

Any primitive that uses the step-penalty channel as the bootstrap anchor for a
Pareto vote on terminal-only-reward substrates reduces to scalar step-penalty
minimization regardless of how δ_m is computed; future candidates must either
tolerate operator silence until first reward fires, or provide an independent
exploration primitive that densifies non-terminal reward signal before the
Pareto comparison is activated.

## Nearest neighbors in the literature

- GVFs / successor features: q_m is a fixed-horizon GVF on a binary cumulant
  (component-level overlap; operator is distinct — no linear scalarization).
- Multi-objective RL / Pareto Q-learning: Pareto partial order on per-channel
  values, but those use discounted value functions not one-step imminence shifts.
- CID (#cid-channel-imminence-differential): direct parent; BLIC eliminates
  CID's decision-time forward model but inherits the step-penalty collapse.
- ATP (#31): same ATP/PRAR failure mode — step-penalty-dominant Pareto vote.

## Artifacts

- `worklogs/runs/20260606-32-auto/train.py`
- `worklogs/runs/20260606-32-auto/hypothesis.md`
- `worklogs/runs/20260606-32-auto/review.md`
- `worklogs/runs/20260606-32-auto/result.json`
- `worklogs/runs/20260606-32-auto/panel.txt`
- Commit: e98b0be56f51b7542bdc90d8cf64f2165e10e1a3
