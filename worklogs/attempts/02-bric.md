---
id: 02
slug: bric
status: failed
sprint: 2026-05-24
verdict_in_one_line: "Solves long-horizon binary-chunk tasks with terminal reward, but at the cost of expensive counterfactual rollouts and a hand-known edit grammar."
side_information: [reachability/reset structure]
nearest_prior: "Hindsight Experience Replay / counterfactual trajectory editing"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 02 — BRIC (Bracketed Reward-Intervention Control)

## One-sentence idea

Use terminal reward only, but assign credit by testing whether replacing
one segment of an anchor trajectory with a donor segment improves the
terminal outcome — accept the donor segment when it does.

## Core primitive

A **reward-intervention bracket** over an anchor τᵃ, donor τᵇ, editable
interval `I`:

\[
B(\tau^a,\tau^b,I) =
\operatorname{sign}\bigl( R(\tau^a[I\!\leftarrow\!\tau^b_I]) - R(\tau^a) \bigr).
\]

If `R(τᵃ[I←τᵇ_I]) > R(τᵃ)` then `τᵇ_I ≻_{τᵃ,I} τᵃ_I`: the donor segment
dominates the anchor segment in this anchor context.

## Improvement operator

An accepted bracket creates a behavior clause `(c_I, τᵃ_I) → τᵇ_I` where
`c_I` is the local context. The policy update is a minimal projection
onto verified clauses:

\[
\pi_{k+1} = \arg\min_{\pi\in\Pi} D(\pi,\pi_k)
\quad\text{s.t.}\quad
\Pr_\pi(\tau^b_I \mid c_I) \ge 1-\epsilon.
\]

No scalar-weighted log-prob update. Reward determines only whether a
concrete intervention bracket is accepted.

## Why it looked promising

- Terminal-only reward causal credit assignment.
- No value backup; no scalar-weighted log-prob.
- Naturally robust to off-policy / mixed-policy data.

## What was tested

Terminal-only binary-sequence task, hidden length-8 chunks, full episode
reward only when an entire chunk is correct. Horizons `H=128` and `H=256`.

| Task | Method | Seeds | Success | Median solve evals | Mean best reward |
|---|---|---:|---:|---:|---:|
| H=128 | BRIC-seg | 10 | 1.00 | 7,228.5 | 1.000 |
| H=128 | CEM | 10 | 0.00 | — | 0.700 |
| H=128 | REINFORCE | 10 | 0.00 | — | 0.169 |
| H=256 | BRIC-seg | 10 | 1.00 | 15,856.5 | 1.000 |
| H=256 | CEM | 10 | 0.00 | — | 0.644 |
| H=256 | REINFORCE | 10 | 0.00 | — | 0.122 |

A bad-grammar ablation (length-6 segments while true chunks were length-8)
broke BRIC: 0/5 solved, mean best reward 0.283.

## Why it failed

The intervention primitive is elegant in a causal sense, but expensive in
practice: every bracket test requires an extra `R(τᵃ[I←τᵇ_I])` rollout. In
robotics this is a physical replay; in LLM/web agents it is a full tool-
using episode or verifier call. It also depends on knowing the right edit
grammar in advance — the bad-grammar ablation collapses entirely.

## Lesson / constraint added

The next candidate must be **passive** or near-passive: it must extract
credit from ordinary trajectories, not require many counterfactual
environment verifications.

## Nearest neighbors in the literature

- Hindsight Experience Replay (Andrychowicz et al. 2017) — relabels
  trajectories rather than splicing, but shares the "use available
  trajectories more than once" motif.
- Trajectory stitching in offline RL (e.g. Decision Diffuser; stitching
  in CQL/IQL analyses).
- Counterfactual policy evaluation (Thomas & Brunskill 2016).

## Artifacts

- `bric_research_prototype.py` — intervention-bracket prototype + baselines
- `bric_research_results.json`
- `bric_h128_learning_curve.png`
