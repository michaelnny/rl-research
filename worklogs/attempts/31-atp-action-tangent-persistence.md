---
id: 31
slug: atp-action-tangent-persistence
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Level-set crossing horizon primitive collapses to shortest-path-to-terminal planner via step-penalty channel dominating the Pareto vote."
side_information: [learned dynamics, vector diagnostics]
nearest_prior: model-based planning (disqualifier family); CHX (17)
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random: null
  hard_n_beat_strong: null
  commit: 65512b562b937499902950bbacb1856af9a22ba9
---

# 31 — ATP (Action-Tangent Persistence)

## One-sentence idea

Train a learned forward model with per-channel firing-probability heads; at each decision step compute the action-conditional k-vector of "persistence horizons" h*[o,a,m] (first step at which predicted channel-m probability crosses a self-tuned threshold); nudge policy logits by Pareto-non-dominance margin over the horizon vectors, never scalarizing.

## Core primitive

For each candidate action a, unroll the forward model f_θ H_max steps using "force a at step 0, then sample π" and read off h*[o,a,m] = min{h : p_m(ô_h) ≥ τ_m} per channel m, where τ_m is the empirical 75th-percentile of p_m over the replay buffer. The result is an integer-valued k-vector. The threshold-crossing operator is a level-set non-linearity — it destroys magnitude information, keeping only the crossing time.

## Improvement operator

At each decision step, compute the Pareto-non-dominance margin δ(o)[a] = n_a − m_a where n_a counts actions whose horizon vector Pareto-dominates h*[o,a,:] (shorter on all channels, strict on ≥ 1) and m_a counts actions it dominates. Apply a logit nudge α·δ(o) and REINFORCE toward the action with the largest margin. No scalar return, no critic, no Bellman backup.

## Why it looked promising

- Theoretically distinct from GVFs/successor features: level-set crossing time is non-linear and magnitude-destroying; cannot be recovered by linear combination of successor features.
- Explicitly designed to fire reward-free from episode step 1 via the forward model (bypasses bootstrap wall).
- Self-tuned threshold τ_m (75th-percentile over replay) intended to absorb the stochastic-transition noise floor that kills CWAI/JFP.
- Reviewer assigned novel-direction; all candidate-shape slots were filled substantively.
- Side-information channel was clean: {learned dynamics, vector diagnostics}.

## What was tested

Vector stage only (2 envs): deep-sea-treasure-concave-v0 and resource-gathering-v0. Budget 120 s/env. One retry (fix-1: syntax fix — variable scope for delta/a_star). Final scores: DST 99.0 vs random 194.0, RG 0.011 vs random 1.331. beat_random=0, beat_strong=0. Commit 65512b562b937499902950bbacb1856af9a22ba9.

## Why it failed

The step-penalty channel (universal, fires every step) dominates the Pareto vote. For any action a, h*[o,a,step-penalty] ≈ (number of steps until the episode ends), which means "shorter persistence horizon for step-penalty" = "reach termination faster." The Pareto comparison then favors actions that terminate episodes quickly — i.e., near-treasure over far-high-value-treasure on DST — reducing to a shortest-path planner to any terminal state. This explains the DST score of 99.0 (below random 194.0): the agent is actively choosing nearby low-value treasures over distant high-value ones. Meanwhile the treasure/gold/gem channels have h* = H_max for all actions (never crosses τ_m before reward appears), contributing nothing to the vote. This is the CHX/CRP/PICAV collapse pattern extended to forward-model-predicted horizons: any "faster channel onset is better" primitive collapses to terminal-speed optimization when step-penalty is universal and dense.

## Lesson / constraint added

Forward-model primitives with "shorter horizon dominates" semantics cannot distinguish quality-of-terminal from speed-to-terminal when a universal step-penalty channel is present — the step-penalty h* encodes time-to-any-terminal, making the Pareto vote a shortest-path-to-termination operator. Rules out the entire "action-conditional time-to-first-channel-event as Pareto comparison key" family on the current substrate.

## Nearest neighbors in the literature

- Model-based planning / shortest-path (disqualifier): h*[o,a,step-penalty] encodes time-to-terminal, Pareto-min over this is shortest-path planning.
- CHX (17): within-trajectory signal-geometry collapse via dominant step-penalty channel — same family of defeat.
- CRP (22): rank-position collapse on terminal-only channels — same structural point from a different angle.
- JFP (12): argmax of Jacobian norm over a forward rollout — also magnitude-based horizon selection defeated by underfitting on terminal-only channels.

## Artifacts

- Hypothesis: worklogs/runs/20260606-20-auto/hypothesis.md
- Review: worklogs/runs/20260606-20-auto/review.md
- Train: worklogs/runs/20260606-20-auto/train.py
- Result: worklogs/runs/20260606-20-auto/result.json
- Panel: worklogs/runs/20260606-20-auto/panel.txt
- Fix: worklogs/runs/20260606-20-auto/fix-1.md
- Commit: 65512b562b937499902950bbacb1856af9a22ba9
