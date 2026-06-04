---
id: 01
slug: frost
status: failed
sprint: 2026-05-24
verdict_in_one_line: "Vector-feasibility repair beats scalar repair on a constructed safety task, but is not reward-native."
side_information: [vector diagnostics, environment instrumentation]
nearest_prior: "constrained MDPs / safe-RL repair"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 01 — FROST (Feasible Residual Operators for Sequential Trajectories)

## One-sentence idea

Use structured per-channel vector feedback to compute a minimal behavior
repair that drives active defects to zero while preserving protected
constraints, then distill the repaired trajectory back into the policy.

## Core primitive

A **vector repair certificate** over a rollout τ:

\[
C(\tau) = \{ d(\tau),\, G(\tau),\, \mathcal E,\, \mathcal K \}
\]

where the defect vector is `d(τ) = [F(τ) − b]_+` against per-channel
budgets `b`, and `G_{i,t} ≈ ∂F_i(τ)/∂e_t` is the local influence of editing
behavior element `e_t` on channel `i`. Channels include success, safety,
energy, latency, smoothness, validity, etc.

## Improvement operator

Given active violated channels `A`, protected channels `P`, editable
behavior elements `e_t`:

\[
\Delta e^* = \arg\min_{\Delta e} \|L\Delta e\|^2
\quad\text{s.t.}\quad
G_A \Delta e \le -\rho d_A,\ \ G_P \Delta e \le 0,\ \
\Delta e \in \mathcal E,\ \ \|\Delta e_t\| \le \epsilon_t.
\]

Then distill: `θ ← argmin_θ Σ_t D(π_θ(h_t),\, e_t + Δe^*_t)`. The policy is
not updated by reward-weighted log-probability or by value backup; it is
updated toward a vector-feasible repair of its own failed behavior.

## Why it looked promising

- Native consumption of vector feedback without scalarization to `wᵀr`.
- Per-channel projection separates safety from goal trade-offs.
- No critic, no Bellman backup, no policy-gradient log-prob update.

## What was tested

2-D point-robot reaching task with circular obstacle, energy budget,
smoothness budget, horizon `H=96`. Event-local vector feedback for safety.

| Method | Success rate | Median iterations | Mean final dist | Mean safety defect | Mean energy defect |
|---|---:|---:|---:|---:|---:|
| FROST-event policy | 1.00 | 86 | 0.136 | 0.000 | 0.000 |
| FROST-event plan-only | 0.67 | 12 | 0.148 | 0.00027 | 0.000 |
| Scalar repair, safety-heavy | 0.00 | 70 | 0.304 | 0.000 | 0.385 |
| Scalar repair, goal-heavy | 0.00 | 45 | 0.147 | 0.000 | 0.951 |
| REINFORCE scalar | 0.00 | 50 | 1.274 | 0.118 | 0.000 |

## Why it failed

FROST is not reward-native. It assumes both structured per-channel feedback
and local repair-influence estimates `∂F_i/∂e_t`. On terminal-only reward
or in a black-box environment without per-channel diagnostics, the
primitive has nothing to operate on. It is closer to constrained
repair / control than to general sparse-reward RL.

This hits the cross-attempt mode "the primitive needs reward correlation
to bootstrap, but reward correlation does not exist on long-horizon
sparse tasks until a deep unrewarded path is traversed" — except FROST
substitutes vector defects for reward correlation, and pays the cost of
assuming dense per-channel signals.

## Lesson / constraint added

Native vector feedback is valuable, but **repair cannot be the core
identity** of the family. The next candidate must start from
trial-and-error reward maximization, not from externally supplied
constraint defects.

## Nearest neighbors in the literature

- Constrained MDPs / Lagrangian-projected policy optimization (Achiam et
  al. 2017, "Constrained Policy Optimization").
- Control-barrier-function safe RL (Cheng et al., Wabersich et al.).
- Predictive shielding.
- Safety-Gymnasium-style cost-channel methods.

The repair-projection viewpoint overlaps materially with these once
"defect" is read as "constraint violation."

## Artifacts

- `frost_event_prototype.py` — event-local repair prototype
- `frost_event_summary.csv`
- `frost_event_trajectory.png`
