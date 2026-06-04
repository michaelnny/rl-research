---
id: 03
slug: kernel-rl
status: failed
sprint: 2026-05-24
verdict_in_one_line: "Passive reward-support mining beats REINFORCE/CEM on a custom favorable benchmark; collapses to zero on DeepSea where reward correlations don't exist yet."
side_information: [event traces, environment instrumentation]
nearest_prior: "GVFs / successor features (passive prediction families)"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 03 — KERNEL-RL / RSK (Reward-Support Kernel Conditioning)

## One-sentence idea

Passively mine compact behavior atoms whose presence statistically
separates reward-bearing experience from non-reward-bearing experience,
then condition the policy on the stable atoms while leaving everything
else exploratory.

## Core primitive

For a behavior atom `z ∈ Z` (context-action, chunk, skill, tool call,
code edit, plan step, latent primitive) and a reward event `e`, the
**reward-support contrast**:

\[
\Delta_e(z) = \mathbb E[\psi_e(\tau)\mid z\in\tau] - \mathbb E[\psi_e(\tau)\mid z\notin\tau].
\]

For terminal-only reward, `ψ(τ) = R(τ) − R̄_batch`. The reward-support
kernel is

\[
K_e = \{ z : \operatorname{LCB}(\Delta_e(z)) > \lambda \ \wedge\ z\text{ confirms across independent batches}\}.
\]

## Improvement operator

Condition the policy on the kernel:

\[
\pi_{k+1} = \arg\min_\pi D_{\mathrm{KL}}(\pi\,\|\,\pi_k)
\quad\text{s.t.}\quad \Pr_\pi(z\mid c_z) \ge 1-\epsilon\ \ \forall z\in K,
\]

leaving the policy unchanged where no kernel atom applies. No scalar-
weighted log-prob update; no elite cloning.

## Why it looked promising

- Sparse-reward native: it triggers only when reward-correlated atoms appear.
- Cheap and passive: no extra environment trials.
- Independent-batch confirmation prevents over-eager promotion.

## What was tested

Terminal-only sparse sequence task with hidden length-8 chunks. Score is
the count of exactly correct chunks. Horizons `H=512` and `H=1024`.

| Task | Method | Seeds | Success | Median solve evals | Mean final score |
|---|---|---:|---:|---:|---:|
| H=512 | KERNEL-RL | 3 | 1.00 | 12,288 | 64/64 |
| H=512 | REINFORCE | 3 | 1.00 | 114,688 | 64/64 |
| H=512 | CEM | 3 | 0.00 | — | 53.3/64 |
| H=1024 | KERNEL-RL | 3 | 1.00 | 12,288 | 128/128 |
| H=1024 | REINFORCE | 3 | 0.00 | — | 118/128 |
| H=1024 | CEM | 3 | 0.00 | — | 79.7/128 |

DeepSea probe (where reward correlation does not exist until a deep
unrewarded path is intentionally traversed):

| N | Random | Q-learning | RSK-context | Frontier-Graph |
|---:|---:|---:|---:|---:|
| 12 | 0/5 | 0/5 | 0/5 | 5/5 |
| 20 | 0/5 | 0/5 | 0/5 | 5/5 |
| 30 | 0/5 | 0/5 | 0/5 | 5/5 |

## Why it failed

The validation was unfair — the chunk task was decomposable in exactly the
way the algorithm needed. The DeepSea probe exposed the underlying
weakness: passive association mining has nothing to mine when no atom
is yet correlated with reward. This is the canonical **"primitive needs
reward correlation to bootstrap, but reward correlation does not exist on
long-horizon sparse tasks"** failure mode.

## Lesson / constraint added

Association mining is not enough. Long-horizon sparse reward needs a
mechanism for **intentionally reaching new temporally deep experience
before reward correlations exist**.

## Nearest neighbors in the literature

- GVFs / successor features (Sutton et al. 2011, Barreto et al. 2017) —
  passive prediction families with the same "compute statistics, condition
  policy on them" shape.
- Implicit-Q-learning style filtering of high-advantage actions.
- Diversity-via-statistics (e.g. DIAYN-without-novelty-bonus
  variants).

## Artifacts

- `kernel_rl_prototype.py`
- `kernel_rl_summary.csv`
- `kernel_rl_learning_curve_H512.png`, `kernel_rl_learning_curve_H1024.png`
- `deepsea_probe.py`, `deepsea_probe_summary.csv`, `deepsea_probe_plot.png`
