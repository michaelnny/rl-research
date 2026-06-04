---
id: 04
slug: carl-frontier-graph
status: failed
sprint: 2026-05-24
verdict_in_one_line: "Tabular frontier graph solves DeepSea where Q-learning and KERNEL-RL fail, but is structurally indistinguishable from Go-Explore + count-based exploration."
side_information: [reachability/reset structure, transition geometry]
nearest_prior: "Go-Explore (Ecoffet et al. 2019)"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 04 — CARL / Frontier-Graph (Controllability-First RL)

## One-sentence idea

Before optimizing reward, learn what parts of the environment are
reproducibly reachable; expand the controllable frontier, compress
reliable routes into skills, and attach reward events only after they
are encountered.

## Core primitive

A **controllability cell**:

\[
\mathcal C = \{ (z,\, \pi_z,\, \rho_z) \}
\]

where `z = f_φ(h)` is an abstract reachable situation, `π_z` is a policy
fragment that reliably reaches `z`, and `ρ_z` is a reproducibility
certificate. A cell is accepted only if `P_{π_z}(f_φ(h_t)=z) ≥ 1−ε`.

## Improvement operator

Frontier expansion:

\[
\operatorname{Expand}(\mathcal C) = \arg\max_{z,a}\ \text{frontier-novelty}(T(z,a))
\quad\text{s.t.}\quad z \in \mathcal C.
\]

When reward is observed it marks reachable cells (`z ↦ reward event
record`) rather than becoming a value backup. Routes are composed:
`π_{z_0→z_k} = π_{z_{k-1}→z_k} ∘ ⋯ ∘ π_{z_0→z_1}`.

## Why it looked promising

- Possible complexity shift under a correct compact abstraction:
  graph expansion in `O(|Z||A|)` versus `Ω(|A|^H)` random discovery.
- Strong on DeepSea where everything else fails.

## What was tested

DeepSea probe at depths `N=12, 20, 30`:

| Depth | Frontier-Graph solved | Median solved episode |
|---:|---:|---:|
| 12 | 5/5 | 25 |
| 20 | 5/5 | 41 |
| 30 | 5/5 | 59 |

## Why it failed

Not failed empirically — failed on the **novelty audit**. The raw frontier
graph is too close to existing exploration and model-based-graph-search
families, especially Go-Explore: remember reachable states, return to
promising states, then explore outward. Without a genuinely new
abstraction-learning-and-improvement principle, calling it a new family
would be dishonest.

This is the cross-attempt mode "passing DeepSea-style monotone-progress
benchmarks is not strong evidence" + the disqualifier-family rule
"Go-Explore with renamed cells."

## Lesson / constraint added

The next promising direction is probably controllability-first, but it
must avoid merely reinventing Go-Explore, options, hierarchical RL,
model-based search, or count-based exploration. The new piece must be
the abstraction-learning principle itself.

## Nearest neighbors in the literature

- Go-Explore / Phasic Go-Explore (Ecoffet et al. 2019, 2021).
- Count-based exploration (Bellemare et al. 2016).
- Successor-feature options (Barreto et al. 2017).
- Model-based-graph search / planning-via-learned-graph
  (e.g. Eysenbach et al. SoRB).

## Artifacts

- `deepsea_probe.py`
- `deepsea_probe_summary.csv`
- `deepsea_probe_plot.png`
