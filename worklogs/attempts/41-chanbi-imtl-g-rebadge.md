---
id: 41
slug: chanbi-imtl-g-rebadge
status: rebadge
sprint: 2026-06-06
verdict_in_one_line: "Angular bisector of per-channel unit-normalized score-function gradients is algebraically identical to IMTL-G (Liu et al., ICLR 2021) for K=2."
nearest_dead_family: I
mode: probe-v1
candidate_json:
  update_family: per-channel-unit-normalized-gradient-aggregation
  memory: null
  nearest_disqualifier: "Existing methods may appear as components; they cannot be the explanation for why the method works."
  ablation_plan: "unnormalized per-channel sum (recover REINFORCE on uniform-weight scalarized return)"
panel_evidence:
  stage: null
  envs: []
  beat_random: 0
  beat_strong: 0
  scores: {}
  ablation_scores: {}
  ablation_delta: {}
  confirmation_seeds: []
  commit: 6402d4df616f4b02b51ffa69ef0c688dd5daea40
---

# 41 - CHANBI (Channel-Bisecting Score-Function Direction)

## One-sentence idea

Update the policy along the angular bisector `d* = (u_1 + u_2)/||u_1 + u_2||` of per-channel unit-normalized score-function gradients, claiming rescaling-invariance (equal projection onto each channel direction) as the load-bearing novel property.

## Core primitive

The per-channel unit-normalized score-function gradient pair `(u_1, u_2)` where `u_c = g^c / ||g^c||` and `g^c = sum_t G^c_t grad_theta log pi(a_t|s_t)`. The spherical sum `d* = (u_1 + u_2)/||u_1 + u_2||` is claimed to be the unique direction invariant under componentwise positive rescaling of per-channel rewards.

## Improvement operator

`theta <- theta + alpha * (sum_c ||g^c||) * d*` where `d*` is the unit angular bisector of the active per-channel unit-normalized gradients.

## Why it looked promising

- Rescaling-invariance is a genuine structural property: `d*` is unchanged under `r^c <- alpha_c r^c` for `alpha_c > 0`.
- The `cos_d_uniform` discriminator would be measurably non-trivial on DST-concave within 30 episodes, giving a fast mechanism-presence test.
- Distinct from MGDA (which uses raw `g^c` and is scale-sensitive) and from scalarization (which operates on reward space, not gradient space).
- Clean ablation: skip per-channel normalization and recover REINFORCE on uniform-weight scalarized return.

## What was tested

Run 20260606-24-auto. No panel run; reviewer-rejected before compute. Stage null.

## Why it failed

Rebadge of IMTL-G (Liu et al., "Towards Impartial Multi-Task Learning," ICLR 2021). For K=2, the equal-projection constraint `d . u_1 = d . u_2` has a unique solution direction proportional to `(u_1 + u_2)` — CHANBI's d* exactly. IMTL-G's defining property is precisely the rescaling-invariance / equal-projection claim CHANBI attributed to itself as novelty. Applying the IMTL-G aggregator to RL per-channel score-function gradients is a substrate choice, not a new mechanism. Family I.

## Lesson / constraint added

Any proposal whose central novelty is "equal projection / rescaling-invariance across vector-reward channels in parameter gradient space" is IMTL-G applied to RL and must be rejected.

## Nearest neighbors in the literature

- IMTL-G (Liu et al., ICLR 2021): exact direction match for K=2; defines the equal-projection property.
- MGDA (Desideri 2012): min-norm point in convex hull of raw gradients; scale-sensitive, not an angular bisector.
- Nash-MTL (Navon 2022): geometric-mean inner products; scale-sensitive.
- CAGrad (Liu 2021): constrained QP near average gradient; scale-dependent constraint.

## Artifacts

Run directory: `worklogs/runs/20260606-24-auto/`
Commit: 6402d4df616f4b02b51ffa69ef0c688dd5daea40
