---
id: 30
slug: cpr-channel-posterior-chebyshev-reweight
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Chebyshev sup-norm projection collapses to step-penalty single-channel cloning when reward-channel posteriors are degenerate throughout bootstrap window."
side_information: [vector diagnostics]
nearest_prior: scalar-weighted-log-prob (step-penalty channel domination)
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: 1d7074bf72b0bfdde2a4baacd22f9320232d1fdd
---

# 30 — CPR (Channel-Posterior Chebyshev Reweight)

## One-sentence idea

Maintain k per-channel replay-reweighted empirical action posteriors `π̂_m(a|o)` and update the policy by minimizing `sup_m KL(π̂_m || π_θ)` — a Chebyshev (worst-channel) projection onto the joint constraint set defined by all channel posteriors — with no scalar channel weight and no fixed convex combination.

## Core primitive

The k-tuple of per-channel empirical action posteriors `π̂_m(a|o) ∝ Σ_{(o_t,a_t)} w_m(t) · 1[a_t=a]` where `w_m(t) = ReLU((Δc_t[m] − μ_m)/σ_m)` is the channel-standardized positive part of the per-step vector increment and `c(o)` is a coarse observation cluster (hash or k-means). Each `π̂_m` is the distribution over actions whose mass concentrates on actions empirically co-occurring with channel `m` firing; the primitive is all k simultaneously.

## Improvement operator

At each gradient step, sample a batch of observations from replay, compute per-observation `m*(o) = argmax_m KL(π̂_m(·|o) || π_θ(·|o))`, and take a gradient step on `KL(π̂_{m*}(·|o) || π_θ(·|o))`. Equivalently: project `π_θ` toward the Chebyshev center of the k empirical posteriors — the unique policy minimizing worst-case channel deviation. No scalar channel weight; the argmax switches `m*` per observation per step.

## Why it looked promising

- The Chebyshev sup-norm aggregation is mathematically non-equivalent to `wᵀr` for any fixed or learned `w` — a Chebyshev center of k distributions is a non-linear functional that cannot be expressed as a linear combination, providing genuine structural distinction from PPO/REINFORCE/GRPO.
- The reviewer verdict was `novel-direction` with a detailed argument for why none of the Sprint-4 disqualifiers apply.
- Side information is vector diagnostics exclusively via `info["vector"]` — the canonical channel for the vector panel envs.
- Step-penalty channel fires immediately from episode start, promising non-degenerate posteriors from the first trajectory — avoiding the pure bootstrap wall of FED/CEC/TPP.
- Monotonic improvement claim is concrete: `sup_m KL(π̂_m || π_θ)` is convex in log-policy space (sup of convex functions), so projected gradient descent is non-increasing per step under fixed posteriors.

## What was tested

Stage: vector (DST + RG). Budget: 120 s. n_retries: 0. Scores: DST 99.0 vs random 194.0 (below random), RG 0.011 vs random 1.331 (below random). beat_random=0, beat_strong=0. Commit: 1d7074bf72b0bfdde2a4baacd22f9320232d1fdd.

## Why it failed

The hypothesis's own falsifier §8a (single-channel domination) was confirmed. On both vector panel envs, the reward/treasure channel posterior `π̂_treasure` is degenerate — effectively empty — throughout the bootstrap window because the treasure channel fires only at the terminal step, and no rewarded trajectories are collected early enough to seed the posterior. With k-1 channels degenerate, `sup_m KL(π̂_m || π_θ)` reduces to the single non-degenerate channel (step-penalty only), making every gradient step a single-channel weighted cloning update — effectively scalar log-prob update weighted by step-penalty magnitude. The Chebyshev structure is invisible when only one posterior has mass. DST's non-zero score (99.0) is consistent with step-penalty avoidance driving short-episode preference, not treasure-seeking. RG's near-zero score (0.011) confirms the same pattern on a second vector env.

This extends the bootstrap-wall ruling (FED #15, CEC #18, PCR #24, TRAC #26) to the per-channel posterior construction family: the aggregation operator (Pareto front, sup-norm, rank-position) does not matter when the channel posteriors themselves are degenerate. The structural distinction from `wᵀr` exists only when all k posteriors are simultaneously non-degenerate.

## Lesson / constraint added

Any primitive built on per-channel empirical posteriors requires all k channels to have non-degenerate posteriors simultaneously; on terminal-only-reward substrates, reward-channel posteriors are degenerate until the first rewarded trajectory appears — ruling out the entire "per-channel posterior + any max/sup aggregation" family without a paired bootstrap mechanism that populates reward-channel posteriors before terminal reward fires.

## Nearest neighbors in the literature

- **REINFORCE / PPO / GRPO** — scalar-weighted log-prob; CPR reduces to these on the single step-penalty channel when reward posteriors are degenerate.
- **Distributionally Robust Optimization (DRO)** — min-max optimization over a set of distributions; CPR's Chebyshev projection is a DRO instance with the constraint set defined by k empirical posteriors.
- **Multi-objective Policy Gradient (Pareto-front variants)** — CPR's Chebyshev aggregation is a specific scalarization of multi-objective policy gradient, though without fixed weights.
- **Maximum Entropy RL / KL-projection methods** — KL minimization from a target distribution is a standard entropy-regularized formulation; CPR's novelty was the per-channel target, not the KL distance itself.

## Artifacts

- `worklogs/runs/20260606-19-auto/train.py`
- `worklogs/runs/20260606-19-auto/result.json`
- `worklogs/runs/20260606-19-auto/panel.txt`
- `worklogs/runs/20260606-19-auto/hypothesis.md`
- `worklogs/runs/20260606-19-auto/review.md`
