---
id: 35
slug: phi-path-homology-invariants
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Lévy-area geometry fires from step 1, but linear regression onto terminal cumulants re-introduces the FED-family bootstrap wall on all envs."
side_information: [transition geometry, vector diagnostics]
nearest_prior: "24 (PCR) / FED-family bootstrap-wall"
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 8b8408ecbad260410cb5b4152dc57c0af1106dbd
---

# 35 — PHI (Path-Homology Invariants)

## One-sentence idea

Use the antisymmetric Lévy-area (order-2 path-signature matrix) in observation-embedding space, projected per-channel through k independent linear regressors trained against terminal vector cumulants, as a reward-independent geometric primitive for a Pareto-non-dominance action-bias.

## Core primitive

For each trajectory τ, embed observations via a frozen random projection `e(o_t) ∈ R^d` and accumulate the order-2 path signature: `A(τ)_{ij} = (1/2) Σ_t (e_i(o_t)·e_j(o_{t+1}) − e_j(o_t)·e_i(o_{t+1}))`. Per (state-cluster s, action a), maintain the empirical mean of per-step area increments `Δa_{ij}(t) = (1/2)(e_i(o_t)e_j(o_{t+1}) − e_j(o_t)e_i(o_{t+1}))` conditioned on `a_t=a` and `cluster(o_t)=s`. For each channel m, fit a linear regressor `P_m: R^{d×d} → R` from `A(τ)` to the terminal cumulant `c_T(τ)[m]`. The per-action k-vector `g[s,a,m] = P_m(E[Δa(t)|s,a])` is the primitive output.

## Improvement operator

Logit nudge `Δℓ(s,a) = α(n_a^{dom}(s) − n_a^{sub}(s))` where `n_a^{dom}` counts actions whose g-row is strictly Pareto-dominated by action a in the coordinate-wise partial order on R^k, and `n_a^{sub}` is the reverse count. No scalar aggregation of channels at any step.

## Why it looked promising

- The Lévy-area is genuinely non-zero from step 1, episode 1 — it does not require any reward to be observed before the primitive produces a non-trivial signal.
- The structural distinction from PFA (#29) was well-articulated: PFA used firing-probability space (near-zero on always-on and terminal-only channels), while PHI uses observation-embedding space.
- The structural distinction from CWAI was articulated: Jacobian column norms shrink under stochastic transitions, while path-integral averaging grows more stable as the buffer grows.
- The k independent regressors avoid scalarization — each channel has its own projector and contributes separately to the Pareto vote.

## What was tested

Stage: core (DoorKey-8x8, KeyCorridor, deep-sea-treasure-concave-v0, resource-gathering-v0), 120 s budget, 4 workers.
- DoorKey-8x8: 0.0 vs random 0.137 (below random)
- KeyCorridor: 0.0 vs random 0.0
- DST: 99.0 vs random 194.0 (below random by 49%)
- RG: 0.011 vs random 1.331 (below random by 99%)
beat_random=0, beat_strong=0. Wallclock: 116.7 s. Commit: 8b8408ecbad260410cb5b4152dc57c0af1106dbd.

## Why it failed

The geometric primitive (Lévy-area) is reward-independent, but the **channel-projection layer** (linear regressors `P_m` fit against terminal cumulants) is not. On DST and RG, the reward/treasure channel is terminal-only, so `P_m` for that channel sees near-zero targets throughout training (sparse terminal events), yielding near-zero R² and near-zero coefficients. The only non-degenerate regressor is for the step-penalty channel, which fires every step; its coefficients encode "trajectories with larger step-penalty-channel cumulant magnitude have a particular area signature" — i.e., shorter episodes. The operator then reduces to preferring actions that lead to shorter episodes, which is actively harmful on DST (choosing the nearest low-value treasure over the far high-value one). On DoorKey and KeyCorridor, no rewarded trajectories arrive within 120 s, leaving all k regressors with noise coefficients.

This is structurally the same as PCR (#24): "a reward-independent primitive coupled to a reward-gated learned projection layer inherits the FED-family bootstrap collapse." The independence property must extend through the **entire pipeline** — including any learned mapping from the geometric feature to channel scores — not just the raw feature computation stage.

## Lesson / constraint added

A geometric or structural trajectory primitive that is reward-independent at the feature level is not sufficient to escape the bootstrap wall if it requires a supervised regression onto terminal channel cumulants to produce an action-discriminating signal; the regression layer must be either reward-free or seeded by a non-terminal information source.

## Nearest neighbors in the literature

- **Rough path theory / path signatures** (Lyons 1998, Chevyrev & Kormilitzin 2016): PHI uses the order-2 signature (Lévy area) as an RL primitive — novel application domain but same mathematical object.
- **PFA (#29 Per-Channel Phase-Flow Asymmetry):** uses antisymmetric area in firing-probability space rather than embedding space; collapses on terminal-only channels for the same structural reason PHI's regressor collapses.
- **PCR (#24 Policy Commitment Recovery):** a reward-independent primitive (commitment-recovery vector) coupled to a terminal-outcome-gated operator — same two-stage failure pattern as PHI.
- **CWAI (alive-promising, Channel-Wise Action Influence):** uses Jacobian column norms of a learned forward model; PHI's path-integral is more stable under stochastic transitions but shares CWAI's reliance on learned channel-to-feature mapping.

## Artifacts

- Hypothesis: `worklogs/runs/20260606-27-auto/hypothesis.md`
- Review: `worklogs/runs/20260606-27-auto/review.md`
- Result: `worklogs/runs/20260606-27-auto/result.json`
- Curator: `worklogs/runs/20260606-27-auto/curator.md`
- Commit: 8b8408ecbad260410cb5b4152dc57c0af1106dbd
