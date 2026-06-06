---
verdict: empirical-signal
nearest_dead_family: none
---

## Verdict reasoning

- **Principle and schema.** COPDEV is a per-step score-function policy gradient weighted by the L1 distance between two per-channel marginal empirical CDF ranks of the cumulative-return process. The schema validates, the ablation is structurally clean (d_t=1 recovers REINFORCE-without-baseline), and the claimed primitive does not map to any dead family. Reviewer accepted for compute.

- **Training-dynamics observable confirmed load-bearing.** The primary discriminator `gradnorm_var = Var_t(||g_t||)/Mean_t(||g_t||)^2` shows a >3000x separation: COPDEV mean 0.426 (max 1.96) vs ablation mean 0.000115 (max 1.0) across the full 120s run on DST-concave. The primitive demonstrably alters per-step gradient-norm variance, exactly as predicted and in the direction predicted. This is the first run since the redesign where the mechanistic discriminator fires unambiguously at the training-dynamics level.

- **Score result: random tie, not beat.** COPDEV final score 194.0 = random baseline 194.0 on deep-sea-treasure-concave-v0 (beat_random=0, beat_strong=0). The ablation scored 99.0 — the nearest-treasure sub-random floor, meaning the ablation pathologically collapsed to T=1 episodes with near-zero gradients (~5e-12) late in training while COPDEV maintained longer episodes and matched the random score. The `ablation_delta = +95.0` shows COPDEV dominated its ablation on the score, but neither arm beat the random baseline.

- **Ablation collapse artifact.** The ablation (REINFORCE d_t=1) did not stay near the nearest-treasure floor in the usual sense; instead it collapsed to T=1 episodes with vanishing gradient norms (~5e-12), which is a degenerate optimization failure rather than a policy-gradient-failure-to-find-treasure. This makes the ablation_delta partially an artifact of the ablation's instability, not purely COPDEV's quality. The gradnorm_var separation is still genuine (COPDEV has nonzero per-step weight variance; ablation's gradient magnitude vanished entirely), but the score comparison is confounded.

- **What this iteration teaches.** Quick-stage training-dynamics observables can discriminate the mechanism from its ablation even when neither arm departs the random score floor. The principle of using a logged gradient-norm-variance scalar as the primary discriminator — one that fires at random init, not after treasure discovery — is the right design pattern for the 120s budget. However, matching random is not scoring above random; the next sharper test should ask whether COPDEV can beat random on a quick env or produce a non-trivial score trajectory, which would require confirming the gradnorm_var separation is actually driving useful policy change and not just reflecting longer episode survival.

## Lesson for the next Researcher

The gradnorm_var discriminator pattern works — design the next probe around a mechanism that both (a) produces this kind of provably-non-uniform per-step weight and (b) has a structural reason to drive the policy toward higher-value trajectories, not merely longer ones; COPDEV showed mechanism presence but not yet mechanism usefulness on the score axis.
