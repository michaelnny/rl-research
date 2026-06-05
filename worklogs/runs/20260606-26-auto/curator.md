---
verdict: failed-structural
nearest_prior_or_disqualifier: 29-pfa / FED-family bootstrap wall
side_information: [vector diagnostics, learned dynamics]
---

## Verdict reasoning

- **Structural distinction from CWAI is genuine**: ACCD uses signed probability shift (ψ_a − ψ_full) between two supervised classifiers rather than a Jacobian column-norm on a deterministic forward model. The primitive is categorically different under variable renaming. However, the substrate collapse is not mediated by the CWAI failure mode (Jacobian shrinkage under stochasticity) but by the bootstrap wall — so structural novelty vs. CWAI does not save it.
- **Bootstrap wall confirmed on the probability-shift family**: The hypothesis correctly predicted failure mode (b) in full detail. On DST and RG, the step-penalty channel has P_a ≈ P_full ≈ 1 (saturated, S ≈ 0) and the reward/treasure/gold channels have P_a ≈ P_full ≈ 0 before the first rewarded trajectory (terminal-only, S ≈ 0). Both channels contribute zero signal to the Pareto vote throughout the bootstrap window, making k_eff = 0 and the operator equivalent to pure base-policy exploration. Scores: DST = 0.0 vs random 194.0, RG = 0.001 vs random 1.331 — identical failure pattern to FED, CEC, ARP, PFA, and related family members.
- **One primitive + one improvement operator**: The architecture is clean (two classifiers + Pareto-non-dominance logit nudge). No stacking disqualifier applies. The failure is substrate-structural, not design-structural.
- **Evidence quality**: 0/2 envs beat random. Zero signal on the vector stage, which is the minimum gate. No evidence supports the "probability-shift survives stochasticity" hypothesis because the experiment was confounded by the bootstrap wall — the distinguishing condition (stochastic envs with non-terminal channel signal) was never realized within 120 s.
- **Failure-mode informativeness**: Rules out the entire "offline-classifier probability-shift as primitive" family on substrates where all channels are either always-saturated or terminal-only. This is a **new constraint** not previously recorded: even a probabilistic primitive that is theoretically sign-preserving under stochasticity inherits the FED bootstrap wall when channel signal is structurally absent before first reward. The family might survive on environments with per-step non-saturated reward-bearing channels, but the current vector substrate does not provide this.

## Lesson for the next iteration

Any offline-classifier-based primitive (probability-shift, firing-probability difference, or posterior ratio) is rendered silent on the current substrate's bootstrap window by the same mechanism as FED/CEC/ARP; the next candidate must either fire on every step from non-reward signal OR provide its own exploration primitive that seeds the channel-firing classifier before reward appears.
