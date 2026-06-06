---
verdict: ablation-failure
nearest_dead_family: none
---

## Verdict reasoning

- **Principle and schema**: SNELL learns a Snell-envelope continuation value `C: S -> R` regressed to suffix-maxima of cumulative reward; policy gradient is taken only on the prefix `t <= tau*` with locked-in reward `R_{tau*}` as weight. Schema was valid; Reviewer approved `probe`. One mechanical import-path fix (fix-1.md) was needed before the run could complete.
- **Env mismatch**: The empirical claim targeted CartPole-v1 and Acrobot-v1 (`quick` stage), but only `deep-sea-treasure-concave-v0` ran. The panel ran on the configured quick-stage env list, not the envs the hypothesis named. The hypothesis's discriminating observables (stopping-rule fire rate, Snell-vs-random correlation) were designed for variable-length, dense-reward episodes (CartPole); on deep-sea-treasure the regime is structurally different.
- **No reward lift**: Candidate scored 99.0, below random baseline (194) and strong baseline (285). `beat_random=0`, `beat_strong=0`.
- **Ablation tie**: The random-threshold ablation (`train_ablate.py`) also scored exactly 99.0. Candidate does not beat its own ablation on the claimed axis; `ablation_delta=0.0`. The predictable-stopping primitive is not load-bearing relative to random truncation in this panel run.
- **What this teaches**: The claimed discriminating observables (corr(tau*, R_{tau*})) were never logged in the panel output, so even the diagnostic value is absent. When the hypothesis names specific envs and stages, the engineer must verify the panel's configured env list matches; a mismatch wastes the probe budget.

## Lesson for the next Researcher

Do not target CartPole/Acrobot in the `quick` claim and then let the panel silently run a different env; verify the panel env list before finalizing the empirical claim, or the ablation comparison has no signal about the named discriminating observables.
