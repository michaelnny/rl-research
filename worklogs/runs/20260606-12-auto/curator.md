---
verdict: alive-weak
nearest_prior_or_disqualifier: cwai-channel-wise-action-influence
side_information: [learned dynamics, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction:** JFP measures the per-(action, channel) argmax-timestep of the gradient norm of a learned cumulant head along an H-step counterfactual rollout — timing of peak sensitivity, not magnitude. CWAI uses 1-step Jacobian column-norm magnitude. These are structurally orthogonal (amplitude vs. phase, 1-step vs. unrolled). No disqualifier family mapping was established by the reviewer or is apparent from the mechanism.
- **Primitive vs. stack:** One primitive (firing-phase vector J(s,a) ∈ R^k) + one improvement operator (Pareto-non-dominance logit nudge over timing vectors). This meets the single-primitive requirement.
- **Evidence quality:** beat_random=0, beat_strong=0. DST scored 99.0 vs random 194.0; RG scored 0.011 vs random 1.331. Both below random. The failure pattern is consistent with the hypothesis's own predicted failure mode (a): cumulant head underfit on terminal-only channels (treasure in DST, gold/gem in RG) within the 120 s budget, causing t_m^* to be an argmax over near-zero gradient noise rather than a genuine phase signal. This is a learning-speed / budget failure, not a structural collapse into a disqualifier family.
- **Failure-mode informativeness:** The failure rules out the specific hyperparameter regime (default learning rates, H=8, no terminal-channel loss weighting) but does not rule out the JFP family. Unlike CHX/PICAV/TCP which collapsed due to fundamental channel-geometry issues (singleton residual, within-trajectory rank degeneration), JFP's forward model is supervised on dense channels (step-penalty fires every step) and the cumulant head failure is correctable by increasing terminal-channel loss weight, using a higher LR for the cumulant head, or increasing H.

## Lesson for the next iteration

A retry of JFP should weight the cumulant-head supervised loss more heavily on terminal-only channels (e.g. 10x weight on treasure/gold/gem channels) and log per-channel Jacobian norms early in training to verify the cumulant head is actually learning before the phase signal is trusted.
