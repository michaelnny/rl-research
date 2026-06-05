---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-24-PCR / bootstrap-wall family
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** ACFC is genuinely structurally distinct from DPC (no state-hash index, whole-episode action-frequency × channel-firing-count concordance matrix rather than per-state sign-vote tensor). However, the practical failure mode is the same bootstrap wall: the concordance matrix C[a,m] contains only step-penalty signal before any rewarded trajectory appears, because goal/treasure/gem channels fire zero times in failed episodes. The mechanism's "dense from episode 1" claim holds only for the step-penalty channel, making the Pareto-non-dominance operator over C reduce to step-penalty-maximization (fastest-termination preference) — a named scalarization rebadge — before any success is observed. This is structurally the same collapse as PCR (#24), where a reward-independent primitive paired with a signal that only becomes discriminating after a rewarded trajectory is equivalent to the FED-family bootstrap wall.
- **Primitive vs stack:** One primitive (concordance matrix C[a,m]) plus one improvement operator (Pareto-non-dominance logit bias). Clean, not a stack. The failure is not architectural.
- **Evidence quality:** Beat random on 0 of 4 envs, beat strong on 0 of 4 envs. DoorKey and KeyCorridor scored 0.0 (below random of 0.137 and 0.0 respectively). DST scored 99.0 vs random 194.0 (near-zero reward, far below random). RG scored 0.011 vs random 1.331. No positive signal anywhere; the hypothesis's own falsifier criterion for operator silence was confirmed on all envs. The failure is not thin evidence of a weak effect — the candidate scored below or at random on every env.
- **Failure mode informativeness:** The failure rules out the entire "whole-episode frequency-histogram concordance as a density-independent side-information channel" family on the current substrate. When terminal-only reward channels dominate k, any cross-episode sign-concordance on channel-count differences degenerates to concordance on step-penalty only (total-steps difference), and the Pareto non-dominance bias reduces to scalar minimization of episode length. This extends the bootstrap-wall ruling beyond state-hash-indexed primitives to any primitive that indexes by whole-episode aggregate statistics when those aggregate statistics are dominated by a universal channel (step-penalty) before the first rewarded trajectory.

## Lesson for the next iteration

ACFC confirms that eliminating the state-hash index does not cure the bootstrap wall — any primitive whose k-channel structure is meaningful only after at least one rewarded trajectory is collected inherits the FED/PCR/CEC bootstrap-wall collapse, regardless of whether it uses a state hash, cluster index, or whole-episode aggregate as its indexing variable.
