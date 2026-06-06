---
verdict: ablation-failure
nearest_dead_family: none
---

## Verdict reasoning

- **Principle and schema.** EFLO proposes using the GAE-style exponentially-averaged forward policy-entropy residual `A_t^H(lambda)` as the sole credit-assignment weight on the score-function gradient, with no reward read. The primitive is coherent, novel, and schema-valid; the Reviewer correctly issued a `probe` verdict.

- **Tertiary falsifier fired on both arms.** The hypothesis defined a clean self-consistency check: EFLO's mean H_t should remain in `[0.5, log(4)]` because the update direction ascends the forward-state-conditional entropy functional, while the c_t=1 ablation should collapse toward 0 (trajectory log-likelihood ascent). Instead, EFLO's mean H_t collapsed identically to the ablation: both reach ~0.0006 by episode 9000 and flat-line near zero. The entropy-flow-ascent principle was not self-consistent on this substrate.

- **Score-axis outcome.** Ablation scored 99.0 on DST-concave; candidate scored 0.0. `ablation_delta = -99.0`. The ablation's log-likelihood ascent, despite collapsing entropy, incidentally exploited the environment's reward structure (locking onto a short rewarding trajectory early), while EFLO's score-function weights driven by near-zero entropy residuals produced an effectively random or degenerate gradient direction that never found reward. The ablation beat the candidate by 99 points.

- **What went wrong mechanistically.** The hypothesis predicted that `A_t^H(lambda)` would be non-trivial (std > 0) from the first episode because per-state entropy varies at random init. What appears to have happened instead: the early entropy collapse (visible at episode 200–400, H_t already dropping sharply in EFLO) suggests that EFLO's score-function gradient, weighted by entropy-flow residuals from early training, actually *reinforces* the actions that decrease entropy fastest — i.e., the update direction inadvertently drives the policy toward lower-entropy states rather than higher ones. The stop-gradient on `c_t` (proof debt item 2) means the update cannot detect this self-defeating dynamic.

- **Lesson.** The EFLO update `g = sum_t A_t^H(lambda) * grad log pi(a_t|s_t)` does not maximize `J^H` in practice because the stop-gradient form creates a fixed-point mismatch: steps with large *positive* entropy-flow (actions leading to higher-entropy next states) and steps with large *negative* entropy-flow are both upweighted in the gradient, but the relative signs conspire with the policy gradient direction in a way that collapses rather than preserves entropy. Without a correction term for the stop-gradient bias (proof debt item 2), the claimed entropy-ascent direction is not the actual gradient direction.

## Lesson for the next Researcher

Do not retry reward-free entropy-flow primitives that use GAE-style stop-gradient weighting on the score-function without first resolving whether the stop-gradient form is a valid pseudo-gradient for the claimed entropy functional — the stop-gradient bias can flip the effective optimization direction, causing entropy collapse in both arms.
