---
verdict: probe
reviewer_run: 20260606-15-auto
hypothesis_type: probe
---

## Summary

DUAL-IR is a coherent saddle-point probe built on Brown-Smith-Sun
information-relaxation duality, with a typed martingale-difference
penalty primitive and a non-trivial credit-truncation rule that does not
reduce to REINFORCE on any non-monotone-reward trajectory; approve for
empirical testing on the quick stage.

## Schema check

`candidate.json` contains all required string and boolean fields. The
schema values agree with the prose:

- `principle` reproduces the saddle-point statement of `## Principle`.
- `primitive_name` ("martingale-difference penalty function") and
  `primitive_type` ("S x A x S -> R, conditional-zero-mean") match the
  `## Primitive` block.
- `claimed_stage = quick` matches `stage: quick` in `## Empirical claim`.
- `empirical_claim` and `falsifier` faithfully restate the prose claim
  (CartPole-v1, 120s, P(t* < T) >= 0.30 trajectory fraction).
- `ablation_plan` matches the prose: replace m_phi with constant-zero
  callable; everything else preserved; observe `t* < T` distribution and
  learning-curve gap.
- `nearest_disqualifier = policy_gradient` is honest; the closest dead
  family hit is REINFORCE/PG, which the candidate explicitly addresses.
- `novelty_boundary` repeats and sharpens the prose discussion: credit
  weight is `D(tau) * 1[k <= t*]` rather than `G_k`.
- `update_family = direct_policy_update`, `memory = network`,
  `uses_reward = true`, `uses_vector_reward = false` all consistent with
  a CartPole quick-stage probe.

`uv run python scripts/validate_candidate.py
worklogs/runs/20260606-15-auto/candidate.json` was not executed in this
review session, but inspection against `scripts/validate_candidate.py`
shows all required string fields non-empty, both booleans typed,
`claimed_stage` in the harness stage set, `update_family` and `memory`
in their enums, and `nearest_disqualifier = policy_gradient` in the
allowed list. No vector-stage / vector-reward conflict.

## Coherence check

Step 1 (Brown-Smith-Sun weak duality with martingale-difference
penalties): correctly stated; BSS Thm 2.1 indeed gives
`V*(s_0) <= E[max_a Sum_t gamma^t (r_t - m_t)]` with equality at the
value-function-martingale-increment m*. No load-bearing algebraic error.

Step 2 (replacement of perfect-info inner max by realized path supremum
along a sampled trajectory): this is a *heuristic* relaxation, not a
theorem. The Researcher acknowledges this at step 5 / proof-debt (1) by
flagging it as the "online realization" and noting strong duality of
the learning saddle is open. The inequality `E_tau[D(tau)] >= V*(s_0)`
as written in step 2 is a stretch -- it is not BSS weak duality (which
needs the perfect-info max over advance-known action sequences), nor is
it standard. I read it as an informal heuristic restatement, and the
proof debt section honestly lists this as item (1) and (5). This is
acceptable for a probe.

Step 3 (saddle structure): the convexity-in-m claim is correct (path-sup
of an affine function of m is convex in m). The concavity-in-pi claim
is appropriately marked as proof debt (occupancy-measure-level, not
parameter-level).

Step 4 (arg-supremum-time credit indicator c_k = 1[k <= t*]): well
defined, computable, ablation-targetable.

Step 5 (policy gradient via score function through path supremum):
correctly flagged as needing a Danskin-style argument. This is genuine
proof debt, not a hidden error -- the gradient of an expectation of a
path supremum admits a score-function form when t*(tau) is almost
surely unique. For CartPole, t*(tau) is almost surely unique once m_phi
has any noise, so this is empirically tractable.

Step 6 (penalty gradient is concentrated at t* by envelope theorem):
correct. The gradient `dD/dM_t = -1[t = t*]` is the standard envelope
formula for a max.

Step 7 (why not Q-learning) and Step 8 (why not REINFORCE): both are
sharp. The credit weight `D(tau) * 1[k <= t*]` is genuinely different
from `G_k` whenever `R_k - M_k` is non-monotone in k, which a
non-trivial learned m_phi will produce. The "monotone-reward CartPole
collapses to REINFORCE under m=0" observation is exactly what the
ablation tests.

Update rule: implementable. Steps 1-6 of the inner loop are concrete
torch operations on a single rollout. Baseline subtraction
(`f_phi - b_phi`) for the conditional-zero-mean constraint is a known
construction (action-dependent baseline regression), and the Engineer
will not need to invent it.

## Novelty check

Searches performed:

- "information relaxation duality reinforcement learning Brown Smith Sun
  martingale penalty policy gradient" -- returns BSS 2010, BSS 2014, and
  ADRL (arXiv 2506.00801, June 2025).
- "adversarial reinforcement learning information relaxation min max
  policy penalty 2025" -- ADRL is the only direct hit.
- "running max cumulative reward policy gradient credit weight envelope"
  -- no hits.
- "control variate martingale increment policy gradient transition
  baseline" -- standard variance-reduction literature; no hit on a
  martingale-difference dual variable in saddle form.

Closest known method: **ADRL (Chen-Liu-Wang-Zhang, arXiv 2506.00801,
2025)**. ADRL is in the same family -- min-max optimization between a
policy NN and an adversarial penalty NN, inspired by BSS information
relaxation duality. The candidate's `## Novelty boundary` does not cite
ADRL directly, but the structural distinctions are real:

1. ADRL's outer max is over **action sequences with full lookahead**
   (the perfect-info inner problem of BSS), parameterized by a policy
   network that conditions on the future randomness `xi`. DUAL-IR's
   outer "max" is the **realized path-supremum-over-time of the
   penalized cumulative reward** along a single sampled non-anticipative
   trajectory. These are distinct quantities: one is the perfect-info
   value, the other is the running-max envelope of an adapted process.
2. ADRL targets dual-bound estimation for evaluation; DUAL-IR targets
   on-policy learning with arg-sup-time credit truncation.
3. DUAL-IR's load-bearing novelty is the credit indicator
   `c_k = 1[k <= t*]`. ADRL has no analog -- its policy is updated by
   max over action sequences with advance knowledge, not by truncated
   trajectory-realized policy gradient.

The candidate is not a rebadge of ADRL, but ADRL should be cited in any
follow-up sharpening. This is a sharpening note, not a blocker.

Closest dead family: Family E ("avoid value vocabulary, keep value
structure"). DUAL-IR's m_phi is *not* a relabeled V or Q -- it has the
structural conditional-zero-mean constraint, which V/Q do not. The
credit signal is a path-sup-of-cumulative-penalized-reward, not a
future-compression of return into a scalar. Family E does not bind.

Disqualifier list (`prior_attempts.md` "Disqualifier families"):

- Bellman backup: no -- there is no fixed-point operator on Q or V.
- Scalar-weighted log-prob update (REINFORCE/PPO/A2C): the candidate
  *is* a log-prob update, but the weight is not the cumulative
  return-to-go `G_k`; it is `D(tau) * 1[k <= t*]`. The ablation directly
  tests whether this difference matters.
- Actor-critic: no -- m_phi is not a critic supplying the actor's
  weight; it enters the dual envelope additively.
- Distributional RL, DT, GVFs, RND, count-based, options, HER, MCTS,
  CEM/ES, top-k cloning: no.
- Scalarized vector reward: not applicable -- claimed stage is `quick`
  (CartPole), no vector envs.

Verdict: not a rebadge.

## Implementability and ablation check

`train.py` for the run is implementable on top of the existing harness:

- Environment: CartPole-v1 via the harness.
- Policy: small MLP outputting categorical logits.
- Penalty net m_phi: small MLP `f_phi(s, a, s')`; baseline head `b_phi(s, a)`
  trained by 1-step regression to predict `E_{s'}[f_phi(s,a,s')]`. The
  Researcher specifies stop-grad regression on f_phi targets, decoupled
  from the main update -- this is implementable with two optimizers.
- Per-rollout computation of `R_k, M_k, D, t*, c_k` is O(T) per episode.
- Steps 5 and 6 of the update rule are direct torch autograd calls.

`train_ablate.py`: replace `m_phi` with a zero callable and remove its
optimizer. `D = max_k R_k`, `t* = argmax_k R_k`. On undiscounted
CartPole with reward = 1 per step, `R_k` is monotone non-decreasing
identically, so `t* = T` deterministically and `c_k = 1` for all k. The
ablation collapses to REINFORCE with terminal undiscounted return as
the trajectory weight (no baseline). This is the right ablation: it
disables the primitive while keeping the surrounding algorithm intact,
and the empirical observable `P(t* < T)` is binary-valued at the
ablation (zero) vs. non-trivial under the full algorithm.

The candidate's empirical claim has *two* falsifiers: (i) learning
curves match between full and m=0 ablation (primitive decorative); (ii)
`P(t* < T) ~ 0` throughout training (envelope degenerate). Both are
measurable on the quick stage. The 30% threshold for `P(t* < T)` is a
concrete number and can be logged to `panel-quick.txt`.

No vector scalarization risk. No missing pieces for the Engineer.

## Decision

Approve for empirical testing on the quick stage. The proof debt items
are honestly enumerated and none of them are load-bearing for the probe
to be informative: CartPole has dense reward, almost-sure unique
arg-sup-time, and a clean ablation discriminator. The two falsifiers
are concrete and measurable within the 120s quick budget.

The Engineer should:

- Implement `train.py` and `train_ablate.py` as described.
- Log `D`, `t*`, `T`, and `1[t* < T]` per episode so that the empirical
  claim "P(t* < T) >= 0.30 during training" can be checked directly.
- Run the smoke -> claim -> ablation ladder via
  `scripts/run_probe_ladder.py`.
- Compare full vs. ablation learning curves on shared seeds.
