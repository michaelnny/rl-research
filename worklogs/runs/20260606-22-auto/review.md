---
verdict: probe
reviewer_run: 20260606-22-auto
hypothesis_type: probe
---

## Summary

PARGRAD is a structurally distinct sharpening of COPDEV (joint bivariate
weak-dominance count with at-least-one-strict, monotone in each
coordinate, against a per-t historical buffer); the prior strict-
dominance bug that silenced p_t on DST-concave's deterministic channel 2
has been correctly relaxed to weak-with-one-strict, restoring the load-
bearing channel-1 reduction and an alive-and-lower-treasure signal. The
unicriterial sanity ablation is preserved. Approve as a probe.

## Schema check

Ran the validator mentally against `scripts/validate_candidate.py`:
- All 15 required string fields are non-empty.
- `uses_reward = true`, `uses_vector_reward = true`.
- `claimed_stage = "quick"`. Quick stage in `harness.STAGES` includes a
  vector env (DST-concave) so the `uses_vector_reward` constraint is
  satisfied.
- `update_family = "direct_policy_update"`, `memory = "replay"`,
  `nearest_disqualifier = "scalarization"` are all in the allowed sets.
- The `scalarization` disqualifier is paired with a `novelty_boundary`
  that contains the substring "scalar" and gives a concrete
  componentwise-monotone-reparam counterexample
  (`r^1 -> log(1 + exp(r^1))`).

Schema-vs-prose alignment:
- `principle` matches `## Principle` (per-step score-function ascent
  with empirical bivariate weak-dominance-with-at-least-one-strict
  count weight).
- `primitive_name` and `primitive_type` match `## Primitive` (typed:
  `(t) -> empirical probability measure on R^2`, evaluated by weak-
  dominance counting on the realized rollout).
- `claimed_stage`, `empirical_claim`, `falsifier` all match
  `## Empirical claim` (gradnorm_var > 0.1, mean_pt drift, score above
  random floor 194, plus the new bivariate-vs-unicriterial test).
- `ablation_plan` matches `## Ablation plan` (random-uniform-per-step
  primary; unicriterial channel-1-only secondary).
- `nearest_disqualifier` and `novelty_boundary` match
  `## Novelty boundary`.

Schema and prose are now consistent.

## Coherence check

Steps 1–4 (bivariate process, weak convergence, dominance count as
L-statistic, rank invariance under componentwise monotone maps) all
follow. The rank-invariance argument is correct under weak dominance
with at-least-one-strict: componentwise strictly-increasing maps
preserve `<=` and the `!= ` clause, so `p_t` is preserved.

Step 5 is now correct. Under weak dominance with at-least-one-strict on
DST-concave (deterministic channel 2 given survival): every alive
buffer entry has `z^2 = y^2`, so the `z^2 <= y^2` clause is trivially
satisfied; the `!=` clause forces strictness in channel 1, giving
`p_t = (1/N_t) * |{j : M^1_{t,j} < M^1_t(tau)}|` over alive-at-t buffer
entries. This is non-trivial whenever some past trajectories have
collected treasure by time t and others have not. The reduction is
explicitly flagged as "dangerously close to unicriterial" and is
addressed by the secondary sanity ablation.

Step 6 (per-step score-function gradient with weight p_t) is a clean
implementable update rule.

Step 7 (monotonicity / Pareto-direction) is correct: under weak
dominance with at-least-one-strict, increasing either coordinate of
y_t weakly increases the count (some buffer entries move from "tie" to
"strict-dominated" or stay strict-dominated), so p_t is monotone non-
decreasing in each coordinate of y_t. That is the structural
distinction from COPDEV's symmetric `|F^1 - F^2|` gap.

Step 8 (gradnorm_var and mean_pt_trend observables) is well-posed; the
shadow-buffer logging idea for the random-uniform arm is sensible.

Steps 9–11 (distinction from Family C, Family A, NSGA-II) hold.

Proof debt items 1–5 are listed and named (Borkar 2008 two-timescale,
Pareto-improvement at stationary points, VC bias-variance for
rectangles, degenerate-channel reduction characterization, bivariate
stochastic dominance connection). All within "empirical signal first,
theorem later" envelope.

The only residual concern is precisely the one the Researcher already
flagged: on DST-concave the bivariate primitive operationally reduces
to a channel-1 marginal rank weight. The unicriterial sanity ablation
is the right way to test this and was retained.

## Novelty check

Closest published methods searched: NSGA-II (population selection,
discrete front-rank, no gradient), Pareto-MTL / multi-task gradient
balancing (scalarized loss balancing, not per-step weighting), order-
statistics policy gradient (per-trajectory scalar rank), prioritized
experience replay (TD-error rank as sampling probability), MO-MCTS,
hypervolume MORL, distributional RL. None match a per-step continuous
empirical bivariate dominance count used as score-function gradient
weight against a per-t historical buffer.

In-loop distinctions:
- COPDEV (run 21) maintains two 1D marginal CDFs and computes
  `|F^1 - F^2|`, symmetric, no Pareto direction; needed a survival-
  rank hack on DST-concave. PARGRAD maintains one 2D measure and uses
  the joint dominance count, monotone in each coordinate. Two
  trajectories with identical marginal ranks but opposite joint
  positions in R^2 give different p_t but identical d_t — operational
  distinction holds.
- PRISM (run 18) is terminal-only KDE on the Pareto front. PARGRAD is
  per-step against the full historical buffer.
- GRADCOMP (run 20) rotates update direction. PARGRAD modifies the
  scalar magnitude of each per-step contribution.

Family A–H exclusions are correct: no state/action bucketing, no
pairwise matching, no within-trajectory geometry, no value/cumulant
prediction, single primitive, not cochain.

Not a rebadge.

## Implementability and ablation check

Implementable against the substrate contract:
- Per-step `p_t` is `O(N)` per t over a deque of `(z1, z2)` pairs;
  `T_max` deques total. Storage `O(T_max * N)` bivariate scalars.
- The pseudocode reads `info["vector"]` per step to build per-channel
  cumulants `M[t, c]`; no scalarization (`wTr_vec`) anywhere.
- Score-function update `g = sum_t p_t * grad log pi(a_t|s_t)` is a
  one-line modification of REINFORCE.
- Logging hooks (`mean_pt`, `var_pt`, `gradnorm_var`,
  `mean_pt_trend.txt`) are all straightforward.

Ablations:
- Primary (random-uniform per-step weight) preserves per-step weight
  non-uniformity (so `gradnorm_var` stays non-zero in both arms,
  isolating the *direction* of `p_t` as the load-bearing property) and
  drops the buffer entirely. Clean structural ablation.
- Secondary (unicriterial channel-1-only rank) directly tests whether
  the joint bivariate structure adds anything above a marginal rank on
  this substrate. This is exactly the right sanity check given the
  step-5 reduction.

No vector-scalarization issue. No missing pieces for the Engineer.

## Decision

Verdict: **probe**.

The strict-dominance pseudocode bug from the prior review is fixed
(weak dominance with at-least-one-strict in derivation step 5,
pseudocode step 3, and `candidate.json.primitive_type`). Schema is
consistent. Mechanism is structurally distinct from COPDEV (joint vs
marginal, directional vs symmetric) and from named methods. Ablation
is clean and the optional unicriterial-rank sanity ablation is
preserved as a Curator-readable test of whether the bivariate primitive
genuinely outperforms its single-channel reduction on DST-concave.

Proof debt (Borkar convergence, Pareto-improvement at stationary
points, finite-buffer bias-variance, degenerate-channel reduction
characterization, stochastic-dominance connection) is explicit and
within "empirical signal first" envelope. The empirical observables
(`gradnorm_var`, `mean_pt_trend`, final hypervolume, bivariate-vs-
unicriterial contrast) are well-posed and substrate-visible within the
120s quick-stage budget.

Approve for Engineer probe run.
