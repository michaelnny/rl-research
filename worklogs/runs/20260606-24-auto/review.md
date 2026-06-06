---
verdict: reject
reviewer_run: 20260606-24-auto
hypothesis_type: probe
---

## Summary

CHANBI's update direction `d* = (g_1/||g_1|| + g_2/||g_2||)/||...||`
applied to per-channel score-function gradients is, for K=2, the same
direction (up to positive scale) as IMTL-G (Liu et al., "Towards
Impartial Multi-Task Learning," ICLR 2021); the proposed novelty boundary
explicitly invokes IMTL-G's defining property (equal projection /
rescaling-invariance) without naming the method. Reject as a
published-method rebadge.

## Schema check

`candidate.json` has all required string fields and the two required
booleans. `claimed_stage = "quick"` matches `## Empirical claim`.
`primitive_name`, `primitive_type`, `principle`, `nearest_disqualifier`,
`empirical_claim`, `falsifier`, and `ablation_plan` all line up with the
prose. The schema is internally consistent. The contradiction is not
internal to the file - it is between the file and the literature.

## Coherence check

The derivation is internally clean. Steps 1-3 set up per-channel
score-function gradients g^c and unit normalizations ghat_c. Step 4
correctly observes that ghat_c is invariant under per-channel positive
rescaling r^c <- alpha_c r^c. Step 5 defines d* as the spherical sum.
Steps 6-7 distinguish from MGDA and Nash-MTL on the rescaling test.
Proof debt (Pareto-improvement, anti-aligned tie-break, convergence)
is explicitly listed and is acceptable for a probe.

The load-bearing algebraic claim - that `d* = (u_1+u_2)/||u_1+u_2||`
has equal projection onto each unit gradient - is correct:
`d* . u_1 = (1 + u_1.u_2)/||u_1+u_2|| = d* . u_2`. This is exactly the
"equal projection onto each gradient direction" property that defines
IMTL-G.

## Novelty check

Searches: "multi-task gradient normalize unit sum direction angular
bisector"; "sum of unit gradients multi-task learning"; "IMTL-G unit
gradient equal projection"; "impartial multi-task equal projection
gradient unit normalize closed form".

Closest method: **IMTL-G** (Liu et al., "Towards Impartial Multi-Task
Learning," ICLR 2021). IMTL-G aggregates per-task gradients
`g = sum_t alpha_t g_t` such that the result has **equal projections
onto each unit task gradient** `u_t = g_t/||g_t||`, i.e.,
`g . u_1 = g . u_2 = ... = g . u_K`. The reference implementation
(JohnLaMaster/Impartial-Multi-Task-Learning, `imtl.py`) literally
constructs `u[i] = g[i] / torch.linalg.norm(g[i])` as the unit gradients
and solves a closed form for alpha.

For K=2 the equal-projection constraint `g . u_1 = g . u_2` is one
linear equation, and within `span(u_1, u_2)` its solution set is a
single 1D line through the origin parallel to `u_1 + u_2`. CHANBI's
choice `d* = (u_1 + u_2)/||u_1 + u_2||` is exactly that line
(positively oriented). IMTL-G's closed-form alpha for K=2 picks a
different point on the same line (a convex combination of g_1, g_2)
but the **direction** is identical. Two policy-gradient updates with
proportional update vectors produce the same trajectory up to a step-
size rescaling, which the candidate already absorbs into the natural
step-size scalar `S = sum_c ||g_c||` in step 6 of the update rule.

The novelty boundary cites Desideri 2012 (MGDA), Sener-Koltun 2018,
Yu 2020 (PCGrad), Roijers-Whiteson 2017, Navon 2022 (Nash-MTL), and
CAGrad 2021. It does **not** cite IMTL-G. The hypothesis attributes
to CHANBI the exact property IMTL-G was designed to achieve - that
the aggregated direction be invariant under per-channel rescaling and
treat tasks "impartially" - and offers it as the structural distinction
from those other methods.

The "applied to RL via per-channel score-function gradients on
info['vector']" framing is a substrate choice, not a novel mechanism;
IMTL-G is gradient-aggregator-agnostic and applies wherever per-task
gradients are defined. Naming a known multi-task-learning aggregator
"CHANBI" and applying it to vector-reward policy gradients does not
clear the novelty bar laid out in `prior_attempts.md` and the
disqualifier list (existing methods may appear as **components**, not
as the explanation).

The proof-debt items 1-2 (Pareto-improvement at first order;
convergence under stochastic estimation) are also addressed in the
IMTL-G paper and follow-up work on impartial / equal-projection
aggregators; the analytical gap CHANBI would close is already partially
closed in that literature.

## Implementability and ablation check

If the rebadge issue did not exist, the update rule is implementable:
two backward passes per rollout (one per channel return-to-go), unit
normalization in parameter space, sum and renormalization, scalar
step-size restoration. `train.py` against the existing harness contract
is straightforward and consumes `info["vector"]` correctly without
scalarization. The ablation (skip per-channel normalization, recover
REINFORCE on uniform-weight scalarized return) is load-bearing for the
"normalization vs no normalization" question and produces the predicted
`cos_d_uniform == 1` discriminator by construction. The empirical claim
(cos_d_uniform < 0.95, mean T > 2, hypervolume >= 194) is testable on
DST-concave at the quick stage within budget.

These are not the issues. The issue is upstream: even on the strongest
possible empirical signal, the conclusion would be "IMTL-G's direction
works on this RL substrate," which is a substrate evaluation of a
known method, not a new family.

## Decision

Triggered criteria from the rejection list:

- "The claimed novelty is only a new index axis, statistic, aggregation
  rule ..." - the aggregation rule (sum of unit gradients with equal-
  projection property) is IMTL-G's, applied to per-channel
  score-function gradients.
- Disqualifier list: "Existing methods may appear as components ...
  They cannot be the **explanation** for why the method works."
  IMTL-G's equal-projection property *is* the explanation CHANBI gives
  for its rescaling-invariance.
- Decisive rebadge hit: the cited "structural difference from MGDA /
  Nash-MTL / CAGrad" is the property that defines IMTL-G, which is
  uncited.

A revise verdict is not appropriate because the fix is not mechanical:
there is no version of "sum of unit per-task gradients with equal
projection" that is not IMTL-G. The Researcher's next turn would need
to either (a) locate a structural property of d* that IMTL-G does not
have (e.g., a non-trivial RL-specific interaction between G^c_t weighting
and unit-normalization that does not collapse to the supervised IMTL-G
case), or (b) abandon the angular-bisector primitive entirely.

For reference, the K=2 algebraic equivalence: IMTL-G requires the
aggregated direction `d` to satisfy `d . u_1 = d . u_2`. CHANBI sets
`d* proportional to (u_1 + u_2)`. Compute: `(u_1 + u_2) . u_1 =
1 + u_1 . u_2 = (u_1 + u_2) . u_2`. Equal. Within `span(u_1, u_2)`
the equal-projection constraint defines a unique direction up to sign;
both methods select the positive one. They are the same direction.
