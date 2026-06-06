---
verdict: reject
reviewer_run: 20260606-14-auto
hypothesis_type: probe
---

## Summary

NORMAL dresses Q-learning in cone-projection vocabulary, but every
component of its update either reduces to Bellman backup with a
state-independent baseline subtraction or is a row-wise shift that
preserves argmax order; the "active-set indicator" is just argmax of A
under a neural net, so the algorithm acts as greedy Q-learning at the
policy level.

## Schema check

`candidate.json` parses and matches the schema fields used by
`scripts/validate_candidate.py` (all required strings, booleans, allowed
enums, claimed_stage=quick valid, nearest_disqualifier=q_learning is in
the allowed set). Prose-to-JSON correspondence is fine: principle,
primitive_name/type, claimed_stage, empirical_claim, falsifier,
ablation_plan, nearest_disqualifier, novelty_boundary all line up with
the hypothesis. No mechanical schema reason to revise.

## Coherence check

Step-by-step on the derivation and update rule:

1. VI / normal-cone characterization of optimality (steps 1-3): correct
   textbook material. Stampacchia VI on the simplex with -A^{pi*} in the
   normal cone is standard. No issue.

2. C2 "advantage consistency cone" (step 4 / step 7 update rule):
   `target = r + gamma * max_{a'} A(s', a') - V_bar`,
   `A(s,a) <- A(s,a) + alpha * (target - A(s,a))`.
   This is identical in form to Q-learning's TD update with a
   *state-independent* scalar `V_bar` subtracted from the target. For
   action selection only the relative order of `A(s, .)` matters; the
   subtraction of `V_bar` is the same constant for every action at s, so
   it cancels out of `argmax_a A(s, a)`. The dynamics of the argmax
   trajectory under this rule are equivalent to those of Q-learning up
   to a slowly varying additive constant.

3. C1 projection (step 7 lines 102-103, line 142-144 in the update
   pseudocode): `A(s, .) <- A(s, .) + max(0, -max_a A(s, .))`. This is
   a *row-wise additive shift*. It strictly preserves the argmax set
   `chi(s) = argmax_a A(s, .)` and the partial order of A(s, .). The
   only thing the shift can ever change is magnitudes that flow into the
   *next* TD target through `max_{a'} A(s', a')`. But that shifted max
   is the same as the unshifted max plus the same constant; the
   constant gets absorbed by the running baseline `V_bar`. Net effect on
   the argmax sequence: zero.

4. "Uniform-on-argmax" policy (line 105, 127-128): with a function-
   approximator advantage `A_theta(s, .)` on a continuous state space
   (CartPole), `argmax_a A_theta(s, .)` is a singleton with probability
   one over the network weights. So `Uniform(chi(s))` reduces to the
   deterministic argmax almost everywhere. The "no epsilon-greedy
   needed" claim collapses: there is no exploration mechanism left.

5. "Self-bootstrapping at zero reward" (step 5): the claimed mechanism
   is that chi propagates from terminal states "via the C2
   consistency-cone projection." But C2 is exactly the Bellman target,
   so this is just standard Q-learning bootstrap from terminal value =
   0; if rewards are identically zero everywhere, there is nothing to
   propagate, contra the prose.

6. The Researcher names the right concern in novelty boundary (g):
   "The ablation that replaces chi with greedy-on-A reduces NORMAL to
   standard advantage-Q; if that ablation matches, family E
   classification is correct and we lose." The above analysis says it
   *will* match by construction, not just empirically.

The proof-debt convergence theorem is fine to defer; that is not the
problem. The problem is that the update rule, as written, is Q-learning
plus order-preserving cosmetics.

## Novelty check

Searches considered: "Dykstra alternating projection Q-learning",
"normal cone policy iteration simplex active set RL", "advantage
baseline subtraction Q-learning". The closest method is plain Q-learning
with a running baseline (a known minor variant; see e.g. average-reward
Q-learning of Schwartz 1993 / Mahadevan 1996, where the average reward
plays the role of `V_bar`). NORMAL's update equation is structurally
that family with a row-wise non-negativity shift bolted on top.

This sits in the disqualifier list as Bellman backup (Q-learning) and
also borders Family E (avoid value vocabulary, keep value structure):
the load-bearing learned object is `A_theta`, a real-valued function
trained by Bellman residual minimization, with the "active-set" relabel
being argmax of A.

## Implementability and ablation check

Implementability is fine; an Engineer could write `train.py` and
`train_ablate.py` against the existing harness contract without
inventing new pieces. CartPole/quick stage is appropriate for a small-
action-space probe. `info["vector"]` is irrelevant here
(uses_vector_reward=false, claimed_stage=quick), so no scalarization
issue.

The ablation plan (uniform-on-argmax -> argmax + epsilon-greedy; remove
C1 shift) is the right ablation in form. The problem is the prediction:
because both differences are no-ops at the policy level under a neural
A_theta on continuous states, the ablation is mathematically guaranteed
to match the main run up to tie-breaking and a small constant in the
TD target. The probe is therefore a setup for an inevitable
ablation-failure verdict, not a real test of a novel primitive. This is
exactly the failure mode the loop is trying to avoid.

## Decision

Reject. Triggered criteria:

- "The central update reduces to ... Q-learning/DQN/TD3/SAC": the C2
  step is Bellman TD with a constant subtracted from the target.
- "The claimed novelty is only a new ... aggregation rule": the C1
  shift and uniform-on-argmax are order-preserving relabels of greedy
  argmax on Q.
- "The primitive is ... [a] partial-order vote, or any other dead family
  ... ": borderline Family E (advantage with relabeled vocabulary).
- "The ablation plan ... would [match the main algorithm by
  construction] rather than disabling/randomizing the primitive": the
  ablation cannot fail to match given the order-preserving nature of
  the supposedly load-bearing C1 step.

A revisable version of this idea would need a primitive whose effect on
the argmax trajectory is not invariant under row-wise additive shifts
of A — for example, an active-set object that survives even when
A(s, .) becomes asymptotically flat, or a projection step that *changes
the relative order* of A(s, .) rather than only its magnitude floor.
That is a real research move and belongs in the next Researcher turn,
not in a reviewer rescue of this probe.
