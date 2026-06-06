---
verdict: probe
reviewer_run: 20260606-21-auto
hypothesis_type: probe
---

## Summary

COPDEV proposes per-step score-function weighting by the L1 distance
between two per-channel rolling-empirical-CDF rank evaluations of the
cumulative-return process; the primitive is typed, the ablation is
clean, and the discriminating observable is a logged training-dynamics
scalar that fires at random initialization, so empirical testing is
warranted despite open proof debt and a non-trivial degenerate-channel
substitute on DST-concave.

## Schema check

`candidate.json` matches the structural schema. Required string and
boolean fields are present and non-empty. `claimed_stage = "quick"` is a
known stage (maps to `deep-sea-treasure-concave-v0`).
`uses_vector_reward = true` is consistent with the quick stage having a
vector env. `nearest_disqualifier = "scalarization"` is paired with a
`novelty_boundary` that explicitly addresses scalarization (the Sklar
rank-invariance argument and the strict-monotone-componentwise
counterexample), so the schema's scalarization-mention rule is
satisfied.

Schema/prose alignment:
- `principle` matches the hypothesis `## Principle`.
- `primitive_name`/`primitive_type` match `## Primitive` (per-(t,c)
  empirical CDF as the typed object; d_t as its evaluation).
- `claimed_stage`, `empirical_claim`, `falsifier` match `## Empirical
  claim`.
- `ablation_plan` matches `## Ablation plan` (replace d_t with 1, keep
  everything else fixed).
- `nearest_disqualifier`/`novelty_boundary` match `## Novelty
  boundary`.

## Coherence check

Steps 1-4 of the derivation are coherent. The bicriterial cumulative
process `(M^1_t, M^2_t)` is well-defined for vector envs, marginal
empirical CDFs converge under Glivenko-Cantelli per (t, c), and Sklar's
theorem licenses calling the L1 distance from the diagonal of the unit
square a copula deviation. Rank invariance under strict-monotone
componentwise reparametrization is correct and is a real structural
distinction from any linear scalarization w^T r.

Step 5 has a substantive wrinkle that the Researcher names: on
DST-concave, channel 2 (the step penalty) is deterministic, so
`F_t^2` collapses to a unit step at -t and the rank evaluation is
trivially 1. The Researcher substitutes a survival-rank
`F̂_t^2(M^2_t) := P(historical episode survived >= t)`, which is *not*
a marginal CDF of M^2_t at time t but rather an episode-length CDF.
This means the principle's rank-invariance property (the load-bearing
non-scalarization argument) does not engage on the only env in the
quick stage; the actually-computed d_t there is `|F̂_t^1(M^1_t) -
survival_t|`. This is acknowledged in step 5 and listed as proof debt
item (5). It is honest proof debt rather than a hidden contradiction:
the empirical question becomes whether the survival-rank substitute is
load-bearing on this substrate even with the degeneracy.

Steps 6 and 8 give a concrete random-init prediction (d_3 ~ 0.3) and a
discriminating observable (`gradnorm_var = Var_t(||g_t||) /
Mean_t(||g_t||)^2`) that follows from per-step weight non-uniformity.
The observable is well-defined and computable from logged rollout
gradient norms.

Step 7's claim about credit-assignment shape is a heuristic argument,
not a derivation. That is fine for a probe.

The proof debt is named explicitly (two-timescale convergence,
identification of the optimized objective, bias of the rolling-buffer
CDF estimator, Pareto-improvement claim, degenerate-channel
characterization). This is the right shape for a probe-first loop.

## Novelty check

Searches for "copula deviation policy gradient empirical CDF rank
weight reinforcement learning" and "copula reinforcement learning
multi-objective vector reward policy gradient" returned no method that
applies a per-step copula-rank-disagreement weight to score-function
ascent. Copula techniques in ML appear in copula-VI, neural Gaussian
copulas, and copula weight initialization — none in the policy-gradient
weight slot.

Closest known relatives, all distinct:
- REINFORCE weights by cumulative return G_t; COPDEV weights by per-step
  rank disagreement, not by any function of total return.
- Advantage / GAE rely on a learned baseline V; COPDEV has no V.
- Distributional RL learns a state-action return distribution;
  COPDEV maintains marginal CDFs of *time-indexed cumulative channel
  values*, not state-action distributions.
- Rank-based PER applies rank to *sampling probability*; COPDEV applies
  it to *gradient weight* on an on-policy rollout.
- Order-statistics / ordergrad rank whole trajectories; COPDEV ranks
  per-step.
- Linear / non-linear scalarization is excluded by Sklar rank
  invariance with a concrete counterexample.

Dead-family check against `prior_attempts.md`:
- Family A (bucketed-tensor + partial-order vote): no state/action
  bucketing; index is (t, c) only; no partial-order vote.
- Family B (pairwise trajectory comparison): no pairing; the buffer
  aggregates many trajectories and is evaluated against a single
  on-policy rollout.
- Family C (within-trajectory geometric statistic): the rank evaluation
  requires a cross-trajectory historical buffer, so it is not a
  within-trajectory geometric primitive.
- Family D (reward-independent + reward-gated): d_t depends on reward
  throughout; there is no gate.
- Family E (avoid value vocabulary): no learned future compression.
- Family G (mechanism stack): single primitive (the per-(t,c) ring
  buffer + rank evaluation).
- Family H (cochain): not applicable.

Within-loop nearest siblings PRISM (run 18, terminal-return KDE on the
Pareto frontier, per-trajectory weight) and GRADCOMP (run 20, Fisher
principal eigenvector) are operationally distinct: PRISM uses terminals
and per-trajectory log-density; COPDEV uses cumulative-process CDFs at
each time index and per-step weight.

Not a rename of any disqualifier in `prior_attempts.md`.

## Implementability and ablation check

The Engineer can implement this against the existing contract:
- Reads `info["vector"]` per step and never sums channels (update rule
  step 1 explicitly forbids summation).
- Maintains O(T_max * 2 * N) ring buffers; lookup O(N) per step.
- Adds a per-step weight to a standard score-function update; backprop
  unchanged.
- Logs `gradnorm_var`, mean d_t, mean episode length, first-rewarded
  episode index — all standard rollout-time scalars.

`train_ablate.py` is the same code path with `d_t = 1` and the buffer
maintenance disabled. The ablation isolates the per-step
rank-disagreement weighting and is therefore load-bearing for the
novelty claim.

Vector-reward consumption: explicit in update rule step 1; not a
scalarization.

Substrate-floor risk (per `prior_attempts.md` budget constraint): the
discriminator `gradnorm_var` is a training-dynamics scalar that fires
at random init, so the candidate does not need to depart the DST=99
floor for the falsifier to fire. This is the right kind of observable
for the 120s budget and the right answer to the floor-clamping pattern
seen in runs 15-20.

## Decision

Verdict: `probe`.

Empirical testing is allowed because (a) the schema validates and
matches the prose, (b) the primitive is one typed object (per-(t,c)
empirical CDF, ring-buffer represented), (c) the update rule is
implementable without inventing missing pieces, (d) the ablation
disables exactly the load-bearing primitive, (e) the discriminator
is a logged scalar that does not require finding treasure inside
budget, and (f) no rebadge or dead-family hit.

The convergence theorem and the identification of the optimized
objective `Phi(pi)` are open and are correctly named under `Proof
debt`. The DST-concave degenerate-channel substitute (survival-rank
for channel 2) is also named as proof debt; the empirical question is
whether the substitute mechanism is still load-bearing for
`gradnorm_var` on this substrate. That question is decidable inside
the 120s budget by the proposed ablation.
