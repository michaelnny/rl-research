---
verdict: probe
reviewer_run: 20260606-18-auto
hypothesis_type: probe
---

## Summary

PRISM proposes a typed primitive (rolling Pareto-frontier KDE on
vector-return space) used as a non-linear, non-scalarizing
per-trajectory weight on the score-function policy gradient. The
update is implementable, the ablation is load-bearing, the empirical
claim names a discriminating observable (`coverage_n`), and the
nearest dead families do not match. Approve as a vector-stage probe.

## Schema check

`candidate.json` is internally consistent and matches the prose:

- `principle` matches `## Principle` (score-function ascent on log-
  density of terminal vector return under rolling Pareto-frontier KDE).
- `primitive_name` = `rolling_pareto_frontier_kde` and `primitive_type`
  describe a typed object `R^m -> R_{>=0}` matching `## Primitive`.
- `claimed_stage` = `vector`, matching `## Empirical claim` and the
  envs cited (DST-concave, RG).
- `falsifier` matches the prose primary/secondary failure modes.
- `ablation_plan` (fixed origin-centered Gaussian, sigma frozen from
  first 8 random rollouts) matches `## Ablation plan` step by step.
- `nearest_disqualifier` = `topk_cloning`; `novelty_boundary` matches
  the CEM/scalarization/MOO-SVGD/Latent-PG distinctions in `##
  Novelty boundary`.
- `uses_vector_reward` = true; `feedback_signal` references
  `info["vector"]`, satisfying the vector-env contract.

I have not executed `validate_candidate.py` here, but the JSON is
well-formed, every required field appears populated, and types/enums
look consistent with prior approved probes. If the validator surfaces a
mechanical issue, the Engineer will fail fast at smoke time.

## Coherence check

The derivation has a clear backbone:

1. Define `μ*` as a KDE on `R^m` whose support is the rolling
   Pareto-non-dominated subset `F_n`. Typed, computable, finite-support.
2. Maximize cross-entropy `E_{y ~ ν_π}[log μ*(y)]` by score-function
   identity, yielding gradient `E_τ[log μ*(y_τ) · Σ_t ∇ log π]`.
   Standard Williams-1992 derivation; this step is correct.
3. Show the per-trajectory weight `log μ*(y)` cannot be written as
   `wᵀy + const`. The three-frontier-points-around-origin
   counterexample is correct: log-sum-of-Gaussians has level sets
   that no linear functional can produce.

Heuristic / open pieces, explicitly listed as proof debt:

- Two-timescale convergence of `(μ*, θ)` (proof debt 1).
- Frontier coverage growth rate (proof debt 2).
- Median-heuristic bandwidth bias (proof debt 3).
- Hypervolume monotonicity (proof debt 4).
- Strict dominance over linear scalarization on non-convex frontiers
  (proof debt 5).

These are theorem-level questions. None is required before compute.

One coherence concern (not a blocker, worth flagging for the Engineer):
`log μ*(y)` is unbounded below for trajectories far from `F_n`. The
weight can therefore push the policy strongly *away* from
unfrequented regions of vector-return space, which is the opposite
of what a frontier-extension argument needs. The Researcher's
implicit assumption is that the rolling window absorbs the first
visit to a new region before the gradient propagates. The running-
mean centering (step 4) and the warmup `w_τ = 1.0` for `|F| < 2`
mitigate this somewhat. This is exactly the kind of question
empirical signal can answer cheaply.

## Novelty check

Searched (mentally and by reference to the candidate's own audit) for
the principle "rolling KDE on multi-objective return space used as
trajectory weight in policy gradient" rather than the name "PRISM."
The closest published methods listed in the candidate (Parisi-Pirotta
continuous Pareto-frontier approximation, MOO-SVGD, latent-conditioned
PG, Pareto Q-Learning, GAIL, RWR/AWR) all have structurally different
primitives:

- Parisi-Pirotta: parametric manifold of policies; PRISM is a single
  policy + non-parametric KDE on `R^m`.
- MOO-SVGD: K-particle ensemble + policy-space repulsion; PRISM has
  no ensemble, kernel acts on returns not policies.
- Latent-conditioned PG: preference input to policy; PRISM has no
  conditioning.
- Pareto Q-Learning: per-(s,a) vector-Q sets; PRISM has no value
  vocabulary.
- CEM / top-k cloning: 0/1 elite indicator; PRISM uses continuous KDE
  density and the weight is non-monotone in any component.
- RWR / AWR: `exp(A/β)` scalar advantage weight; PRISM's weight is
  `log μ*(y)`, a non-monotone vector functional, with no value/
  advantage estimator anywhere.

Against the dead families in `prior_attempts.md`:

- Family A (bucketed-tensor + partial-order vote): PRISM has no
  bucketing and no partial-order vote. KDE on `R^m` is continuous.
- Family B (pairwise trajectory comparison): no pair matching;
  trajectories are weighted against the running aggregate.
- Family C (within-trajectory geometry): PRISM uses terminal vector
  return only.
- Family D (reward-independent + reward-gated): PRISM is reward-
  dependent throughout.
- Family E (rename value vocabulary): no value vocabulary. The only
  learned object is `π_θ`. Frontier KDE is a *target measure*, not a
  compressed-future-return scalar.
- Family F (hand-engineered prior): no event vocabulary, no
  hand-coded structure beyond "Pareto dominance on `R^m`."
- Family G (mechanism stack): one primitive (`μ*`); the score-
  function gradient is standard, the novelty is fully isolated in
  the trajectory weight.
- Family H (cochain): not applicable.

Disqualifier check:

- Not Bellman backup, no Q/V function, no critic.
- Not PPO/REINFORCE *with reward*: the trajectory weight is
  log-density of the realized return under an exogenous target, not
  the discounted return itself. The score-function carrier is
  REINFORCE-shaped; the load-bearing object is `μ*`.
- Not CEM/top-k: continuous KDE weight, not 0/1 elite indicator;
  weight non-monotone in scalar fitness.
- Not scalarized vector reward: counterexample in the candidate is
  correct (level sets of log-sum-of-Gaussians are not affine).
- Not actor-critic, RND, options, GVFs, distributional RL,
  Decision Transformer, RLHF/DPO, HER, reward machines.

Verdict: not a rebadge.

## Implementability and ablation check

Implementability against the substrate contract:

- Entry point `train(env_id, seed, time_budget_s) -> PolicyFn` is
  achievable: the Engineer writes a small policy network, an episode
  loop that reads `info["vector"]` for vector envs, a `deque` of
  recent terminal returns, an O(N^2) Pareto filter (cheap for `N`
  in tens to low hundreds), and a log-sum-exp evaluation. None of
  this requires inventing missing pieces.
- Vector consumption: explicit `r_vec_t = info["vector"]` for vector
  envs. The fallback `r_vec_t = (r_t, -1)` for sparse envs is
  irrelevant to the claimed stage (`vector`); the Engineer should
  guard the sparse-env branch out for this probe to avoid masking
  failure with a synthetic vector.
- Score-function gradient is one `torch.autograd` line per episode.
- Median-heuristic bandwidth and log-sum-exp are standard.

Ablation strength:

- `train_ablate.py` replaces `μ*` with a fixed isotropic Gaussian
  centered at the origin, sigma frozen from first 8 rollouts. This
  preserves vector-return computation, score-function update, and
  baselining, and removes only the frontier-tracking. The ablation
  weight is monotone in `−||y||²`, which deliberately points the
  *wrong* way relative to any frontier signal — strengthening the
  discriminator.
- The candidate observable `coverage_n = |F_n|` is degenerate under
  the ablation (no frontier maintained), so the test reduces to
  hypervolume comparison + the candidate's own coverage growth.
- A second sanity ablation (`h → ∞`) is named but optional; the
  primary ablation is sufficient.

Empirical claim is concrete and falsifiable: `coverage_n >= 3` by end
of vector-stage training plus hypervolume strictly above
nearest-treasure-collapse on at least one of DST-concave or RG. The
falsifier (collapse to `|F_n| = 1` throughout) is exactly the runs
13-17 failure mode and would clearly indicate the principle does not
fire on this substrate.

## Decision

Approve as `probe`. Empirical testing is justified despite open
theorem work because:

1. The primitive is typed, single, and computable.
2. The update follows from the principle by standard score-function
   identity; any heuristic gaps are listed as proof debt.
3. The novelty is concentrated in one object (`μ*`) whose ablation is
   load-bearing and whose effect is observable (`coverage_n`).
4. None of the eight dead families or fifteen disqualifier shapes
   match.
5. The vector stage is the right substrate: the principle exists only
   when `m ≥ 2`, and the runs 13-17 failure pattern is the exact
   thing the non-monotone weight is designed to break.

Engineer: when implementing, please ensure (a) `info["vector"]` is
read directly rather than reconstructed from scalar reward, (b) the
sparse-env synthetic-vector fallback is *not* used in the vector
stage runs, (c) the running-mean baseline is taken over a window
comparable to `N`, and (d) the Pareto filter and KDE evaluation are
on `y_τ` (the discounted vector return), not on per-step rewards.
