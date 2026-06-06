---
verdict: probe
reviewer_run: 20260606-20-auto
hypothesis_type: probe
---

## Summary

GRADCOMP rotates the per-rollout REINFORCE direction toward the top
eigenvector of the empirical Fisher (computed cheaply via the rank-T
score-Gram trick), with a gradient-magnitude-annealed slerp angle and a
cold-start magnitude floor; the revised version targets the `sparse`
stage where REINFORCE has g_tau = 0 until first reward, removing the
prior scalarization concern, and the principle / primitive / update /
ablation are coherent and implementable.

## Schema check

`candidate.json` validates structurally against
`scripts/validate_candidate.py`: all required string fields are
non-empty, both booleans typed, `update_family="direct_policy_update"`,
`memory="none"`, `nearest_disqualifier="policy_gradient"`, and
`claimed_stage="sparse"` is a known stage. `uses_vector_reward=false`
is consistent with `STAGES["sparse"] = ["MiniGrid-DoorKey-8x8-v0",
"MiniGrid-KeyCorridorS3R3-v0"]`, both `ENV_TYPE = scalar`, so no
substrate-rule conflict.

Prose-vs-schema agreement:
- `principle`, `primitive_name`, `primitive_type` match
  `## Principle` and `## Primitive` exactly.
- `claimed_stage="sparse"` matches the empirical-claim header
  `stage: sparse` and the named envs DoorKey-8x8 + KeyCorridorS3R3.
- `empirical_claim` and `falsifier` match `## Empirical claim`
  (primary, secondary, tertiary falsifier conditions all reflected).
- `ablation_plan` matches `## Ablation plan` (random-unit-vector
  replacement preserving slerp + eta + floor + hyperparameters), plus
  the secondary eta-equiv-1 sanity ablation.
- `nearest_disqualifier` and `novelty_boundary` match
  `## Novelty boundary` (NPG inverse-Fisher vs GRADCOMP top-eigenvector
  rotation, TRPO/PPO trust region, ES random perturbation, no curiosity
  reward modification).

## Coherence check

The derivation reads cleanly:

- Step 1 (empirical Fisher PSD, rank <= T) is standard.
- Step 2 (REINFORCE score-weighted sum) is standard and now keyed to
  the natively scalar reward of the sparse-stage MiniGrid envs.
- Step 3 (cold-start failure mode of REINFORCE: R_t = 0 -> g_tau = 0)
  is exactly the regime DoorKey/KeyCorridor produces under random init.
- Step 4 (slerp between v1 and g_hat with floor magnitude
  ||g_tau||_* = max(||g_tau||, c)) introduces a single load-bearing
  primitive: one unit vector v1(tau) per rollout in parameter tangent
  space.
- Step 5 (v1 non-zero whenever >= 2 distinct (s,a) with non-orthogonal
  scores) is correct; DoorKey episodes always have T >= 5.
- Step 6 (not curiosity / not reward-modifying) is correct.
- Step 7 (sigmoid annealing eta = sigma((||g_tau|| - c)/c)) is
  heuristic but explicit.
- Step 8 (slerp preserves unit norm; magnitude controlled solely by
  ||g_tau||_*) is correct.
- Step 9 (NPG amplifies low-Fisher; GRADCOMP rotates toward high-Fisher;
  NPG produces 0 update at g_tau = 0 while GRADCOMP's floor c does not)
  is the cleanest novelty boundary.

The Gram-trick reduction `v1 = (sum_t alpha_t g_t) / ||.||` from the top
eigenvector of `G_{ij} = g_i^T g_j` is a standard rank-T eigendecomp of
the empirical Fisher and is correct.

The proof debt items (convergence under annealed eta, effective-rank
monotonicity in the cold phase, alignment limit at convergence,
Gram-estimator variance, first-reward-time bound) are named and
appropriate for post-empirical theorem work, not pre-empirical blocking.

The principal structural caveat: at warm operation eta -> 1 GRADCOMP
reduces to REINFORCE exactly, so the entire novelty lives in the
cold-phase rotation and how often it fires before the policy departs
the floor. The sparse stage is the natural test bed for this exact
regime, and the empirical claim and discriminating observables target
it directly.

## Novelty check

Searches: "policy gradient rotation Fisher principal eigenvector
REINFORCE slerp", "policy gradient top eigenvector empirical Fisher
update direction", "rank-1 inverse Fisher policy gradient 2026". The
nearest published method is rank-1-inverse-Fisher NPG approximations
(e.g. Huo et al. 2026), which still operate on F^{-1} (low-Fisher
amplification, opposite spectral end). I did not find a published method
that performs slerp between the REINFORCE direction and the top
eigenvector of F with a magnitude floor for cold start.

Dead-family check against `prior_attempts.md`:
- Family A (bucketed-tensor + partial-order vote): not applicable.
- Family B (pairwise trajectory comparison): not applicable.
- Family C (within-trajectory state-geometric statistic): hypothesis
  correctly argues v1 is a parameter-tangent statistic of policy
  variation, not a state-geometric one.
- Family D (reward-free + reward-gated firing): hypothesis correctly
  argues GRADCOMP is "always-fire with rotating direction"; the floor
  c keeps the cold-phase update non-zero.
- Family E (avoid value vocabulary, keep value structure): no value /
  Q / advantage; only theta is learned.

Standard disqualifier list: not Q-learning, Bellman, PPO clip, MCTS,
RND, count-based, options, HER, successor features, distributional RL,
DT conditioning, RLHF/DPO. Not a vector-scalarization rebadge (sparse
envs are natively scalar). Verdict: not a rebadge.

## Implementability and ablation check

Implementability: The Engineer can write
`worklogs/runs/<run_id>/train.py` exposing the standard `train(env_id,
seed, time_budget_s) -> PolicyFn` signature with a torch MLP policy,
per-step score collection via autograd, the score-Gram matrix
`G in R^{T x T}`, one Lanczos / power iteration on G, the slerp on the
unit sphere, the gradient-floor mechanism, and the sigmoid eta
annealer. All ops are standard. No invented missing pieces. T is
bounded by `MAX_EPISODE_STEPS = 2000` so G is at most 2000x2000 (or
practically smaller after early termination) and the cost is
negligible relative to backprop.

The sparse-stage envs are natively scalar reward, so no `info["vector"]`
consumption is needed and no scalarization rule is triggered.

Ablation: replacing v1(tau) with a uniformly random unit vector
xi ~ Uniform(S^{n-1}) resampled per rollout, while keeping slerp + eta
annealer + floor + all hyperparameters, is a load-bearing test of the
*trajectory-informed* nature of v1. The discriminating observable
`v1 . g_hat` (positive drift in GRADCOMP, zero in expectation in the
random-direction ablation) is operationally testable. The secondary
sanity ablation (eta == 1 always = pure REINFORCE, no compass) cleanly
isolates whether the rotation does anything beyond the gradient floor.
These ablations are strong.

## Decision

`probe`. The two issues from the prior `revise` review are addressed:
(1) `claimed_stage` is now `sparse` with DoorKey-8x8 and
KeyCorridorS3R3, both scalar-reward envs in `STAGES["sparse"]`; the
empirical claim, falsifier, and ablation observables are all rewritten
in those terms. (2) The disallowed `R_t = sum(info["vector"])`
reduction is gone; the update rule consumes the natively scalar
environment reward only.

The principle (rotation toward high-Fisher direction) is structurally
distinct from NPG (inverse-Fisher amplification of low-Fisher
direction), TRPO/PPO (KL trust region), curiosity (reward-modifying),
and ES (random perturbation). The primitive is one typed unit vector
in parameter tangent space per rollout, computable in O(T*n) via the
score-Gram trick. The update rule is implementable as written. The
ablation isolates the trajectory-informed nature of v1 against a random
unit vector with the same floor and slerp scaffolding. The proof debt
is explicitly listed (convergence, effective-rank monotonicity,
alignment limit, estimator variance, first-reward-time bound) and is
not a pre-empirical block. Empirical signal on whether the cold-phase
Fisher walk reaches first reward earlier than REINFORCE on DoorKey is
exactly what should justify investing in those theorems.
