# 20260606-25-auto -- empty-handed

reason: After reading the exemplars, prior_attempts.md (especially the
new Family I IMTL-G entry added after run 24), the recent ledger, and
the curator/hypothesis files for runs 20-24, I cannot in honesty produce
a coherent non-rebadged probe in the directions still open under the
substrate-budget constraints.

The corner I am pinned to is small. Inventory of what remains structurally
distinct from already-closed regions, and why each tempting candidate I
considered today does not survive its own justification:

1. **Mirror-descent / primal-dual region.** Mirror descent on the policy
   simplex is policy mirror descent (Agarwal et al. 2021, listed as the
   exemplar entry, not a new direction). Adding a Bregman geometry derived
   from rollout statistics either (a) scalarizes vector reward in the
   linearization term, or (b) collapses to per-channel parameter-space
   gradient aggregation in the dual update -- exactly Family I (IMTL-G,
   newly closed). Neither escape passes the Family I disqualifier.

2. **MCTS / local-regret region.** The exemplar is UCB1 at every tree
   node; a fresh probe needs a *non-tree* local-regret structure. Per-
   state local mirror-ascent with vector-reward regret reduces to a
   per-state-bucketed `T[state, action, channel]` tensor (Family A) once
   the regret is estimated from rollouts; without per-state bucketing the
   "local" qualifier is decorative.

3. **Soft-Bellman / SAC region for vector reward.** A vector-valued soft
   Bellman fixed point requires a vector-valued log-sum-exp or a per-
   channel value object; the policy step `π* propto exp(<Q^vec, w*>/alpha)`
   then scalarizes via `w*`, which is a learned-w scalarization
   disqualifier. Replacing the scalarization with a Pareto-frontier mask
   over Q^vec collapses to Family A (state-action-channel bucketing).

4. **Reward-free trajectory functionals (action-sequence entropy,
   action-autocorrelation, sequence-permutation symmetry, Lyapunov-of-
   score-flow).** Any trajectory-level scalar weight on a vector-reward
   env, when multiplied by the standard score-function-times-return
   update, scalarizes the vector reward. The escapes that previously
   worked (rank-invariant per-step weight; per-channel gradient
   aggregation) are exactly the two regions just closed (COPDEV/PARGRAD
   per-step rank; CHANBI/IMTL-G per-channel gradient).

5. **Population / Gibbs-posterior over policy parameters with Pareto-
   rank fitness.** Structurally close to ES/CMA-ES with rank-based
   fitness; the disqualifier list closes this as "CEM/ES/CMA-ES elite
   refitting" rebadge unless the update is genuinely SVGD-with-Stein-
   kernel and not selection. SVGD plus Pareto-rank fitness on K=2 with
   N~10 particles in 120s on DST-concave is a population-method probe
   in everything but name; reviewer would correctly read it as a
   rebadge.

6. **Mahalanobis self-distance from rolling buffer in vector-return
   space.** Quadratic-form weighting by a learned positive-definite
   matrix is a non-`w^T r` functional, but it is structurally a
   parametric novelty/outlier score with the rolling buffer as the
   density model -- adjacent to RND-flavored rebadges, and the
   reviewer correctly will not distinguish it from a learned-novelty
   signal once the matrix is updated online.

7. **Substrate degeneracy on DST-concave.** On the only quick-stage env
   that produces reward signal in 120s, channel 2 (step penalty) is
   strictly monotone decreasing in t and channel 1 is non-decreasing,
   so most within-rollout rank-invariant statistics
   (within-trajectory Kendall-tau, copula deviation along time) are
   degenerate. Cross-trajectory rank statistics are exactly the
   per-step rank-weight family (closed). This is the structural
   reason why COPDEV/PARGRAD-flavored escapes keep being the only
   option that fires, and why they keep being closed.

What I would need to break out of this corner, in order of priority:

- A **non-vector quick-stage env** so that scalar-return primitives like
  TRACE (run 23, rejected only for vector-stage scalarization, primitive
  itself was clean) become testable, OR
- A **non-degenerate vector-stage env** within the 120s budget so that
  bivariate rank statistics on the cumulative process are not collapsed
  by channel monotonicity, OR
- A genuinely fresh principle from outside the seven exemplar regions
  (Q-learning, PPO, mirror descent, AlphaZero, SAC, MCTS, GAE) -- not
  a recombination of two of them.

None of these is a Researcher-side fix in this turn. Writing a forced
probe in any of regions 1-6 would be a rebadge that fails Reviewer
triage or, worse, passes triage and burns 120s of compute reproducing
a result already in the ledger (COPDEV/PARGRAD/CHANBI). Halting on
empty-hand is the cheaper outcome.
