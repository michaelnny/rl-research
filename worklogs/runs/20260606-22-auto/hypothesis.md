# 20260606-22-auto -- PARGRAD (Per-Step Pareto-Rank Score-Function Gradient) [probe]

## Principle

Update the policy by score-function ascent where each step's gradient
weight is the **fraction of historical bivariate cumulative-return
points weakly dominated by the current rollout's cumulative-return
point at the same trajectory time index, with at least one coordinate
strict** -- a per-step weight that is rank-invariant under componentwise
strictly-monotone reparametrization (non-scalarizing), is bounded in
`[0, 1]`, fires at random initialization without any treasure being
found, AND has a structural Pareto-progress direction (high weight
means the current rollout Pareto-dominates many historical rollouts at
this trajectory time).

## Primitive

The **per-time-index empirical bivariate cumulative-return measure**

  `μ̂_t : Borel(ℝ²) → [0,1]`,
  `μ̂_t(B) := (1/N_t) · |{ j ∈ buffer_t : (M^1_{t,j}, M^2_{t,j}) ∈ B }|`

defined for each trajectory time index `t ∈ {0, …, T_max}`, where
`buffer_t` is a rolling window of the last `N` historical bivariate
cumulative-return pairs `(M^1_{t,j}, M^2_{t,j})` observed at trajectory
time `t` across past trajectories, and `M^c_t := Σ_{k ≤ t} γ^k · r^c_k`
is the cumulative discounted reward of channel `c` up to time `t`.
Domain: Borel sets of `ℝ²`. Codomain: `[0, 1]` (probability mass).

This is **one typed mathematical object**: a finite-support discrete
probability measure on `ℝ²` for each `t`, implicitly represented by a
small per-`t` ring buffer of size `N` storing past bivariate cumulative-
return pairs. The Pareto-rank weight is the **weak-dominance-with-at-
least-one-strict** count

  `p_t(τ) := μ̂_t({ z ∈ ℝ² : z ≼ y_t(τ) and z ≠ y_t(τ) })`

with `y_t(τ) = (M^1_t(τ), M^2_t(τ))` and `≼` componentwise weak
dominance (`z ≼ y` iff `z^1 ≤ y^1` and `z^2 ≤ y^2`). Equivalently,
`p_t(τ) = (1/N_t) · |{ j : z_j^1 ≤ y_t^1 and z_j^2 ≤ y_t^2 and (z_j^1,
z_j^2) ≠ y_t }|`. This is the standard Pareto-dominance relation `≺`
used in multi-objective optimization (Deb 2002 §I.B; Coello-Lamont-
Van Veldhuizen §1.3): `y` Pareto-dominates `z` iff `y` is at least as
good in every coordinate and strictly better in at least one. The
measure is the single primitive; `p_t` is its evaluation on the
lower-Pareto-cone of the realized rollout's `t`-th cumulative-return
point.

This measure is **not** a value function (it does not predict any
return-to-go), **not** a Q-function (no state-action argument), **not**
an advantage estimator, **not** a state-action distribution, **not** a
state-bucketed tensor (no state index, no action index). It is a
statistic of the bivariate cumulative-return process indexed only by
trajectory time `t`.

## Derivation sketch

1. **Bivariate cumulative-return process.** For a 2-channel vector-reward
   MDP with per-step `r_vec_t = (r^1_t, r^2_t) ∈ ℝ²`, a trajectory `τ`
   under policy `π_θ` induces a bivariate cumulative process
   `(M^1_t, M^2_t)_{t=0..T}` with `M^c_t = Σ_{k ≤ t} γ^k r^c_k`. This
   is a 2D random walk indexed by trajectory time.

2. **Empirical bivariate measure at each time index.** For each fixed
   time `t`, the population-level joint distribution
   `ν^π_t := Law((M^1_t, M^2_t) | π)` under `π_θ` is well-defined on
   `ℝ²`. The empirical measure `μ̂_t` from a ring buffer of past
   trajectories' values at time `t` converges weakly to `ν^π_t` under
   stationarity of the on-policy distribution (Glivenko-Cantelli on
   bivariate empirical CDFs; Vapnik-Chervonenkis 1971 for the rectangle
   class).

3. **Pareto-rank as L-statistic.** Define the per-step Pareto-rank
   weight `p_t(τ) := μ̂_t({ z ∈ ℝ² : z ≼ y_t(τ), z ≠ y_t(τ) })` where
   `y_t(τ)` is the realized rollout's bivariate cumulative-return at
   time `t` and `≼` is **componentwise weak dominance**. Equivalently
   `p_t(τ) = (1/N_t) · |{j : z_j^1 ≤ y_t^1 and z_j^2 ≤ y_t^2 and
   (z_j^1, z_j^2) ≠ y_t}|`. This is the **standard Pareto-dominance
   count** (Deb 2002): `y` Pareto-dominates `z` iff `y` is at least as
   good in every coordinate and strictly better in at least one. The
   `at-least-one-strict` clause excludes ties as self-dominations.

4. **Why p_t is not scalarization.** `p_t(τ)` is invariant under any
   componentwise strictly-increasing reparametrization
   `(r^1, r^2) ↦ (φ^1(r^1), φ^2(r^2))`: weak dominance and
   the `≠` clause are both preserved by componentwise strictly-monotone
   maps, and `p_t` depends only on the dominance pattern. Any linear or
   non-linear scalarization `f(r^1, r^2)` that is **not** purely a
   function of dominance patterns is sensitive to such
   reparametrizations. Concrete counterexample: replace
   `r^1_t ← log(1 + exp(r^1_t))`. Then `p_t(τ)` is unchanged (weak
   dominance and the `≠` clause are both preserved); any
   `wᵀ(r^1, r^2)` or any non-linear `f(r^1, r^2)` that is monotone in
   `r^1` changes. Hence `p_t ≠ f(r^1, r^2)` for any deterministic
   scalarizing function `f`.

5. **Why p_t is non-zero at random init AND on degenerate-channel
   substrates.** At random init, the buffer accumulates a spread of
   `(M^1_t, M^2_t)` values. The current rollout's `y_t(τ)` is sampled
   from approximately the same distribution, so `p_t` is approximately
   distributed in `[0, 1]` by symmetry of Pareto-dominance under
   exchangeable bivariate samples. For DST-concave, channel 2 is
   deterministic given survival: every trajectory still alive at time
   `t` has the same `M^2_t = -(1 - γ^{t+1})/(1-γ)` value (note we only
   push `(M^1_t, M^2_t)` to `ring_buffer[t]` for trajectories that
   reached time `t`). Under **weak dominance with at-least-one-strict**,
   `z^2 ≤ y^2` is then satisfied with equality for every alive buffer
   entry, the strictness must come from channel 1, and `p_t` reduces to

     `p_t(τ) = (1/N_t) · |{ j : M^1_{t,j} < M^1_t(τ) }|`,

   the **fraction of historical alive-at-`t` trajectories whose
   channel-1 cumulative is strictly below the current rollout's**.
   This is non-trivial at random init because some historical episodes
   have already collected treasure by time `t` and some have not.
   (Under the previously-stated *strict* dominance the channel-2
   equality killed `p_t` identically -- this revision restores the
   load-bearing reduction.)

6. **Per-step policy gradient.** The PARGRAD update is
   `∇_θ J̃ := E_τ[ Σ_{t=0}^{T} p_t(τ) · ∇_θ log π_θ(a_t | s_t) ]`,
   i.e., score-function ascent with **per-step weight `p_t`**. The
   weight is the **empirical probability of Pareto-dominance** at each
   step, not the cumulative return. Each gradient term up-weights
   actions that contributed to a step where the rollout Pareto-dominates
   a large fraction of historical bivariate cumulative-return points at
   the same `t`.

7. **Pareto-direction property.** Unlike COPDEV's
   `|F̂_t^1 - F̂_t^2|` (rank-disagreement, symmetric), `p_t` has a
   structural direction: `p_t` is **monotone non-decreasing** in each
   channel coordinate of `y_t(τ)`. Increasing either `M^1_t` or `M^2_t`
   weakly increases `p_t` (more historical points are Pareto-
   dominated; ties become strict). So the gradient `∇_θ J̃` rewards
   trajectories that improve **either or both** channels at any step,
   without summing or weighting them. This is the **directional**
   property COPDEV lacked.

8. **Discriminating training-dynamics observable.** Two scalars logged
   per rollout:
   (i) `gradnorm_var = Var_t(||g_t||) / Mean_t(||g_t||)²`,
       the cv² of per-step gradient norms. Non-zero by construction
       when `p_t` varies across `t`.
   (ii) `mean_pt_trend := <p_t> averaged over the rollout`,
       which should **drift upward** over training if the policy is
       moving toward Pareto-dominating trajectories. Under the
       random-uniform ablation (where `p_t ← Uniform[0,1]` per step),
       `mean_pt_trend` (computed for logging only against the same
       buffer in the ablation) stays at ~0.5 in expectation regardless
       of training. So `mean_pt_trend` is a **directional**
       discriminator not available to COPDEV. Its expected drift is
       the Pareto-progress signature.

9. **Why this is not a within-trajectory geometric statistic
   (Family C).** Family C uses path geometry of the cumulant trace
   *within a single trajectory* (hull, Lévy area, signature, spectrum)
   to drive logits. PARGRAD uses **cross-trajectory empirical
   dominance** at each `t` against a historical buffer. The weight
   `p_t(τ)` cannot be computed from `τ` alone -- it requires the
   buffer. Rank invariance under componentwise monotone maps
   (point 4) excludes the geometric-statistic angle (which is
   sensitive to magnitudes).

10. **Why this is not Family A bucketed-tensor + partial-order vote.**
    Family A maintains a tensor `T[bucket, action, channel]` indexed
    by some bucketing of *observations*. PARGRAD has no state-
    bucketing, no action-bucketing, no channel index. The data
    structure is `T_max` ring buffers, each indexed by trajectory time
    `t` only. There is no *vote* -- `p_t` is a real-valued L-statistic
    (a count fraction), not a partial-order indicator. No Pareto/
    Kemeny/sup-norm/strict-superset rule is applied to drive a logit
    nudge. The gradient weight is a continuous real number used in
    score-function ascent.

11. **Why this is not NSGA-II / non-dominated sort.** NSGA-II is an
    *evolutionary* method that sorts a *population* of *whole
    individuals* into Pareto fronts and selects elites for
    reproduction. PARGRAD has a single policy and no population; it
    does not compute Pareto fronts; it computes a per-step empirical-
    dominance count as a continuous gradient weight on a single on-
    policy rollout. NSGA-II's selection is a discrete top-k operation,
    while PARGRAD's update is a per-step score-function gradient with
    continuous weights.

12. **Proof debt (open).** (i) Convergence of the PARGRAD stochastic
    gradient under the rolling-buffer dynamics (two-timescale
    Borkar 2008). (ii) Identification of the optimized objective:
    conjecture, the stationary objective is
    `J̃(π) := E_τ[Σ_t F̃^π_t(y_t(τ))]` where
    `F̃^π_t(y) := P(Z ≼ y, Z ≠ y)` for `Z ~ ν^π_t`, the **expected sum
    of bivariate Pareto-rank along trajectories**. (iii) Pareto-
    improvement claim: do maximizers of `J̃` lie on the Pareto frontier
    of the achievable vector-return set? Marked open. (iv) Bias and
    variance of the empirical bivariate measure on a finite ring
    buffer.

## Update rule

```
Inputs: env (vector reward, m=2), policy π_θ, discount γ,
        learning rate α, buffer size N (per t), max time T_max
Init:   θ random; for each t ∈ {0,...,T_max}:
        ring_buffer[t] = deque(maxlen=N), empty.

For each episode:
    1. Roll out τ = (s_0, a_0, r_vec_0, ..., s_T) under π_θ, T ≤ T_max.
       Read r_vec_t per-component from info["vector"] for the vector
       env (DST-concave: m=2; channel 1 = treasure, channel 2 = step
       penalty). NEVER sum the channels; never compute wᵀr_vec.

    2. Compute cumulative per-channel values:
         For t = 0..T, c ∈ {1, 2}:
             M[t, c] = Σ_{k ≤ t} γ^k · r_vec_{k, c}

    3. Compute per-step Pareto-rank weight p_t (weak dominance with
       at-least-one-strict; standard Pareto-dominance, Deb 2002):
         For t = 0..T:
             y1, y2 = M[t, 1], M[t, 2]
             N_t = len(ring_buffer[t])
             if N_t == 0:
                 p_t = 0.5     # symmetric prior at empty buffer
             else:
                 dom_count = 0
                 for (z1, z2) in ring_buffer[t]:
                     if (z1 <= y1) and (z2 <= y2) and not (z1 == y1
                                                          and z2 == y2):
                         dom_count += 1
                 p_t = dom_count / N_t                         # in [0,1]

    4. Push current values into the rolling buffer:
         For t = 0..T:
             ring_buffer[t].append((M[t, 1], M[t, 2]))

    5. Score-function policy gradient with per-step weight:
         g_t = ∇_θ log π_θ(a_t | s_t)  for each t = 0..T
         g_θ = Σ_t p_t · g_t
         θ ← θ + α · g_θ

    6. Logging observables (load-bearing for ablation discrimination):
         - mean_pt = mean(p_t for t in 0..T)       # Pareto-progress signal
         - var_pt  = var(p_t for t in 0..T)        # weight non-uniformity
         - gradnorm_var = Var_t(||g_t||) / Mean_t(||g_t||)²
         - mean episode length T
         - first-rewarded-episode-index (when channel-1 first non-zero)
         - mean_pt_trend.txt: append mean_pt per episode for
           Curator inspection of Pareto-progress drift over training.
```

The load-bearing primitive is the **per-`t` empirical bivariate measure
`μ̂_t`**, expressed as a ring buffer per `t`. The Pareto-rank weight
`p_t` is its evaluation by weak-dominance-with-at-least-one-strict
counting on the realized rollout. Total storage: `O(T_max · N)`
bivariate scalars. Per-step lookup: `O(N)` (linear scan). Negligible
compared to backprop.

## Empirical claim

stage: quick

claim: On the **quick** stage (deep-sea-treasure-concave-v0; vector
reward `(r^1, r^2)` with channel 1 = treasure pickup at terminal and
channel 2 = `-1` step penalty), PARGRAD should produce:
(a) **`gradnorm_var > 0.1`** within the first 50 episodes (well within
    120s budget), strictly larger than the random-rank ablation's
    `gradnorm_var` by at least 2× (replicating COPDEV's load-bearing
    training-dynamics signal);
(b) **a measurably upward-drifting `mean_pt_trend`** over training: the
    rollout-average `mean_pt` should grow from ~0.25 (early,
    exchangeable buffer) toward a higher value (say > 0.4) as the
    policy concentrates on Pareto-dominating trajectories. Under the
    random-uniform ablation (`p_t ← Uniform[0,1]`), `mean_pt_trend`
    (computed for logging only against the same buffer) stays at ~0.5
    throughout (no drift toward dominance);
(c) **a final hypervolume score on DST-concave that is strictly above
    the random-baseline floor (194)**. PARGRAD is designed to pull the
    policy toward dominating trajectories, so the score should move
    above random unless the gradnorm direction does not in fact
    correspond to useful policy updates on this substrate.

The quick stage is the appropriate test because (i) DST-concave's
2-channel vector reward exactly instantiates the bivariate Pareto-rank
mechanism; (ii) the discriminating observable `gradnorm_var` is a
training-dynamics scalar that fires at random init; (iii) the new
directional observable `mean_pt_trend` distinguishes PARGRAD from
COPDEV's symmetric mechanism.

falsifier:

**Primary** (presence): if `gradnorm_var` for PARGRAD equals the
random-rank ablation's within seed variance during the first 50
episodes, the per-step weight is empirically constant across `t` and
the primitive is decorative.

**Secondary** (directionality, this is the new test): if PARGRAD's
`mean_pt_trend` does **not** drift above 0.5 over training (i.e., the
policy does not concentrate on dominating trajectories) -- even when
`gradnorm_var` is non-zero -- then the Pareto-direction property is
not load-bearing in practice.

**Tertiary** (substrate signal): if PARGRAD's final DST-concave score
matches random (194) within seed variance, the Pareto-rank gradient
weight produces no measurable shift in policy return on this substrate
within budget.

**Reduction-to-unicriterial** (sanity): if PARGRAD matches the
unicriterial channel-1-only sanity ablation (see Ablation plan) on
`mean_pt_trend` and final DST-concave score within seed variance, the
bivariate primitive collapses on this deterministic-channel-2 substrate
to a single-channel rank weight, and the joint-bivariate structure is
not load-bearing here -- a neutral but informative outcome that
demands re-testing on a non-degenerate vector substrate before claiming
directional Pareto-progress.

## Ablation plan

### Primary ablation (random-uniform per-step weight)

Replace the **per-step Pareto-rank weight `p_t`** with a **uniform
random number sampled fresh per step**:

In `train_ablate.py`:
1. Skip step 3's dominance count entirely; do not read
   `ring_buffer[t]`.
2. Sample `p_t ~ Uniform(0, 1)` independently for each `t = 0..T` per
   rollout.
3. Skip step 4 entirely (no buffer maintenance).
4. Apply the same score-function update with the random `p_t`.

This preserves: per-step score-function gradient computation, all
hyperparameters, the policy architecture, the rollout mechanism, the
*non-uniformity of per-step weights across `t`* (so `gradnorm_var`
should remain non-zero in the ablation). It removes: the *systematic
Pareto-direction* of `p_t`. The ablation's `mean_pt_trend` (logged for
diagnostic purposes by recomputing the dominance count against a
shadow buffer maintained for logging only) stays at ~0.5 in
expectation.

### Sanity ablation (channel-1-only unicriterial rank), retained per
### reviewer recommendation

Reviewer flagged that on DST-concave the joint-bivariate primitive
**reduces** to `(1/N_t) · |{ j : M^1_{t,j} < M^1_t(τ) }|` (alive-at-`t`
trajectories with lower channel-1 cumulative; see derivation step 5),
which is dangerously close to a unicriterial channel-1 rank weight.
This sanity ablation explicitly tests whether the bivariate structure
is load-bearing on this substrate.

In a second ablation file (or as a second arm under
`train_ablate.py`):

1. Maintain a **single-channel** ring buffer `ring_buffer1[t]` of past
   `M^1_{t,j}` values only. No channel-2 or joint structure stored.
2. Compute `p_t = (1/N_t) · |{ j : M^1_{t,j} < M^1_t(τ) }|`, the
   marginal channel-1 rank below the current rollout.
3. Apply the same score-function update with this unicriterial weight.

Predicted contrast: if PARGRAD matches this unicriterial arm on
`mean_pt_trend` and on final hypervolume, the bivariate structure adds
nothing on DST-concave (because channel 2 is deterministic given
survival) -- a Curator-readable null on the bivariate claim, even if
the primary ablation already shows PARGRAD outperforms random-uniform.
This null does not falsify the *directional Pareto-rank* mechanism
generally; it falsifies the *bivariate* claim on this specific
substrate and demands a non-degenerate vector retest.

### Discriminator predictions on DST-concave (within 120s budget)

- **Discriminator (i) -- gradnorm_var**: PARGRAD `gradnorm_var > 0.1`
  consistently. The random-uniform ablation should also be non-zero;
  this observable is **not** the primary discriminator -- it confirms
  the per-step-weight scaffolding is intact in both arms.

- **Discriminator (ii) -- mean_pt_trend (PRIMARY for this probe)**:
  PARGRAD should show `mean_pt_trend` drifting upward over training.
  The random-uniform ablation's `mean_pt_trend` (logged for diagnostic
  comparison via a shadow buffer) should hover near `0.25`–`0.5` with
  no systematic drift. The drift difference in the last 30% of
  training versus first 30% is the load-bearing comparison.

- **Discriminator (iii) -- final DST-concave score**: PARGRAD should
  exceed the random-floor 194; the random-uniform ablation should
  hover at or below random.

- **Discriminator (iv) -- bivariate-vs-unicriterial reduction**:
  PARGRAD's drift and final score on DST-concave compared against the
  unicriterial channel-1 arm. Equal performance flags the substrate-
  level reduction; strictly better performance for PARGRAD would be a
  surprising positive on a substrate where the reduction was expected.

If the random-uniform ablation matches PARGRAD on `mean_pt_trend`, the
Pareto-direction primitive is not load-bearing. If PARGRAD shows
strictly upward `mean_pt_trend` AND a higher DST-concave score than
random-uniform, the rank-based gradient weight is causally responsible
for the Pareto-progress dynamics; whether the **bivariate** structure
matters above and beyond a unicriterial channel-1 rank is settled by
the second sanity ablation.

## Novelty boundary

Closest known methods:

(a) **REINFORCE / vanilla policy gradient** (Williams 1992). REINFORCE
    weights every step's score by the cumulative return `G_t`. PARGRAD
    weights every step by the empirical-dominance count `p_t`, which
    is rank-invariant and depends on a cross-trajectory bivariate
    historical buffer. Different per-step weight, structurally
    distinct from any function of cumulative return.

(b) **COPDEV (run 21, this loop)**. COPDEV uses
    `|F̂_t^1(M^1_t) - F̂_t^2(M^2_t)|`, the L¹ distance between two
    marginal CDF rank evaluations -- a **symmetric** rank-disagreement
    statistic with no Pareto-direction. PARGRAD uses the *joint*
    empirical bivariate measure's weak-dominance-with-at-least-one-
    strict count, which is **monotone** in each channel coordinate
    and has a structural Pareto-direction. The primitives are
    different: COPDEV maintains 2 marginal CDFs; PARGRAD maintains 1
    bivariate measure.

(c) **NSGA-II / non-dominated sort** (Deb 2002). NSGA-II is an
    *evolutionary* multi-objective method that sorts a *population* of
    *whole individuals* into Pareto fronts and uses front-rank for
    selection. PARGRAD borrows the *dominance relation* `(z ≼ y, z ≠
    y)` from Deb 2002 §I.B but applies it as a **per-step continuous
    gradient weight** against a historical buffer, not as a discrete
    selection operator on a population. PARGRAD has a single policy,
    no population, no front-sorting.

(d) **MO-MCTS / Pareto-MCTS** (Wang-Sebag 2012). These extend MCTS to
    multi-objective settings with non-dominated tree-node selection.
    PARGRAD has no tree search, no node selection, no model.

(e) **Pareto-Frontier KDE / PRISM (run 18, this loop)**. PRISM uses
    a rolling KDE of *terminal* vector returns on the Pareto frontier
    and a per-trajectory log-density weight. PARGRAD uses *per-time-
    step* bivariate empirical-dominance counts against the *full*
    historical buffer (not just non-dominated points), per-step (not
    per-trajectory).

(f) **Order-statistics / ordergrad / Pass@K policy gradient** (Yu
    2014). These rank *whole trajectories* by *scalar* total return.
    PARGRAD uses per-step bivariate Pareto-rank, not per-trajectory
    scalar rank.

(g) **Distributional RL** (Bellemare 2017; Dabney 2018). Distributional
    RL learns a return distribution `Z(s, a)` per (state, action).
    PARGRAD has no learned distribution; only an empirical bivariate
    measure of *time-indexed cumulative channel values*, not state-
    action distributions. No distributional Bellman operator.

(h) **Rank-based prioritized experience replay** (Schaul 2015). PER
    uses TD-error rank to set the *probability of sampling a
    transition from the replay buffer*. PARGRAD has no replay buffer,
    no TD-error, and the rank is applied as a **per-step gradient
    weight** on an on-policy rollout -- different mechanism slot.

(i) **Linear / non-linear scalarization of vector reward** (explicit
    disqualifier). PARGRAD is **not** scalarization: `p_t` is invariant
    under componentwise strictly-monotone reparametrization of either
    channel (weak dominance and the `≠` clause are both preserved by
    such maps). Counterexample: replace
    `r^1_t ← log(1 + exp(r^1_t))`. Then `p_t` is unchanged; any
    `wᵀ(r^1, r^2)` changes.

(j) **Multi-objective MDP linear scalarization** (Roijers-Whiteson
    2017, §3). The standard MO-MDP approach scalarizes the vector
    reward by a fixed weight vector `w` (or by Chebyshev / piecewise-
    linear scalarization). PARGRAD does not scalarize.

(k) **Hypervolume-based MORL / HV-PG** (Beume 2007). Hypervolume is a
    *terminal* metric on the achievable set used to evaluate a Pareto-
    frontier approximation. PARGRAD's `p_t` is a per-step empirical-
    dominance count along trajectories, not a hypervolume of any
    frontier set.

(l) **Quantile regression / quantile RL** (Dabney 2018). Quantile RL
    learns the quantile function of the return distribution at each
    state. PARGRAD does not learn quantiles or any state-conditioned
    distribution.

(m) **GRADCOMP (run 20, this loop)**. GRADCOMP rotates the parameter
    update direction toward the rollout's Fisher-principal-eigen-
    direction. PARGRAD does not modify the *direction* of the update;
    it modifies the *magnitude* of each per-step score-function
    contribution. Different slot.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: PARGRAD has no
  state/action bucketing. The data structure is `T_max` ring buffers,
  each indexed by trajectory time `t` only. No partial-order *vote* is
  applied to drive a logit nudge; the Pareto-rank `p_t` is a real-
  valued L-statistic used as a continuous gradient weight in score-
  function ascent.
- **Family B (pairwise trajectory comparison)**: PARGRAD does not pair
  trajectories; the buffer aggregates many trajectories.
- **Family C (within-trajectory geometric statistic)**: PARGRAD's
  weight `p_t` depends on the cross-trajectory historical buffer at
  `t`. Rank invariance under componentwise monotone maps further
  excludes the geometric-statistic angle.
- **Family D (reward-independent + reward-gated)**: `p_t` is reward-
  dependent throughout (uses cumulative bivariate returns), and the
  application is also reward-dependent.
- **Family E (avoid value vocabulary)**: No value, no Q, no advantage,
  no return-to-go compression.
- **Family F (hand-engineered structural priors)**: No vocabulary, no
  symbol grammar; the only data structure is a ring buffer per `t`.
- **Family G (mechanism stack)**: Single primitive -- the per-`t`
  empirical bivariate measure.
- **Family H (cochain complexes)**: Not applicable.

The structural difference from all named methods and dead families is
the **per-step empirical Pareto-rank (weak dominance with at-least-
one-strict, the standard Pareto-dominance relation) against a per-`t`
rolling historical bivariate buffer, used as a directional rank-
invariant score-function gradient weight**.

## Proof debt

1. **Convergence under non-stationary buffer.** Conjecture: under
   PARGRAD updates with rolling buffers of fixed size N per `t` and
   Robbins-Monro step sizes for `θ`, the iterate `θ_n` converges to a
   stationary point of the modified objective
   `J̃(π) := E_τ[ Σ_t F̃^π_t(y_t(τ)) ]`,
   where `F̃^π_t(y) := P_{Z ~ ν^π_t}(Z ≼ y, Z ≠ y)` is the bivariate
   Pareto-dominance probability at trajectory time `t`. Strategy:
   two-timescale Borkar 2008 with the buffer on the slow timescale.

2. **Pareto-improvement claim (load-bearing).** Conjecture: at a
   stationary point `(π*, F̃*)` of the joint dynamics, `π*` is
   **Pareto-non-dominated** in the achievable bivariate vector-return
   set. Rigorous proof open; the empirical `mean_pt_trend` observable
   is the substrate test of this conjecture.

3. **Bias and variance of the empirical bivariate measure on a finite
   buffer.** The bivariate empirical CDF from a buffer of size N has
   bias `O(1/N)` and variance `O(1/N)` against the true bivariate
   marginal `ν^π_t` (Vapnik-Chervonenkis 1971 for the rectangle class
   with VC-dim 4 in `ℝ²`).

4. **Degenerate-channel reduction on DST-concave.** When channel 2 is
   deterministic given survival, weak-dominance-with-at-least-one-
   strict reduces to the channel-1-strict alive-and-lower count
   `(1/N_t) · |{j : M^1_{t,j} < M^1_t(τ)}|`. Open: characterize when
   this reduction is structurally distinct from a single-channel rank
   weight (handled by the unicriterial-ablation sanity check) and when
   it remains a genuine bivariate primitive (non-degenerate vector
   substrate retest).

5. **Connection to bivariate stochastic dominance.** Conjecture: the
   stationary maximizers of `J̃` correspond to policies whose induced
   bivariate cumulative-return distribution is **first-order
   bivariate-stochastically-dominant** over any non-stationary
   competitor (Atkinson-Bourguignon 1982; Hadar-Russell 1974). Marked
   open.

The empirical probe will reveal whether the per-step Pareto-rank
gradient weight produces (a) a non-zero `gradnorm_var` (mechanism
presence), (b) an upward-drifting `mean_pt_trend` (Pareto-progress
direction), and (c) a final DST-concave score above the random floor
(useful policy update), and (d) whether the bivariate primitive is
load-bearing above the unicriterial channel-1 rank on this substrate.
Positive (b) and (c) on the quick stage would justify investing in
proof items (1), (2), and (5).
