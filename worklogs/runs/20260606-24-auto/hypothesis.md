# 20260606-24-auto -- CHANBI (Channel-Bisecting Score-Function Direction) [probe]

## Principle

Update parameters along the direction `d* = (ĝ^1 + ĝ^2) / ||ĝ^1 + ĝ^2||`,
the **unit-vector angular bisector** of the per-channel score-function
gradients `ĝ^c := g^c / ||g^c||`, so that the update is invariant under
any componentwise positive rescaling of channels (`r^c ← α_c r^c`,
`α_c > 0`) -- i.e., the load-bearing primitive is the *normalized
per-channel score-function gradient pair* and the update rule is their
spherical sum, not any linear scalarization `wᵀr` with fixed or
adaptive weights.

## Primitive

The **per-channel unit-normalized score-function gradient field**

  `ĝ : Π × Trajectory → (S^{|θ|-1} ∪ {0})^K`,
  `ĝ^c(τ; θ) := g^c(τ; θ) / ||g^c(τ; θ)||` if `||g^c|| > 0`, else `0`,

where for each channel `c ∈ {1, ..., K}` the per-channel score-function
gradient is

  `g^c(τ; θ) := Σ_{t=0}^{T} G^c_t(τ) · ∇_θ log π_θ(a_t | s_t)`,
  `G^c_t(τ) := Σ_{k ≥ t} γ^{k-t} · r^c_k`.

Domain: a single rollout `τ` and policy parameters `θ`. Codomain: a
`K`-tuple of unit vectors (or zero vectors) in the parameter manifold.

This is **one typed mathematical object**: a `K`-tuple of unit vectors
on the policy parameter sphere, computed from a single rollout's
per-channel return-to-go and the standard score function. The
normalization `g^c / ||g^c||` strips away any per-channel return-scale
information, leaving only the *direction in parameter space along which
channel c's score-function gradient points*.

This is **not** a value function (no future-compression learned), **not**
a Q-function (no state-action argument), **not** an advantage estimator
(no baseline), **not** a per-step weight (operates on parameter-space
vectors, not on per-step real numbers), **not** a state-bucketed tensor
(no buckets), **not** an experience replay buffer (no cross-trajectory
memory), and **not** a learned weight `w` over channels (the
combination is a unit-sum, not a convex combination). The aggregation
`d* = (ĝ^1 + ĝ^2) / ||ĝ^1 + ĝ^2||` is a **deterministic non-linear
function of the gradient unit vectors** with no free parameters.

## Derivation sketch

1. **Setup.** A vector-reward MDP exposes per-step `r_vec_t ∈ ℝ^K` from
   `info["vector"]`. For each channel `c ∈ {1, ..., K}` define the
   discounted return-to-go `G^c_t := Σ_{k ≥ t} γ^{k-t} r^c_k` and the
   per-channel score-function gradient
   `g^c(τ) := Σ_t G^c_t · ∇_θ log π_θ(a_t|s_t) ∈ ℝ^{|θ|}`.

2. **Per-channel unit gradients.** Define `ĝ^c := g^c / ||g^c||` when
   `||g^c|| > 0`, else `0`. Each `ĝ^c` is the unit vector on `S^{|θ|-1}`
   in the direction in which infinitesimally moving `θ` increases
   channel `c`'s expected return at first order. By construction, `ĝ^c`
   is **invariant under positive rescaling** `r^c ← α_c r^c`, `α_c > 0`:
   `g^c ← α_c g^c`, hence `ĝ^c` unchanged.

3. **Spherical-sum direction.** Define `d* := (Σ_c ĝ^c) / ||Σ_c ĝ^c||`
   (with the sum restricted to channels where `||g^c|| > 0`). This is
   the **unit angular bisector** of the active channel directions. For
   `K = 2` and acute angle, `d*` is the unique unit vector lying
   exactly between `ĝ^1` and `ĝ^2`. By construction `ĝ^c · d* > 0` for
   every active channel (whenever the active gradients span an acute
   cone), so `d*` is a *simultaneous-improvement direction* for all
   active channels at first order.

4. **Why d* is not scalarization w^T r.** Under the rescaling
   `(r^1, r^2) ↦ (α_1 r^1, α_2 r^2)` with `α_1, α_2 > 0`, both
   `ĝ^1, ĝ^2` are unchanged, hence `d*` is unchanged. **No** scalarization
   `w^T r = w_1 r^1 + w_2 r^2` with fixed or adaptive `w` has this
   invariance: under the rescaling, `w_1 α_1 r^1 + w_2 α_2 r^2 ≠
   const · (w_1 r^1 + w_2 r^2)` in general, so the resulting
   score-function gradient direction depends on `(α_1, α_2)`. Even
   adaptive scalarization (e.g., MGDA's
   min-norm-convex-combination) produces a `d_MGDA = Σ_c α_c g^c` whose
   direction depends on `||g^c||`. CHANBI's `d*` does not.

5. **Why d* is not MGDA / multi-gradient descent.** MGDA computes
   `d_MGDA = argmin_{d ∈ conv({g^c})} ||d||²`, the min-norm point in
   the convex hull of *raw* (non-normalized) per-channel gradients,
   then ascends along it. The MGDA direction depends on
   `||g^c||` ratios -- if one channel has a 100× larger gradient norm,
   MGDA tilts heavily toward it. CHANBI normalizes before summation:
   each channel contributes equally regardless of `||g^c||`. This is
   structurally distinct: MGDA optimizes *worst-case* improvement
   (Pareto-stationary direction at minimum-norm); CHANBI optimizes
   *equal-rate* improvement (angular center, scale-invariant). Under
   `r^c ← α_c r^c`, MGDA's direction changes; CHANBI's does not.

6. **Why d* is not Nash bargaining.** Nash bargaining computes
   `argmax_d Π_c (g^c · d)_+` (geometric mean of per-channel
   improvements). The Nash direction *is* sensitive to per-channel
   rescaling at finite step sizes (the geometric mean depends on
   absolute scales of `g^c · d`). CHANBI is exactly invariant.

7. **Why d* fires at first treasure on DST-concave.** On
   DST-concave, channel 2 (`r^2 = -1`) yields `g^2 = -Σ_t (T-t)
   · ∇log π`, a non-zero gradient on every rollout. Channel 1
   (treasure) yields `g^1 = treasure · γ^T · ∇log π_θ(a_T|s_T)`, zero
   when no treasure is found, non-zero when found. Random walks on the
   11×11 DST grid hit treasure within ~5-10 steps on roughly 28% of
   trajectories at random init, so within the first 10-30 episodes
   both channels' gradients are non-zero and `d*` is well-defined.

8. **Why this is not Family A (bucketed-tensor + partial-order vote).**
   No state/action bucketing, no tensor, no partial-order. The
   primitive is two unit vectors in parameter space.

9. **Why this is not Family C (within-trajectory geometric statistic).**
   `ĝ^c` is a parameter-space vector (norm-normalized score-function
   gradient), not a path-geometry statistic of the state/cumulant
   trace. It is reparametrization-invariant in the policy but not in
   geometric features of the trajectory.

10. **Why this is not GRADCOMP (run 20).** GRADCOMP rotates the
    update toward the principal eigenvector of the empirical Fisher
    matrix `Σ_t score_t score_t^T`. This is a *single-channel*
    operation on the score-function moments. CHANBI operates on the
    *per-channel* gradient *directions*, not on second-moment
    statistics of the score.

11. **Why this is not COPDEV/PARGRAD (runs 21-22).** COPDEV/PARGRAD
    use per-step rolling-buffer rank weights as scalar coefficients
    on `∇log π_t`. CHANBI does not use per-step weights -- the entire
    per-step weighting is the standard `G^c_t`. The novelty is in
    *aggregating* `g^1, g^2` into `d*` via the spherical sum, not
    in modifying per-step weights. Different mechanism slot
    (parameter-space vector aggregation vs. per-step coefficient
    weighting).

12. **Proof debt.** (i) Convergence: under what conditions does
    `θ ← θ + α d*` converge to a Pareto-stationary policy? Conjecture:
    when both channels are active and the angle between `ĝ^1, ĝ^2`
    is acute, `d* · ∇_θ E[G^c | π] ≥ 0` for both `c`, hence the
    update is a Pareto-improvement direction at first order. The full
    convergence theorem requires bounding the expected angle's
    behavior under stationary policies. (ii) Stationary-point
    characterization: `d* = 0` iff `ĝ^1 = -ĝ^2`, the *anti-aligned*
    case (channels in direct conflict). On DST-concave this can occur
    if channel 1 (push toward treasure) is exactly opposed to channel
    2 (avoid step penalty); however generically `ĝ^1 · ĝ^2 > -1` and
    `d*` is well-defined. (iii) Variance: `Var[d*]` over rollouts is
    bounded by the variance of `ĝ^c` on the unit sphere; rigorous
    bound open.

## Update rule

```
Inputs: env (vector reward, K channels read from info["vector"]),
        policy π_θ, discount γ, learning rate α, K = 2 (DST-concave)
Init:   θ random.

For each episode:
    1. Roll out τ = (s_0, a_0, info_0, r_0, ..., s_T) under π_θ.
       For t = 0..T-1: read r_vec_t = info_t["vector"] ∈ ℝ^K
       (DST-concave: K=2; channel 1 = treasure pickup terminal,
       channel 2 = -1 step penalty).
       NEVER sum the channels; never compute w^T r_vec.

    2. Compute discounted per-channel returns-to-go:
         For t = 0..T-1, c ∈ {1, ..., K}:
             G_c[t] = Σ_{k ≥ t} γ^{k-t} · r_vec_k[c]

    3. Compute per-channel score-function gradients (parameter-shaped
       vectors, one per channel):
         For c ∈ {1, ..., K}:
             g_c = Σ_t G_c[t] · ∇_θ log π_θ(a_t | s_t)
                                                 # parameter-shaped vec

    4. Normalize per-channel gradients (skip zero ones):
         For c ∈ {1, ..., K}:
             if ||g_c|| > eps:
                 ghat_c = g_c / ||g_c||
                 active_c = True
             else:
                 ghat_c = 0                       # zero param vector
                 active_c = False

    5. Spherical-sum direction:
         s = Σ_{c: active_c} ghat_c              # parameter-shaped vec
         if ||s|| > eps:
             d_star = s / ||s||
         else:
             d_star = 0                           # anti-aligned channels

    6. Parameter update:
         θ ← θ + α · (Σ_{c: active_c} ||g_c||) · d_star
         # The scalar Σ ||g_c|| restores natural step-size
         # in the absence of normalization; without it, the update
         # would be parameter-norm-invariant but step-size-blind.

    7. Logging observables (load-bearing for ablation discrimination):
         - cos_g1g2 = ghat_1 . ghat_2  (NaN if either inactive)
         - n_active = |{c: active_c}|             # 0, 1, or 2
         - ||g_c|| per channel                    # raw norms
         - cos_d_uniform = d_star . d_unif where d_unif =
              (g_1 + g_2) / ||g_1 + g_2|| is the unnormalized-sum
              direction (the ablation's d).
              This is the PRIMARY discriminator: equals 1 iff
              ||g_1|| = ||g_2|| or one channel inactive; strictly
              < 1 otherwise. Logged per rollout.
         - mean episode length T
         - first-rewarded-episode-index (when channel 1 first nonzero)
```

The load-bearing primitive is the **per-channel unit-normalized
score-function gradient pair** `(ĝ^1, ĝ^2)`. Storage: 2 parameter-shaped
vectors per rollout (transient). Per-rollout cost: 2 score-function
gradient computations per channel, K extra norm-and-normalize operations
-- negligible. The single new computation beyond REINFORCE is the
per-channel separation of `G_t` into per-channel `G^c_t` and the
post-aggregation normalization step.

## Empirical claim

stage: quick

claim: On the **quick** stage (deep-sea-treasure-concave-v0; K = 2
vector reward via `info["vector"]`), CHANBI should produce:

(a) **Both channels become active within the first 30 episodes**: at
    random init on DST-concave, ~28% of trajectories find treasure
    within ~5 steps; channel 1's gradient is non-zero on those
    rollouts. By episode 10-30, the rolling fraction of active-both
    rollouts should exceed 20%. Logged as `n_active == 2` per rollout.

(b) **`cos_d_uniform` strictly bounded away from 1.0** in the
    active-both regime, with mean `< 0.9` and median typically
    `0.5 - 0.8`: when `||g^1||` and `||g^2||` differ by 10× or more
    (which is generic since channel 2 accumulates linearly with episode
    length while channel 1 is a sparse terminal pickup), the
    unnormalized sum direction `d_unif = (g^1 + g^2)/||g^1 + g^2||`
    is dominated by the larger-norm channel, while CHANBI's `d*` gives
    each channel equal angular weight. The difference between these
    two directions is the operator's signature.

(c) **A measurable shift in mean episode length compared to the
    ablation**: CHANBI tilts the update equally toward both channels'
    improvement, so it should not collapse to T=1 (the ablation's
    failure mode in run 21) while channel 2's gradient dominates.
    Channel 1's normalized gradient pulls toward longer trajectories
    (treasure is several steps deep), preserving exploration even
    when channel 2's raw gradient is large. Predicted: CHANBI's mean
    T over training stays in `[3, T_max]`; the unnormalized-sum
    ablation's mean T may collapse toward 1 as channel 2 dominates.

(d) **Final hypervolume score on DST-concave at or above the random
    floor (194)**: CHANBI is designed not to scalarize, so it should
    not regress below random; preserving exploration via the
    norm-balanced direction may produce a small improvement. The
    primary discriminator is `cos_d_uniform`, not the score; matching
    random while showing the operator signature is positive evidence.

The quick stage is appropriate because (i) DST-concave provides the
2-channel vector-reward substrate the primitive operates on; (ii) random
trajectories find treasure within budget so both channels become active
within the first ~30 episodes; (iii) the discriminator
`cos_d_uniform` is a logged training-dynamics scalar that becomes
non-trivial as soon as both channels are active (not requiring the
policy to *learn* anything); (iv) the rescaling-invariance is a
structural property that distinguishes CHANBI from any scalarization
ablation regardless of substrate-specific tuning.

falsifier:

**Primary** (mechanism presence): if `cos_d_uniform >= 0.95` on
average across the training run when both channels are active, then
the per-channel normalization makes no measurable difference to the
update direction -- the operator is decorative. Detectable from the
first ~30 episodes once both channels activate.

**Secondary** (mechanism direction): if CHANBI's mean episode length
matches the ablation's within seed variance throughout training,
the rescaling-invariance produces no behavioral difference on this
substrate.

**Tertiary** (substrate signal): if CHANBI's final hypervolume on
DST-concave is **strictly worse than the random baseline 194 by
more than seed variance**, the angular-bisector direction actively
harms learning on this substrate (e.g., by under-weighting the
channel-2 gradient that would otherwise concentrate the policy on
faster trajectories).

**Cold-start collapse** (substrate-confounded null): if CHANBI's
`n_active == 2` rolling fraction stays at 0 throughout training
(no rollouts find treasure), the substrate did not exercise the
two-channel primitive within budget -- a substrate-budget null,
not a mechanism falsification. In that case retest on a
non-degenerate vector substrate.

## Ablation plan

### Primary ablation (unnormalized per-channel sum = uniform-weight scalarization)

Replace the **per-channel normalization** with the raw sum:

In `train_ablate.py`:
1. Steps 1-3 identical (per-channel returns and gradients).
2. Skip step 4's normalization.
3. In step 5 set `s = g_1 + g_2` directly (no per-channel scaling).
4. `d_ablate = s / ||s||` if `||s|| > 0` else `0`.
5. Step 6: `θ ← θ + α · ||s|| · d_ablate = θ + α · (g_1 + g_2)`.

This is exactly REINFORCE on the **uniform-weight scalarized return**
`G_t = G^1_t + G^2_t`. It preserves: per-channel return decomposition
(used internally), the score-function form, all hyperparameters, the
policy architecture, the rollout mechanism. It removes: the *per-channel
norm normalization* before aggregation -- i.e., the rescaling-invariance.

### Predicted contrasts on DST-concave (within 120s budget)

- **Discriminator (i) -- `cos_d_uniform`**: this is logged in the
  candidate run (per derivation). In the ablation, by definition,
  `d_ablate = d_unif`, so `cos_d_uniform_ablation` ≡ 1. The
  candidate's value should be strictly < 1 once both channels are
  active and `||g^1|| / ||g^2||` is not exactly 1. **This is the
  primary load-bearing discriminator** and fires from the first
  treasure-finding episode (within ~10-30 episodes at random init
  on DST-concave).

- **Discriminator (ii) -- mean episode length T over training**:
  CHANBI normalizes channel 2's contribution, so it should not
  collapse to T=1 even when channel 2's raw gradient dominates. The
  ablation (uniform-weight `G^1 + G^2` REINFORCE) is exactly the
  collapse regime that killed COPDEV's ablation in run 21. Predicted
  contrast: CHANBI mean T stays > 2; ablation mean T drifts toward 1.

- **Discriminator (iii) -- final hypervolume**: CHANBI should match
  or modestly improve random (194); ablation may collapse to the
  nearest-treasure floor 99 or below (as in run 21).

If `cos_d_uniform` stays ≥ 0.95 throughout the run when both channels
are active (i.e., `||g^1||` and `||g^2||` happen to be nearly equal
in magnitude on this substrate), the per-channel normalization
operator is decorative on DST-concave. If `cos_d_uniform < 0.9`
consistently AND mean T differs measurably between arms, the
primitive is causally responsible.

### Sanity ablation (random per-channel sign flip)

Optional second arm: replace the spherical sum with `s = ε_1 ĝ^1 +
ε_2 ĝ^2` where `ε_c ∈ {+1, -1}` are sampled uniformly per rollout.
This randomizes the direction of each channel's normalized gradient.
If CHANBI matches the random-sign-flip ablation, the *normalization*
matters but not the *channel-positive-direction-aggregation*. (This
is a diagnostic, not load-bearing for the primary claim.)

## Novelty boundary

Closest known methods:

(a) **REINFORCE / vanilla policy gradient** (Williams 1992). Standard
    REINFORCE uses scalar `G_t = Σ_c G^c_t` (i.e., uniform-weight
    scalarization on vector envs). CHANBI separates per-channel
    gradients and angular-bisects in parameter space. The unnormalized-
    sum ablation is exactly REINFORCE-on-uniform-scalarization;
    CHANBI's normalization is the structural difference.

(b) **MGDA / multi-gradient descent algorithm** (Désidéri 2012;
    Sener-Koltun 2018). MGDA computes
    `d_MGDA = argmin_{d ∈ conv({g^c})} ||d||²` -- the minimum-norm
    point in the convex hull of *raw* per-channel gradients. The
    direction is `Σ_c α_c^* g^c` for the optimal Pareto-improving
    weights `α^*`, which depends on `||g^c||`. CHANBI normalizes
    *first* then sums: `d* = Σ_c (g^c / ||g^c||)`, which is *not* a
    convex combination of `g^c` (the implicit weights are
    `1/||g^c||`, not in the simplex unless rescaled). Under
    `r^c ← α_c r^c`, MGDA's direction changes (since `||g^c||`
    changes), CHANBI's does not.

(c) **PCGrad / gradient-surgery for multi-task learning** (Yu 2020).
    PCGrad projects each task gradient onto the orthogonal complement
    of the others' direction when they conflict, then sums. CHANBI
    does not project. PCGrad operates on raw `g^c` (so is not
    rescaling-invariant); CHANBI normalizes first.

(d) **Multi-objective scalarization (linear, Chebyshev,
    augmented-weighted-Chebyshev)** (Roijers-Whiteson 2017). Linear
    scalarization is `wᵀr` with fixed `w`. CHANBI does not scalarize
    at the reward level; its per-step weights remain `G^c_t` per
    channel; aggregation is at the parameter-gradient level via
    norm-normalization. Chebyshev scalarization
    `min_c (G^c - ref^c)/w_c` requires reference points and weights;
    CHANBI uses neither.

(e) **Nash-MTL / Nash bargaining for multi-task learning** (Navon
    2022). Nash-MTL solves
    `argmax_d Π_c (g^c · d)_+` (geometric-mean improvement) with a
    constraint on `||d||`. CHANBI uses **angular bisection** (sum of
    unit vectors), which is the *arithmetic mean of unit gradients*,
    not the geometric mean of inner products. Under `r^c ← α_c r^c`,
    Nash-MTL's direction depends on the rescaling
    (the geometric mean of inner products is not norm-invariant);
    CHANBI's does not.

(f) **MGDA-UB / Pareto-Set Learning** (various 2020-2023). These
    refine MGDA via upper bounds or condition-numbered preconditioners;
    all use raw `g^c` and inherit the scale-sensitivity. CHANBI's
    angular-bisector primitive is structurally distinct from any
    convex-hull projection of raw per-channel gradients.

(g) **Cosine-similarity-based multi-task gradient combination**
    (Suteu-Guo 2019; CAGrad 2021). CAGrad finds the direction `d`
    minimizing the worst-case angle between `d` and per-task
    gradients, subject to `d` being close to the average gradient.
    CHANBI's primitive is the explicit angular bisector
    `Σ_c ĝ^c`, the mean of unit vectors. CAGrad and CHANBI both use
    angular geometry, but CAGrad's optimization formulation is a
    constrained QP solving for `d` near the average; CHANBI's `d*` is
    explicit and closed-form. Crucially, CAGrad still has a
    scale-dependent `||g^c||` weighting in its constraint
    (`d` close to the average `g^c`); CHANBI normalizes `g^c` to unit
    norm first. This is the structural difference. (Engineer note:
    do not implement CAGrad's constrained QP; the load-bearing
    primitive is the closed-form angular bisector.)

(h) **GRADCOMP (run 20, this loop)**. GRADCOMP rotates the update
    toward the principal eigenvector of the empirical Fisher matrix
    `Σ_t score_t score_t^T`. This is a *second-moment* operation on a
    single (scalar-return) gradient. CHANBI operates on
    *per-channel first-moment* gradient directions. Different
    mechanism slot.

(i) **COPDEV / PARGRAD (runs 21-22)**. COPDEV uses per-step
    rolling-buffer rank weights as scalar coefficients on `∇log π`.
    CHANBI uses no rolling buffer, no per-step weights beyond
    standard `G^c_t`, and no rank statistics. The novelty is at the
    parameter-space vector-aggregation level.

(j) **Distributional / quantile RL** (Bellemare 2017; Dabney 2018).
    CHANBI has no return distribution and no quantile estimator.

(k) **Successor features / GVFs** (Barreto 2017; Sutton 2011).
    CHANBI does not learn cumulant predictors; per-channel
    return-to-go `G^c_t` is computed online from the realized
    rollout, not learned.

(l) **Hypervolume gradient** (Emmerich 2020; HV-PG). Hypervolume
    gradients differentiate the dominated-volume of a Pareto front
    approximation. CHANBI uses no hypervolume and no front
    approximation.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: CHANBI has no
  bucketing of any kind. The data structure is `K = 2`
  parameter-shaped vectors per rollout. No partial-order vote; the
  aggregation is a real-valued unit-vector spherical sum.
- **Family B (pairwise trajectory comparison)**: CHANBI works on a
  single rollout's per-channel gradients; no cross-trajectory pairing.
- **Family C (within-trajectory geometric statistic)**: `ĝ^c` lives
  on the policy parameter sphere, not in observation/cumulant trace
  geometry. Path-geometry statistics (hull, signature, Lévy) have no
  analogue here.
- **Family D (reward-independent + reward-gated)**: `g^c` is reward-
  dependent throughout; the application is not gated.
- **Family E (avoid value vocabulary)**: No value function, no
  Q-function, no advantage. The score function is the standard
  policy-gradient primitive; the novelty is in *aggregation*, not in
  return prediction.
- **Family F (hand-engineered structural priors)**: No vocabulary
  or symbol grammar; the primitive is closed-form.
- **Family G (mechanism stack)**: One primitive -- the per-channel
  unit-normalized score-function gradient pair. No three components.
- **Family H (cochain complex)**: Not applicable.

The structural difference from all named methods and dead families is
the **per-channel unit-normalized score-function gradient pair as a
single primitive, with the closed-form spherical-sum direction
`d* = Σ_c ĝ^c / ||Σ_c ĝ^c||` as the update rule**. This is
operationally distinct from MGDA (uses raw `g^c`, not normalized;
optimizes min-norm not angular-bisector), Nash-MTL (geometric-mean
inner products, not arithmetic-mean unit vectors), CAGrad (constrained
QP near average `g^c`, scale-dependent), PCGrad (orthogonal projection,
no normalization), and any scalarization `wᵀr` (CHANBI is invariant
under componentwise positive rescaling, no scalarization is).

## Proof debt

1. **Pareto-improvement at first order.** Conjecture: when
   `(ĝ^1, ĝ^2)` span an acute angle (`ĝ^1 · ĝ^2 > 0`),
   `d* · ĝ^c > 0` for both `c = 1, 2`. Proof sketch:
   `ĝ^c · d* = ĝ^c · (ĝ^1 + ĝ^2) / ||ĝ^1 + ĝ^2|| = (1 + ĝ^1·ĝ^2) /
   ||ĝ^1 + ĝ^2||` for `c ∈ {1, 2}` (by symmetry). When
   `ĝ^1 · ĝ^2 > 0`, this is positive. **At a stationary policy** of
   both expected channel returns (`E[g^c] = 0` for both `c` in
   expectation), the conjecture is vacuous. The rigorous improvement
   theorem (in expectation, not just per-rollout) is open; it requires
   bounding the rollout variance of `d*` and showing
   `E[d* · ∇_θ E[G^c]] > 0` along the iterate.

2. **Convergence under stochastic rollout estimation.** Standard
   Robbins-Monro analysis applies for the bounded-step iterate
   `θ ← θ + α · S · d*` with `S = Σ_c ||g^c||` (the natural step-
   size scalar). Open: characterize the stationary set as the **set
   of Pareto-stationary policies** where `ĝ^1 = -ĝ^2` (anti-aligned
   gradients) or all `||g^c|| = 0` (jointly stationary).

3. **Anti-aligned channel handling.** When `ĝ^1 · ĝ^2 = -1`, the
   spherical sum vanishes (`d* = 0`). The update halts. This is the
   *Pareto-stationary* set under the angular-bisector flow. Open:
   characterize the anti-aligned set on DST-concave (does it occur
   generically?) and design a tie-break rule (e.g., randomly pick one
   channel to descend along, or fall back to the larger-norm
   channel) without breaking rescaling-invariance.

4. **Connection to natural gradient.** Conjecture: the spherical-sum
   direction `d*` is *not* equal to the natural gradient
   `F^{-1} g_avg` for any per-channel-mixed Fisher `F` -- normalizing
   per-channel gradients before averaging is structurally different
   from preconditioning. The proof debt: characterize when CHANBI's
   `d*` and natural-gradient `F^{-1} g_avg` agree (probably only at
   the trivial isotropic-Fisher case).

5. **Connection to KL-regularized natural multi-objective methods.**
   Open: identify whether CHANBI's update can be written as the
   policy mirror-descent step on some Bregman geometry over per-
   channel value functionals. Conjecture: yes, with the Bregman
   potential `Σ_c sqrt(||∇φ^c||²)` for per-channel value functionals
   `φ^c`, but the mapping is non-trivial.

The empirical probe will reveal whether the rescaling-invariance
(a structural property derived from the closed-form primitive)
produces a measurable separation on the `cos_d_uniform` discriminator
on DST-concave within 120s; positive separation on this scalar AND a
measurably different mean-episode-length trajectory would justify
investing in proof items (1) and (2).
