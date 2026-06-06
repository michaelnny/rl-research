# 20260606-18-auto -- PRISM (Pareto-Rolling-Imitation Score-function Match) [probe]

## Principle

In a vector-reward MDP, the policy is updated by score-function ascent
on the **log-density of the realized terminal vector return under a
rolling kernel-density estimate (KDE) of the Pareto-non-dominated
subset of past terminal returns**, which makes the load-bearing
per-trajectory weight a *non-monotone, non-linear, non-scalarizing*
functional of the full vector return that strictly increases with
proximity to the moving Pareto frontier and strictly decreases for
interior (dominated) returns.

## Primitive

The **rolling Pareto-frontier target measure**

  `μ* : ℝ^m → ℝ_{≥0}`,
  `μ*(y) = (1/|F_n|) · Σ_{p ∈ F_n} K_h(y − p)`

where `m` is the vector-reward dimension, `K_h` is a Gaussian kernel
with bandwidth `h > 0` on `ℝ^m`, and `F_n ⊆ ℝ^m` is the **rolling
Pareto-non-dominated subset** of the last `N` terminal vector returns:

  `F_n := { p ∈ {y_{n−N+1}, …, y_n} : ¬∃ q in window with q ≻ p }`

with `≻` componentwise strict dominance and `y_t = Σ_k γ^k r_vec_k` the
realized terminal cumulative vector return of episode `t`.

This is **one typed object**: a finite-support density on `ℝ^m`
implicitly represented by the running window's Pareto-nondominated
subset and the bandwidth `h`. Codomain: a probability density in
`ℝ^m`. Neither a value, nor a Q, nor a policy, nor a partial-order
table. The primitive is a *measure on the vector-return space*, not on
the state-action space.

## Derivation sketch

1. **Vector-MDP achievable set.** For a vector-reward MDP `(S, A, P,
   r_vec, γ)` with `r_vec : S × A → ℝ^m`, the **achievable set**
   `M := {μ(π) : π ∈ Π} ⊆ ℝ^m` of expected terminal vector returns is
   convex (Roijers-Whiteson 2017, §2.2), and its upper Pareto frontier
   `∂_+ M` is the Pareto-optimal subset.

2. **Frontier as imitation target.** The classic MORL goal is to
   produce a policy whose terminal-vector-return distribution `ν_π`
   places mass on `∂_+ M`. We *invert* this: define a **synthetic
   target measure** `μ*` whose density is concentrated on the
   currently-known frontier (the rolling Pareto-non-dominated subset
   `F_n`), with Gaussian KDE smoothing to make the density positive
   in a neighborhood of `F_n`. The policy is then updated to **match
   ν_π to μ\***.

3. **Imitation-as-score-function.** Maximizing the cross-entropy
   `CE(ν_π, μ*) = E_{y ∼ ν_π}[log μ*(y)]` is, by the standard
   score-function identity, computable as a policy-gradient with
   per-trajectory weight `log μ*(y_τ)`:
   `∇_θ CE(ν_π_θ, μ*) = E_τ[ log μ*(y_τ) · Σ_t ∇_θ log π_θ(a_t | s_t) ]`,
   where `y_τ` is the realized terminal vector return of trajectory
   `τ`. This is the load-bearing identity (see Sutton et al. 1999 for
   the derivation, applied to log-density of an *exogenous* target
   rather than to discounted return).

4. **Why this is not scalarization.** The per-trajectory weight
   `w(y_τ) := log μ*(y_τ)` is a *non-linear* functional of the full
   vector return: by KDE definition,
   `log μ*(y) = log Σ_{p ∈ F_n} exp(−‖y − p‖² / (2h²)) − const`
   = `−min_{p ∈ F_n} ‖y − p‖² / (2h²) + log-sum-exp correction`
   = (in the small-bandwidth limit) the **negative squared distance
   to the nearest Pareto-non-dominated point in the window**. This
   is *not* of the form `wᵀ r_vec_total` for any fixed or
   adaptive-but-constant-per-trajectory `w`: the weight depends
   non-linearly on `y_τ` through the min-of-squared-distances.

5. **Why this is not CEM/elite-cloning.** CEM/top-k cloning re-trains
   a policy on the elite trajectories under a *scalar* fitness
   ranking. PRISM uses a *vector* rank: trajectories are weighted by
   *density at frontier*, which is not a 0/1 elite indicator but a
   continuous density. Two trajectories with the same scalar fitness
   can receive very different PRISM weights if one lies near a
   frontier corner (high `μ*`) and the other lies in a dominated
   interior pocket (low `μ*`).

6. **Why this is not Decision-Transformer/return-conditioned.** PRISM
   has no return conditioning fed to the policy as input; the policy
   maps state → action distribution unmodified, and the frontier
   structure enters only at update-time as a per-trajectory weight.

7. **Resistance to time-step pull.** On DST-concave, the failure
   pattern across runs 13–17 was that any reward-magnitude-weighted
   gradient pulls the policy to nearest treasure (terminal `y =
   (1, −1)` in normalized coords). Under PRISM, once *any* trajectory
   reaches a far treasure (terminal `y = (124, −19)` or similar) and
   enters the rolling window, the rolling Pareto frontier `F_n`
   contains both nearest and far points. New episodes ending at
   nearest treasure are now *interior* (dominated by far treasures
   when `m ≥ 2` and the trade-off is non-trivial), so their
   `log μ*(y)` weight drops. Episodes reaching new far treasures get
   the highest weight. The mechanism is **explicitly non-monotone in
   any single component** — the weight does not increase with treasure
   value alone, only with proximity to the *currently-rare* corners
   of the frontier.

8. **Discriminating observable.** Define `coverage_n := |F_n|` (the
   number of distinct Pareto-non-dominated points in the rolling
   window). Under PRISM, `coverage_n` should **grow with training**
   (the policy should produce diverse non-dominated terminals). Under
   the random ablation (replace `μ*` with a fixed uniform density),
   the weight is constant across trajectories and the score-function
   update is a high-variance unbiased estimator of `∇_θ E[1] = 0` —
   no learning signal — so `coverage_n` stays at random-baseline
   level and may even decline as the policy overfits to noise.

9. **Bandwidth `h` as the only free knob.** `h` controls the
   trade-off between frontier-localization (small `h`: weight is sharply
   peaked at exact frontier points, giving CEM-like behavior) and
   frontier-broadening (large `h`: weight is nearly uniform, giving
   REINFORCE-like behavior). The probe fixes `h` via the
   median-heuristic on the rolling window (Schölkopf-Smola 2002).

10. **Proof debt.** Open theorems: (i) under the PRISM update with
    matched-step-sizes, the policy-induced terminal-return distribution
    `ν_π_n` converges in Wasserstein-2 distance to a measure supported
    on the upper Pareto frontier `∂_+ M`. (ii) The non-stationarity of
    the target `μ*` (which depends on the policy's own past returns)
    requires a two-timescale analysis: `μ*` should be slow relative
    to the policy parameter `θ`. Standard Borkar 2008 two-timescale
    SGD applies if the target update is `O(1/t)` slower; this is
    open.

## Update rule

```
Inputs: env, policy π_θ, vector-reward dim m, window size N,
        bandwidth_factor c (PRISM uses median-heuristic; c = 1.0 default)
Init:   θ random; rolling window W = deque(maxlen=N); pareto_frontier F = []

For each episode:
    1. Roll out τ = (s_0, a_0, r_vec_0, ..., s_T) under π_θ.
       Read r_vec_t from info["vector"] for vector envs;
       for sparse scalar envs, set r_vec_t = (r_t, -1) ∈ ℝ²
       (treasure value + step-marker; second component is the
       constant -1 step penalty, ensuring r_vec is always vector).

    2. Compute terminal vector return:
         y_τ = Σ_t γ^t · r_vec_t                  ∈ ℝ^m

    3. Update rolling window and frontier:
         W.append(y_τ)
         F = {p ∈ W : ¬∃ q ∈ W with q ≻ p}        # Pareto-non-dominated subset

    4. Compute trajectory weight:
       If |F| < 2:
         w_τ = 1.0                                 # warmup: REINFORCE
       Else:
         h = c · median({||p − q|| : p, q ∈ F, p ≠ q}) + ε
         log_μ*(y_τ) = log_sum_exp({−||y_τ − p||² / (2h²) : p ∈ F})
                      − log|F| − (m/2) log(2π h²)
         # Center weight on the running mean for variance reduction:
         w_τ = log_μ*(y_τ) − running_mean(log_μ*(y_·) over last N)

    5. Score-function policy gradient:
         g_θ = w_τ · Σ_t ∇_θ log π_θ(a_t | s_t)
         θ ← θ + α · g_θ

    6. Logging observables (load-bearing for ablation discrimination):
         - coverage_n = |F|                       # frontier diversity
         - mean log_μ*(y_τ) over recent episodes  # frontier-density proxy
         - per-component spread: (max − min) of y_τ_i over the window
```

The load-bearing primitive is the **rolling Pareto-frontier KDE
target measure `μ*`**. The score-function gradient direction is
*always* the standard score-function (no novel gradient form); the
novelty is purely in the **per-trajectory weight `w_τ = log μ*(y_τ)`**
and what `μ*` is.

## Empirical claim

stage: vector

claim: On the **vector** stage (deep-sea-treasure-concave-v0 and
resource-gathering-v0), PRISM should exhibit **growing frontier
diversity** `coverage_n` over training, ending at `coverage_T ≥ 3`
(at least 3 distinct Pareto-non-dominated terminal vector returns
discovered), AND should achieve hypervolume strictly above the
nearest-treasure-collapse score (DST > 99, RG > 0.011) on at least
one of the two envs within the 120s budget. The vector stage is the
appropriate test because (i) the reward is genuinely multi-channel
and the frontier KDE was *designed* to consume per-component vector
returns; (ii) the runs 13–17 convergent-failure pattern (DST
collapse to score 99) is exactly the failure mode PRISM's
non-monotone weight is designed to break — by making nearest-
treasure trajectories *less* informative once any single far-treasure
trajectory enters the window.

falsifier: **Primary**: if PRISM's `coverage_n` stays at 1 (only
nearest-treasure terminal in the rolling Pareto front) throughout
training, the frontier-extension mechanism never fires and the
weight is effectively constant — same as a REINFORCE baseline. This
is structural failure of the principle on this substrate.
**Secondary**: if PRISM matches its random-density ablation on both
hypervolume *and* on `coverage_n`, the rolling-frontier target is
decorative — the algorithm's behavior is determined by score-
function variance alone, not by frontier-imitation.

## Ablation plan

Replace the **rolling Pareto-frontier KDE target `μ*`** with a
**fixed isotropic Gaussian centered at the origin of vector-return
space**:

  `μ*_ablate(y) := (2π σ²)^{−m/2} · exp(−‖y‖² / (2σ²))`

with σ chosen as the standard-deviation of the first-N random
rollout returns (a one-shot data-driven scale, then frozen). The
log-weight becomes `log μ*_ablate(y_τ) = −‖y_τ‖² / (2σ²) + const`,
which is **monotone in `−‖y_τ‖²`**: trajectories whose terminal
vector return is *closer to the origin* get higher weight — the
*opposite* of any reasonable RL signal, but with the same
*magnitude* of variance as PRISM, so any learning that occurs is
attributable to score-function noise alone.

In `train_ablate.py`:
1. Skip steps 3 (no rolling window, no frontier).
2. At episode 1, collect N=8 rollouts and compute
   `σ = std(||y_τ||) over those 8 rollouts`.
3. Thereafter, weight `w_τ = −‖y_τ‖² / (2σ²) − running_mean(...)`.
4. Apply the same score-function update.

This preserves: vector-return computation, score-function update,
running-mean baselining, all hyperparameters. It removes: the
*frontier-tracking* component of `μ*`. The `coverage_n` observable
should be **1** (or `random-baseline level`) throughout training,
because the ablation has no mechanism to encourage frontier
extension.

If PRISM matches the ablation on hypervolume AND on `coverage_n`,
the rolling-frontier mechanism is decorative. If PRISM exhibits
strictly higher `coverage_n` *and* strictly higher hypervolume, the
primitive is causally responsible for the lift. If PRISM has higher
`coverage_n` but matched hypervolume, the primitive shapes the
return distribution without changing scalarized return — interesting
but not a hypervolume win.

A second sanity ablation (cheap, optional): bandwidth `h → ∞`
collapses the KDE to a uniform density on the convex hull of the
window, which makes `log μ*` constant — pure REINFORCE. If PRISM
also matches this, the bandwidth-localization is the decorative
piece, not the frontier-tracking. (This second ablation is
informative but not the primary discriminator.)

## Novelty boundary

Closest published methods:

(a) **Multi-Objective Reinforcement Learning with Continuous Pareto
    Frontier Approximation** (Parisi et al. 2014, AAAI; Pirotta et
    al. 2015 JAIR). They parameterize a *manifold of policies* whose
    image in vector-return space approximates the Pareto frontier,
    and gradient-ascend on a manifold-quality metric (e.g., area or
    spread). PRISM uses a **single policy** (not a manifold), and
    the load-bearing object is a **target measure on vector-return
    space** (not a manifold of policies). Parisi-Pirotta's primitive
    is a parametric family `π_ρ` indexed by `ρ`; PRISM's primitive is
    a non-parametric KDE on `ℝ^m`.

(b) **MOO-SVGD / Stein Variational Pareto Front** (Liu et al. 2021
    NeurIPS). Particle-based: K policies are jointly updated via SVGD
    in policy-parameter space, with a kernel-similarity repulsion in
    vector-return space. Primitive: K particles + a kernel on policy
    parameters. PRISM has **no particle ensemble** and **no
    repulsion in policy space**; the kernel acts on vector-return
    space, not on policies, and the policy is single (no `K`). The
    mechanism is imitation toward a frontier *measure*, not particle
    repulsion.

(c) **Latent-Conditioned Policy Gradient** (Tongzeng et al. 2023).
    A single policy conditioned on a latent preference variable `z`,
    trained to map preferences to Pareto-optimal policies. PRISM has
    **no preference conditioning** — the policy is plain `π_θ(a|s)`,
    and the frontier structure enters only as a *training-time
    weight*, not as a *runtime input*.

(d) **Pareto Q-Learning** (Van Moffaert-Nowé 2014). Learns a *set* of
    vector-Q-values per state-action, with an action-selection rule
    based on Pareto-non-dominance over the set. Primitive: per-(s,a)
    set of vector returns. PRISM has **no per-state-action set** and
    **no value vocabulary** — the only learned object is the policy
    `π_θ`, with a per-trajectory KDE weight at update-time.

(e) **CEM / Top-K Trajectory Cloning** (Rubinstein 1997; explicit
    disqualifier in `prior_attempts.md`). CEM weights trajectories by
    a 0/1 elite indicator (above some scalar threshold). PRISM
    weights by a **continuous KDE density** on vector-return space.
    Two trajectories with the same scalar fitness can have very
    different PRISM weights based on *where in vector-return space*
    they fall relative to the frontier corners. CEM's weight is a
    *step function*; PRISM's is a *smooth density*. The frontier
    construction itself is also non-scalar.

(f) **Scalarized Vector Reward `wᵀr` with adaptive `w`** (explicit
    disqualifier). PRISM's weight `log μ*(y_τ)` is a non-linear,
    *non-monotone-in-each-component* function of `y_τ`. There is no
    `w ∈ ℝ^m` such that `log μ*(y) = wᵀy + const`: counterexample,
    consider three frontier points placed equilaterally around the
    origin in `ℝ^m` — the level sets of `log μ*` are *non-linear*
    (concentric circles around each frontier point, summed). No
    linear functional has these level sets.

(g) **Imitation learning / GAIL** (Ho-Ermon 2016). GAIL minimizes a
    JS divergence between policy occupancy and an expert occupancy
    using an adversarially-trained discriminator. PRISM's "expert" is
    a **synthetic non-parametric KDE on vector-return space**, and
    the divergence is **cross-entropy with respect to terminal
    vector returns**, not state-action occupancy. There is no
    discriminator network and no adversarial training.

(h) **Reverse-KL imitation / behavioral cloning on Pareto-elite
    rollouts** would be the closest "obvious" reduction. PRISM is
    *not* this: BC weighs the score-function *uniformly* across the
    elite set; PRISM uses a *KDE density* that weights interior-of-
    frontier-cluster trajectories higher than corner-of-frontier
    ones, which is the opposite of what BC on the frontier set would
    do (BC would treat all elites equally).

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: PRISM has no
  bucketing — there is no per-state, per-cluster, or per-channel
  tensor. The KDE `μ*` is over `ℝ^m`, not over states or actions.
  There is no partial-order vote; the weight `log μ*(y_τ)` is a
  continuous real-valued density.
- **Family B (pairwise trajectory comparison)**: PRISM compares each
  trajectory to a *running aggregate density* (the rolling frontier
  KDE), not to a *paired sibling trajectory*. There is no
  pair-matching condition; every trajectory contributes to and is
  weighted by the same rolling KDE.
- **Family C (within-trajectory geometric statistic)**: PRISM's
  weight is a function of the *terminal* vector return only, not a
  within-trajectory path statistic. Hull / Lévy area / spectral
  coefficient are not used.
- **Family D (reward-independent + reward-gated)**: PRISM's weight
  is *reward-dependent throughout* (the KDE consumes vector-reward
  cumulants), not reward-gated.
- **Family E (avoid value vocabulary, keep value structure)**: PRISM
  has no value vocabulary and no value structure. There is no
  learned scalar quantity that compresses future return; the only
  learned object is the policy.

The structural difference from all named methods and dead families
is the **rolling Pareto-frontier KDE target measure `μ*` on vector-
return space, used as a non-linear non-scalarizing per-trajectory
weight on the score-function policy gradient**. Removing the
rolling window collapses to "per-episode i.i.d. weight" (cardinality
ablation); replacing the frontier with a fixed Gaussian collapses
to a non-vector ablation (the primary ablation); raising bandwidth
to ∞ collapses to REINFORCE (sanity ablation).

## Proof debt

1. **Wasserstein-Pareto convergence (open).** Conjecture: under
   matched-time-scale step sizes and PRISM updates, the policy-
   induced terminal-vector-return distribution `ν_π_n` converges in
   Wasserstein-2 distance to a measure supported on the upper Pareto
   frontier `∂_+ M`. Proof strategy: (i) `μ*` converges to a
   stationary measure once `F_n` stabilizes on `∂_+ M`; (ii) in the
   stationary limit, the cross-entropy gradient `∇_θ CE(ν_π, μ*)`
   has zeros precisely at `ν_π = μ*`; (iii) standard SGD-on-cross-
   entropy convergence applies once the target is stationary. The
   open piece is the joint dynamics of `μ*` and `θ`, requiring a
   two-timescale Borkar 2008-style argument with the target on the
   slow timescale.

2. **Frontier-coverage rate.** Show that `coverage_n` grows at least
   logarithmically in `n` under PRISM. The intuition: each new
   frontier point generates a new local mode in `μ*` whose basin is
   non-empty (because the bandwidth is positive), so subsequent
   gradient steps explore that basin and may discover further
   frontier points. The open piece is bounding the time between
   frontier discoveries.

3. **Bias of the median-heuristic bandwidth.** The bandwidth `h` is
   set by the median-heuristic on `F_n`, which is a stochastic
   estimator. The bias of `log μ*(y_τ)` under this stochastic `h`
   is an open question, analogous to the bias analysis of
   median-heuristic MMD (Garreau-Jitkrittum-Kanagawa 2018) but on
   the vector-return space.

4. **Hypervolume monotonicity.** Conjecture: under PRISM, the
   expected dominated hypervolume `HV(ν_π_n)` is monotone-non-
   decreasing in `n`. This is the load-bearing improvement claim
   if PRISM is to be a *new* MORL method; it is open and would
   require a Pareto-improvement-flow theorem analogous to
   Kakade-Langford for the scalar case.

5. **Comparison with linear scalarization.** Open: prove that on
   *non-convex* Pareto frontiers (such as DST-concave by
   construction), PRISM's expected hypervolume strictly dominates
   any linear-scalarization-with-fixed-`w` approach. The
   intuition: linear scalarization can only recover *convex hull
   vertices* of `M`, but PRISM's KDE-density weight admits
   non-convex frontier points in `F_n`, so the policy can be
   pulled toward them.

The empirical probe will reveal whether the rolling-KDE / score-
function dynamics produces measurably higher frontier coverage than
random-Gaussian weighting on the vector substrate within the 120s
budget; a positive `coverage_n` signal on DST or RG would justify
investing in proof item (1).
