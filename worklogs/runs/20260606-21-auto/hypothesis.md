# 20260606-21-auto -- COPDEV (Per-Step Copula Deviation Policy Gradient) [probe]

## Principle

Update the policy by score-function ascent where each step's gradient
is weighted by the **per-step empirical copula deviation between the
two reward channels' cumulative processes** — i.e., the absolute
difference between the marginal-empirical-CDF positions of `M^1_t`
(cumulative treasure channel) and `M^2_t` (cumulative step-penalty
channel) — making the load-bearing primitive a non-linear,
**rank-invariant**, **non-scalarizing**, **non-monotone-in-each-component**
functional of the bicriterial cumulative return process.

## Primitive

The **per-channel marginal empirical CDF**

  `F̂_t^c : ℝ → [0,1]`,
  `F̂_t^c(x) := (1/N) · Σ_{j ∈ buffer} 1[ M^c_{t,j} ≤ x ]`

defined for `c ∈ {1, 2}` (one CDF per reward channel) over a rolling
buffer of the last `N` historical cumulative-channel values observed
**at time index `t`** across past trajectories. Domain: cumulative-
return scalar values per channel, indexed by trajectory time `t`.
Codomain: `[0, 1]` (probability rank).

This is **one typed object**: a finite-support empirical CDF on `ℝ`
per `(t, c)` pair, implicitly represented by a small per-(t, c) ring
buffer of recent cumulative values. The two CDFs together define the
per-step copula deviation
  `d_t(τ) := |F̂_t^1(M^1_{t,τ}) - F̂_t^2(M^2_{t,τ})|`,
which is the per-step gradient weight. The CDFs are the **single
mathematical object** computed and updated; the deviation `d_t` is its
*evaluation*, not a separate primitive.

The CDFs are **not** value functions, **not** Q-functions, **not**
advantage estimators, **not** state-action tables, **not** density
estimates of states. They are marginal probability ranks of the
*per-channel cumulative reward process viewed at trajectory time `t`*.

## Derivation sketch

1. **Bicriterial cumulative-return process.** For a 2-channel vector-
   reward MDP with per-step `r_vec_t = (r^1_t, r^2_t) ∈ ℝ²`, the
   trajectory induces a bivariate cumulative process
   `(M^1_t, M^2_t)_{t=0..T}` with `M^c_t := Σ_{k≤t} γ^k r^c_k`. This
   is a 2D random walk indexed by trajectory time.

2. **Marginal empirical CDFs at each time index.** For each fixed time
   `t`, the population-level marginal CDFs `F_t^c(x) := P(M^c_t ≤ x)`
   under `π_θ` are well-defined. The empirical estimators `F̂_t^c`
   computed from a rolling buffer of past trajectories' values at time
   `t` converge in sup-norm to `F_t^c` (Glivenko-Cantelli, applied per
   `(t, c)`). For DST-concave with deterministic step-penalty channel
   2 (`r^2_k ≡ -1`), `F_t^2` is a degenerate point mass at `-t`, so
   `F̂_t^2` collapses to a unit step at `-t` and `F̂_t^2(M^2_t) = 1`
   trivially — *but only when r^2 is exactly deterministic*. On RG
   (resource-gathering, vector stage), channel 2 is non-degenerate and
   `F̂_t^2` is non-trivial. Hence the construction is non-degenerate
   on at least one substrate env (RG); for DST-concave we use a
   **rank-shuffled** version of channel 2 (described in step 5).

3. **Copula deviation as rank-disagreement.** By Sklar's theorem, the
   joint distribution of `(M^1_t, M^2_t)` factors into marginals and a
   copula `C_t : [0,1]² → [0,1]`. The points
   `(F_t^1(M^1_t), F_t^2(M^2_t))` are uniform-marginal samples from
   `C_t`. The **diagonal of the unit square** is the comonotone copula
   (perfect rank-correlation); deviations from this diagonal measure
   *how rank-anticorrelated* the two channels are at this trajectory
   step. The absolute deviation
   `d_t(τ) := |F̂_t^1(M^1_t) - F̂_t^2(M^2_t)| ∈ [0, 1]`
   is the **L¹ distance from the comonotone diagonal**. It is large
   when one channel's value is in a high rank and the other in a low
   rank — a *bicriterial-divergence* signal.

4. **Why this is not scalarization.** `d_t` is a function of the **ranks**
   `F̂_t^1(M^1_t)` and `F̂_t^2(M^2_t)`, not of `M^1_t, M^2_t` directly.
   Replacing `r^c_t` with any strict monotone transformation
   `φ^c(r^c_t)` leaves `d_t` exactly invariant (rank-CDF is invariant
   under strictly-increasing reparametrization of each marginal). No
   linear functional `wᵀr` has this property — `wᵀ(φ¹(r^1), φ²(r^2))`
   is not invariant under `φ^c`. So `d_t ≠ wᵀr_t` for any fixed,
   adaptive, or learned `w ∈ ℝ²`.

5. **Per-step policy gradient.** The COPDEV update is
   `∇_θ J = E_τ[ Σ_{t=0}^{T} d_t(τ) · ∇_θ log π_θ(a_t | s_t) ]`,
   i.e., score-function ascent with **per-step weight `d_t`**. Note
   the cumulative return `R_τ` is not used; the weight is purely a
   **bicriterial rank-disagreement** at each step. For DST-concave
   (where channel 2 is deterministic and `F̂^2_t` collapses), we use
   a **rank-shuffled channel 2**: instead of using `M^2_t` as-is, we
   compute the **per-trajectory rank** of `t` against historical
   episode lengths, i.e., `F̂_t^2(M^2_t) := (rank of episode-survived-
   to-time-t in buffer) / N`. This makes channel 2's CDF position
   reflect "how unusual is this episode's survival to time t?".

6. **Why d_t is non-zero at random init on DST-concave.** At random
   init, episodes have length distribution concentrated around 5-10
   (random walk in 11-row deep grid with terminal at depth 5+). For a
   trajectory at step t=3, `F̂_t^2(M^2_3)` is the historical fraction
   of trajectories that survived ≥ 3 steps — a non-trivial number
   (e.g., 0.7 if 70% of historical trajectories were at least 3
   steps long). Channel 1 cumulative `M^1_3 = 0` typically, so
   `F̂_t^1(0)` is the historical fraction with cumulative treasure ≤ 0
   at step 3, which is ≈ 1.0. Therefore `d_3 = |1.0 - 0.7| = 0.3` —
   a non-trivial, finite gradient weight even at random init when no
   reward is found.

7. **Why this changes nearest-treasure trajectory dynamics on DST-
   concave.** Nearest-treasure trajectories have length T=5, with
   `M^1` jumping from 0 to ~0.7 at terminal. Under COPDEV, the per-
   step weights are: t=0..3: d_t reflects "how many trajectories
   have survived this long with zero treasure?" (mid-range); t=4
   (terminal): d_t = |F̂^1(0.7) - F̂^2(survived 5 steps)|, a sharp
   spike when 0.7 is rare among historical M^1_4 values. The *shape*
   of credit assignment is non-uniform across t — the mechanism
   *up-weights* the actions taken at moments of bicriterial rank-
   disagreement, *not* by their reward-weight. This produces a
   measurably different per-step gradient norm distribution than
   REINFORCE (which weights all steps by terminal G_t).

8. **Discriminating training-dynamics observable.** Define
   `gradnorm_var_t := Var_t(||g_t||) / Mean_t(||g_t||)²`
   (squared coefficient of variation of per-step gradient norms,
   averaged over a rollout). Under COPDEV, this is bounded away from
   zero (non-uniform per-step weights → varying gradient norms across
   t). Under the **uniform-weight ablation** (replace `d_t ← 1`,
   recovering REINFORCE-no-baseline), this is **identically zero by
   construction** for `||g_t||` (modulo score-function magnitude). The
   discriminator is a **logged scalar** that fires at random init,
   independent of whether the policy ever finds a treasure.

9. **Why this is not a within-trajectory geometric statistic
   (Family C).** Family C uses path geometry (hull, Lévy area,
   spectrum, signature) of the *state/cumulant trace* per `(s, a)` to
   drive logits. COPDEV uses **marginal probability ranks** of the
   cumulative reward channels at each `t`, computed against a
   **historical buffer across trajectories**. The CDF-rank evaluation
   requires a cross-trajectory empirical buffer — it is **not a
   within-trajectory statistic**. Rank invariance (point 4) excludes
   the geometric-statistic angle.

10. **Why this is not a Family A bucketed tensor.** There is no
    per-state, per-action, per-channel tensor. The CDFs `F̂_t^c` are
    indexed only by `(t, c)` — at most `T × 2` ring buffers, each
    storing the last `N` scalar values of one channel's cumulative
    return at one time index. No state-bucketing, no action-bucketing,
    no partial-order vote.

11. **Proof debt (open).** (i) Convergence of the COPDEV stochastic
    gradient to a stationary point of some objective `J̃(π_θ)`. The
    weight `d_t` is non-stationary (depends on rolling buffer), so
    the analysis requires a two-timescale argument with the buffer on
    the slow timescale (Borkar 2008). (ii) The functional `J̃` being
    optimized: conjecture, `J̃(π) = E_τ[Σ_t d_t(τ; F)] · log π_θ(a_t|s_t)`
    with `F` the stationary marginal CDFs at policy `π`, and the
    fixed-point characterization of (π*, F*) at convergence.
    (iii) Bias of the rolling-buffer CDF estimator and its propagation
    to gradient bias.

## Update rule

```
Inputs: env (vector reward, m=2), policy π_θ, discount γ,
        learning rate α, buffer size N (per (t, c) pair), max time T_max
Init:   θ random; for each (t, c) ∈ {0,...,T_max} × {1, 2}:
        ring_buffer[t, c] = deque(maxlen=N), empty.

For each episode:
    1. Roll out τ = (s_0, a_0, r_vec_0, ..., s_T) under π_θ, T ≤ T_max.
       Read r_vec_t per-component from info["vector"] for the vector
       env (DST-concave: m=2; channel 1 = treasure, channel 2 = step
       penalty). NEVER sum the channels.

    2. Compute cumulative per-channel values:
         For t = 0..T, c ∈ {1, 2}:
             M[t, c] = Σ_{k ≤ t} γ^k · r_vec_{k, c}

    3. Compute per-step copula deviation d_t:
         For t = 0..T:
             # Empirical CDF rank using current ring_buffer
             rank1 = (count of M[t, 1] >= b for b in ring_buffer[t, 1]) / |buffer|
                     OR 0.5 if buffer empty
             # For DST-concave (degenerate channel 2),
             # use survival-rank instead of M[t, 2]:
             # survival[t] = number of historical episodes that
             # survived ≥ t steps / |all_episodes|
             rank2 = survival_cdf[t]  # scalar in [0, 1]
             d_t = |rank1 - rank2|

    4. Push current values into the rolling buffer:
         For t = 0..T:
             ring_buffer[t, 1].append(M[t, 1])
         survival_cdf updated: increment count of "survived ≥ t" for
         every t in 0..T.

    5. Score-function policy gradient with per-step weight:
         g_θ = Σ_t d_t · ∇_θ log π_θ(a_t | s_t)
         θ ← θ + α · g_θ

    6. Logging observables (load-bearing for ablation discrimination):
         - mean(d_t over t)            # per-rollout mean weight
         - var(d_t over t)              # per-rollout weight spread
         - gradnorm_var = Var_t(||g_t||) / Mean_t(||g_t||)²
                                        # cv² of per-step gradient norms
         - mean episode length          # discriminating training-dynamics scalar
         - first-rewarded-episode-index # if/when treasure first found
```

The load-bearing primitive is the **per-(t,c) marginal empirical CDF
F̂_t^c**, expressed as a small ring buffer per (t, c). The per-step
gradient weight `d_t` is its evaluation on the realized rollout. Total
storage: `O(T_max · 2 · N)` scalars. Per-step lookup: `O(N)` (or
`O(log N)` with a sorted structure). Negligible compared to backprop.

## Empirical claim

stage: quick

claim: On the **quick** stage (which maps to deep-sea-treasure-concave-v0
under the panel's quick configuration; vector reward, dense step
penalty), COPDEV should produce a **strictly larger
`gradnorm_var = Var_t(||g_t||)/Mean_t(||g_t||)²`** than its uniform-
weight ablation **within 60 seconds of training** (well within the
120s budget). This observable is logged per-rollout and aggregated
into the panel-level `result.json`. Additionally, COPDEV's **mean
episode length** trajectory should diverge measurably from the
ablation's: COPDEV should commit to terminal-disagreement-maximizing
trajectories at a different rate than REINFORCE-with-uniform-weights.

The quick stage is the appropriate test because (i) the panel runs
quick on DST-concave (per recent runs), where the bicriterial channel
structure is exactly the substrate for COPDEV's primitive; (ii) the
discriminating observable `gradnorm_var` is computable from logged
gradient norms and is *non-zero at random init*, so it does not require
the policy to find any treasure to fire; (iii) the nearest-treasure
trajectory regime (length 5, terminal treasure ≈0.7) provides a
non-degenerate cumulative-return process through which the per-channel
CDFs can be estimated within the budget.

falsifier: **Primary**: if `gradnorm_var` for COPDEV equals the
ablation's `gradnorm_var` within seed variance (i.e., the per-step
weight `d_t` is empirically constant across t, so per-step gradient
norms are uniform), the copula-deviation primitive is decorative. This
should be detectable in the first ~50 episodes (well within budget).
**Secondary**: if COPDEV's mean episode length matches the ablation's
within seed variance throughout training, the mechanism produces no
measurable effect on training dynamics on this substrate. **Tertiary**:
if COPDEV's hypervolume on DST-concave is strictly worse than the
ablation's (i.e., the rank-disagreement weighting drives the policy
**away** from useful regions), the principle is harmful on this
substrate.

## Ablation plan

Replace the **per-step copula-deviation weight `d_t`** with **uniform
weight 1** (recovering REINFORCE-without-baseline):

In `train_ablate.py`:
1. Skip step 3 entirely (no CDF computation, no rank lookup).
2. Skip step 4 entirely (no buffer maintenance).
3. In step 5, set `d_t ≡ 1` for all t.
4. Apply the same score-function update.

This preserves: per-step score-function gradient computation, all
hyperparameters, the policy architecture, the rollout mechanism. It
removes: the *per-step rank-disagreement weighting*.

Predicted contrast on DST-concave (within 120s budget):

- **Discriminator (i) — `gradnorm_var`**: COPDEV's per-step weights
  `d_t` vary across t (CDF ranks are non-uniform across trajectory
  steps), so per-step gradient norms are non-uniform → `gradnorm_var
  > 0.1` typically. Ablation has `d_t ≡ 1`, so per-step gradient
  norms equal `||score_t||` only — the ratio `Var_t(||score_t||) /
  Mean_t(||score_t||)²` is determined purely by the *score function
  variability* across t, which is low for a near-uniform random
  policy. Predicted ratio: COPDEV > ablation by at least 2× within
  the first 50 episodes. **This is a logged scalar and is the primary
  load-bearing discriminator that fires at random init.**

- **Discriminator (ii) — mean episode length**: COPDEV's d_t-weighted
  gradient should produce systematically different commitment behavior
  than REINFORCE's uniform weighting. Whether this is shorter or
  longer trajectories depends on the substrate; the prediction is
  *measurably different*, with confidence interval non-overlapping
  the ablation's, within the budget.

- **Discriminator (iii) — final hypervolume**: Both arms are likely to
  end near DST=99 (nearest-treasure floor), but COPDEV may shift the
  final score by ±10 (i.e., to 89 or 109) due to different commitment
  rates. The hypervolume **is not** the primary discriminator; the
  load-bearing observable is `gradnorm_var`, which fires at random
  init and is a logged training-dynamics scalar.

A second sanity ablation (cheap, optional): replace `d_t` with a
**fixed random shuffle** of `{0.0, 0.5, 1.0}` per step (ignoring the
buffer). If COPDEV matches this random-d_t ablation on `gradnorm_var`
but the *mean episode length* differs, the rank-structure of `d_t`
matters; if it matches on both, the magnitude-distribution of d_t is
all that matters.

If the uniform-weight ablation matches COPDEV on `gradnorm_var`, the
copula-deviation mechanism is not load-bearing. If COPDEV exhibits
strictly higher `gradnorm_var` AND a measurably different mean episode
length, the primitive is causally responsible for the dynamics.

## Novelty boundary

Closest known methods:

(a) **REINFORCE / vanilla policy gradient** (Williams 1992). REINFORCE
    weights every step's score by the *cumulative return* `G_t` (or
    return-to-go). COPDEV weights every step by the **per-step
    bicriterial rank-disagreement `d_t`**, which is *not* a function
    of the cumulative scalar return. The weight is rank-invariant
    (Sklar's theorem application), and depends on a cross-trajectory
    empirical-CDF buffer. This is a structurally distinct per-step
    weight.

(b) **Policy gradient with baseline / Advantage** (Sutton 1999).
    Subtracts `b(s_t)` or uses `Â_t = G_t - V(s_t)` to reduce variance.
    COPDEV has no baseline subtraction and no learned value function;
    `d_t` is computed from a rolling buffer of cumulative-channel
    values, not from any learned function over states.

(c) **GAE** (Schulman 2016). GAE weights are `(γλ)`-weighted sums of
    TD residuals over a learned `V`. COPDEV has no V, no TD residual,
    no exponential interpolation. The weight is a copula deviation,
    a known statistical concept (Sklar 1959; Nelsen 2006 *Introduction
    to Copulas*).

(d) **Distributional RL** (Bellemare 2017). Distributional RL learns
    a **return distribution** `Z(s, a)` per (state, action). COPDEV
    has no learned distribution; only marginal CDFs of *time-indexed
    cumulative channel values*, which are **not** state-action
    distributions. There is no distributional Bellman operator.

(e) **Rank-based prioritized experience replay** (Schaul 2015). Uses
    the **rank of TD-error** to set the *probability of sampling a
    transition from the replay buffer*. COPDEV has no replay buffer,
    no TD-error, and the rank is applied as a **per-step gradient
    weight**, not as a sampling probability. The mechanism slot is
    different: PER changes which transitions are sampled; COPDEV
    changes the gradient weight at each step of an on-policy rollout.

(f) **Order-statistics policy gradient (ordergrad / PolicyBoost)**
    (Yu 2014; recent ordergrad repo). These rank **whole trajectories**
    by total return and apply weights to the score-function based on
    trajectory-rank (e.g., Pass@K, Top-M@K, CVaR, quantile objectives).
    COPDEV applies a **per-step** weight, not a per-trajectory weight,
    and the weight depends on the **bicriterial copula** (two channels'
    joint structure), not on a single-objective rank of total return.

(g) **Quantile regression / quantile RL** (Dabney 2018). Learns the
    *quantile function* of the return distribution. COPDEV does not
    learn quantiles or any distributional object — it computes *online
    empirical CDF ranks* of *observed* cumulative channel values via
    a small ring buffer. No learned function over states or actions.

(h) **Linear / non-linear scalarization of vector reward** (explicit
    disqualifier). COPDEV is **not** scalarization: per Sklar's
    theorem, `d_t` is invariant under any strictly-monotone componentwise
    transformation of `(r^1, r^2)`, while *any* linear or non-linear
    scalarization `f(r^1, r^2)` is sensitive to such transformations.
    Counterexample: replace `r^1_t ← exp(r^1_t)`. Then `d_t` is
    unchanged (CDF ranks are invariant under monotone reparametrization);
    *any* scalarization weight changes. So `d_t ≠ f(r^1, r^2)` for any
    deterministic scalarizing function `f`.

(i) **PRISM (run 18, this loop)**. PRISM uses a rolling KDE of
    *terminal vector returns* on the Pareto frontier and a per-
    trajectory log-density weight. COPDEV uses **per-time-step
    marginal CDFs** of *cumulative* channel values (not terminals).
    The weight is **per-step** (not per-trajectory) and is the L¹
    distance from the comonotone copula (not log-density of a
    Pareto-frontier KDE). Different mathematical object, different
    application granularity (per-step vs. per-trajectory), different
    application slot.

(j) **GRADCOMP (run 20, this loop)**. GRADCOMP uses the per-rollout
    Fisher principal eigenvector as a parameter-space rotation
    direction. COPDEV has no Fisher computation and no parameter-space
    rotation. The weight `d_t` modifies the *magnitude* of each
    per-step score-function contribution, leaving the score-function
    direction unchanged.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: COPDEV has no
  state/action bucketing. The CDFs are indexed by `(t, c)` only —
  trajectory time and reward channel. There is no partial-order vote
  on the rank values; `d_t` is a real-valued L¹ distance.
- **Family B (pairwise trajectory comparison)**: COPDEV does not pair
  trajectories. The CDFs aggregate across the rolling buffer of
  many trajectories; the weight `d_t(τ)` is a one-trajectory
  evaluation against the buffer.
- **Family C (within-trajectory geometric statistic)**: COPDEV's
  weight `d_t` is **NOT** a within-trajectory geometric statistic.
  It depends on cross-trajectory historical CDFs `F̂_t^c`, which are
  not available within a single trajectory in isolation. The path
  geometry of the cumulative process is not used (no hull, no Lévy
  area, no signature).
- **Family D (reward-independent + reward-gated)**: `d_t` is
  reward-dependent throughout (uses cumulative channels), and the
  application is also reward-dependent (the gradient weight is `d_t`,
  not a gate).
- **Family E (avoid value vocabulary)**: No value function, no
  cumulant prediction, no return compression. The CDFs are
  observational empirical statistics, not learned future-compressions.
- **Family H (cochain complexes)**: Not applicable.

The structural difference from all named methods and dead families is
the **per-step copula deviation between per-channel cumulative-return
empirical CDFs as a rank-invariant gradient weight**. This is
operationally distinct from Bellman backups (no operator), policy-
gradient-with-advantage (no advantage), distributional RL (no
distribution learned), prioritized replay (no replay buffer), and
order-statistics methods (per-step not per-trajectory; bicriterial
not unicriterial).

## Proof debt

1. **Convergence under non-stationary CDF buffer.** Conjecture: under
   COPDEV updates with a rolling buffer of fixed size N, and standard
   Robbins-Monro step sizes for `θ`, the iterate `θ_n` converges to a
   stationary point of the modified objective
   `J̃(π) := E_τ[Σ_t d_t(τ; F̂_τ) · 1_{a_t = a_t}]`
   in the joint dynamics with the buffer's slow timescale. Strategy:
   two-timescale Borkar 2008 argument with the buffer on the slow
   timescale; the open piece is identifying the stationary objective
   `J̃` and characterizing its stationary points.

2. **Identifying the optimized objective.** What does COPDEV optimize?
   At stationarity, the buffer's CDFs `F̂_t^c` converge to the
   policy-induced marginals `F^c_t(·; π_θ)`. The gradient becomes
   `E_τ[Σ_t |F^1_t(M^1_t) - F^2_t(M^2_t)| · ∇log π_θ(a_t|s_t)]`,
   which is the score-function gradient of the functional
   `Φ(π) := E_τ[Σ_t |F^1_t(M^1_t; π) - F^2_t(M^2_t; π)|]`,
   the **expected total copula deviation along trajectories**.
   Open question: characterize the maximizers of `Φ` and their
   relation to Pareto-optimal vector returns.

3. **Bias of the rolling-buffer CDF estimator.** The empirical CDF
   `F̂_t^c` from a buffer of size N has bias `O(1/N)` and variance
   `O(1/N)` against the true marginal CDF `F^c_t`. Open: bound the
   propagation to the gradient estimator's bias and variance, and
   determine the buffer-size schedule that matches the policy
   step-size schedule for joint convergence.

4. **Pareto-improvement claim (conditional).** Conjecture: maximizers
   of `Φ` lie on or near the Pareto frontier of the achievable vector-
   return set `M ⊆ ℝ²`. Intuition: the copula deviation is maximized
   on rank-anticorrelated trajectories, which are Pareto-non-comparable
   to each other and lie on the frontier. This would justify COPDEV
   as a bicriterial-Pareto-search method without scalarization. Open
   and load-bearing.

5. **Variance of d_t under degenerate channels.** When channel 2 is
   deterministic (DST-concave), `F̂_t^2` collapses and `d_t` reduces
   to a simpler function (the **survival-rank** variant in step 5 of
   the update rule). Open: characterize when this degenerate-case
   reduction is equivalent to a known method and when it remains
   distinct.

The empirical probe will reveal whether the per-step gradient-norm-
variance observable `gradnorm_var` separates COPDEV from its uniform-
weight ablation on DST-concave within 120s; a positive separation
on this logged scalar would justify investing in proof items (2)
and (4).
