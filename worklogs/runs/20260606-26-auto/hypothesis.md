# 20260606-26-auto -- UNIRANK (Per-Step Marginal Survival-Rank Score-Function Gradient) [probe]

## Principle

Update the policy by score-function ascent where each step's per-step
gradient weight is the **empirical CDF rank of the realized rollout's
channel-1 cumulative-return value `M^1_t` against a per-`t` rolling
buffer of historical channel-1 cumulatives at the same trajectory time
index, restricted to trajectories that reached time `t`** -- a
*unicriterial*, rank-invariant per-step weight that is simultaneously
(i) the structural reduction of PARGRAD's bivariate Pareto-rank weight
on substrates where channel 2 is deterministic given survival, and
(ii) a primitive whose empirical comparison to COPDEV's bivariate
mechanism on DST-concave produces load-bearing corpus evidence on
whether the bivariate-rank region is substrate-degenerate.

## Primitive

The **per-(t) marginal survival-conditioned empirical CDF**

  `F̂_t : ℝ → [0, 1]`,
  `F̂_t(x) := (1 / N_t) · |{ j ∈ alive_buffer_t : M^1_{t,j} ≤ x }|`

defined for each trajectory time `t ∈ {0, ..., T_max}`, where
`alive_buffer_t` is a rolling window of size `N` storing the realized
channel-1 cumulative `M^1_{t,j} := Σ_{k ≤ t} γ^k r^1_{k,j}` of past
trajectories that **reached time `t`** (i.e., did not terminate before
`t`). Domain: real-valued channel-1 cumulative-return scalars at time
`t`. Codomain: `[0, 1]` (probability rank).

This is **one typed mathematical object**: a finite-support empirical
CDF on `ℝ` per `t`, implicitly represented by a single ring buffer of
scalar `M^1_t` values per time index. The per-step weight is
`q_t(τ) := F̂_t(M^1_t(τ))`, the rank quantile of the current rollout's
channel-1 cumulative within the alive cohort at time `t`.

This is **not** a value function (no learned predictor of return-to-go),
**not** a Q-function (no state-action argument; indexed only by `(t)`),
**not** an advantage estimator (no baseline subtraction; the weight is
multiplicative in `[0,1]`, not additive), **not** a state-action tensor
(no buckets), **not** a reward bonus (the weight enters as a gradient
coefficient, not as a return modifier). It is purely a *cross-trajectory
marginal rank* statistic restricted to the alive cohort.

## Derivation sketch

1. **Substrate-degeneracy hypothesis (load-bearing for closure).** On
   DST-concave under the panel's quick configuration, channel 2
   (`r^2 ≡ -1` per surviving step) is deterministic given survival:
   for any trajectory that reaches time `t`, `M^2_t = -(1-γ^{t+1})/(1-γ)`
   is identical across all such trajectories. PARGRAD's
   weak-dominance-with-at-least-one-strict count `p_t` therefore
   reduces (run 22 derivation step 5) to
   `p_t(τ) = (1/N_t) · |{ j alive at t : M^1_{t,j} < M^1_t(τ) }|`,
   the strict-below rank of the current rollout's channel-1 cumulative
   within the alive cohort. The conjecture under test is whether **this
   reduction is empirically faithful to PARGRAD's bivariate primitive**
   on DST-concave; equivalently, whether the *bivariate channel-2
   information* contributes any signal beyond the unicriterial
   alive-rank weight on this substrate.

2. **Unicriterial weak-rank weight.** Define the unicriterial weight
   `q_t(τ) := F̂_t(M^1_t(τ))`, the **weak-below rank** of `M^1_t(τ)`
   in the alive cohort buffer at time `t` (`F̂_t` includes the `≤`
   tie). For DST-concave with `r^1 = 0` for non-terminal steps and
   `r^1 = treasure_value` only at the terminal pickup, alive-cohort
   trajectories at time `t` have `M^1_t = 0` for those still
   navigating and `M^1_t = γ^t · treasure` for those that just
   terminated at `t`. The buffer hence contains a mixture of `0`s
   and `γ^t · treasure` values; `q_t` is non-trivial.

3. **Why q_t is rank-invariant under monotone reparametrization.**
   `F̂_t` depends on `M^1_t` only through ordering, hence
   `q_t(τ) = F̂_t(M^1_t(τ))` is invariant under any componentwise
   strictly-monotone transformation `r^1 ← φ(r^1)`. By the same
   counterexample as PARGRAD/COPDEV (replace `r^1 ← log(1+exp(r^1))`),
   `q_t ≠ wᵀr_vec` for any deterministic scalarizing `w`. UNIRANK is
   therefore in the rank-invariant non-scalarization region (same
   feature class as PARGRAD/COPDEV).

4. **Why q_t fires at random init on DST-concave.** At random init,
   ~28% of rollouts find treasure within ~5 steps; the alive_buffer at
   time `t` contains a non-trivial mix of zero-treasure and
   treasure-found cumulatives. `q_t = 1` for treasure-found rollouts
   above the cohort median; `q_t < 1` for zero-treasure rollouts in
   the cohort. The mechanism is non-zero from rollout 1 of training.

5. **Why this is the structural reduction of PARGRAD on DST-concave.**
   With `M^2_t` constant across the alive cohort, PARGRAD's
   `p_t = |{ j : M^1_j ≤ M^1_t and M^2_j ≤ M^2_t and (M^1_j, M^2_j)
   ≠ (M^1_t, M^2_t) }|/N_t`. The `M^2_j ≤ M^2_t` clause is satisfied
   with equality for every alive `j`, so the strictness must come
   from channel 1 alone, giving
   `p_t = |{ j : M^1_j < M^1_t }| / N_t`. The strict version is
   `p_t = F̂_t^-(M^1_t)` (strict-below CDF). UNIRANK's `q_t` uses the
   weak-below CDF (`≤`), differing from PARGRAD's reduction only by
   the tie convention. Both are unicriterial channel-1 rank weights.

6. **Why this is the unicriterial reduction of COPDEV on DST-concave.**
   COPDEV's per-channel rank-disagreement
   `d_t = |F̂_t^1(M^1_t) - F̂_t^2(M^2_t)|` reduces, with channel 2
   deterministic given survival, to
   `d_t = |F̂_t^1(M^1_t) - 1|` (since `F̂_t^2` is a unit step at the
   constant alive-cohort value, `F̂_t^2(M^2_t) = 1` for every alive
   member). Up to the affine sign-flip `d_t ↔ 1 - q_t`, this is the
   complement of UNIRANK's weight. UNIRANK and `1 - COPDEV` share
   their information content on DST-concave alive cohorts.

7. **Per-step policy gradient.** The UNIRANK update is
   `g_θ = E_τ[ Σ_t q_t(τ) · ∇_θ log π_θ(a_t | s_t) ]`.

8. **Discriminating training-dynamics observable.** The same
   `gradnorm_var = Var_t(||g_t||) / Mean_t(||g_t||)²` scalar that fired
   on COPDEV (run 21, mean 0.426 vs ablation 0.000115). Predicted:
   UNIRANK reproduces a `gradnorm_var > 0.1` separation against the
   uniform-weight ablation, directly because `q_t` is a non-uniform
   per-step weight indexed by `t`.

9. **Why this is not Family A (bucketed-tensor + partial-order vote).**
   No state-bucketing, no action-bucketing, no channel index, no
   partial-order vote. The data structure is `T_max` ring buffers,
   each indexed by `t` only, each holding scalar `M^1_t` values. No
   tensor shape with state/action/channel axes; the weight `q_t` is a
   continuous real number used in score-function ascent.

10. **Why this is not Family B (pairwise comparison).** UNIRANK does
    not pair trajectories; the buffer aggregates many trajectories
    and the weight is the rank of one against the cohort.

11. **Why this is not Family E (avoid value vocabulary, keep value
    structure).** UNIRANK does not learn a future-compression of
    return: `q_t` is computed from the realized observation of `M^1_t`,
    is multiplicative not predictive, and replaces no V/Q/advantage.

12. **Closure semantics.** UNIRANK's empirical comparison to COPDEV
    (run 21, gradnorm_var 0.426, score 194 = random) and PARGRAD
    (run 22, score 95 < random) is the *load-bearing corpus output*
    of this probe:
    - **Outcome A (UNIRANK reproduces COPDEV's gradnorm_var
      separation, score within seed of 194)**: the bivariate primitive
      was substrate-degenerate on DST-concave; bivariate-rank region
      collapses to unicriterial-rank region; both are dead pending
      a non-degenerate vector substrate.
    - **Outcome B (UNIRANK fails to reproduce gradnorm_var separation
      OR scores notably below 194)**: COPDEV's bivariate channel-2
      information is load-bearing; the unicriterial reduction loses
      signal; bivariate-rank region remains structurally distinct
      and genuinely open.
    - **Outcome C (UNIRANK strictly beats random 194 on DST-concave)**:
      a single-channel rank weight is a useful primitive on this
      substrate, opening a unicriterial-rank direction the COPDEV
      ablation (REINFORCE) was unable to test (because REINFORCE's
      ablation collapsed to T=1 trajectories with vanishing
      gradients, run 21 curator).

13. **Proof debt.** (i) Convergence under the rolling-buffer
    dynamics; standard two-timescale Borkar argument. (ii) The
    optimized objective: conjecture `J̃(π) = E_τ[Σ_t F^π_t(M^1_t)]`,
    the expected sum of survival-conditioned channel-1 ranks, which
    rewards survival to time `t` jointly with above-cohort channel-1
    progress. (iii) Bias of the alive-cohort buffer estimator under
    selection effects (only alive-at-`t` trajectories enter buffer).

## Update rule

```
Inputs: env (vector reward, m=2 on DST-concave; UNIRANK reads ONLY
        channel 1, info["vector"][0]), policy π_θ, discount γ,
        learning rate α, buffer size N, max time T_max
Init:   θ random; for each t ∈ {0, ..., T_max}:
        alive_buffer[t] = deque(maxlen=N), empty.

For each episode:
    1. Roll out τ = (s_0, a_0, info_0, ..., s_T) under π_θ. T ≤ T_max.
       Read r1_t = info_t["vector"][0] per step. Channel 2 is read
       but NOT used in the weight (it is read only for the ablation
       comparison logging; see below).

    2. Compute per-step channel-1 cumulative:
         For t = 0..T:
             M1[t] = Σ_{k ≤ t} γ^k · r1_k

    3. Compute per-step weight q_t against the alive_buffer:
         For t = 0..T:
             N_t = len(alive_buffer[t])
             if N_t == 0:
                 q_t = 0.5         # symmetric prior at empty buffer
             else:
                 # weak-below rank (CDF with <= tie convention)
                 below_count = |{ x in alive_buffer[t] : x <= M1[t] }|
                 q_t = below_count / N_t

    4. Push M1[t] to alive_buffer[t] for every t the trajectory
       reached:
         For t = 0..T:
             alive_buffer[t].append(M1[t])

    5. Score-function policy gradient:
         g_θ = Σ_t q_t · ∇_θ log π_θ(a_t | s_t)
         θ ← θ + α · g_θ

    6. Logging observables (load-bearing for closure semantics):
         - gradnorm_var = Var_t(||g_t||) / Mean_t(||g_t||)^2
                       (the COPDEV-replication discriminator)
         - mean_qt        = mean(q_t over rollout t in 0..T)
         - var_qt         = var(q_t over rollout t in 0..T)
         - mean_qt_trend  = append per-episode mean_qt to a file for
                          drift analysis
         - mean episode length T
         - first-rewarded-episode-index (when r1 first nonzero)
         - shadow_d_t = |q_t - 1| for each t  -- the COPDEV-equivalent
                          weight under the substrate-degenerate
                          reduction (logging only; not used in update)
         - shadow_pt_strict = (1/N_t)·|{x in buffer : x < M1[t]}|  --
                          the PARGRAD-strict-reduction weight
                          (logging only)
```

## Empirical claim

stage: quick

claim: On the **quick** stage (DST-concave, vector reward; UNIRANK
reads channel 1 only and treats channel 2 strictly as an unused
alongside-channel) UNIRANK should produce, within 120s budget:

(a) **`gradnorm_var > 0.1`** consistently after the first 30 episodes,
    reproducing COPDEV's mechanism-presence signature (run 21:
    gradnorm_var mean 0.426).

(b) **Mean episode length T > 2** at end of training (avoiding the
    REINFORCE-collapse-to-T=1 failure mode that the COPDEV ablation
    hit in run 21).

(c) **A final hypervolume score on DST-concave that satisfies one of
    three load-bearing outcomes**:

    - Outcome A (substrate-degenerate-confirmation, score within ±20
      of random 194): bivariate-rank region collapses to unicriterial;
      run 21 COPDEV's empirical-signal was a unicriterial signal in
      disguise; closes the bivariate-rank region for the corpus.

    - Outcome B (substrate-degenerate-falsification, score notably
      below 194 or above random *and* `gradnorm_var` separation
      absent): COPDEV's bivariate channel-2 information was
      load-bearing on DST-concave; bivariate-rank region remains
      open; UNIRANK is not a useful reduction.

    - Outcome C (unicriterial-rank wins, score > 200): a single-
      channel rank weight is a useful primitive; opens unicriterial-
      rank direction not tested by COPDEV's ablation. This would be
      a fresh empirical-signal probe-result.

The quick stage is the appropriate test because (i) DST-concave under
the panel's quick configuration is the same env where COPDEV produced
its empirical-signal observable; (ii) the discriminating
`gradnorm_var` scalar fires from the first ~30 episodes regardless of
treasure discovery rate; (iii) the closure semantics depend on direct
comparison against logged COPDEV (run 21) and PARGRAD (run 22)
outcomes.

falsifier: **Primary** (mechanism presence): if `gradnorm_var` for
UNIRANK is below 0.05 mean and statistically indistinguishable from
the uniform-weight ablation throughout training, the unicriterial
rank-weight produces no per-step gradient-norm non-uniformity --
the primitive is decorative even at the training-dynamics level
(would be a structurally informative null because COPDEV did
produce gradnorm_var ≈ 0.426 on the same substrate).

**Secondary** (closure outcome A or B determination): a comparison of
UNIRANK's `gradnorm_var` against COPDEV's logged 0.426 directly
adjudicates Outcome A vs B. Equivalent gradnorm_var (within seed
variance, say 0.3-0.5 mean) confirms substrate-degeneracy (Outcome
A); strictly lower (< 0.15) supports Outcome B. The Curator should
explicitly log this comparison.

**Tertiary** (substrate signal): if UNIRANK's final score is strictly
worse than the uniform-weight ablation (REINFORCE on channel 1
return-to-go) by more than seed variance, the rank reweighting is
actively harmful on this substrate.

## Ablation plan

### Primary ablation (uniform per-step weight)

Replace the **per-step rank weight `q_t`** with **constant `q_t ≡ 1`**:

In `train_ablate.py`:
1. Skip step 3's CDF computation entirely.
2. Skip step 4's buffer maintenance.
3. In step 5, set `q_t ≡ 1` for all `t`.
4. Apply the same score-function update.

This recovers REINFORCE-without-baseline on channel-1 return-to-go
(noting that `Σ_t 1·∇log π_θ(a_t|s_t)` is REINFORCE with weight ≡ 1,
not weighted by `G_t`, so this is REINFORCE-without-cumulative-return
— the *minimal* ablation removing only the rank weight, not also
adding back any return weighting). This is the same ablation
structure as COPDEV/PARGRAD's primary ablation; it preserves the
score-function form and the rollout mechanism while removing the
*per-step rank-weight non-uniformity*.

### Sanity ablation (random-uniform per-step weight)

Optional second arm: replace `q_t` with `q_t ~ Uniform(0,1)` sampled
fresh per step. This preserves per-step weight non-uniformity but
removes the *systematic rank correlation*. Predicted: UNIRANK's
`gradnorm_var` should match the random-uniform ablation's roughly,
but the *systematic correlation between `q_t` and `M^1_t`-rank*
distinguishes UNIRANK in `mean_qt_trend` (drifts upward as policy
finds higher-rank trajectories) from random-uniform (stays at 0.5).
This sanity ablation is logging-only on `mean_qt_trend`; it is
**not** the load-bearing comparison.

### Discriminator predictions on DST-concave (within 120s budget)

- **Discriminator (i) -- gradnorm_var (PRIMARY for closure)**:
  UNIRANK should show `gradnorm_var > 0.1` mean across training.
  Uniform-weight ablation should show `gradnorm_var ≈ 0` (similar to
  COPDEV's run 21 ablation at 0.000115). If UNIRANK gradnorm_var
  matches COPDEV's 0.426 within a factor of 2-3, the substrate-
  degeneracy hypothesis is empirically confirmed.

- **Discriminator (ii) -- mean episode length T**: UNIRANK should
  not collapse to T=1 (COPDEV ablation's failure mode in run 21).
  The rank weight is bounded in `[0, 1]` and non-zero on average,
  so gradient magnitudes should not vanish to ~5e-12 the way the
  REINFORCE ablation in run 21 did.

- **Discriminator (iii) -- final hypervolume score**: as described
  in claim (c) above; partitions the result into Outcomes A/B/C.

If `gradnorm_var` for UNIRANK matches the uniform-weight ablation's
within seed variance, the rank weight provides no non-uniformity at
all -- structurally informative null. If `gradnorm_var` matches
COPDEV's 0.426, substrate-degeneracy confirmed (Outcome A). If
UNIRANK's score strictly exceeds 194, unicriterial-rank weight is a
useful primitive in its own right (Outcome C).

## Novelty boundary

Closest known methods:

(a) **REINFORCE / vanilla policy gradient** (Williams 1992).
    Weights every step's score by the cumulative return `G_t`.
    UNIRANK weights by the per-`t` empirical CDF rank `q_t` of the
    realized rollout's channel-1 cumulative within the alive cohort.
    The weight is rank-invariant under monotone channel-1
    reparametrization; REINFORCE's `G_t` is not.

(b) **COPDEV (run 21, this loop)**. COPDEV uses
    `|F̂_t^1(M^1_t) - F̂_t^2(M^2_t)|`, a bivariate rank-disagreement.
    UNIRANK uses `F̂_t(M^1_t)` -- the unicriterial channel-1 rank.
    UNIRANK is the structural reduction of COPDEV's primitive on
    substrates with deterministic-given-survival channel 2 (up to
    the affine `q_t = 1 - d_t` flip). They are mathematically
    distinct primitives; the load-bearing question is whether they
    are *empirically* distinct on DST-concave.

(c) **PARGRAD (run 22, this loop)**. PARGRAD uses bivariate
    weak-dominance-with-at-least-one-strict count. UNIRANK uses the
    unicriterial channel-1 strict-below count's weak relaxation.
    The UNIRANK weight is the structural reduction of PARGRAD on
    DST-concave (run 22 derivation step 5).

(d) **Order-statistics / ordergrad / Pass@K policy gradient** (Yu
    2014, Liu 2024 GRPO). These rank *whole trajectories* by *scalar*
    total return and apply weights based on trajectory rank.
    UNIRANK uses *per-step* time-indexed cumulative ranks against a
    per-`t` rolling buffer. Different granularity (per-step vs.
    per-trajectory).

(e) **Rank-based prioritized experience replay** (Schaul 2015).
    PER uses TD-error rank to set the *probability of sampling*
    transitions. UNIRANK has no replay buffer (the alive_buffer is
    a small per-`t` rolling history of scalar cumulatives, not a
    transition buffer), no TD-error, and the rank is a *per-step
    gradient weight*, not a sampling probability.

(f) **Quantile RL / IQN / Distributional RL** (Dabney 2018;
    Bellemare 2017). These learn the *quantile function* of the
    return distribution. UNIRANK does not learn quantiles or any
    distributional object -- it computes online empirical CDF ranks
    of *observed* alive-cohort cumulatives via a small ring buffer.
    No learned function over states or actions.

(g) **Linear / non-linear scalarization** (explicit disqualifier).
    UNIRANK's `q_t` is invariant under any monotone reparametrization
    `r^1 ← φ(r^1)`; no `wᵀr_vec` is. UNIRANK does not use channel 2
    in any way (not as a multiplicative factor, not as an additive
    weight, not as a buffer dimension). The probe is technically
    a *single-channel* probe on a vector env -- which is
    *scalarization with weight `w = [1, 0]`*, the disqualifier
    explicitly listed in `prior_attempts.md`. **This is the central
    novelty-boundary tension**: see point (k) below.

(h) **GRADCOMP (run 20, this loop)**. GRADCOMP rotates the parameter
    update direction toward the rollout's Fisher principal
    eigenvector. UNIRANK does not rotate; it reweights per-step
    score-function magnitudes.

(i) **Self-Imitation Learning (SIL)** (Oh 2018). SIL replays past
    above-threshold trajectories and trains the policy to imitate
    via max-margin. UNIRANK has no replay, no imitation loss, no
    threshold; the buffer holds scalar cumulatives only.

(j) **Order-statistics REINFORCE / quantile-weighted REINFORCE**
    (proposed in Glasserman 2004 for Monte Carlo control variates,
    not for policy gradient). The closest published variant is
    **quantile-weighted REINFORCE for risk-sensitive control**
    (Tamar 2015; Chow 2017 CVaR-policy-gradient): per-trajectory
    weights derived from the trajectory's return quantile.
    UNIRANK uses *per-step* cumulative-return ranks at time `t`
    against a *per-`t`* alive-cohort buffer. Different granularity
    and different cohort definition (per-step time-indexed
    survival cohort, not per-trajectory whole-return rank).

(k) **The vector-vs-scalar boundary (load-bearing).** UNIRANK reads
    channel 1 only from `info["vector"]`. By the strict
    `prior_attempts.md` disqualifier, "scalarized vector reward
    `wᵀr` for any fixed or learned `w`" is dead -- and `w = [1, 0]`
    is a fixed scalarization. **This probe acknowledges this
    tension**: UNIRANK is structurally a scalarization with weight
    `[1, 0]`. However, the *load-bearing claim* of this probe is
    **closure-semantic**: comparing UNIRANK's empirical signature
    against COPDEV's (run 21) directly tests whether the bivariate
    primitive's previously-logged empirical-signal was load-bearing
    on the bivariate channel-2 information OR was a unicriterial
    signal in disguise. The probe's contribution to the corpus is
    the closure outcome, not a claim of novelty in the
    scalarization region. The unicriterial primitive `q_t` itself is
    rank-invariant (not a `wᵀr` for any `w` at the *weight* level)
    even though the *channel-1-only* substrate access is a fixed
    scalarization. This distinction is honest about the
    disqualifier overlap.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: UNIRANK has
  no state/action bucketing. The data structure is `T_max` ring
  buffers indexed by `t` only.
- **Family B (pairwise comparison)**: UNIRANK aggregates many
  trajectories in the buffer; no pairing.
- **Family C (within-trajectory geometric statistic)**: UNIRANK's
  weight depends on cross-trajectory historical buffer at `t`. Not
  a within-trajectory statistic.
- **Family D (reward-independent + reward-gated)**: `q_t` is
  reward-dependent throughout (uses cumulative channel 1); no gate.
- **Family E (avoid value vocabulary)**: No value, no Q, no
  advantage. The buffer holds raw observations of `M^1_t`.
- **Family F (hand-engineered priors)**: No vocabulary or grammar.
- **Family G (mechanism stack)**: One primitive only.
- **Family H (cochain complexes)**: Not applicable.
- **Family I (per-channel parameter-space gradient aggregation
  IMTL-G)**: UNIRANK does *not* aggregate per-channel gradients in
  parameter space; it uses a unicriterial per-step weight on a
  single score-function gradient. Different mechanism slot.

The structural novelty bar is **deliberately modest** for this probe
because the load-bearing corpus contribution is the closure test of
the bivariate-rank region's substrate-degeneracy on DST-concave, not
a claim of fresh family invention.

## Proof debt

1. **Convergence under non-stationary alive-cohort buffer.** Standard
   two-timescale Borkar argument with the buffer on the slow
   timescale; conjecture: UNIRANK iterates converge to a stationary
   point of the modified objective
   `J̃(π) = E_τ[Σ_t F^π_t(M^1_t)]` where `F^π_t` is the alive-cohort
   stationary marginal CDF of `M^1_t` under `π`.

2. **Substrate-degeneracy formal characterization (load-bearing for
   the closure test).** Conjecture: when channel 2 is deterministic
   given survival (DST-concave), the bivariate weak-dominance count
   `p_t` reduces exactly to `F̂_t^1(M^1_t)` minus a tie-correction
   term of order `O(1/N_t)` (the strict-vs-weak boundary). Open:
   prove the empirical equivalence of UNIRANK's `q_t` and PARGRAD's
   `p_t` up to `O(1/N)` on DST-concave alive cohorts, formalizing
   the run 22 derivation step 5 reduction.

3. **Selection bias of the alive-cohort estimator.** The buffer
   only contains trajectories that reached time `t`, so `F̂_t` is a
   *survival-conditioned* estimator. Open: characterize the bias
   of `F̂_t` against the unconditional marginal `F^π_t` and the
   propagation to gradient estimator bias.

4. **Closure status on the substrate-degenerate region.** If
   Outcome A (UNIRANK reproduces COPDEV's gradnorm_var and score)
   is observed, the corpus closure statement is: *the bivariate-
   rank-weight region (COPDEV/PARGRAD/IMTL-G-related-but-distinct)
   reduces to the unicriterial-rank region on substrates with
   deterministic-given-survival channel structure; both are dead on
   DST-concave; bivariate-rank requires a non-degenerate vector
   substrate to remain meaningful*. This conditional closure
   statement requires the empirical comparison to be load-bearing,
   which is what this probe provides.

5. **Variance of the rank-weight gradient.** Open: characterize
   `Var[g_θ^UNIRANK]` vs. `Var[g_θ^REINFORCE]` and identify
   conditions for variance reduction.

The empirical probe directly produces (1)-(4)'s discriminating
evidence on DST-concave within 120s; the closure statement (4) is
the load-bearing corpus contribution that UNIRANK's panel run is
specifically designed to deliver, regardless of whether the score
outcome is Outcome A, B, or C.
