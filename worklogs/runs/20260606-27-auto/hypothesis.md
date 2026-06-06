# 20260606-27-auto -- EFLO (Entropy-Flow GAE Policy Update) [probe]

## Principle

Apply GAE-style bias-variance-interpolated exponential averaging to
**policy-entropy residuals** `δ^H_t := H(π_θ(·|s_{t+1})) - H(π_θ(·|s_t))`
along the rollout, and use the resulting forward-entropy-flow estimator
`Â_t^H(λ)` as the **sole** per-step weight on the score-function update;
the policy thereby ascends a discounted forward-state-conditional
policy-entropy functional with a single λ ∈ [0,1] trading off the bias
of the one-step entropy residual against the variance of the
infinite-horizon entropy-flow.

## Primitive

The **GAE-style forward-entropy-residual estimator**

  `Â^H : Trajectory × Π × [0,1] → ℝ^{T}`,
  `Â_t^H(τ; θ, λ) := Σ_{l=0}^{T-t-1} (γλ)^l · δ^H_{t+l}(τ; θ)`,
  `δ^H_t(τ; θ) := H(π_θ(·|s_{t+1})) - H(π_θ(·|s_t))`.

Domain: a single rollout `τ = (s_0, a_0, …, s_T)`, the current policy
parameters `θ`, and the bias-variance hyperparameter `λ`. Codomain: a
real-valued vector of length `T` (one weight per realized step).

This is **one typed mathematical object**: a `R^T`-valued functional of
(trajectory, policy, λ) computed from the policy's own state-conditional
action-entropy at consecutive realized states, exponentially smoothed
forward in time. The full primitive is the entropy-flow estimator; the
policy update follows from it directly (see step 5 of the derivation).

This is **not** a value function (no learned future-compression), **not**
a Q-function (no state-action argument; entropy depends only on state),
**not** an advantage estimator (no baseline subtraction; the residual is
an entropy difference, not a reward-based difference), **not** a critic
(no separate parameters; `H(π_θ(·|s))` is read off the policy network
directly), **not** a state-bucketed tensor (no buckets), **not** a count
or visitation density (no state-counting), **not** a learned novelty
predictor (no auxiliary network), **not** a per-channel gradient
aggregator (single scalar, no channel index), **not** a per-step rank
weight (the weight is real-valued from policy entropies, not a CDF
position), **not** a reward bonus (entropy enters as a *gradient
weight*, not as an additive intrinsic reward).

## Derivation sketch

1. **Setup.** Let `π_θ : S → Δ(A)` be a softmax policy. For any state
   `s` define `H(π_θ(·|s)) := -Σ_a π_θ(a|s) log π_θ(a|s)`, the standard
   Shannon entropy of the policy's action distribution at `s`. Along a
   rollout `τ = (s_0, a_0, …, s_T)`, define the **policy-entropy
   residual** `δ^H_t := H(π_θ(·|s_{t+1})) - H(π_θ(·|s_t))`.

2. **GAE construction (Schulman 2016, §3, applied to a non-value
   residual).** GAE defines `Â_t^GAE(γ,λ) := Σ_{l=0}^∞ (γλ)^l δ_{t+l}`
   where `δ_t := r_t + γV(s_{t+1}) - V(s_t)` is the TD residual on a
   value function. Replacing the value-residual `δ_t` with the
   **entropy-residual** `δ^H_t` gives the **entropy-flow GAE estimator**
   `Â_t^H(λ) := Σ_l (γλ)^l δ^H_{t+l}`. This is the same exponential-
   averaging machinery applied to a structurally different residual.

3. **Bias-variance interpolation property.** For `λ = 0`,
   `Â_t^H(0) = δ^H_t = H(π_θ(·|s_{t+1})) - H(π_θ(·|s_t))`, the **single-
   step forward-entropy-difference** (one-step low-variance, biased-by-
   discount). For `λ = 1`, `Â_t^H(1) = Σ_l γ^l δ^H_{t+l}`, the **full
   discounted-sum forward-entropy-flow** (Monte-Carlo, low-bias high-
   variance). Intermediate `λ ∈ (0,1)` interpolates, exactly as in GAE.

4. **Telescoping identity for `λ = 1`.** When `λ = 1`, the sum
   telescopes: `Σ_l γ^l δ^H_{t+l} = (1-γ) Σ_l γ^l H(π_θ(·|s_{t+l+1}))
   + γ^∞ H(π_θ(·|s_∞)) - H(π_θ(·|s_t))`. Up to discounting, this is a
   *forward-discounted-mean entropy* minus the current entropy — i.e.,
   the **entropy advantage** of step `t`. This is the entropy-only
   analogue of the value-advantage `G_t - V(s_t)` that GAE-1 estimates.

5. **Policy update from the principle.** The principle "ascend the
   discounted forward-state-conditional entropy functional"
   `J^H(θ) := E_τ[Σ_t γ^t H(π_θ(·|s_t))]` has, by the policy-gradient
   theorem applied to a state-occupancy-determined objective, the
   gradient
   `∇_θ J^H(θ) = E_τ[Σ_t Â_t^H(λ) · ∇_θ log π_θ(a_t|s_t)
                + Σ_t γ^t ∇_θ H(π_θ(·|s_t))]`,
   where the first term is the **score-function-credit** for entropy
   propagated through the rollout (the EFLO update direction) and the
   second is the **direct entropy gradient** at visited states. The
   load-bearing primitive is the first term; its weight is `Â_t^H(λ)`,
   which is exactly the GAE-style estimator. Treating `Â_t^H(λ)` as a
   stop-gradient quantity (per GAE's standard convention) gives the
   pseudo-gradient
   `g_θ^EFLO := E_τ[Σ_t Â_t^H(λ) · ∇_θ log π_θ(a_t|s_t)]`.
   This is the **EFLO update**.

6. **Why the update is reward-independent.** `Â_t^H(λ)` is a function of
   only `(s_0, …, s_T, θ)` — never of reward. The substrate provides
   `info["vector"]` on vector envs and scalar reward on scalar envs;
   EFLO reads neither. This sidesteps the scalarization disqualifier
   *and* the bootstrap-wall constraint (the per-step weight is non-zero
   at random init regardless of reward discovery).

7. **Why the primitive fires at random init.** At a random softmax
   policy, `H(π_θ(·|s))` is positive and varies smoothly with `s`.
   `δ^H_t` has expectation `O(1/√d_θ)` in the random-feature regime
   (small) but standard deviation `O(σ_logit)` where `σ_logit` is the
   logit-scale of the random init (typically 0.1–1.0). Therefore
   `Var(δ^H_t) > 0` strictly, and `Â_t^H(λ)` is a non-trivial random
   variable from rollout 1 of training. The mechanism does not need any
   reward signal to begin altering the policy.

8. **Why this is not Family A (bucketed-tensor + partial-order vote).**
   No state-bucketing, no action-bucketing, no channel index, no
   partial-order vote, no tensor. Storage is `O(T)` per rollout (a
   single scalar `Â_t^H(λ)` per step).

9. **Why this is not Family C (within-trajectory geometric statistic).**
   `Â_t^H(λ)` is a function of the *policy's own action-distribution*
   at consecutive states — a quantity defined on the **policy
   manifold**, not on the observation/cumulant trace. It is invariant
   under any state reparametrization that preserves softmax-policy
   logits; path-geometric statistics (hull, signature, Lévy area) are
   not.

10. **Why this is not Family I (per-channel parameter-space gradient
    aggregation).** EFLO does not split the gradient into per-channel
    components. It computes one scalar weight per step from the policy's
    own entropy. There is no `K`-tuple of unit vectors, no IMTL-G
    closed-form aggregator, no normalization-then-sum.

11. **Why this is not SAC / soft-Bellman / max-entropy RL** (the
    nearest disqualifier). SAC adds `α H(π(·|s_t))` into the **soft-
    Bellman target** of a learned **soft Q-function**, then policy-
    iterates against the soft Q. EFLO has no Q-function, no Bellman
    operator, no soft-Bellman target, and no per-step entropy *added to
    reward*. Its per-step weight is the **GAE-exponentially-averaged
    forward entropy DIFFERENCE between consecutive visited states** —
    a temporal entropy-flow estimator, not a single-step entropy
    bonus. The structural distinction: SAC's `H(π(·|s_t))` enters at the
    *value-bootstrap* level; EFLO's `Â_t^H(λ)` enters at the *credit-
    assignment-weight* level on the existing score-function gradient.

12. **Why this is not entropy-regularized PG** (A2C/PPO entropy bonus).
    Standard entropy-regularized PG adds `α · ∇_θ H(π(·|s_t))` to the
    gradient at each visited state — a *direct* entropy-gradient
    contribution that does not depend on the *future* entropy of the
    rollout. EFLO replaces the standard return-weighted score with a
    forward-entropy-flow-weighted score; the per-step weight is a
    **temporal sum** of entropy residuals, not a per-step entropy
    derivative, and it is **multiplicative** on `∇log π_θ(a_t|s_t)`,
    not an *additive* entropy gradient term.

13. **Why this is not GAE-with-renamed-residual.** The GAE formula is
    machinery; the *residual* is the load-bearing object. GAE's
    residual is a value-function TD residual `r_t + γV(s_{t+1}) -
    V(s_t)` — a *learned* quantity tied to a value predictor. EFLO's
    residual is `H(π_θ(·|s_{t+1})) - H(π_θ(·|s_t))` — a *closed-form*
    quantity derived from the policy itself, **with no learned
    predictor**, **no value target**, **no Bellman bootstrapping**, and
    **no reward**. The exponential-averaging is GAE's machinery; the
    entropy-flow residual is the genuinely new primitive.

14. **Why this is not RND/count-based exploration.** RND uses a
    randomly-initialized predictor network whose error on a state is
    the intrinsic reward; count-based methods use state-visitation
    counts. EFLO uses neither: the entropy is the policy's own action-
    distribution entropy, and the residual is a temporal *difference*
    of entropy, not a state-novelty score. No predictor network, no
    counts, no kNN, no density model.

15. **Proof debt** (acknowledged, not assumed): (i) full convergence of
    EFLO iterates to a stationary point of `J^H(θ)` (the discounted
    forward-state-conditional entropy functional); (ii) characterization
    of when `Â_t^H(λ)` as a stop-gradient produces a low-bias estimate
    of the true `J^H` gradient; (iii) variance bound vs. the
    `λ=1` Monte-Carlo and `λ=0` one-step variants.

## Update rule

```
Inputs: env (any reward shape — EFLO does NOT read reward),
        policy π_θ (softmax over discrete actions), discount γ,
        learning rate α, GAE-trade-off λ ∈ [0,1]
Init:   θ random.

For each episode:
    1. Roll out τ = (s_0, a_0, ..., s_T) under π_θ. Reward is NOT
       read by EFLO; we depend only on the state-trajectory and the
       policy itself.

    2. Compute per-state policy entropy:
         For t = 0..T:
             H_t = -Σ_a π_θ(a | s_t) · log π_θ(a | s_t)
             # standard Shannon entropy of the softmax distribution

    3. Compute per-step entropy residuals:
         For t = 0..T-1:
             delta_H_t = H_{t+1} - H_t

    4. GAE-style forward-entropy-flow weight (right-to-left
       backward recursion, exactly as canonical GAE):
         A_T = 0
         For t = T-1 down to 0:
             A_t = delta_H_t + γ * λ * A_{t+1}
         # A_t = Σ_{l=0}^{T-t-1} (γλ)^l · delta_H_{t+l}

    5. Detach (stop-gradient) the entropy-flow weight from θ:
         For t = 0..T-1:
             c_t = A_t.detach()    # treat as a fixed scalar weight

    6. Score-function policy update:
         g_θ = Σ_t c_t · ∇_θ log π_θ(a_t | s_t)
         θ ← θ + α · g_θ

    7. Logging observables (load-bearing for ablation/closure):
         - mean(c_t over t)              # rollout-mean entropy-flow
         - std(c_t over t)               # weight non-uniformity
         - mean(H_t over t)              # mean per-state entropy
         - mean(|delta_H_t|) over t      # raw entropy-residual scale
         - cosine alignment between EFLO gradient and the
           uniform-weight (c_t ≡ 1) gradient g_U = Σ_t 1 · ∇log π_t,
           computed once per rollout — strictly < 1 whenever c_t is
           non-uniform; equals 1 only in the trivial case
         - mean episode length T
         - first-rewarded-episode-index (for reference; not used in
           the update)
```

The load-bearing primitive is the **GAE-style forward-entropy-residual
estimator** `Â_t^H(λ)`, computed in step 4 by the canonical right-to-left
GAE recursion. Per-step cost: one extra log-softmax-entropy computation
per visited state, dominated by the standard policy forward pass. No
auxiliary network, no replay buffer, no per-step return-to-go.

## Empirical claim

stage: quick

claim: On the **quick** stage (deep-sea-treasure-concave-v0 under the
panel's quick configuration; 4-action, ~11×11 grid, vector reward —
EFLO does NOT read reward and is environment-reward-shape-agnostic)
EFLO with `λ = 0.95` should produce, within 120s budget:

(a) **`std(c_t) > 0.1 · |mean(c_t)| + 1e-3`** consistently after the
    first 30 episodes — the entropy-flow weight is non-uniform across
    steps. Detectable from rollout 1 since `δ^H_t` is non-zero at random
    init; the rolling mean over 30 episodes stabilizes within budget.

(b) **Mean cosine alignment between EFLO gradient and the
    uniform-weight (c_t≡1) gradient strictly less than 0.95** across
    training, confirming that the entropy-flow weight genuinely
    reshapes the score-function direction (not just rescales it).

(c) **Mean per-state entropy `H_t` does NOT collapse to 0** within
    budget — i.e., the EFLO update preserves a non-trivial action
    entropy at visited states, which is the structural signature of an
    entropy-flow-maximizing update vs. a degenerate log-likelihood
    maximizer (the c_t≡1 ablation is expected to collapse `H_t → 0`
    rapidly because Σ_t ∇log π_θ(a_t|s_t) is the gradient of trajectory
    log-likelihood, which deterministically concentrates the policy).

(d) **Final hypervolume score on DST-concave at or above the random
    floor (194)** as a *secondary* outcome. EFLO reads no reward, so
    score-axis improvement is not the primary claim — the primary
    claim is the entropy-flow signature (a)–(c). Three load-bearing
    score outcomes:
    - **Outcome A** (score within ±20 of 194): EFLO drives a non-
      degenerate exploration policy that is not strictly worse than a
      random walk on this env. Combined with (a)–(c), this would
      confirm the principle "forward-entropy-flow ascent yields
      non-degenerate exploration on DST-concave."
    - **Outcome B** (score notably below 194 by >30): the entropy-flow
      ascent actively harms hypervolume on this env (e.g., by
      preferring ineffective broad exploration over the actual reward
      structure). The primitive fires (a)–(c) but is misaligned with
      the panel's reward-based scoring on this env.
    - **Outcome C** (score notably above 194, > 220): the entropy-flow
      ascent **incidentally discovers reward** because broad
      exploration concurs with treasure on DST-concave. This would be
      a fresh empirical-signal result: a no-reward update beats random
      on a reward env.

The quick stage is appropriate because (i) it gives the smallest
within-budget environment to measure the training-dynamics
discriminators (a)–(c), which fire from rollout 1 regardless of reward
discovery; (ii) DST-concave's 4-action, ~120-state grid is small enough
that policy-entropy `H_t` and its forward-flow `Â_t^H(λ)` are well-
defined and stable; (iii) the mechanism's claim of reward-independence
is best demonstrated on a reward-bearing env where reward is *available
but not used*, so that the score axis still partitions the outcomes
into A/B/C.

falsifier:

**Primary** (mechanism presence): if `std(c_t) ≤ 0.05 · |mean(c_t)|`
across training (i.e., the entropy-flow weight is essentially constant
across steps), then the GAE-of-entropy structure is empirically flat —
λ-interpolation produces no per-step variation — and the primitive is
decorative. Detectable in the first ~30 episodes.

**Secondary** (mechanism direction): if the cosine alignment between
EFLO and the c_t≡1 ablation's gradient direction stays ≥ 0.95
throughout training, the entropy-flow weight does not actually reshape
the gradient direction relative to the (degenerate) log-likelihood
ascent. Detectable per-rollout from rollout 1.

**Tertiary** (substrate signal): if EFLO's mean per-state entropy `H_t`
collapses toward 0 within 30 episodes (i.e., the policy becomes
deterministic-on-rollouts), the forward-entropy-flow ascent failed to
preserve the very entropy it claims to ascend — the principle is not
self-consistent on this substrate.

**Quaternary** (score-axis null): if EFLO scores strictly below 194 by
>30 *and* (a)–(c) all fire, the primitive is mechanistically present
but score-axis-misaligned (Outcome B); not a structural falsification
but useful corpus signal.

## Ablation plan

### Primary ablation (uniform per-step weight)

Replace the **per-step entropy-flow weight `c_t = Â_t^H(λ).detach()`**
with the **constant 1**:

In `train_ablate.py`:
1. Steps 1–2 identical (rollout, per-state entropy computation — though
   the entropy values are not used in the update).
2. Skip step 3 (no entropy residuals).
3. Skip step 4 (no GAE recursion).
4. In step 5, set `c_t ≡ 1` for all `t`.
5. Step 6: `g_θ = Σ_t 1 · ∇_θ log π_θ(a_t | s_t)`, identical otherwise.

Predicted ablation behavior: `Σ_t ∇log π_θ(a_t|s_t)` is the gradient of
the **rollout's trajectory log-likelihood** `log P_θ(τ) = Σ_t log
π_θ(a_t|s_t)`. Ascending this gradient pushes the policy to put all
mass on the realized actions at each visited state — i.e., **policy
entropy collapse**. Within ~10–30 episodes the ablation policy should
become near-deterministic, with mean H_t → 0 and severely reduced
exploration. This is the expected pathological signature of removing
the entropy-flow weight: the score-function form alone (without a
meaningful credit weight) degenerates to log-likelihood maximization.

### Discriminator predictions on DST-concave (within 120s budget)

- **Discriminator (i) — std(c_t)**: candidate `std(c_t) > 0`
  consistently; ablation `std(c_t) ≡ 0` by construction. This is the
  *trivial* discriminator (definitional).

- **Discriminator (ii) — cosine alignment**: candidate gradient
  direction differs from ablation's whenever c_t is non-uniform, with
  `cos < 0.95` typical at random init. Stable over training only if
  the policy retains entropy non-uniformity across states.

- **Discriminator (iii) — mean H_t over training (PRIMARY for
  separation)**: candidate's mean H_t should remain in `[0.5, log(4)]`
  (4 actions; max entropy ≈ 1.39); ablation's mean H_t should drop
  toward 0 within 30 episodes. **This is the load-bearing structural
  contrast**: EFLO preserves entropy because its update direction
  ascends the entropy functional; the c_t≡1 ablation collapses entropy
  because its update direction is log-likelihood maximization.

- **Discriminator (iv) — final hypervolume**: open. EFLO at or above
  194 (random) with H_t preserved is positive evidence; ablation at or
  near 0 (collapsed-deterministic) is the predicted pathology.

### Sanity ablation (random per-step weight) — diagnostic only

Optional second arm: replace `c_t` with `c_t ~ Uniform(-1, 1)` sampled
iid per step. This preserves per-step weight non-uniformity (matches
discriminator i) but removes the entropy-flow correlation. Predicted:
random-weight ablation's mean H_t stays near random init (no
systematic pull) and cosine alignment with c_t≡1 has mean ≈ 0
(uncorrelated). EFLO's cos-alignment should be a *systematic* deviation
correlated with entropy-flow direction — not a random deviation.

If the candidate's `std(c_t)`, `cos < 0.95`, AND `mean H_t > 0.5`
discriminators ALL match ablation's signature within seed variance,
the entropy-flow weight is empirically inert. If at least
discriminator (iii) — entropy preservation vs. collapse — separates
candidate from ablation, the primitive is causally responsible for
preventing entropy collapse, and the principle is empirically
operative.

## Novelty boundary

Closest known methods:

(a) **GAE / Generalized Advantage Estimation** (Schulman 2016). GAE
    is the exemplar from `worklogs/exemplars.md`. GAE's residual is
    `δ_t = r_t + γV(s_{t+1}) - V(s_t)`, a **value-function TD residual**
    requiring a learned `V`. EFLO's residual is `δ^H_t = H(π_θ(·|s_{t+1}))
    - H(π_θ(·|s_t))`, a **policy-entropy difference** requiring no
    learned predictor. Same exponential-averaging machinery
    (`Σ_l (γλ)^l δ_{t+l}`); structurally distinct residual, different
    optimized functional (forward-state-conditional entropy vs.
    expected discounted return). The principle "bias-variance
    interpolation of a temporal residual" is GAE's; the residual itself
    is the genuinely new primitive.

(b) **SAC / Soft-Actor-Critic / Max-Entropy RL** (Haarnoja 2018;
    Ziebart 2010). SAC's principle is `J(π) = E[Σ_t r_t + α H(π(·|s_t))]`,
    optimized via a soft-Bellman fixed-point on a learned **soft
    Q-function**. EFLO has no Q-function, no Bellman backup, no soft-
    Bellman fixed point. SAC's per-step entropy enters at the
    **value-bootstrap level** (in the soft Q-target). EFLO's per-step
    entropy enters at the **credit-assignment-weight level** (as a
    GAE-style temporal estimator multiplying the score-function
    gradient). SAC requires reward; EFLO does not.

(c) **Entropy-regularized PG / A2C/PPO entropy bonus** (Mnih 2016,
    Schulman 2017). These add `α · ∇_θ H(π(·|s_t))` per visited state
    to the policy gradient, an **additive single-step** entropy-
    derivative term. EFLO uses `Â_t^H(λ) · ∇log π_θ(a_t|s_t)` — a
    **multiplicative temporal-flow-weighted** score-function term. The
    structural difference: the entropy enters the gradient as a *per-
    step credit weight derived from forward entropy-flow*, not as an
    additive single-step regularizer. Reduction check: at λ → 0,
    `Â_t^H ≈ δ^H_t = H_{t+1} - H_t` is a *forward* one-step entropy
    difference, still distinct from the local `∇H(π(·|s_t))` of A2C/PPO.

(d) **State-visitation entropy maximization** (Hazan 2019; Liu-Abbeel
    2021 / APT; Seo 2021 / RE3). These maximize `H(d_π)` where `d_π`
    is the state visitation distribution, requiring density estimation
    (kNN, learned density model, neural-net prediction). EFLO maximizes
    `E_τ[Σ_t γ^t H(π_θ(·|s_t))]` — a discounted state-conditional
    *action-entropy* functional, computed in closed form from the
    policy network's output. No density estimator, no kNN, no
    auxiliary network.

(e) **Curiosity / RND / count-based novelty** (Pathak 2017, Burda 2018,
    Bellemare 2016). These add an intrinsic *reward* `r^int(s)` to the
    environment return. EFLO does not modify reward (the update reads
    no reward at all). The entropy-flow enters as a *gradient weight*,
    not as a reward bonus. No predictor network, no visitation counts,
    no novelty score.

(f) **TRACE (run 23, this loop, rejected)**. TRACE used the squared
    consecutive-state log-likelihood gap `(log π(a_t|s_t) -
    log π(a_t|s_{t+1}))²` as a per-step weight. EFLO uses the GAE-style
    exponentially-averaged forward-entropy-RESIDUAL of the policy
    `Σ_l (γλ)^l (H(π(·|s_{t+l+1})) - H(π(·|s_{t+l})))`. Distinct
    primitives: TRACE is a per-action squared log-prob gap (no
    temporal averaging, no entropy); EFLO is a temporal sum of state-
    conditional entropy differences. Different domains
    (action-specific log-likelihoods vs. state-only entropies) and
    different temporal structure (single-step squared vs. exponential
    forward sum).

(g) **COPDEV / PARGRAD / UNIRANK (runs 21, 22, 26)**. These use
    rolling-buffer cumulative-return rank/CDF weights from past
    trajectories on bivariate vector channels. EFLO uses no buffer,
    no cumulative-return statistics, no channel structure, and no
    rank/CDF. Its weight is computed entirely within the current
    rollout from the policy's own entropy at visited states.

(h) **CHANBI (run 24, rejected)**. CHANBI normalized per-channel
    score-function gradients and aggregated via spherical sum (Family
    I). EFLO has no per-channel decomposition, no parameter-space
    aggregation, no normalization-then-sum. Its primitive is one
    real-valued scalar per step, not a parameter-shaped vector pair.

(i) **GRADCOMP (run 20)**. GRADCOMP rotated the parameter update
    toward the empirical Fisher's principal eigenvector (a second-
    moment direction modification). EFLO preserves the score-function
    direction at each step and modifies the *per-step magnitude* via a
    forward-entropy-flow weight. Different mechanism slot.

(j) **REINFORCE / vanilla policy gradient** (Williams 1992). REINFORCE
    weights every step by `G_t = Σ_{k≥t} γ^{k-t} r_k`. EFLO weights
    every step by `Â_t^H(λ)`, computed entirely from the policy's own
    entropy at visited states with NO reward. The c_t≡1 ablation is
    NOT REINFORCE — it is **trajectory-log-likelihood ascent**, a
    pathological direction. To recover REINFORCE one would set
    `c_t = G_t`; we do not perform this comparison because it is the
    standard well-known PG, available as the panel's strong baseline.

(k) **Maximum-entropy IRL** (Ziebart 2008). MaxEnt IRL infers a reward
    function from expert demonstrations under a maximum-entropy
    trajectory model. EFLO has no expert, no IRL, and no inference
    over reward; it ascends a closed-form policy-entropy functional.

(l) **Diversity-is-all-you-need / DIAYN** (Eysenbach 2018). DIAYN
    learns skill-conditioned policies by maximizing mutual information
    between skills and trajectories via a discriminator. EFLO has no
    skills, no discriminator, no MI estimator.

(m) **Mirror descent / Natural Policy Gradient** (Kakade 2002).
    NPG/PMD precondition the policy gradient by the inverse Fisher.
    EFLO does not precondition — it modifies the per-step weight via
    a temporal entropy-flow estimator. No Fisher computation, no
    inversion.

(n) **Trust-region methods (TRPO/PPO)**. TRPO/PPO clip the policy
    likelihood ratio. EFLO has no clipping, no trust region, no ratio.

(o) **MaxEnt option discovery / hierarchical RL**. EFLO has no
    option/skill structure; one flat policy, one update direction.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: EFLO has no
  bucketing of any kind. The data structure is the policy network and a
  per-step real-valued weight `c_t` derived from policy entropies.
- **Family B (pairwise trajectory comparison)**: EFLO works on a single
  rollout; no pairing across trajectories.
- **Family C (within-trajectory geometric statistic)**: `c_t` is a
  function of the *policy's action-distribution entropy* on consecutive
  visited states — a quantity on the policy manifold. Path geometry of
  the state/cumulant trace plays no role.
- **Family D (reward-independent + reward-gated)**: EFLO is reward-
  independent (yes), but it is **not reward-gated**. The update
  `g_θ = Σ_t c_t · ∇log π_t` is non-zero whenever the rollout is non-
  trivial; there is no gate that "fires when reward appears."
- **Family E (avoid value vocabulary)**: EFLO has no learned future-
  compression of return. The entropy `H(π_θ(·|s))` is not a value
  function in disguise — it does not predict future return, does not
  satisfy a Bellman equation, and depends only on the current policy
  output at `s`. It is not a renamed V, Q, or advantage.
- **Family F (hand-engineered structural priors)**: No vocabulary, no
  symbol grammar, no event types — entropy is canonical and parameter-
  free.
- **Family G (mechanism stack)**: One primitive — the GAE-style
  forward-entropy-flow estimator. The policy update follows
  deterministically from it.
- **Family H (cochain complex)**: Not applicable.
- **Family I (per-channel parameter-space gradient aggregation)**:
  EFLO has no per-channel decomposition; one scalar weight per step,
  one summed gradient.

The structural difference from all named methods and dead families is
the **GAE-style exponentially-averaged forward-entropy-residual of the
policy itself, used as the sole credit-assignment weight on the score-
function update**. This is operationally distinct from GAE (uses
*entropy* residuals, not value residuals; no learned `V`), SAC (no
Q-function, no Bellman target, entropy enters as gradient weight not as
reward augmentation), entropy-regularized PG (multiplicative temporal
weight, not additive single-step regularizer), and state-entropy
maximization (uses state-conditional *action* entropy in closed form,
not visitation-density entropy).

## Proof debt

1. **Convergence to stationary points of `J^H`.** Conjecture: under
   Robbins-Monro step sizes and bounded `Â_t^H(λ)`, the EFLO iterate
   converges to a stationary point of the discounted forward-state-
   conditional policy-entropy functional `J^H(θ) := E_τ[Σ_t γ^t
   H(π_θ(·|s_t))]`. Strategy: standard stochastic-approximation analysis
   for stop-gradient pseudo-gradients with bounded bias.

2. **Stop-gradient bias bound.** The "true" gradient of `J^H` differs
   from `g_θ^EFLO` by a term involving `∇_θ H(π_θ(·|s_t))` propagated
   through the entropy-residual chain. Open: bound this bias as a
   function of `λ`, the rollout horizon, and the policy-entropy
   curvature; characterize when the stop-gradient form is asymptotically
   unbiased.

3. **Stationary-point structure on a tabular MDP.** Conjecture: on a
   finite tabular MDP, the stationary policies of `J^H` are exactly the
   **state-uniform-action policies** weighted by reachability — i.e.,
   policies that maximize forward state-conditional action entropy
   subject to the state-occupancy distribution induced by the policy.
   This is the entropy-only analogue of the optimal value function's
   characterization. Open: prove uniqueness or characterize the
   stationary set's geometry.

4. **Bias-variance trade-off for Â_t^H(λ).** GAE's bias-variance
   theorem (Schulman 2016, §3) holds for value residuals under a
   stochastic-process assumption on the value function. Open: prove the
   analogous bound for entropy residuals, where the "value" is replaced
   by the entropy functional.

5. **Connection to maximum-entropy occupancy.** Conjecture: EFLO's
   stationary policy is related but not equal to the max-entropy state-
   visitation policy of Hazan 2019. The connection is via the chain
   `H(d_π) = H(s) + H(a|s) + H(s|s,a) - H(s,a,s)` (decomposition of
   trajectory entropy). EFLO maximizes one component (`H(a|s)`
   discounted-summed); Hazan maximizes another (`H(s)`). Open: derive
   the precise relationship and characterize when they agree.

The empirical probe will reveal whether the entropy-flow signature
discriminators (a)–(c) fire on DST-concave within 120s; positive
firing on (a)–(c) AND a measurably non-collapsed mean entropy `H_t`
relative to the c_t≡1 ablation would justify investing in proof items
(1), (2), and (4).
