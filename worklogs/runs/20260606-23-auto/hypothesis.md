# 20260606-23-auto -- TRACE (Transition-Reflective Asymmetry Coupled Estimator) [probe]

## Principle

Update the policy by score-function ascent where each step's per-step
weight is the **squared log-likelihood gap of the realized action between
the source state and the successor state under the current policy**, multiplied
by the unicriterial return-to-go — yielding a credit-assignment rule that
*amplifies updates on state-discriminating decisions* whose source-state
log-likelihood differs sharply from the same action's successor-state
log-likelihood, so the load-bearing primitive is reward-independent in form,
fires on every step at random init, does not require non-degenerate vector
channel structure, and reduces to REINFORCE-with-baseline only when the
policy is action-distribution-invariant across consecutive states.

## Primitive

The **policy-induced consecutive-state action-log-likelihood gap field**

  `δ : Π × S × A × S → ℝ`,
  `δ_θ(s, a, s') := log π_θ(a | s) − log π_θ(a | s')`

with squared evaluation `c_t(τ) := δ_θ(s_t, a_t, s_{t+1})²` along the
realized rollout. Domain: a triple `(s, a, s')` with `s, s' ∈ S` and
`a ∈ A`. Codomain: `ℝ`. The squared form is bounded below by 0, smooth
in θ where `π_θ` is, and equals zero exactly on
**state-action-invariant policies** (policies whose conditional `π(a|·)`
is identical across consecutive states actually visited).

This is **one typed mathematical object**: a real-valued function of a
state-action-state triple parameterized by the current policy. It is not
a value function (no learned future-compression), not a Q-function (no
state-action *value* — it is a *log-likelihood difference* of a single
fixed action `a`), not an advantage estimator (no baseline subtraction,
no return inside `δ`), not a critic (no separate parameters), not a
state-bucketed tensor (no buckets — values are on the continuous policy
manifold), and not a count-based novelty (no visitation counts at all).

The primitive lives in the **policy's own log-likelihood landscape**: for
fixed `a`, the function `s ↦ log π_θ(a|s)` is the standard categorical
log-density, and `δ_θ(s,a,s')` is the **directional difference of this
density** between two specific points `(s, s')`. There is no kernel, no
extra learned function, no auxiliary head — `δ` is fully determined by
the existing policy network.

## Derivation sketch

1. **Setup.** Let `π_θ : S → Δ(A)` be a stochastic policy. Along a
   rollout `τ = (s_0, a_0, r_0, s_1, …, s_T)`, define for each `t ∈
   {0, …, T-1}` the **policy commitment gap**
   `δ_t := log π_θ(a_t | s_t) - log π_θ(a_t | s_{t+1})`.
   Note `δ_t` is well-defined for any policy that assigns positive
   probability to `a_t` at *both* `s_t` and `s_{t+1}` (true for any
   softmax policy).

2. **Zero-locus characterization.** `δ_t = 0` iff
   `π_θ(a_t | s_t) = π_θ(a_t | s_{t+1})` — the policy assigns equal
   probability to the realized action at the two consecutive states. A
   policy is **commitment-invariant** if `δ_t ≡ 0` for every realized
   transition; this includes (a) any state-independent policy
   `π(a|s) = π(a)` and (b) any policy whose action-conditional is
   constant on the *closed walk* of the trajectory.

3. **Weighted score-function functional.** Define the
   `δ²`-coupled REINFORCE objective
   `J̃(θ) := E_τ[Σ_{t=0}^{T-1} G_t · δ_t² · log π_θ(a_t | s_t)]`
   with `G_t := Σ_{k≥t} γ^{k-t} r_k` the unicriterial scalar return-to-go.
   This is a well-defined functional on the parameter manifold for any
   smoothly parameterized softmax policy.

4. **Gradient identity (proof debt).** Treating `δ_t²` as a
   *stop-gradient* per-step weight evaluated at the current θ
   (frozen on the forward pass; not differentiated through), the
   gradient of `J̃` reduces by the score-function identity to
   `∇_θ J̃(θ) ≈ E_τ[Σ_t G_t · δ_t² · ∇_θ log π_θ(a_t | s_t)]`.
   This is the **TRACE update direction**. The "≈" marks the proof
   debt: stop-gradient on δ_t² is a pseudo-gradient, not the
   gradient of `J̃` viewed as a functional of θ. (Alternative: a
   true-gradient variant treats δ_t² as differentiable, adding a
   `2 δ_t · (∇_θ log π_θ(a_t|s_t) − ∇_θ log π_θ(a_t|s_{t+1}))` term;
   we choose the stop-gradient form for primitive cleanliness, leaving
   the comparison to proof debt.)

5. **Why δ_t² fires at random init on DST-concave.** For a small softmax
   policy with random weights over a 4-action MiniGrid-style env or
   DST-concave (4 actions on an 11-row depth grid with a few dozen
   reachable states), at random init `δ_t` is approximately a Gaussian
   random variable with variance `O(σ²)` (where σ is the logit scale),
   hence `δ_t²` is `O(σ²)`, *strictly positive in expectation, varying
   per step*. The mechanism is non-zero from step 1 of training without
   requiring any reward to be observed.

6. **Why δ_t² is not a function of reward and not a baseline.** `δ_t²`
   depends only on `(s_t, a_t, s_{t+1}, θ)` — it has no reward argument
   and no return-conditioning. It is therefore **not** a baseline (a
   baseline is *subtracted*; δ_t² is *multiplied*) and not an advantage
   (advantage = `Q − V`; δ_t² is a single positive quantity). Replacing
   reward `r_k` with any monotone-equivalent reward leaves `δ_t²`
   exactly unchanged at fixed θ.

7. **Why δ_t² is not curiosity / RND / count-based.** Curiosity adds an
   intrinsic *reward* that enters via `G_t`; δ_t² is a *gradient
   weight*, not a reward addition. RND uses a separate randomly-
   initialized predictor network; δ_t² uses **only the policy's own
   log-likelihoods**. Count-based bonuses depend on state visitation
   frequency; δ_t² is independent of visitation.

8. **Why δ_t² has structural credit-assignment direction.** A `δ_t²`-
   large step is one where the policy's conditional `π(a_t | ·)` is
   sharply different at `s_t` versus `s_{t+1}`, meaning `a_t` is a
   *state-discriminating* action — the policy "commits" to it at `s_t`
   in a way it wouldn't at `s_{t+1}`. On DST-concave's "row-where-down-
   becomes-treasure" decision, `a_t = down` from a high row has very
   different policy probability than `a_t = down` from a low row;
   trajectories that hit such commitment moments contribute the
   largest `δ_t² · G_t` weight to the gradient. The primitive
   *concentrates credit assignment on load-bearing decisions*
   identified by their state-discriminating character — without
   needing reward to identify them, only to *scale* the credit.

9. **Why this is not Family A (bucketed tensor + partial-order vote).**
   No state-bucketing, no action-bucketing, no channel index, no
   partial-order vote, no tensor. The data structure is the policy
   network itself; the gradient weight is a real number from policy
   log-likelihoods.

10. **Why this is not Family C (within-trajectory geometric statistic).**
    Family C uses path geometry (hull, Lévy area, signature) of the
    *observation/cumulant trace* per (s, a). TRACE uses a function of
    the *policy's own log-likelihood* on consecutive states — a
    quantity defined on the policy manifold, not on the
    state-cumulant trace. It is invariant under any reparametrization
    of states that preserves softmax-policy logits.

11. **Why this is not GRADCOMP/Fisher-rotation (run 20).** GRADCOMP
    rotates the *direction* of the parameter update toward the top
    Fisher eigenvector. TRACE preserves the standard score-function
    *direction* `∇_θ log π_θ(a_t|s_t)` and modifies the per-step *weight*.
    Different mechanism slot.

12. **Why this is not COPDEV/PARGRAD (runs 21, 22).** COPDEV/PARGRAD use
    per-step weights derived from cumulative-return *empirical CDF/measure
    rolling buffers* on the bivariate channel structure. TRACE has no
    rolling buffer, no CDF/measure, no channel structure (it operates on
    scalar return). Its per-step weight is purely a function of the
    policy's log-likelihoods at the realized consecutive states.

13. **Proof debt.** (i) Convergence of TRACE updates to a stationary
    point of `J̃`. (ii) Identification of fixed points: are stationary
    `J̃`-maximizers also stationary points of vanilla REINFORCE's `J`?
    Conjecture: yes when `δ_t² > 0` everywhere on-policy (because the
    gradient direction is a positive-weight reweighting), and the only
    spurious extra fixed points are the **commitment-invariant**
    policies `δ_t ≡ 0` (where TRACE's gradient vanishes regardless of
    return). (iii) Stop-gradient vs. full-gradient comparison: bound
    the bias of the stop-gradient form against the true `∇_θ J̃`.

## Update rule

```
Inputs: env (vector reward; we use SCALAR return — uses channel 1 if
        vector, else native scalar), policy π_θ, discount γ, lr α
Init:   θ random.

For each episode:
    1. Roll out τ = (s_0, a_0, r_0, ..., s_T) under π_θ.
       For DST-concave (vector env), reward is read from
       info["vector"][0] (treasure channel only). The mechanism does
       NOT scalarize multi-channel reward; it operates on whichever
       SINGLE channel the panel exposes (channel 1 = treasure on
       DST-concave). Channel 2 (step penalty) is unused.

    2. Compute returns-to-go:
         G_t = Σ_{k≥t} γ^{k-t} · r_k         for t = 0..T-1

    3. Compute per-step commitment gap and squared weight:
         For t = 0..T-1:
             logp_st_at  = log π_θ(a_t | s_t)        # standard
             logp_stp_at = log π_θ(a_t | s_{t+1})    # policy *at next state*
                                                    # for the SAME action a_t
             δ_t  = logp_st_at - logp_stp_at         # signed gap
             c_t  = δ_t.detach() ** 2                # stop-gradient weight

    4. Per-step score gradient:
         g_t = ∇_θ log π_θ(a_t | s_t)               # standard score

    5. TRACE policy update:
         g_θ = Σ_t G_t · c_t · g_t
         θ ← θ + α · g_θ

    6. Logging observables (load-bearing for ablation discrimination):
         - mean(c_t over t)               # rollout-averaged weight
         - std(c_t over t)                # weight non-uniformity
         - mean(|δ_t|) over t             # raw asymmetry magnitude
         - cosine alignment between TRACE gradient and the
           uniform-weight (REINFORCE) gradient g_R = Σ_t G_t · g_t,
           computed once per rollout — flat 1.0 for ablation,
           strictly < 1.0 for TRACE when c_t is non-uniform
         - mean episode length T
         - first-rewarded-episode-index (when G_0 first > 0)
```

The load-bearing primitive is the **policy commitment-gap field
δ_θ(s,a,s')**, evaluated as `c_t := δ_t²` per step. Storage:
`O(T)` per rollout. Per-step cost: one extra forward pass through the
policy network at `s_{t+1}` for the same action `a_t` — i.e., a single
extra log-softmax lookup per step (negligible). The only logged
training-dynamics scalar required for the primary discriminator is the
**rollout-mean cosine alignment** between the TRACE and uniform-weight
gradient vectors; this fires from rollout 1 at random init.

## Empirical claim

stage: quick

claim: On the **quick** stage (deep-sea-treasure-concave-v0 under the
panel's quick configuration; 4-action, ~11-row grid, vector reward
where channel 1 = terminal treasure), TRACE should produce:
(a) **`mean(c_t) > 0`** strictly bounded away from 0 within the first
    20 episodes, with `std(c_t)/mean(c_t) > 0.5` (per-step weight
    non-uniformity), confirming the primitive fires at random init;
(b) **rollout-mean cosine alignment between TRACE gradient and the
    REINFORCE gradient strictly less than 1.0** (≤ 0.9 typical, with
    seed variance), confirming the per-step weighting genuinely changes
    the gradient direction (not just magnitude);
(c) **a mean-episode-return trajectory measurably distinct from the
    uniform-weight ablation's** (REINFORCE-c_t≡1) within 120s budget,
    with a non-zero `ablation_delta` on the panel's hypervolume score
    on DST-concave.

The quick stage is the appropriate test because (i) DST-concave provides
4 actions × ~11 reachable depth-rows × scalar reward, exactly the
substrate where state-discriminating actions (down vs. left vs. right)
have policy-conditional differences; (ii) the load-bearing observable
**cosine alignment** is a logged training-dynamics scalar that fires at
random init regardless of reward discovery; (iii) the asymmetry weight
does **not** depend on channel 2 being non-degenerate (avoiding the DST-
concave reduction trap that killed PARGRAD on run 22).

falsifier:

**Primary** (mechanism presence): if `mean(c_t)` is essentially zero or
the cosine alignment between TRACE and REINFORCE gradients is ≥ 0.95
within seed variance throughout training, the per-step weight `c_t`
is empirically constant or trivial and the primitive is decorative.
This is detectable in the first ~30 episodes, well within budget.

**Secondary** (mechanism direction): if TRACE's `mean_episode_return`
trajectory matches the uniform-weight ablation's within seed variance
throughout training despite a nonzero cosine-alignment gap, the
gradient-weight primitive produces no measurable effect on policy
behavior on this substrate.

**Tertiary** (substrate signal): if TRACE's final hypervolume on DST-
concave is **strictly worse** than both the random baseline (194) and
the uniform-weight ablation, the credit-assignment direction is
actively harmful — the asymmetry-weighting drives the policy *away*
from useful regions.

## Ablation plan

### Primary ablation (uniform per-step weight)

Replace `c_t := δ_t²` with the **constant 1**:

In `train_ablate.py`:
1. In step 3, skip computing `logp_stp_at` and `δ_t`; set `c_t ≡ 1`.
2. All other steps identical: returns-to-go, score gradient, parameter
   update.

This preserves: per-step score-function gradient computation, the
unicriterial return-to-go, all hyperparameters, the policy
architecture, the rollout mechanism. It removes: the *per-step
asymmetry-weighting* — the ablation is exactly REINFORCE-without-
baseline.

### Secondary ablation (random per-step weight, retained for diagnostic)

Replace `c_t := δ_t²` with **`c̃_t ~ Exponential(λ=1)` sampled iid per
step**. This preserves the *non-uniformity* of per-step weights but
removes the *correlation with policy commitment*. The discriminator:
TRACE's cosine-alignment-vs-REINFORCE should differ from the
random-weight ablation's in a *systematic, policy-correlated* way (the
TRACE alignment should drift down toward 0 on rollouts where the
policy is genuinely commitment-asymmetric, while the random-weight
alignment is noise-only).

### Discriminator predictions on DST-concave (within 120s budget)

- **Discriminator (i) — mean(c_t) and std(c_t)**: TRACE has
  `mean(c_t) ≈ σ²` (random init; σ = logit scale), `std(c_t)/mean ≥ 0.5`.
  Uniform ablation has `mean(c_t) = 1` and `std(c_t) = 0` by
  construction.

- **Discriminator (ii) — cosine alignment between TRACE and REINFORCE
  gradients (PRIMARY)**: TRACE gradient direction differs from
  uniform-weight REINFORCE direction whenever `c_t` is non-uniform across
  steps. The cosine should be in `(0.5, 0.95)` typical at random init,
  drifting either toward 1 (if policy becomes commitment-invariant) or
  staying away (if policy remains commitment-asymmetric).

- **Discriminator (iii) — mean_episode_return on DST-concave**: open
  question. If TRACE concentrates credit on state-discriminating
  actions effectively, episodes that find treasure become amplified.
  Expected outcome: TRACE matches or beats ablation by 5–20 hypervolume
  points; matching random (194) would be a partial success, beating
  random would be the substrate signal.

If the uniform-weight ablation matches TRACE on all three discriminators
within seed variance, the asymmetry-weight primitive is not load-
bearing. If TRACE shows strictly non-trivial cosine alignment AND a
measurably different return trajectory, the primitive is causally
responsible.

## Novelty boundary

Closest known methods:

(a) **REINFORCE / vanilla policy gradient** (Williams 1992). REINFORCE
    weights every step's score by `G_t`. TRACE weights by `G_t · δ_t²`
    where `δ_t² = (log π(a_t|s_t) - log π(a_t|s_{t+1}))²` is a
    policy-state-conditional log-likelihood gap. Different per-step
    weight; reduces to REINFORCE only when the policy is state-
    invariant on consecutive states.

(b) **Policy gradient with baseline / Advantage** (Sutton 1999).
    Baselines *subtract* `b(s_t)`; advantage uses `Â_t = G_t - V(s_t)`.
    TRACE *multiplies* by `δ_t²` ≥ 0; the weight is purely additive in
    *credit weight*, not in *return level*. No value function is
    learned.

(c) **GAE / TD-residual exponential averaging** (Schulman 2016). GAE
    weights are `(γλ)`-weighted TD residuals over a learned `V`. TRACE
    has no V, no TD residual, no exponential interpolation.

(d) **Stein Variational Policy Gradient (SVPG)** (Liu-Wang 2017). SVPG
    maintains a *population* of policies updated by Stein variational
    gradient flow with a kernel. TRACE has a single policy, no
    population, no kernel, no Stein operator.

(e) **Curiosity / RND / count-based bonuses** (Pathak 2017; Burda 2018).
    These add an intrinsic *reward* `r_int(s)` to the environment
    return. TRACE does not modify reward. The asymmetry quantity
    `δ_t²` enters as a *gradient weight*, not as a reward bonus, and
    depends on the policy network's outputs (not on a separate
    randomly-initialized predictor).

(f) **Self-Imitation Learning (SIL)** (Oh et al. 2018). SIL replays
    past good trajectories and trains the policy to imitate them via
    a max-margin objective. TRACE does not replay, does not imitate,
    and uses no past-trajectory buffer.

(g) **Detailed-balance-violation / time-asymmetry RL**. Detailed
    balance involves the *transition kernel* `P(s'|s,a)` and its
    reverse `P(s|s',a)`, neither of which TRACE accesses. TRACE uses
    only the **policy log-likelihoods** at consecutive states for the
    *same action* — a *policy-state-conditional* log-likelihood gap,
    not a kernel time-reversal.

(h) **Natural Policy Gradient / Fisher-preconditioned PG** (Kakade
    2002; GRADCOMP run 20). NPG/GRADCOMP modify the *direction* of
    update via Fisher (inverse or top eigenvector). TRACE preserves
    the score-function direction `∇_θ log π_θ(a_t|s_t)` per step and
    modifies the *per-step magnitude* of the contribution.

(i) **COPDEV (run 21) / PARGRAD (run 22)**. COPDEV/PARGRAD use
    per-step weights derived from rolling buffers of cumulative
    bivariate channel returns. TRACE has no buffer, no CDF/measure,
    no channel structure (it uses scalar return-to-go only). The
    weight depends solely on the policy's own log-likelihoods at the
    realized consecutive states.

(j) **Orthogonal gradient descent for continual learning**
    (Farajtabar 2020). OGD projects new-task gradients orthogonal to
    past-task gradients to preserve old-task performance. TRACE does
    not project — it reweights — and operates within a single task,
    on a single rollout.

(k) **Reverse-KL / forward-KL policy distillation**. KL-divergence
    minimization between policies requires *two distributions*; TRACE
    uses one policy evaluated at two states, with a *fixed action* and
    a *log-likelihood gap*, not a divergence between distributions.

(l) **AWR / Advantage-Weighted Regression** (Peng 2019). AWR
    distillation weight is `exp(β · A(s,a))` where A is an advantage.
    TRACE uses no advantage and no exponential weighting; the weight
    is `δ_t²`, a quadratic function of policy log-likelihoods.

(m) **Successor features / GVFs** (Barreto 2017; Sutton 2011). These
    learn predictive cumulants of the trajectory. TRACE learns no
    predictive object; the only object is the policy itself.

(n) **Distributional RL** (Bellemare 2017). Distributional RL learns
    a return distribution per (s,a). TRACE has no return distribution.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: TRACE has no
  bucketing of any kind. The only object is the policy network and a
  per-step real-valued weight derived from its log-likelihoods.
  No partial-order, no vote, no tensor.
- **Family B (pairwise trajectory comparison)**: TRACE works on a
  single rollout at a time; no pairing across trajectories.
- **Family C (within-trajectory geometric statistic)**: `δ_t² ` is
  not a geometric statistic of the *state/cumulant* trace. It is a
  function of the *policy's log-likelihoods* on consecutive states,
  which is a quantity on the policy manifold, not on the
  observation/cumulant trace. The signature/hull/Lévy-area path
  geometry has no analogue here.
- **Family D (reward-independent + reward-gated)**: `δ_t²` is
  reward-independent in form (yes), but the application is **not
  reward-gated**: every step contributes `G_t · δ_t² · g_t`, and the
  update is non-zero whenever `G_t` and `δ_t²` are jointly non-zero
  along the trajectory. There is no gate that "fires when reward
  appears." The mechanism is best described as a *positive-weight
  reweighting of REINFORCE*, not a gated reward-free primitive.
  (This distinction matters: Family D's failure mode is the gate
  inheriting the bootstrap wall; TRACE's failure mode would be
  REINFORCE's failure mode — which on quick-stage DST-concave is
  *not* a bootstrap wall, since `G_0` is nonzero on the ~28% of
  random trajectories that find any treasure.)
- **Family E (avoid value vocabulary, keep value structure)**: TRACE
  has no learned future-compression. It uses raw `G_t` (Monte-Carlo
  return-to-go, not bootstrapped), and the asymmetry weight `δ_t²`
  is an *online policy quantity*, not a learned predictor.
- **Family F (hand-engineered structural priors)**: No vocabulary,
  no symbol grammar, no event types — `δ_t²` is determined by the
  policy network at runtime.
- **Family G (mechanism stack)**: One primitive — the
  consecutive-state action-log-likelihood gap. No three named
  components.
- **Family H (cochain complex)**: Not applicable.

The structural difference from all named methods and dead families is
the **squared policy-state-conditional log-likelihood gap of the
realized action between consecutive states, used as a non-negative
per-step gradient weight on the unicriterial return-to-go score-
function update**. This is operationally distinct from Bellman backups
(no operator), advantage-weighted PG (no advantage, multiplicative not
exponential weight), Stein/SVPG (no kernel/population), curiosity (no
reward modification), and Fisher-preconditioned PG (no direction
modification).

## Proof debt

1. **Convergence to stationary points of `J̃`.** Conjecture: under
   Robbins-Monro step sizes, the TRACE iterate converges to a
   stationary point of the modified objective
   `J̃(θ) := E_τ[Σ_t G_t · δ_t(θ)² · log π_θ(a_t|s_t)]` viewed with
   `δ_t²` as a stop-gradient quantity. Strategy: standard stochastic-
   gradient-with-fixed-bounded-bias analysis applies because `δ_t²`
   is bounded for any softmax policy with logits in a bounded set.
   Open: characterize when the stop-gradient bias is negligible
   relative to the score-function gradient noise.

2. **Stationary-point structure.** Conjecture: stationary points of
   `J̃` are either (i) stationary points of vanilla REINFORCE's `J`
   restricted to the support `{δ_t² > 0}`, or (ii) commitment-
   invariant policies where `δ_t ≡ 0` on-policy. The latter set is
   non-empty (state-independent policies are commitment-invariant
   under any environment), so TRACE has *spurious* fixed points
   beyond REINFORCE's. Open: bound the basin of attraction of
   spurious fixed points and characterize when they are avoided.

3. **Stop-gradient vs. full-gradient comparison.** The "true" gradient
   of `J̃` includes a `2 δ_t · (∇log π(a_t|s_t) - ∇log π(a_t|s_{t+1}))`
   term that the stop-gradient form omits. Open: bound the bias and
   determine whether the full-gradient form admits a cleaner fixed-
   point characterization.

4. **Variance reduction relative to REINFORCE.** Conjecture: weighting
   by `δ_t²` provides a variance-reduction-style benefit by *down-
   weighting* steps where the policy is commitment-flat (low δ_t²) —
   essentially, only the *informative* policy decisions contribute to
   the gradient. Open: bound the variance of the TRACE gradient
   estimator vs. REINFORCE and identify conditions under which it is
   strictly lower.

5. **Connection to Fisher information.** The empirical Fisher
   `F̂_t := Σ_t g_t g_t^T` of the rollout has eigenvalues that depend
   on the per-step score magnitudes `||g_t||`. Conjecture:
   `δ_t² · ||g_t||²` is correlated with the *Fisher contribution* of
   step `t` to the rollout's `F̂`. Open: formalize a Fisher-weighted-
   PG interpretation of TRACE.

The empirical probe will reveal whether the cosine-alignment
discriminator separates TRACE from its uniform-weight ablation on the
quick stage within 120s; positive separation on the cosine scalar AND
a measurably different return trajectory would justify investing in
proof items (1), (2), and (4).
