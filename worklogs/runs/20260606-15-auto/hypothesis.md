# 20260606-15-auto -- DUAL-IR (Dual Information-Relaxation Learning) [probe]

## Principle

The optimal value `V*(s_0)` is the saddle point of an **information-relaxation
dual** `min_m max_π E_τ[max_t (R_t(τ) − M_t(τ))]` where `M` is the cumulative
sum of a **martingale-difference penalty** `m` on transitions, and learning
proceeds by alternating descent on `m` and ascent on `π` using **realized
arg-supremum-of-time** credit assignment, which assigns positive policy
gradient signal only to actions taken **on or before** the realized
trajectory's penalized-return-maximizing time-step.

## Primitive

The **martingale-difference penalty function**

  `m_θ : S × A × S → ℝ`

with the conditional-zero-mean constraint

  `E_{s' ~ p(·|s,a)}[m_θ(s, a, s')] = 0`  for all `(s, a)`.

This is **one typed object**: a real-valued function on transitions whose
conditional mean given the pre-action state-action pair is zero. The
constraint is structural — `m_θ(s, a, s')` is implemented as
`f_θ(s, a, s') − b_θ(s, a)` where `b_θ(s, a)` is a small running-average
network tracking `E[f_θ(s, a, s'')]` over sampled `s''`. Domain: state
× action × next-state. Codomain: ℝ. The primitive is **not** a value, not
an advantage, not a Q, not a policy. It is a **transition-conditional
martingale increment** — a control variate that satisfies a measurability
constraint by construction.

## Derivation sketch

1. **Information-relaxation duality (Brown-Smith-Sun 2010 §3, Operations
   Research 58, Thm 2.1).** For any finite-horizon MDP, any martingale-
   difference penalty `m` (i.e., `E[m_t | F_{t-1}] = 0`) gives the
   inequality
   `V*(s_0) ≤ E_{τ ~ perfect-info-relaxation}[max_{a_0,...,a_T} Σ_t γ^t (r_t − m_t)]`,
   where the inner max is over **deterministic action sequences** chosen
   with full advance knowledge of all randomness (the "information
   relaxation"). Equality holds when `m*` equals the martingale increments
   of `V*` along trajectories: `m*_t = γ V*(s_{t+1}) − E[γ V*(s_{t+1}) | s_t, a_t]`.

2. **Online realization.** In the online (no perfect-info) setting we
   replace the inner perfect-info max over action sequences by the
   **realized cumulative-penalized-reward supremum along an actually
   sampled trajectory**:
   `D(τ) := max_{0 ≤ t ≤ T} (R_t(τ) − M_t(τ))`
   where `R_t = Σ_{k≤t} γ^k r_k` and `M_t = Σ_{k≤t} γ^k m_θ(s_k, a_k, s_{k+1})`.
   `D(τ)` is the **dual envelope** for trajectory `τ`. By optional sampling
   for martingales, `E_τ[D(τ)] ≥ V*(s_0) − V_π(s_0) + V_π(s_0) = V*(s_0)`
   when π is suboptimal and the inequality is exact at the saddle.

3. **Saddle structure.** The dual functional `L(π, m) := E_τ[D(τ)]` is
   convex in m (by linearity of M_t in m and convexity of the path
   supremum) and the supremum-then-expectation in π is **not** linear in
   π (it is the path-sup of a return process, which is concave in π for
   stochastic policies on finite-horizon MDPs by Cherikh-Sigaud 2003-style
   argument over occupancy measures, marked as proof debt). The saddle
   point `(π*, m*)` exists when V* is finite.

4. **Action-credit signal.** Define the **realized arg-supremum-time**
   `t*(τ) := argmax_{0 ≤ t ≤ T} (R_t(τ) − M_t(τ))`. This is the time at
   which the penalized cumulative reward peaks along the realized
   trajectory. Define the per-step **credit indicator**
   `c_k(τ) := 1[k ≤ t*(τ)]`. Actions taken **after** t* contributed only
   to a *decline* in the dual envelope and so receive no policy gradient
   credit; actions taken on or before t* are the actions that **drove**
   the dual envelope up to its trajectory-realized maximum.

5. **Policy gradient.** The dual objective `L(π, m)` is differentiable in
   π via the score-function trick on the policy distribution, with
   gradient
   `∇_θ L(π_θ, m) = E_τ[ Σ_k c_k(τ) · D(τ) · ∇_θ log π_θ(a_k | s_k) ]`
   (proof debt: standard score-function differentiability does not
   immediately apply through a path-supremum; the rigorous form requires
   a Danskin-style argument for path-sup of cumulative processes, marked
   below). The structural property: the **credit weight** is the
   *realized supremum value* `D(τ)` truncated by the supremum-time
   indicator `c_k` — neither the cumulative return nor an expected
   advantage.

6. **Penalty gradient.** With π fixed, `L(π, m_θ)` is minimized in m_θ
   by reducing the dual envelope: `∂L/∂θ_m = E_τ[ ∂D(τ)/∂M(τ) · ∂M(τ)/∂θ_m ]`,
   and `∂D/∂M_t = −1[t = t*(τ)]` by the envelope theorem. So the m-update
   is: at each rollout, **subtract** from `m_θ(s_{t*}, a_{t*}, s_{t*+1})`
   a small step proportional to `D(τ)`. This is the dual descent step.

7. **Why not Q-learning.** The action ordering at a state `s` under
   DUAL-IR is determined by *which actions appear in trajectories with
   the largest `D(τ)` and at indices `k ≤ t*(τ)`*. Two actions can have
   the same expected return (same Q) but different `D(τ)` distributions:
   one whose trajectories realize the path-sup early (high credit) vs.
   one whose trajectories realize the sup at terminal (low or zero credit
   beyond t*). DUAL-IR strictly prefers the first; Q-learning is
   indifferent. **The relative ordering of actions changes.**

8. **Why not policy-gradient.** Vanilla REINFORCE/PPO/A2C credit signal
   is the cumulative return (or advantage) from time k onward,
   `G_k(τ) = Σ_{j ≥ k} γ^{j-k} r_j`. DUAL-IR credit signal is
   `c_k(τ) · D(τ)`, the **realized path-supremum value** truncated by
   the supremum-time indicator — a **non-monotone**, **path-history**-
   dependent, and **trajectory-global** quantity. When `m_θ ≡ 0`, DUAL-IR
   reduces to `c_k(τ) · max_t R_t(τ)`, which on any monotone-non-decreasing
   reward trajectory (e.g., undiscounted CartPole) collapses to terminal
   return weighting (REINFORCE without baseline) — but the **martingale
   penalty `m_θ`** is what generically introduces a non-trivial t* < T
   and hence a non-trivial credit truncation.

## Update rule

```
Inputs: env, discount γ, policy π_θ, penalty net m_φ (with baseline b_φ
        absorbing the conditional mean), learning rates α_π, α_m
Init:   θ, φ random; replay-free; one rollout per update

For each episode:
    1. Roll out τ = (s_0, a_0, r_0, s_1, ..., s_T) under π_θ.
    2. Compute per-step penalty: μ_k = m_φ(s_k, a_k, s_{k+1})
       where m_φ(s,a,s') = f_φ(s,a,s') - b_φ(s,a)
       and b_φ is a small head trained on-policy to predict
       E_{s' ~ p(·|s,a)}[f_φ(s,a,s')] via stop-grad regression
       on f_φ targets (1-step bootstrap, decoupled from the main update).
    3. Compute discounted partial sums:
         R_k = Σ_{j ≤ k} γ^j r_j         (cumulative reward)
         M_k = Σ_{j ≤ k} γ^j μ_j         (cumulative penalty)
    4. Compute dual envelope and arg-supremum-time:
         D = max_k (R_k - M_k)
         t* = argmax_k (R_k - M_k)
    5. Policy gradient (ascent on D):
         For each k ≤ t*:
             g_θ += D · ∇_θ log π_θ(a_k | s_k)
         θ ← θ + α_π · g_θ
    6. Penalty gradient (descent on D, envelope theorem):
         g_φ = D · ∇_φ μ_{t*}     # only the t* term matters
         φ ← φ - α_m · g_φ
       (also update the baseline b_φ via 1-step regression to maintain
        the conditional-zero-mean constraint approximately)
```

The credit-truncation `k ≤ t*` is the load-bearing mechanism: it makes
the policy gradient depend on the **arg-sup-time** of the path-supremum,
which is invariant under shifts of `(R_k − M_k)` but sensitive to the
*shape* of the trajectory-cumulative process, not its terminal value.

## Empirical claim

stage: quick

claim: On the **quick stage** (CartPole-v1, MountainCar-v0 with dense
reward shaping or a short cap), DUAL-IR should learn at a rate
**comparable to or faster than** matched-architecture REINFORCE on the
panel's quick budget (120s), with the **arg-supremum-time `t*`** being
strictly less than the terminal time `T` for at least 30% of episodes
during training (a measurable signal of non-trivial dual envelope
shape). The quick stage is appropriate because (i) reward is dense, so
the cumulative sum `R_k` varies meaningfully along trajectories and the
supremum over time is non-degenerate; (ii) episode lengths are variable
(CartPole: 1-500 steps), so `t*` discriminates short-failure vs. long-
success trajectories; (iii) the cold-start failure of LYRA (run 13) is
avoided because dense reward generates a non-trivial dual envelope from
step 1.

falsifier: If DUAL-IR's learning curve on CartPole is **identical** to
the REINFORCE ablation (i.e., the m≡0 ablation matches DUAL-IR on
mean-episodic-return curves within shared seeds), the martingale-
difference primitive is decorative and the algorithm collapses to
REINFORCE. If the empirical fraction of episodes with `t* < T` stays
near zero throughout training, the dual envelope is degenerate and the
credit-truncation mechanism never fires. Either of these outcomes
falsifies the probe.

## Ablation plan

Replace the **martingale-difference penalty `m_θ`** with `m ≡ 0`
identically (no penalty network, no baseline, freeze m_θ at zero). The
rest of the algorithm runs unchanged: still compute `D(τ) = max_k R_k`
and `t* = argmax_k R_k`, still apply credit-truncated policy gradient
weighted by `D · 1[k ≤ t*]`.

On undiscounted CartPole with strictly-positive per-step reward, `R_k`
is monotone non-decreasing, so `t* = T` and `c_k = 1` everywhere. In
this degenerate ablation, DUAL-IR reduces to **REINFORCE with the
terminal undiscounted return as the trajectory weight** (no baseline,
no advantage). If this ablation matches or beats DUAL-IR, the
martingale-penalty primitive is decorative and the algorithm provides
no benefit beyond plain REINFORCE.

The discriminating empirical observable is the trajectory fraction
`P(t* < T)` during training: this is identically zero for the m≡0
ablation on monotone-reward CartPole, and should be substantially
positive for DUAL-IR if the penalty primitive is load-bearing. If
DUAL-IR shows `P(t* < T) > 0` *and* a learning advantage over the
ablation, the primitive is causally responsible for the advantage; if
`P(t* < T) > 0` but no learning advantage, the credit-truncation is
real but not useful (a different kind of falsification, also marked).

Implementable in `train_ablate.py` by replacing the `m_φ` network and
its optimizer with a constant-zero callable and removing step 6 of the
update rule.

## Novelty boundary

Closest known methods:

(a) **Brown-Smith-Sun 2010 information relaxation duality** (Operations
    Research). BSS use the dual representation `V* = inf_m E[max_τ
    perfect-info return − M_τ]` to compute *upper bounds* on V* for
    *evaluating* given policies, with m hand-crafted from approximate
    value functions. **DUAL-IR uses the dual as the optimization
    principle for online policy learning**: m is a learned neural
    network updated jointly with the policy via stochastic saddle-point
    descent on realized rollouts, and the action-credit signal is the
    realized arg-supremum-time of the dual envelope. BSS is an
    evaluation/bound technique; DUAL-IR is a learning algorithm.

(b) **REINFORCE / policy gradient with baseline** (Williams 1992;
    Sutton et al. 1999). PG credit weight is the cumulative return-to-
    go `G_k`. DUAL-IR credit weight is the **arg-sup-time-truncated
    realized supremum value** `D(τ) · 1[k ≤ t*(τ)]`. These two weights
    differ on any trajectory whose `R_k − M_k` is non-monotone in k,
    which is generic when m_φ is a non-trivial learned penalty. The
    ablation `m ≡ 0` collapses DUAL-IR to a degenerate REINFORCE
    variant on monotone-reward envs but not on general envs.

(c) **Dual / convex-conjugate policy iteration** (Mehta-Meyn 2009;
    Wang 2017 occupancy-LP duality). These use the *Lagrangian* of the
    LP-MDP, where the dual variables are *advantage functions* (Wang)
    or *occupancies*. DUAL-IR's dual variable is a **martingale-
    difference function on transitions**, with a measurability
    constraint, not an advantage on state-action pairs. The
    measurability/conditional-zero-mean constraint is structural to the
    information-relaxation interpretation and absent in LP-MDP duality.

(d) **Generalized Advantage Estimation (GAE; Schulman 2016)**. GAE is
    a **bias-variance interpolant** between TD residuals at different
    horizons, with credit weights `(γλ)^l`. DUAL-IR's credit weight is
    a **0/1 indicator** based on the realized arg-sup-time, not an
    exponential interpolation. Even at λ=1, GAE credits the full
    suffix; DUAL-IR credits only the prefix up to t*.

(e) **Risk-sensitive martingale RL** (Vukobratović et al. 2020 ACM
    AI in Finance; Doob-decomposition variance penalty). They use the
    Doob decomposition of *cumulative reward* for variance penalization
    in the *objective*. DUAL-IR uses the martingale-difference function
    on *transitions* as the **dual variable** in a saddle-point
    optimization for the **mean** value, not for risk.

(f) **Distributional RL** (Bellemare et al. 2017). Distributional RL
    learns the return distribution; DUAL-IR has no distributional
    primitive — only a scalar realized supremum.

(g) **Decision Transformer / return-conditioned policies**. DUAL-IR
    has no return conditioning and no transformer. The credit-
    truncation `k ≤ t*` is computed online from the realized rollout,
    not as an input conditioning.

Nearest dead family from `prior_attempts.md`:
- **Family A (bucketed-tensor + partial-order vote)**: DUAL-IR has no
  bucketing and no partial-order vote. The primitive is a continuous-
  valued function on transitions.
- **Family E (avoid value vocabulary)**: DUAL-IR does not learn a value
  function; m is not a relabeled V or Q. m has a structural
  measurability constraint (conditional zero mean) that V/Q do not have,
  and the action-credit signal is path-supremum-based, not expected-
  return-based.

The structural difference from all of the above is the **credit-
truncation by realized arg-supremum-time `t*`** combined with the
**martingale-difference measurability constraint** on the penalty.
Either alone would collapse: m without t* gives a control-variate
REINFORCE; t* without m gives a degenerate envelope on monotone-reward
trajectories. The combination is the load-bearing mechanism.

## Proof debt

1. **Existence of saddle point.** Show that `L(π, m) = E_{τ ~ π}[D(τ; m)]`
   admits a saddle point `(π*, m*)` on the joint product space of policies
   × bounded martingale-difference functions, with `L(π*, m*) = V*(s_0)`.
   The Brown-Smith-Sun 2010 weak-duality direction gives `L(π, m) ≥ V*(s_0)`
   for all (π, m); equality at `m*` of value-function-martingale-increment
   form is BSS Thm 2.1. Strong duality of the *learning saddle* (m_θ in a
   neural-net class, π_θ in a neural-net class) is open.

2. **Score-function differentiability through a path supremum.** The
   policy gradient `∇_θ L(π_θ, m) = E_τ[Σ_k c_k(τ) · D(τ) · ∇_θ log π_θ(a_k|s_k)]`
   requires interchanging differentiation and the path-supremum. This is
   a Danskin-style argument for cumulative processes; the rigorous
   condition is that t*(τ) is almost-surely unique, which holds when the
   reward distribution has a continuous component but not in pure-
   deterministic settings (proof debt: a tie-breaking rule for t* and
   its effect on the gradient estimator's bias).

3. **Convergence of alternating saddle descent.** Standard saddle-point
   convergence requires monotonicity (`L` is convex in m, concave in π
   for some structure on Π). DUAL-IR has the convexity in m but the
   concavity in π is only at the level of occupancy measures, not
   policy parameters. The right convergence theorem is open; the most
   we can claim is local convergence around a (π*, m*) where the
   Hessian saddle structure is non-degenerate.

4. **Baseline-network bias.** The constraint `E_{s'}[m_θ(s,a,s')] = 0`
   is enforced softly via subtracting `b_θ(s,a)`. The bias of the
   resulting `μ_k` and its propagation to the dual envelope's gradient
   estimator is a quantitative open question; analogous to the bias
   analysis of Q-learning's bootstrap target but distinct because the
   constraint is on the conditional mean, not the fixed point of an
   operator.

5. **Improvement theorem.** Conjecture: if at iterate `(π_n, m_n)` the
   sample dual gap `L(π_n, m_n) − V_{π_n}(s_0)` is positive, the joint
   update `(π_{n+1}, m_{n+1})` strictly decreases this gap in
   expectation, converging to zero (hence to a saddle) under standard
   step-size schedules. This is the load-bearing improvement claim and
   is currently open.

The empirical probe will indicate whether the joint saddle iteration
is well-behaved on a substrate before any of these theorems are
attempted; a positive result on CartPole would justify investing in
proof item (5).
