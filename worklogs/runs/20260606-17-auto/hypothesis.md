# 20260606-17-auto -- SNELL (Snell-Envelope Stopping for Predictable Credit Truncation) [probe]

## Principle

The optimal policy is improved by score-function ascent on a
**Snell-envelope-truncated** rollout: a learned **predictable**
(F_t-measurable) continuation-value `C(s)` regressed toward the
**suffix-maximum of cumulative reward** induces a stopping time
`τ* = min{t : R_t ≥ C(s_t)}`, and policy gradient is taken using
**only the prefix `t ≤ τ*` of each trajectory** with weight equal to
the locked-in cumulative reward `R_{τ*}` — never bootstrapping a
Bellman max.

## Primitive

The **Snell continuation value**

  `C : S -> R`,
  `C(s) := E_pi[ max_{u >= t+1} R_u  |  s_t = s ]`

where `R_u = sum_{k <= u} gamma^k r_k` is the cumulative discounted
reward up to time `u`. By the Snell envelope theorem (Snell 1952;
Peskir-Shiryaev 2006 *Optimal Stopping and Free-Boundary Problems* §1),
`C` is the **least supermartingale dominating the path-suffix maxima**
of the cumulative-reward process under pi. It is a single
state-indexed scalar function, but it is **not** a value function: its
fixed-point equation has the maximum **inside** the expectation
(`E[max(R_{t+1}, R_{t+2}, ...) | s_t]`), whereas the Bellman value
`V(s) = E[r + gamma max V(s') | s]` has the max **outside**. This
order-of-operators distinction is the structural separator from Q/V.

## Derivation sketch

1. **Snell envelope.** For an adapted, integrable reward-cumulant
   process `(R_t)_{t=0..T}`, the **Snell envelope** is the process
   `U_t = ess sup{ E[R_tau | F_t] : tau a stopping time, t <= tau <= T }`.
   Snell 1952 proved `U_t = max(R_t, E[U_{t+1} | F_t])`, i.e. `U` is
   the smallest supermartingale dominating `(R_t)`.

2. **Optimal stopping time.** `tau^* = min{ t : U_t = R_t }` =
   `min{ t : R_t >= E[U_{t+1} | F_t] }` is a **predictable** stopping
   time: the decision to stop at `t` uses only `F_t`. This is the
   crucial structural distinction from a *path-supremum-arg* (which
   needs the full future).

3. **Continuation value.** Define `C(s) := E[U_{t+1} | s_t = s]`. By
   stationarity of the Markov state and law of total expectation,
   `C(s)` is well-defined as a state-indexed scalar function and
   satisfies the **suffix-supermartingale fixed-point**
   `C(s) = E_pi[ max(R_{t+1}, R_{t+1} + gamma C(s_{t+1})) | s_t = s ]`,
   where the inner max takes "stop at t+1" vs. "continue beyond t+1".

4. **Why this is not a Bellman value.** The right-hand side has the
   form `E[max(A, B)]`, **not** `max(A, E[B])`. The two expressions
   differ by Jensen's inequality: `E[max(A,B)] >= max(E[A], E[B])`.
   The Snell envelope is a **strict** upper bound on the Bellman
   value of any single-step continuation policy. Operator: `(T_Snell f)(s) = E_pi[max(R, f(s'))]` is an averaging-over-max operator, whereas the Bellman optimality operator `(T* f)(s) = max_a E[r + gamma f(s')]` is a max-over-averaging operator. These are different fixed points.

5. **Stopping rule operationalization.** Online: after observing
   `(s_t, R_t, s_{t+1})`, accept stopping if `R_t >= C(s_t)`; otherwise
   continue. The stopping time `tau^*` of a realized rollout is the
   first such `t`. By the Snell theorem, `E[R_{tau^*}] = E[U_0]
   = sup_tau E[R_tau]`, the optimal-stopping value.

6. **Predictable credit truncation.** Apply REINFORCE-style score-function
   ascent **only on the prefix `t <= tau^*`** of the rollout, with
   the locked-in reward `R_{tau^*}` as a constant trajectory weight:
     `g_theta = sum_{t=0}^{tau^*} R_{tau^*} * grad_theta log pi_theta(a_t | s_t)`.
   Actions taken after `tau^*` are evidentially neutral to the
   stopped-cumulative reward and receive zero gradient.

7. **Why this changes optimization dynamics.** The realized stopping
   time `tau^*` is a deterministic function of `(s_0, R_0, ..., s_t,
   R_t)` through `C`, so its distribution **shifts** with `C` over
   training. Empirically: `tau^*` decreases on rollouts with early
   high reward (lock in early), increases on rollouts where `R_t` is
   slowly accumulating (continue). The expected truncation time
   `E[tau^*]` is a measurable observable that changes monotonically
   as `C` learns. Random truncation has fixed mean `T/2` and zero
   correlation with `R_t`. The discriminating signal is the empirical
   **correlation `corr(tau^*, R_{tau^*})` over rollouts**: under SNELL
   it should be strongly positive (Snell stops at high-R points);
   under random truncation it is near zero by construction.

8. **C-update rule.** `C(s_t)` is regressed (mean-squared-error,
   Robbins-Monro step) toward the realized **suffix-maximum**
     `target_t := max(R_{t+1}, R_{t+2}, ..., R_T)`
   of the rollout. This is **not** a Bellman target (no `max_a`,
   no bootstrap into another `C(s')`). It is a **direct supervised
   regression on a path statistic** that is exactly the optimal-stopping
   continuation value by definition.

9. **Convergence sketch (proof debt).** The fixed-point equation
   `C(s) = E_pi[max(R_{t+1}, R_{t+1} + gamma C(s_{t+1})) | s_t = s]`
   defines an averaging-of-max operator. Under bounded reward and
   stationary `pi`, this operator is a contraction in sup-norm with
   modulus `gamma` (proof debt: needs verification that the
   averaging-of-max preserves contraction under bounded `R`). The
   stochastic regression update converges by standard Robbins-Monro
   when `pi` is fixed; the joint policy-and-`C` iteration is the
   open question.

10. **Falsifier-by-construction.** If random-time-truncation matches
    SNELL's reward, the predictable-stopping primitive is decorative.
    But the *measurable observable* `corr(tau^*, R_{tau^*})` will be
    near zero for random truncation and (claim) positive for SNELL,
    so the ablation tie cannot be coincidental on this observable.

## Update rule

```
Inputs: env, policy pi_theta, continuation-value net C_phi, gamma,
        learning rates alpha_pi, alpha_C
Init:   theta, phi random; C_phi initialized to 0

For each episode:
    1. Roll out tau = (s_0, a_0, r_0, s_1, ..., s_T) under pi_theta.
       Compute R_t = sum_{k <= t} gamma^k r_k for all t.

    2. Compute predictable stopping time tau^*:
         tau^* = min { t : R_t >= C_phi(s_t) }   # default tau^* = T if no such t

    3. Locked-in reward:
         W = R_{tau^*}                          # scalar trajectory weight

    4. Truncated score-function policy gradient:
         g_theta = sum_{t=0..tau^*} W * grad_theta log pi_theta(a_t | s_t)
         theta <- theta + alpha_pi * g_theta

    5. Continuation-value regression target (suffix-maximum):
         For each t in {0, ..., T-1}:
             target_t = max(R_{t+1}, R_{t+2}, ..., R_T)
         L_C = sum_t (C_phi(s_t) - target_t)^2
         phi <- phi - alpha_C * grad_phi L_C

    6. Logging observables (load-bearing for ablation discrimination):
         - mean(tau^*) / T                  # fraction of trajectory used
         - corr(tau^*, R_{tau^*})           # rank correlation over batch
         - C_phi(s_0) - V_pi(s_0) gap      # Snell-vs-value gap proxy
```

For environments with vector reward (`info["vector"]`), the cumulative
`R_t` is replaced with the **scalar** `sum_components(r_vec_t)` for the
stopping-time computation only — the Snell envelope is defined over a
scalar process, so we use the per-step component-sum as the
optimization target. This is **not** scalarization in the disqualifier
sense: the policy update weight is `R_{tau^*}` (component-summed
locked-in reward), which is a *path statistic* of the trajectory,
not a fixed weighted return. (For vector-aware probes, future work
could replace `max` in the Snell recursion with a Pareto-maximum, but
this probe uses scalar Snell to match the panel substrate.)

## Empirical claim

stage: quick

claim: On the **quick** stage (CartPole-v1, Acrobot-v1), SNELL should
match or exceed REINFORCE on mean episodic return within the 120s
budget, AND should exhibit the discriminating observables:
(i) mean(tau^*)/T strictly less than 1 in at least 30% of episodes
during training (i.e., the stopping rule actually fires);
(ii) corr(tau^*, R_{tau^*}) > 0.3 over batched rollouts (i.e., Snell
stops preferentially at high-R points). The quick stage is the right
test because (a) reward is dense, so `R_t` varies meaningfully across
`t` and the Snell envelope has non-trivial shape; (b) episode length
is variable (CartPole 1-500), so `tau^*` discriminates well-running
episodes from failing ones; (c) the cold-start failure of LYRA (run
13) and the env-mismatch of DUAL-IR (run 15) are avoided.

falsifier: If SNELL's mean-episodic-return curve is **identical** to
REINFORCE within seed variance, AND/OR the discriminating observable
(ii) is below 0.1, the predictable-stopping primitive is decorative.
A second falsifier: if random-truncation ablation matches SNELL on
both reward AND on observable (i), the structural-causal claim fails.

## Ablation plan

Replace the **Snell continuation value `C_phi`** with a **fixed
threshold sampled uniformly from observed `R_t` per rollout**:

In `train_ablate.py`:
1. Remove the `C_phi` network entirely.
2. At each rollout, draw `theta_random ~ Uniform(0, max_t R_t)`.
3. Set the ablation stopping time as
   `tau_random = min{ t : R_t >= theta_random }` (default `tau_random = T`).
4. Apply the same truncated-REINFORCE update with locked-in
   reward `R_{tau_random}` and prefix `t <= tau_random`.

This preserves: trajectory truncation, locked-in-reward weighting,
prefix gradient. It removes: the *learned, predictable, state-conditional*
stopping rule.

If the ablation matches SNELL's mean episodic return AND its
`corr(tau^*, R_{tau^*})` observable, the predictable-state-conditional
primitive is decorative. If SNELL exhibits a positive correlation while
the ablation has near-zero correlation BUT mean returns are identical,
the primitive shapes trajectories without changing the optimization
target — partial falsification (interesting but not load-bearing).
If SNELL is strictly better on reward AND on correlation, the primitive
is causally responsible.

A second sanity ablation: replace `R_{tau^*}` (locked-in reward) with
`R_T` (terminal cumulative reward) but keep prefix-truncation by `tau^*`.
If this matches SNELL, the locked-in lock-in is decorative and only
the truncation matters.

## Novelty boundary

Closest known methods:

(a) **Snell-envelope optimal stopping in finance** (Snell 1952;
    Karatzas-Shreve 1998 §2.2; Peskir-Shiryaev 2006). Used to price
    American options. The Snell envelope is computed by backward
    induction on a *known* reward process, with no policy learning
    involved. SNELL adapts the Snell envelope as the **target of a
    learned continuation value** that drives a **score-function
    policy update**, which is novel application of the structure
    to RL.

(b) **DUAL-IR (run 15, this loop)**. DUAL-IR uses the *realized
    arg-supremum-time* `t* = argmax_t (R_t - M_t)` of the path supremum.
    This is **anticipative** — it uses full-trajectory information.
    SNELL uses a **predictable** stopping time `tau^*` measurable with
    respect to `F_t`. The mathematical category (anticipative vs.
    causal) is fundamentally different.

(c) **Q-learning / DQN** (Watkins 1989). Q satisfies the
    **max-outside-expectation** Bellman fixed-point. SNELL's `C`
    satisfies a **max-inside-expectation** Snell fixed-point. By
    Jensen, these are not equal even in expectation. The action-selection
    in Q is `argmax_a Q(s,a)`; the action-selection in SNELL is via
    the policy `pi_theta` updated by score-function — there is no
    argmax-over-actions in the C-update.

(d) **REINFORCE** (Williams 1992). REINFORCE weights every step by the
    full return `R_T` (or return-to-go `G_t`). SNELL weights the prefix
    `t <= tau^*` by the locked-in `R_{tau^*}` and assigns zero weight
    to the suffix `t > tau^*`. The gradient direction differs from
    REINFORCE's whenever `tau^* < T`.

(e) **Optimal-stopping RL / Becker-Cheridito-Jentzen 2019**
    (deep optimal-stopping for high-dim American options). They learn
    a stopping-decision network for *evaluating* given option-style
    payoffs. SNELL learns a continuation value to *truncate gradients
    for policy improvement* — a different mechanism applied to a
    different problem.

(f) **Policy gradient with baseline** (Sutton et al. 1999).
    Subtracting a baseline `b(s_t)` from `G_t` reduces variance but
    does not change the gradient direction. SNELL changes the
    gradient direction by truncating *and* substituting the realized
    locked-in reward `R_{tau^*}` for the cumulative `G_t`.

(g) **Truncated rollouts / horizon-truncated PG**. Some PG variants
    truncate at a *fixed* horizon `H` to reduce variance. SNELL's
    horizon is **adaptive**, **state-conditional**, and **learned**;
    the fixed-horizon variant has zero correlation with `R_t`.

(h) **GAE** (Schulman 2016). GAE is a `(gamma lambda)`-weighted sum of
    TD residuals, requiring a learned scalar V. SNELL has no V and no
    TD residual; its weight is a 0/1 indicator times a path supremum-related
    quantity.

(i) **Decision Transformer / return-conditioned**. Disqualifier under
    `prior_attempts.md`. SNELL has no return conditioning at the input
    level; the stopping rule is computed at policy-update time, not
    fed to the policy.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: SNELL has no
  bucketing and no partial-order vote. `C` is a real-valued state
  function.
- **Family C (within-trajectory geometric statistic)**: `C` is **NOT**
  a within-trajectory statistic — it is a learned function regressed
  toward a path statistic (the suffix-maximum). The within-trajectory
  object is `tau^*`, but it is not a geometric statistic (hull, area,
  etc.); it is a stopping time.
- **Family E (avoid value vocabulary)**: `C` does **not** satisfy the
  Bellman equation. Its fixed-point operator (averaging-of-max) is
  structurally different from the Bellman operator (max-of-averaging).
  This is the load-bearing distinction. The C2 variant of Q-learning
  cited in run 14 (NORMAL) was shown to be order-preserving on `A(s,.)`;
  SNELL's `C` is order-changing on policy gradients because the
  truncation indicator `1[t <= tau^*]` is **not** an additive shift on
  the gradient — it is a multiplicative on/off.

The structural difference from all of the above is **predictable-stopping
truncation** of the trajectory by a **Snell-envelope continuation value**
satisfying a **max-inside-expectation** fixed-point distinct from any
Bellman fixed-point.

## Proof debt

1. **Contraction of the Snell operator.** Show that the Snell-style
   operator `(T_Snell f)(s) := E_pi[max(R_{t+1}, R_{t+1} + gamma f(s_{t+1})) | s_t = s]`
   on the bounded-function space `B(S)` is a contraction in sup-norm
   with modulus `gamma`. The argument is straightforward by
   `|max(a, x) - max(a, y)| <= |x-y|` and standard
   gamma-discounted contraction; the open piece is the policy-dependent
   convergence rate.

2. **Stochastic-regression convergence.** Show that the SGD update
   `phi <- phi - alpha grad_phi (C_phi(s_t) - target_t)^2` with the
   suffix-maximum target converges to the Snell continuation value
   under standard Robbins-Monro conditions. The target distribution
   is non-stationary (depends on `pi`), so this requires a
   two-timescale argument.

3. **Improvement theorem.** *Conjecture:* under the SNELL update, the
   policy iterate `pi_n` satisfies
   `J(pi_{n+1}) >= J(pi_n) - O(epsilon_n)` where `epsilon_n` is the
   regression error of `C_phi` to the true Snell value. Open: this
   would be the load-bearing improvement claim, analogous to
   Kakade-Langford for trust regions.

4. **Bias of locked-in reward gradient.** The score-function gradient
   weighted by `R_{tau^*} * 1[t <= tau^*]` is a biased estimator of
   `grad_theta E[R_tau^*(theta)]` because `tau^*` depends on `theta`
   through `s_t` distribution. The exact bias is open and analogous
   to (but distinct from) the bias analysis of the actor-critic
   advantage estimator.

5. **Empirical-correlation lower bound.** Prove that
   `corr(tau^*, R_{tau^*}) > 0` under any non-trivial `C_phi`. This
   is the discriminating-observable proof that random truncation
   cannot replicate SNELL's behavior.

The empirical probe will reveal whether the Snell-stopping primitive
produces a measurably different rollout-length distribution from
random truncation; a positive result on observables (i) and (ii) on
CartPole would justify investing in proof item (3).
