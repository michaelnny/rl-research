# 20260606-16-auto -- TEAR (Trajectory-Empirical Adjoint Reflection) [probe]

## Principle

The optimal policy is identified by the **discrete-time Pontryagin
maximum principle** applied to the *trajectory-empirical* transition
operator: a vector-valued **adjoint co-state** `λ_t ∈ ℝ^k` is propagated
backward along each realized rollout by a linear adjoint recursion (no
max, no Bellman backup), with terminal boundary `λ_T` set to the
realized terminal vector reward, and the per-step Hamiltonian
`H_t(a) = ⟨r_vec_t, w⟩ + ⟨λ_{t+1}, φ(s_{t+1}) − E_a[φ(s')]⟩`
becomes the score-function weight for `∇_θ log π_θ(a_t|s_t)`; the
policy improves by ascending the *Hamiltonian* of the adjoint, never by
backing up a max-of-Q.

## Primitive

The **trajectory-empirical adjoint co-state**

  `λ : {episodes} × {0, …, T} → ℝ^k`

mapping (rollout τ, time index t) to a `k`-dimensional real vector,
where `k` is the dimension of the state-feature embedding `φ : S → ℝ^k`
(for vector envs, `k` is set to match the reward-vector dimension so
each component of λ tracks the gradient of one reward channel). λ
satisfies the **discrete-time adjoint recursion**

  `λ_T := r_vec_T`  (terminal reward vector, padded to ℝ^k by zeros if k
   exceeds the reward dimension; reward dim ≤ k by construction)
  `λ_t := r_vec_t + J_t^T · λ_{t+1}`,
   `J_t := ∂φ(s_{t+1})/∂φ(s_t) ≈ I + (φ(s_{t+1}) − φ(s_t)) ⊗ φ(s_t) / ‖φ(s_t)‖²`
   (rank-1 trajectory-empirical Jacobian; see derivation step 4).

Codomain: a single vector-valued path `t ↦ λ_t`, one per episode. This
is **one typed object** per rollout: a vector co-state path. Not a
table, not a network output over states, not a distribution.

## Derivation sketch

1. **Pontryagin discrete-time MP.** For a deterministic discrete-time
   system `s_{t+1} = f(s_t, a_t)` with stage cost `r_t = r(s_t, a_t)`,
   Pontryagin's maximum principle (Bertsekas 2017 *Dynamic Programming
   and Optimal Control*, Vol. I, §4.4) states that an optimal control
   sequence `(a_0*, …, a_{T-1}*)` satisfies, for all t and a:
   `H_t(s_t*, a, λ_{t+1}*) ≤ H_t(s_t*, a_t*, λ_{t+1}*)`,
   where `H_t(s, a, λ) := r(s,a) + λ^T f(s,a)` and λ obeys the backward
   adjoint recursion `λ_t = ∇_s H_t = ∇_s r + (∇_s f)^T λ_{t+1}`,
   `λ_T = ∇_s g(s_T)`.

2. **Stochastic extension via score-function.** For stochastic
   `a_t ~ π_θ(·|s_t)`, replace the pointwise Hamiltonian maximum with a
   **soft maximum via score-function ascent**: at each rollout, the
   parameter gradient is
   `∇_θ J = E_τ[Σ_t H_t(s_t, a_t, λ_{t+1}) · ∇_θ log π_θ(a_t|s_t)]`.
   This is the **stochastic Pontryagin gradient** (Recht 2018 §3 has a
   continuous-time version; the discrete adaptation is straightforward
   under standard score-function regularity).

3. **Vector reward = vector co-state.** When the reward is vector-valued
   `r_vec_t ∈ ℝ^m`, the adjoint becomes vector-valued
   `λ_t ∈ ℝ^k` with `k ≥ m`, and the recursion becomes the
   *componentwise* backward equation: each channel of λ propagates the
   gradient-of-cumulative-reward in that channel. The Hamiltonian
   `H_t(a) := ⟨r_vec_t, e_proj⟩ + ⟨λ_{t+1}, J_t · e_a⟩` couples ALL
   channels of λ jointly without scalarizing them: the score-function
   weight is a SCALAR per (a, t), but it is computed by an inner
   product of two vectors `(λ_{t+1}, J_t e_a)` whose components carry
   per-channel future credit. This is **NOT** scalarization `wᵀr` — the
   weight vector `λ_{t+1}` is *not constant in t* and is *not a learned
   policy of preferences*; it is the trajectory-empirical adjoint of the
   vector reward.

4. **Trajectory-empirical Jacobian.** The unknown transition Jacobian
   `∇_s f(s,a)` is replaced along each realized trajectory by its
   **rank-1 trajectory-empirical estimator**
   `J_t := I + (φ(s_{t+1}) − φ(s_t)) · φ(s_t)^T / (‖φ(s_t)‖² + ε)`
   (Sherman-Morrison-style local linearization). This avoids learning a
   separate transition model: J_t is computed in O(k) per step from the
   realized (φ(s_t), φ(s_{t+1})) pair. The estimator is **biased but
   consistent** under sufficient state-feature variation along the
   trajectory; bias analysis is proof debt.

5. **Cold-start property.** Unlike LYRA (run 13), TEAR does **not**
   multiplicatively tilt by reward. Even at zero scalar reward, the
   vector reward channel often has non-zero per-step entries (e.g., DST
   has a constant −1 step-penalty channel, RG has constant penalties).
   The terminal boundary `λ_T = r_vec_T` is non-zero whenever the env
   exposes any per-step or terminal vector reward, and the backward
   adjoint then carries a non-trivial signal even when the *scalar*
   sum of reward channels is zero.

6. **Order-changing property.** Unlike NORMAL (run 14), the
   action-selection signal `H_t(a) = ⟨r_vec(s_t,a), 1⟩ + λ_{t+1}^T J_t e_a`
   is **not** invariant under row-wise shifts of any value-like
   function: it depends on the *direction* `J_t e_a`, which is a
   per-action vector projected onto the per-trajectory adjoint
   direction. Shifting all components of λ by the same constant
   changes the inner product by `c · ⟨J_t e_a, 1⟩`, which varies with
   a. So the relative ordering of actions can change.

7. **Why not actor-critic.** A standard actor-critic computes a
   *scalar* critic `V(s)` or `Q(s,a)` over all states (a learned
   function) and backs it up via Bellman. TEAR's adjoint λ_t is
   **per-trajectory** (NOT a function of state across episodes), is
   propagated by a **linear backward recursion** (no max), and is
   **vector-valued** (one component per reward channel). The score-
   function weight `H_t(a_t)` is a per-step Hamiltonian, not a
   cumulative return-to-go.

8. **Why not GAE / advantage estimation.** GAE is a (γλ)-weighted sum
   of TD residuals, where the residual `δ_t = r_t + γ V(s_{t+1}) −
   V(s_t)` requires a learned scalar V. TEAR has no V; the per-step
   weight is a Hamiltonian inner product `⟨λ_{t+1}, J_t e_{a_t}⟩ +
   ⟨r_vec_t, 1⟩` where the first term is a vector-vector pairing, not
   a (γλ)-residual sum.

9. **Why not SVG.** Stochastic Value Gradient (Heess et al. 2015)
   requires a learned dynamics model and uses the reparametrization
   gradient through it. TEAR uses a *trajectory-empirical* rank-1
   Jacobian computed from the realized (s_t, s_{t+1}) pair at each
   step; no model is learned. The score-function gradient (not
   reparametrization) is used. SVG is a reparametrization-gradient
   method; TEAR is a Pontryagin-Hamiltonian-weighted score-function
   method.

10. **Proof debt.** Two open theorems: (i) consistency of the rank-1
    trajectory-empirical Jacobian as an estimator of the true Jacobian
    under stationarity assumptions; (ii) convergence of the
    Pontryagin-weighted score-function gradient to a local-optimum of
    `J(π_θ)` as a Robbins-Monro stochastic-gradient procedure. These
    are pursued only after empirical signal.

## Update rule

```
Inputs: env, policy π_θ, feature dim k, learning rate α, ε for J stability
Init:   θ random; φ(s) = either identity (tabular) or a fixed random projection
        for image/grid obs into ℝ^k. For DST/RG, φ(s) = state vector itself.

For each episode:
    1. Roll out τ = (s_0, a_0, r_vec_0, s_1, ..., s_T) under π_θ.
       Read r_vec from info["vector"] for vector envs;
       for sparse scalar envs, set r_vec_t = (r_t, 1) ∈ ℝ² so the second
       channel is a constant time-marker (always non-zero — guarantees
       λ_T ≠ 0 even on zero-reward sparse trajectories).

    2. Compute features φ_t = φ(s_t) for t = 0..T.

    3. Compute trajectory-empirical Jacobians (rank-1):
         For t = 0..T-1:
           Δφ_t = φ_{t+1} − φ_t
           denom = ‖φ_t‖² + ε
           J_t  = I_k + (Δφ_t · φ_t^T) / denom   (k × k matrix; for
                  efficiency, never materialize — apply via two
                  inner products)

    4. Backward adjoint recursion:
         λ_T = pad_to_k(r_vec_T)        # zero-pad if reward dim < k
         For t = T-1 down to 0:
           λ_t = pad_to_k(r_vec_t) + J_t^T λ_{t+1}
                 # J_t^T v = v + φ_t · (Δφ_t · v) / denom
                 # one O(k) per step

    5. Per-step Hamiltonian weights:
         For t = 0..T-1:
           H_t = ⟨r_vec_t, 1_m⟩ + ⟨λ_{t+1}, J_t · onehot(a_t, |A|)_proj⟩
           where onehot(a_t)_proj ∈ ℝ^k is a fixed random projection of
           the action-one-hot into ℝ^k; J_t v is computed in O(k).
           # H_t is a SCALAR weight per (s_t, a_t) on this episode.

    6. Score-function policy ascent:
         g_θ = Σ_t H_t · ∇_θ log π_θ(a_t | s_t)
         θ ← θ + α · g_θ

# At eval: act greedily under π_θ, OR optionally compute λ over a fresh
# rollout and act argmax_a H_t(a). For probe, act stochastically (sample).
```

The load-bearing primitive is the **per-trajectory adjoint vector
λ_t**, computed once per episode by an O(T · k) backward pass. The
score-function weight H_t replaces the cumulative return G_t (REINFORCE)
or the advantage Â_t (PPO/A2C). It is computed without a learned
critic and without Bellman backup.

## Empirical claim

stage: vector
claim: On the **vector** stage (deep-sea-treasure-concave-v0,
resource-gathering-v0), TEAR should achieve hypervolume strictly above
the random baseline (DST random=194.0, RG random=1.331) on at least one
of the two envs within the 120s budget. The vector stage is the
appropriate test because (i) the reward is genuinely multi-channel and
the adjoint co-state was *designed* to carry per-channel credit; (ii)
DST has a constant per-step penalty channel that guarantees `r_vec_t ≠
0` from step 1, avoiding LYRA's cold-start failure; (iii) the
hypervolume metric rewards Pareto-front coverage, which is exactly what
a vector-valued adjoint should produce by trading off per-channel
gradients along the trajectory.

falsifier: If TEAR's hypervolume is at-or-below the random baseline on
both vector envs (DST ≤ 194.0 AND RG ≤ 1.331), the principle is
falsified. Likewise, if the ablation (replace the trajectory-empirical
adjoint λ_t with a random vector resampled at each step) matches TEAR's
hypervolume, the adjoint primitive is decorative and the algorithm
collapses to "REINFORCE with random per-step weights" (which is a known
high-variance estimator of policy gradient, not a new algorithm).

## Ablation plan

Replace the **trajectory-empirical backward adjoint** with a **random
i.i.d. vector resampled at each step**. Concretely in `train_ablate.py`:

1. Skip steps 3 and 4 of the update rule (no Jacobian, no backward
   recursion).
2. At each step t, sample `λ̃_t ~ N(0, I_k)` independently (i.i.d.
   across t and across episodes; no temporal coupling).
3. Compute the score-function weight as
   `H_t = ⟨r_vec_t, 1_m⟩ + ⟨λ̃_{t+1}, J_t · onehot(a_t)_proj⟩`, with
   `J_t` still computed from features (so the per-action projection
   structure is preserved) — OR, even simpler, replace the second term
   by `⟨λ̃_{t+1}, onehot(a_t)_proj⟩` (no J).
4. Apply the same score-function update.

If the ablation matches TEAR's hypervolume on the vector stage, the
backward adjoint primitive is not load-bearing and TEAR collapses to
REINFORCE with a noisy weight (a published variance-reduction trick at
best). If TEAR is strictly better, the adjoint structure carries
genuine credit-assignment information.

A **second** sanity ablation: replace `λ_T = r_vec_T` with `λ_T = 0`
(no terminal boundary). The adjoint then satisfies `λ_t = pad(r_vec_t)
+ J_t^T λ_{t+1}` with zero terminal — a degenerate equation that
collapses to a forward-cumulative weighted by Jacobian transposes. If
this also matches, the boundary condition is the load-bearing piece (a
weaker but informative signal).

## Novelty boundary

Closest known methods:

(a) **Pontryagin's maximum principle for continuous-time control**
    (Pontryagin et al. 1962; Bertsekas 2017 §4.4). PMP is a
    necessary-conditions framework for *deterministic* optimal control
    with *known dynamics*; TEAR adapts the discrete-time PMP to
    *stochastic* MDPs with *unknown dynamics* by using a
    trajectory-empirical Jacobian and a score-function ascent. The
    novelty is the trajectory-empirical Jacobian and the use of the
    Hamiltonian as the score-function weight in place of cumulative
    return.

(b) **Stochastic Value Gradient (SVG) / SVG(∞)** (Heess et al. 2015).
    SVG learns a transition model and uses reparametrization gradients
    through it. TEAR has NO learned model — only a rank-1 empirical
    Jacobian estimated per-step from realized transitions — and uses
    SCORE-FUNCTION (not reparametrization) gradients. SVG requires
    differentiable models; TEAR works with discrete actions and
    arbitrary envs.

(c) **Actor-critic / advantage methods** (Mnih et al. 2016; Schulman
    et al. 2016). Critic is a *scalar* learned function over states;
    advantage is a (γλ)-weighted sum of scalar TD residuals. TEAR has
    NO learned critic, NO TD residual, NO scalar value function — λ is
    a per-trajectory backward-propagated vector with terminal boundary
    `r_vec_T`. The score-function weight is a Hamiltonian inner
    product, not an advantage.

(d) **Pontryagin-RL / iLQR / DDP** (Tassa et al. 2012, Levine 2014).
    These methods compute Hessians or full Jacobians from a learned or
    given dynamics model and use Newton-style updates. TEAR uses a
    rank-1 trajectory-empirical Jacobian and a first-order
    score-function update. No Hessian, no inverse, no model.

(e) **Generalized Advantage Estimation** (Schulman 2016). GAE is an
    exponentially-weighted sum of TD residuals over a single scalar
    value function. TEAR has neither V nor TD residuals, and the
    weight is computed by a backward LINEAR recursion on a vector
    co-state, not by a forward (γλ)-weighted geometric sum.

(f) **Vector / multi-objective RL via scalarization w^T r**
    (Roijers-Whiteson 2017 survey). Disqualifier under
    `prior_attempts.md`. TEAR is **not** scalarization: λ_t varies
    *per-step* and *per-trajectory*, and is determined by the backward
    adjoint of the vector reward, not by a fixed or learned weight.
    Even though the score-function weight `H_t` is a scalar (it has
    to be, to multiply a parameter gradient), the *path* by which it
    is computed depends on the vector structure of `r_vec` non-
    linearly through the backward recursion. Crucially, `H_t` is not
    of the form `w^T r_t` for any t-independent or policy-independent w.

Nearest dead family from `prior_attempts.md`:
- **Family A (bucketed-tensor + partial-order vote)**: TEAR has no
  bucketing and no partial-order vote. λ_t is a real-valued vector
  computed by a backward linear recursion on a single trajectory.
- **Family C (within-trajectory geometric statistic)**: λ_t IS a
  within-trajectory statistic, but it is **not a geometric path
  statistic** (hull/Lévy area/spectral coefficient). It is the
  solution of a backward LINEAR recursion driven by the realized
  trajectory's transitions. The closest analog would be a discrete-
  time Doob martingale, but the adjoint is not a martingale — it is
  predictable.
- **Family E (avoid value vocabulary, keep value structure)**: TEAR
  does NOT learn a function over states; λ_t is per-trajectory and
  exists only during the rollout's backward pass. There is no Q, no V,
  no advantage as a learned function. The score-function weight is a
  per-step Hamiltonian, not a relabeled return.

The structural difference from all of the above is the **per-trajectory
backward LINEAR adjoint recursion with the realized terminal-reward
boundary condition**. This is neither a Bellman operator (no max,
linear), nor a learned function (per-trajectory only), nor a path
geometric statistic (driven by recursion, not by sample geometry).

## Proof debt

1. **Consistency of the rank-1 trajectory-empirical Jacobian.** Show
   that as the trajectory length T → ∞ and under stationarity of the
   feature process φ(s_t), the rank-1 estimator
   `J_t = I + (φ_{t+1} − φ_t) φ_t^T / ‖φ_t‖²`
   is a consistent estimator (in some operator-norm sense) of the
   true conditional Jacobian `∇_s E[φ(s_{t+1}) | s_t]`. This is
   analogous to but distinct from the OLS bias analysis (the
   estimator is biased, particularly when the trajectory does not
   span the feature space — this is the failure mode to monitor).

2. **Convergence of stochastic Pontryagin score-function gradient.**
   Show that the update
   `θ ← θ + α · Σ_t H_t · ∇_θ log π_θ(a_t|s_t)`
   with `H_t` defined by the backward recursion converges to a local
   optimum of `J(π_θ) = E_τ[Σ_t r_vec_t · 1_m]` under standard
   Robbins-Monro step-size conditions and bounded-Jacobian
   regularity. The proof strategy combines (i) the discrete-time PMP
   first-order conditions; (ii) the score-function regularity for the
   stochastic-gradient interchange; (iii) a martingale-variance bound
   on `H_t` driven by the eigenvalues of the empirical-Jacobian
   product `J_T J_{T-1} ⋯ J_1`. The bound on the latter requires a
   contraction-type argument that is currently open.

3. **Hypervolume monotone improvement.** Conjecture: along the TEAR
   policy-update trajectory `θ_n`, the expected vector return
   `μ_vec(π_θ_n) := E_τ[r_vec_total(τ)]` traces a curve in `ℝ^m`
   whose **dominated hypervolume** with respect to a fixed reference
   point is monotone-non-decreasing in expectation. This is the
   load-bearing claim if TEAR is to be a *new* multi-objective RL
   method; it is open and would require a Pareto-improvement-flow
   theorem analogous to Kakade-Langford for the scalar case.

The empirical probe will reveal whether the joint backward-adjoint /
score-function dynamics is well-behaved on the vector substrate before
any of these theorems are pursued; a hypervolume gain on at least one
of DST/RG would justify investing in proof item (3).
