# 20260606-20-auto -- GRADCOMP (Gradient-Compass via Fisher-Principal Rotation) [probe]

## Principle

Update the policy along a **rotation between the principal eigen-
direction of the empirical Fisher information and the REINFORCE
score-function gradient**, parameterized by a single angular knob;
this turns the trajectory's own *policy-variation direction* into a
load-bearing structural primitive that is non-zero at random init
(when the REINFORCE direction has near-zero magnitude on sparse
reward) and gradually aligns with reward-driven gradient as training
proceeds.

## Primitive

The **per-rollout Fisher principal direction**

  `v₁ : T → S^{n−1}_θ`,
  `v₁(τ) := argmax_{‖u‖=1} ⟨u, F̂_τ u⟩`

where `T` is the space of realized rollouts, `S^{n−1}_θ` is the unit
sphere in the `n`-dimensional policy-parameter tangent space at the
current `θ`, and `F̂_τ` is the **rank-≤T empirical Fisher information
of the rollout** under the current policy:

  `F̂_τ := Σ_{t=0..T} g_t g_t^T`,
  `g_t := ∇_θ log π_θ(a_t | s_t) ∈ ℝ^n`.

Codomain is the unit sphere — a single direction in parameter space
per rollout, computable in `O(T · n)` time without materializing the
`n × n` matrix `F̂_τ` (top eigenvector by rank-T symmetric power
iteration on the score-Gram matrix `G ∈ ℝ^{T×T}` with `G_{i,j} = g_i^T
g_j`, then `v₁ = (Σ_t α_t g_t) / ‖·‖` with `α` the top eigenvector of
`G`).

This is **one typed object**: a unit vector in policy-parameter
tangent space, defined per rollout. It is **not** a value function,
**not** a Q-function, **not** a per-state policy modification, and
**not** an advantage estimator.

## Derivation sketch

1. **Empirical Fisher.** For a stochastic policy `π_θ` with score
   `g_t := ∇_θ log π_θ(a_t|s_t)`, the empirical Fisher of a rollout
   `τ = (s_0, a_0, ..., s_T)` is
   `F̂_τ = Σ_t g_t g_t^T ≽ 0` (positive semidefinite, rank ≤ T).
   Its eigenstructure characterises the **directions in parameter
   space along which the policy varies across this trajectory**:
   `v₁(τ)` is the direction of maximal joint score variation.

2. **REINFORCE direction.** The classical score-function gradient is
   `g_τ := Σ_t R_t · g_t ∈ ℝ^n` where `R_t` is the cumulative
   discounted return-to-go computed from the **scalar environment
   reward** `r_t` (the sparse-stage envs `MiniGrid-DoorKey-8x8-v0`
   and `MiniGrid-KeyCorridorS3R3-v0` produce a scalar reward natively;
   no vector reduction is performed). This is a **reward-weighted
   sum** of the same per-step scores.

3. **Two orthogonal failure modes of REINFORCE.** (a) When `R_t ≈ 0`
   for all `t` (cold start, sparse reward, random init), `g_τ ≈ 0` and
   no parameter update happens — the policy stalls on the random
   floor. **This is the dominant failure mode on DoorKey/KeyCorridor**:
   reward is identically zero on every episode until the agent first
   reaches the goal, which under a random policy can take dozens to
   hundreds of episodes. (b) When the per-step scores are highly
   correlated (e.g., the policy is in a low-curvature basin), `g_τ`
   lies along a single direction and the update reduces effective
   parameter dimension.

4. **Compass principle.** Define the **rotated update direction**
   `d_τ ∈ ℝ^n` by spherical interpolation (slerp) between `v₁(τ)` and
   `ĝ_τ := g_τ / (‖g_τ‖ + ε)`:
   `d_τ := slerp(v₁(τ), ĝ_τ, η_t)`,
   `η_t ∈ [0, 1]` is the **compass angle** (η_t = 0: pure Fisher
   direction; η_t = 1: pure REINFORCE direction). The update is
   `θ ← θ + α · ‖g_τ‖_⋆ · d_τ`,
   where `‖g_τ‖_⋆ := max(‖g_τ‖, c)` with floor `c > 0` to ensure a
   non-zero update at cold start. (At cold start, `‖g_τ‖_⋆ = c` and
   `d_τ ≈ v₁(τ)`; the algorithm walks in the Fisher principal
   direction.)

5. **Why v₁ is load-bearing at random init on sparse-reward envs.**
   When the agent has never seen reward, `R_t ≡ 0` so `g_τ = 0` and
   REINFORCE produces no update — the agent random-walks the maze
   indefinitely. But `F̂_τ ≠ 0` whenever the policy is stochastic and
   at least two distinct (s, a) pairs occur in the rollout (always
   true for DoorKey, where episodes have ≥ 5 steps before timeout).
   So `v₁(τ)` is **always defined and non-zero**, and walking along it
   is a *trajectory-informed* perturbation in parameter space — not a
   random walk.

6. **Why v₁ is not a curiosity bonus.** v₁(τ) does **not** depend on
   reward, count, or novelty. It is purely a function of the
   policy's own per-step scores along the realized trajectory. It is
   not an exploration *bonus* added to a return — it is the
   *direction* of the parameter update itself.

7. **Compass annealing.** `η_t` anneals from `η_0 ∈ [0, 0.3]` toward
   `1` as `‖g_τ‖` exceeds the floor `c`. Concretely:
   `η_t := σ((‖g_τ‖ − c) / c)` with σ a sigmoid. While the policy is
   on the floor (no reward yet seen), η ≈ 0 and the update follows
   v₁; once REINFORCE gradient magnitude exceeds the floor (after
   first reward), η → 1 smoothly.

8. **Why slerp, not linear interpolation.** Slerp preserves unit-norm
   on the sphere, so the *magnitude* of the update is controlled
   solely by `‖g_τ‖_⋆`, not by interpolation magnitude artefacts.
   `d_τ` is always a unit vector. (Linear interpolation would shrink
   the magnitude near η = 0.5 and is a known pathology in directional
   blending.)

9. **Connection to NPG, but not equal.** Natural Policy Gradient
   transforms the *magnitude* of the gradient by `F^{−1}`: `g^NPG =
   F^{−1} g_τ`. GRADCOMP transforms the *direction* by rotating
   *toward* the Fisher's top eigenvector (the opposite spectral end
   from F^{−1}'s amplification). NPG amplifies low-Fisher directions
   (poorly-explored geometry); GRADCOMP slides toward high-Fisher
   directions (well-explored geometry, where the policy can express
   change cleanly). Crucially NPG also produces zero update when
   `g_τ = 0` (it inverts a zero vector); GRADCOMP's gradient floor
   makes the cold-start update non-zero.

10. **Proof debt.** Open theorems: (i) under annealed η_t → 1 and
    standard step sizes, the GRADCOMP iterate converges to a
    stationary point of `J(π_θ)`. The intuition: in the limit η = 1,
    GRADCOMP is REINFORCE; for η_t bounded away from 0, the update
    is biased but the bias is controlled by `‖g_τ‖`. (ii) On the
    cold-start regime `g_τ ≈ 0`, the Fisher-direction walk
    `θ_{n+1} = θ_n + α c v₁(τ_n)` increases the *effective rank* of
    the policy's score-Gram across rollouts, formalized via a
    monotonicity argument on the spectrum of `E_τ[F̂_τ]`. (iii)
    Convergence rate comparison with NPG when both are applied.

## Update rule

```
Inputs: env (scalar-reward sparse stage: DoorKey-8x8 or KeyCorridorS3R3),
        policy π_θ (n params), discount γ, learning rate α,
        gradient floor c, eps for stability
Init:   θ random

For each episode:
    1. Roll out τ = (s_0, a_0, r_0, ..., s_T) under π_θ.
       Compute return-to-go: R_t = Σ_{k≥t} γ^{k-t} r_k for all t,
       where r_k is the SCALAR env reward (sparse stage envs are
       natively scalar — no info["vector"] reduction is performed).

    2. Per-step scores:  g_t = ∇_θ log π_θ(a_t | s_t)        # ℝ^n

    3. REINFORCE direction:
         g_τ = Σ_t R_t · g_t                                  # ℝ^n
         ‖g_τ‖_⋆ = max(‖g_τ‖, c)
         ĝ_τ = g_τ / (‖g_τ‖ + eps)                            # near-zero if cold

    4. Fisher principal direction via score-Gram (rank-T trick):
         G ∈ ℝ^{T×T}, G_{i,j} = g_i^T g_j                     # T*n flops
         (μ, α_vec) = top_eig_sym(G)                          # one Lanczos iter
         v₁ = (Σ_t α_vec[t] · g_t) / ‖·‖                      # ℝ^n unit

    5. Compass angle:
         η = σ((‖g_τ‖ − c) / c)                               # ∈ (0, 1)

    6. Slerp on unit sphere:
         (sign convention: flip v₁ → −v₁ if v₁·ĝ_τ < 0 so they point
         the same half-space; this only fires when ‖g_τ‖ > eps)
         ω = arccos(clip(v₁ · ĝ_τ, −1+eps, 1−eps))            # angle between
         d_τ = (sin((1−η)ω) / sin(ω)) v₁ + (sin(ηω) / sin(ω)) ĝ_τ
         (if ω near 0: d_τ = ĝ_τ; if ‖g_τ‖ < eps: d_τ = v₁)

    7. Parameter update:
         θ ← θ + α · ‖g_τ‖_⋆ · d_τ

    8. Logging observables:
         - ‖g_τ‖             # REINFORCE magnitude (cold ≈ 0)
         - μ                 # principal Fisher eigenvalue
         - v₁ · ĝ_τ          # cosine alignment of compass with REINFORCE
         - η                 # current angle (0=Fisher, 1=REINFORCE)
         - first_reward_ep   # episode index of first non-zero return
```

The load-bearing primitive is `v₁(τ)`. Computing it costs `O(T²)` for
the Gram matrix and `O(T)` for one Lanczos iteration — negligible
compared to backprop. The update rule reduces to REINFORCE when η = 1
(post-warmup) and to a Fisher-direction walk when η = 0 (cold start).

## Empirical claim

stage: sparse

claim: On the **sparse** stage (`MiniGrid-DoorKey-8x8-v0`,
`MiniGrid-KeyCorridorS3R3-v0`), GRADCOMP should reach the **first
reward-bearing episode** (first episode with non-zero return) **strictly
earlier** than REINFORCE on average across seeds, AND should match or
exceed REINFORCE on mean episodic return within the 120s budget. The
sparse stage is the natural test bed because reward is identically zero
until the agent first stumbles into the goal, which is exactly the
cold-start regime where REINFORCE has `g_τ = 0` and stalls on the
random-walk floor while GRADCOMP walks in the trajectory-informed
Fisher principal direction. Discriminating observables on the sparse
stage:
(i) **Cold-phase update magnitude.** During the pre-first-reward phase
(when `‖g_τ‖ < c` for both arms), GRADCOMP should produce strictly
larger parameter updates (`‖θ_{n+1} − θ_n‖`) than REINFORCE, which
should produce zero-magnitude updates by construction.
(ii) **Alignment drift.** The cosine alignment `v₁ · ĝ_τ` (logged on
each warm rollout, after the first reward is found) should drift from
near-zero (random alignment in early training) toward positive values
(≥ 0.3) as training progresses — a sign that the Fisher principal
direction has become correlated with the reward-driven direction.
(iii) **First-reward episode index.** Median episode index of the
first non-zero return, across seeds, should be strictly lower for
GRADCOMP than for REINFORCE on DoorKey-8x8 (REINFORCE under random
init typically takes 50–200 episodes to first reach the goal; GRADCOMP
should take fewer because the Fisher walk is non-trivial).

falsifier: **Primary.** If GRADCOMP's median first-reward-episode
index on DoorKey-8x8 is **identical within seed variance** to REINFORCE
(i.e., the Fisher-direction cold walk does not accelerate first
discovery), the rotation primitive is not load-bearing in the regime
it was designed for. **Secondary.** If observable (ii) — the alignment
`v₁ · ĝ_τ` — stays near zero throughout training, the Fisher direction
is uncorrelated with the reward direction at convergence and the
rotation cannot help. **Tertiary.** If GRADCOMP's mean episodic return
on KeyCorridorS3R3 is strictly worse than REINFORCE's (i.e., the
cold-phase walk drives the policy *away* from useful regions of
parameter space), the floor mechanism is harmful rather than helpful.

## Ablation plan

Replace the **Fisher principal direction `v₁(τ)`** with a **uniformly
random unit vector `ξ ∼ Uniform(S^{n−1})`**, redrawn at each rollout.

In `train_ablate.py`:
1. Skip step 4 (no Gram matrix, no Lanczos).
2. At each rollout, sample `ξ ∼ N(0, I_n)`; set `v₁ ← ξ / ‖ξ‖`.
3. Apply the **same** slerp interpolation, the same η-annealer, the
   same parameter update.

This preserves: the gradient floor `c`, the compass-annealing schedule,
the slerp blending, the score computation, all hyperparameters. It
removes: the *trajectory-informed* nature of the Fisher principal
direction.

Predicted contrast on the sparse stage:
- On warm rollouts (η ≈ 1, post-first-reward), both arms reduce to
  REINFORCE; the ablation tie is expected and not falsifying.
- On cold rollouts (η ≈ 0, pre-first-reward), GRADCOMP walks along
  `v₁(τ)`, a trajectory-informed direction. The ablation walks along
  a random unit vector, which is a pure random walk in parameter
  space.
- The **discriminating observable** is observable (ii):
  `v₁ · ĝ_τ` should grow positive for GRADCOMP across training, but
  should stay at zero in expectation for the random-direction
  ablation.
- The **discriminating reward observable** is the *median episode at
  which the first non-zero return is observed on
  `MiniGrid-DoorKey-8x8-v0`*: GRADCOMP should reach this earlier than
  the random ablation if the Fisher direction encodes useful early-
  training structure.

A second sanity ablation (cheap): set `η ≡ 1` always (pure REINFORCE,
no compass). If GRADCOMP matches, the rotation mechanism is
decorative — only the gradient floor `c` is doing anything. This
isolates whether the principle's value is in the Fisher direction or
just in the floor.

If the random-direction ablation matches GRADCOMP on the
discriminating observables AND on first-reward-episode index, the
Fisher-principal mechanism is not load-bearing. If GRADCOMP exhibits
strictly higher `v₁ · ĝ_τ` AND strictly earlier first-reward episode,
the primitive is causally responsible.

## Novelty boundary

Closest known methods:

(a) **Natural Policy Gradient / NPG** (Kakade 2002; Bagnell-Schneider
    2003; Amari 1998 for the natural-gradient identity). NPG computes
    `g^{NPG} = F^{−1} g_τ` — it amplifies *low-Fisher* directions
    (where the policy is least expressive) and dampens *high-Fisher*
    directions. GRADCOMP rotates the update *toward* the **top
    Fisher eigenvector** — the opposite spectral end. The two methods
    produce different parameter trajectories: NPG drags the update
    toward poorly-explored geometry; GRADCOMP slides toward
    well-explored geometry where the policy can change cleanly.
    Crucially, NPG's update is **invariant to reparametrization**;
    GRADCOMP's is **not** — it depends on the parameter-space metric.
    NPG also has no cold-start mechanism: when `g_τ = 0`, NPG also
    produces zero update. GRADCOMP's gradient floor `c` ensures a
    non-zero update at cold start, which NPG fundamentally cannot.

(b) **TRPO / PPO** (Schulman 2015, 2017). These constrain the update
    by a KL-distance trust region (i.e., a Fisher-quadratic
    constraint). GRADCOMP is **not** a trust-region method: there is
    no KL constraint, no clipped ratio, no inner backtrack. The
    update direction is reshaped pre-step.

(c) **REINFORCE / vanilla policy gradient** (Williams 1992). REINFORCE
    is the η = 1 limit of GRADCOMP. The novelty is the rotation
    *toward* `v₁` for η < 1.

(d) **Curiosity / RND / count-based exploration**. These add an
    intrinsic *reward* `r_int(s)` to the environment reward, which
    enters via the magnitude `R_t` of the score weight. GRADCOMP does
    **not** modify reward at all — the update rule's dependence on
    reward is identical to REINFORCE's. The cold-start mechanism is
    **directional**, not reward-modifying. This distinction is
    especially important on DoorKey/KeyCorridor where curiosity-style
    bonuses are the standard sparse-reward attack.

(e) **Stein Variational Policy Gradient (SVPG; Liu-Wang 2017)**.
    SVPG maintains a *population* of policies and updates them by
    Stein variational gradient flow with a kernel that introduces
    repulsion. GRADCOMP has **a single policy**, no population, no
    repulsion, and no kernel. The Fisher principal direction is a
    function of *the single policy's own rollout*, not of a pairwise
    interaction across particles.

(f) **K-FAC / Shampoo / second-order methods** (Martens-Grosse 2015).
    These approximate `F^{−1}` by a Kronecker-factored structure and
    apply it as a preconditioner. GRADCOMP does not invert F or any
    approximation; it uses a single eigenvector of `F` (computed via
    the Gram trick) without inversion or full eigendecomposition.

(g) **Random search / Evolution Strategies (Salimans et al. 2017)**.
    ES perturbs parameters by random Gaussian noise and uses
    finite-difference gradient estimates. The cold-start ablation of
    GRADCOMP (random unit-vector replacement) reduces to a flavor of
    ES — but GRADCOMP itself does **not**: `v₁(τ)` is a deterministic
    function of the realized score vectors, not a sampled
    perturbation.

(h) **Power iteration on the policy gradient covariance** (e.g.,
    PCA-based variance reduction by Tucker et al. 2018). These
    decompose the gradient covariance to identify and project out
    high-variance directions. GRADCOMP does **not** project out — it
    rotates *toward* the top eigenvector. The structural use is
    opposite.

(i) **Score-function regularizers / control variates** (Greensmith et
    al. 2004 baseline analysis). These modify the *weight* on the
    score, not the *direction* of the parameter update. GRADCOMP
    keeps the score-function update form but rotates its direction.

(j) **Rank-1 inverse-Fisher NPG approximations** (Huo et al. 2026).
    These approximate `F^{−1}` by a rank-1 outer product; still
    inverse-Fisher preconditioning. GRADCOMP uses a rank-1 piece of
    `F` itself (not its inverse) and rotates rather than
    preconditions.

Nearest dead family from `prior_attempts.md`:

- **Family A (bucketed-tensor + partial-order vote)**: GRADCOMP has
  no bucketing — the primitive is a unit vector in continuous
  parameter space, not a discrete tensor. There is no partial-order
  vote.
- **Family B (pairwise trajectory comparison)**: GRADCOMP works on a
  single rollout at a time; no pair-matching across trajectories.
- **Family C (within-trajectory geometric statistic)**: The Fisher
  principal direction `v₁(τ)` IS a within-trajectory statistic, but
  it is **not** a geometric path statistic of the
  observation/cumulant trace. It is a statistic of the **policy's
  parametric variation along the trajectory** — a tangent-space
  object on the policy manifold, not a state-space object. The
  load-bearing structure is the policy's parameter geometry, not the
  state's signal geometry. (Family C dies because state-geometric
  statistics collapse to "shorter is better" on terminal-only-reward
  envs; v₁ is independent of reward and cannot collapse this way.)
- **Family D (reward-independent + reward-gated)**: `v₁(τ)` is
  reward-independent, but the rotation `slerp(v₁, ĝ_τ, η)` is
  **never gated on reward** — even at η = 0 (cold), the update is
  applied (with floor magnitude). The mechanism is not "compute when
  reward-free, fire when reward-present"; it is "always-fire with
  a smoothly-rotating direction."
- **Family E (avoid value vocabulary, keep value structure)**:
  GRADCOMP has no value, no Q, no advantage, no return-compression.
  The only learned object is `θ`. The cold-start mechanism `v₁`
  carries trajectory information through the *score Gram structure*,
  which is a statistic of policy parametric geometry, not of value.

The structural difference from all named methods and dead families is
the **per-rollout rotation of the parameter update toward the Fisher
principal eigenvector with a gradient-magnitude-annealed angle**.
This is operationally distinct from inverse-Fisher (NPG), trust-
region projection (TRPO/PPO), Hessian preconditioning (K-FAC), and
random perturbation (ES); it is conceptually distinct from any
reward-modification (curiosity/RND) or trajectory-weight (PRISM,
SNELL, DUAL-IR, TEAR) variant of REINFORCE.

## Proof debt

1. **Convergence under annealed η.** Conjecture: under the schedule
   `η_t = σ((‖g_τ‖ − c)/c)` and Robbins-Monro step sizes, GRADCOMP
   converges to a stationary point of `J(π_θ)`. Strategy: in the
   regime `‖g_τ‖ ≫ c` (warm), the update is `α · ‖g_τ‖ · ĝ_τ + O(c)`,
   which is REINFORCE plus a vanishing perturbation. Standard
   stochastic-gradient-with-bounded-bias convergence applies. The
   open piece is: in the cold regime (`‖g_τ‖ ≤ c`), the update is
   `α · c · v₁(τ)`, which is biased but is also bounded in magnitude.
   The total bias accumulated over the cold-start phase is `O(c · N_cold)`
   where `N_cold` is the number of cold episodes. If `N_cold` is
   finite (the policy eventually leaves the floor), the total bias
   is finite and the warm-phase REINFORCE convergence absorbs it.
   Open: bound `N_cold` in expectation under what reward
   conditions.

2. **Fisher principal direction's correlation with reward direction
   at convergence.** Conjecture: at a local optimum `θ*`,
   `v₁(τ) → ĝ_τ` (perfect alignment), so the rotation has no effect.
   This would imply GRADCOMP and REINFORCE have the same fixed
   points. Open: prove the limit alignment under stationarity of the
   on-policy distribution.

3. **Effective-rank monotonicity in the cold phase.** Conjecture:
   along the cold-phase iterate `θ_n+1 = θ_n + αc v₁(τ_n)`, the
   expected effective rank `r_eff(E_τ[F̂_τ(θ_n)]) :=
   tr(E[F̂])² / tr(E[F̂]²)` is monotonically non-decreasing in `n`.
   This is the load-bearing structural claim: the cold-phase walk
   *increases policy expressivity* without reward signal. Strategy:
   show that walking in the top-eigenvector direction in parameter
   space increases the spread of the spectrum of `E[F̂]` by a Weyl
   inequality on rank-1 perturbations of symmetric matrices.

4. **Variance of the Gram-matrix top-eigenvector estimator.** The
   `v₁(τ)` estimator from a single rollout has variance scaling with
   `1/T` (number of score samples). Open: bound this variance and
   its propagation to the parameter update under the standard
   policy-gradient regularity conditions.

5. **First-reward-time bound.** Open: under a stochastic policy and
   GRADCOMP's cold-phase iterate on a deterministic-MDP sparse-reward
   problem (DoorKey-style), bound the expected episode index of the
   first non-zero return relative to the REINFORCE baseline. The
   informal claim is that the Fisher walk explores the reachable
   policy manifold faster than a stalled REINFORCE.

The empirical probe will reveal whether the rotation primitive
produces a measurably earlier first-reward-episode on the sparse
stage; positive observables (ii) and (iii) on DoorKey/KeyCorridor
would justify investing in proof items (3) and (5).
