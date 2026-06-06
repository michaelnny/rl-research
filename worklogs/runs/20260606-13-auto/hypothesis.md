# 20260606-13-auto -- LYRA (Lyapunov Reward Asymmetry) [probe]

## Principle

The optimal policy is characterized as the maximizer of the *top
Lyapunov-exponent gap* `Δ(π) := λ₁(π) − λ₂(π)` of the reward-tilted
policy-cocycle `M_t = exp(β r(s_t, a_t)) · γ P^π`, and policy
improvement is implemented as direct gradient ascent on `Δ(π)`
estimated online via QR factorization of perturbation vectors carried
along a single rollout.

## Primitive

The **policy-indexed Oseledec leading-pair frame**

  `Φ : Π × Ω → St(2, ℝ^{d})`

mapping `(π, trajectory ω)` to the orthonormal 2-frame
`(u_1(π, ω), u_2(π, ω)) ∈ ℝ^{d} × ℝ^{d}` whose Lyapunov
exponents along the cocycle `M_t(ω) = exp(β r(s_t, a_t)) · γ P^π_{s_t →·}`
are `λ_1(π) ≥ λ_2(π)`. Codomain is the Stiefel manifold of orthonormal
2-frames in the state-feature space `ℝ^{d}` (tabular: `d = |S|`,
Φ = identity; otherwise a random/learned encoder Φ : obs → ℝ^d).
One typed object: a 2-frame in a Stiefel manifold that is *stationary
in distribution* under the cocycle action by the multiplicative ergodic
theorem (Oseledec 1968; Ruelle 1979).

## Derivation sketch

1. **Cocycle.** Fix a policy π, discount γ ∈ (0,1), and tilt β > 0.
   Along a trajectory `(s_0, a_0, s_1, …)` under π, define the
   row-stochastic-row-tilted matrix-valued cocycle
   `M_t := γ · diag(e^{β r(·, a_t)}) · P^π`,
   where `P^π_{s,s'} = Σ_a π(a|s) p(s'|s,a)`. The product `Π_{k=0..t} M_k`
   acts on functions `f : S → ℝ` and represents the reward-tilted
   discounted forward-evolution operator under π.

2. **Multiplicative ergodic theorem.** Under stationarity of the
   Markov chain `(s_t)` with invariant measure μ_π and `E_{μ_π}[log⁺‖M_0‖] < ∞`,
   Oseledec's theorem gives the existence almost surely of Lyapunov
   exponents `λ_1(π) ≥ λ_2(π) ≥ …` and an Oseledec splitting
   `ℝ^{d} = E_1(ω) ⊕ E_2(ω) ⊕ …` such that for `v ∈ E_i(ω) \ E_{i+1}(ω)`,
   `lim (1/t) log ‖M_t … M_0 v‖ = λ_i(π)` ω-a.s.

3. **Gap functional.** Define the policy-indexed Lyapunov gap
   `Δ(π) := λ_1(π) − λ_2(π) ≥ 0`. By Oseledec's filtration, Δ(π) > 0
   precisely when the leading exponent is *simple* — i.e., the
   reward-tilted cocycle has a unique dominant growth direction. By
   the **Furstenberg-Kifer formula** (Furstenberg-Kifer 1983; Guivarc'h-
   Raugi 1985 for stationary cocycles), Δ(π) controls the
   *exponential rate of memory loss* of the cocycle: large gap = fast
   alignment of any initial vector with `u_1(π)` along the trajectory.

4. **Why gap, not exponent.** The top exponent λ_1(π) alone equals
   `log ρ(γ P^π · diag(e^{β r̄_π}))` where `r̄_π(s) = Σ_a π(a|s) r(s,a)`,
   which by the **Donsker-Varadhan variational principle** is the
   risk-sensitive return at temperature β (Howard-Matheson 1972;
   Anantharam-Borkar 1986). Maximizing λ_1 alone collapses to risk-
   sensitive RL, *published*. The novelty is the gap Δ: it is **not** a
   risk functional, and it is **not** a value functional. It is a
   *spectral coherence* of the reward-induced flow under π.

5. **Gap maximization principle (proof debt).** Conjecture: at fixed
   β, the policy `π_β^* := argmax_π Δ(π)` satisfies (i) `π_β^* → π_*`
   as β → ∞ on finite ergodic MDPs where π_* is unique-greedy at
   every state, and (ii) for finite β, π_β^* is the policy that
   maximally concentrates reward-tilted trajectory mass along a
   single dominant mode in the reward landscape. Heuristic
   justification: maximizing Δ = λ_1 - λ_2 forces the second
   eigenmode to grow strictly slower than the first, which means the
   cocycle's range collapses to the leading direction faster — the
   reward-tilted occupancy concentrates on high-reward
   sub-state-spaces. Counter-example check: on degenerate MDPs where
   λ_1 = λ_2 (e.g., decoupled-component chains), Δ ≡ 0 for every π
   and the principle is uninformative; this is recorded as a known
   degeneracy, not a bug.

6. **Online estimation via QR.** The leading-pair frame `(u_1, u_2)`
   is computed online by the **Benettin-Galgani-Strelcyn QR algorithm**
   (Benettin et al. 1980; Dieci-Russell-Van Vleck 1997 for stochastic
   cocycles): maintain a 2-frame `Q_t ∈ St(2, d)`; at each step,
   `Z_{t+1} := M_t Q_t`; `(Q_{t+1}, R_{t+1}) := qr(Z_{t+1})`;
   accumulate `log R_{t+1}[i,i]` to estimate `λ_i`. Convergence is
   classical: `(1/T) Σ log R_t[1,1] → λ_1` and similarly for λ_2,
   ω-a.s. (Goldsheid-Margulis 1989).

7. **Policy gradient on Δ.** Parameterize π_θ. By the spectral
   perturbation lemma (Kato 1980 §II.2.6) and the chain rule on the
   Stiefel manifold, `∇_θ λ_i = E[u_i^T (∂M / ∂θ) u_i]` (when λ_i
   is simple). Hence `∇_θ Δ = E[u_1^T (∂M / ∂θ) u_1 − u_2^T (∂M / ∂θ) u_2]`,
   computable along a rollout from the current QR estimate of
   (u_1, u_2). The single-sample sample-path estimator approximates
   `(M_t v)(s) ≈ γ exp(β r_t) ⟨Φ(s_{t+1}), v⟩ Φ(s_t)` (rank-1
   stochastic estimator of the operator action; analogous to TD's
   single-sample Bellman estimator and flagged as proof debt). Under
   π_θ, `∂M_t/∂θ` enters via the score-function identity on `P^π`,
   yielding `∂(M_t v)(s)/∂θ ≈ γ exp(β r_t) ⟨Φ(s_{t+1}), v⟩ Φ(s_t) ·
   ∇_θ log π_θ(a_t|s_t)` along the realized transition. This is the
   **score-function form** of the Lyapunov-gap gradient.

## Update rule

```
Inputs: env, β > 0, γ < 1, learning rate α, frame dim k = 2
Init:   policy params θ; orthonormal frame Q ∈ ℝ^{d × 2}, columns u1, u2
        log-exponent accumulators L = [0, 0]; step count T = 0

For each rollout step t (a_t ~ π_θ(·|s_t), s_{t+1} ~ P, scalar r_t):
    # 1. Local cocycle action on the current 2-frame, using the
    #    rank-1 sample-path estimator M_t v ≈ γ exp(β r_t) ⟨Φ(s_{t+1}), v⟩ Φ(s_t)
    phi_s   = Φ(s_t)            # ℝ^d
    phi_sp  = Φ(s_{t+1})        # ℝ^d
    tilt    = γ * exp(β * r_t)  # scalar
    Z       = tilt * outer(phi_s, phi_sp.T @ Q)   # d × 2; columns are tilt·⟨phi_sp, u_i⟩·phi_s

    # 2. QR step: track leading 2-frame and accumulate log singular values
    Q_new, R = qr(Z)                  # Q_new ∈ St(2, d), R upper-tri 2×2
    L += log(abs(diag(R)))            # running sums for λ_1, λ_2
    u1, u2   = Q_new[:, 0], Q_new[:, 1]

    # 3. Spectral-perturbation gradient: per-parameter scalar coefficient
    #    g_i = γ exp(β r_t) · (u_i^T Φ(s_t)) · (Φ(s_{t+1})^T u_i) for i ∈ {1,2}
    score_t = ∇_θ log π_θ(a_t | s_t)  # parameter-shaped vector
    c1 = tilt * (u1 @ phi_s) * (phi_sp @ u1)   # scalar
    c2 = tilt * (u2 @ phi_s) * (phi_sp @ u2)   # scalar

    # 4. Gradient ascent on the gap Δ = λ_1 − λ_2
    θ ← θ + α * (c1 − c2) * score_t
    Q ← Q_new
    T += 1

# Periodic logging
λ_1, λ_2 ← L / T   # current Lyapunov estimates
gap_t   ← λ_1 − λ_2
```

The cocycle action on a feature vector `v ∈ ℝ^d` is
`(Mv)(·) ≈ γ exp(β r) · ⟨Φ(s'), v⟩ · Φ(s)` where Φ is a fixed
state-feature map (e.g., random-features, tile-coding, or a small
learned encoder). For tabular environments d = |S|, Φ = identity, and
the estimator is exact at the per-step level.

## Empirical claim

stage: sparse
claim: On the sparse stage (MiniGrid DoorKey + KeyCorridor), LYRA's
gap-ascent should produce a policy whose mean episodic return
improves *faster* than a matched-architecture REINFORCE baseline,
with the empirical Lyapunov gap Δ_t increasing monotonically over
training. Sparse-reward environments exercise the principle because
the multiplicative tilt `exp(β r)` makes rare positive-reward
transitions dominate the cocycle's growth direction, and
gap-ascent should concentrate occupancy on the rare-reward mode.
The sign of success is beat_strong = 1 on at least one sparse env
within the stage time budget.

falsifier: If the empirical Lyapunov gap Δ_t fails to increase
during training (i.e., `Δ_T ≤ Δ_0` consistently), or if it does
increase but episodic return does not improve correspondingly, the
principle (gap maximization → policy improvement) is falsified on
this substrate. Likewise, if the ablation (random 2-frame replacing
QR-tracked frame) performs as well, the Oseledec primitive is
decorative.

## Ablation plan

Replace the **online QR-tracked Oseledec 2-frame** with a fixed
**random orthonormal 2-frame** Q ∈ ℝ^{d × 2} resampled at the start
of training and never updated. The score-function gradient computation
proceeds identically using this random frame in place of (u_1, u_2):
`c_i = γ exp(β r_t) · (q_i^T Φ(s_t)) · (Φ(s_{t+1})^T q_i)` with
`q_i` the frozen random columns. If the random-frame ablation matches
LYRA's learning curve, the Lyapunov primitive is not load-bearing —
the algorithm collapses to "REINFORCE projected onto a random
2-frame," which is a known variance-reduction trick at best. If the
gap-ascent direction genuinely depends on the leading Oseledec
eigenpair, the random-frame version should be strictly worse, and
ideally not improve on the empirical gap Δ_t at all.

Implementable in `train_ablate.py` by replacing `(Q_new, R) = qr(Z)`
with a no-op (Q stays at the random initialization) and recomputing
the gradient with the frozen Q.

## Novelty boundary

Closest published methods:

(a) **Risk-sensitive RL via Perron eigenvalue / Howard-Matheson
    1972**: optimizes λ_1 alone (the top Lyapunov exponent of the
    reward-tilted cocycle), which is the risk-sensitive return.
    Structural difference: LYRA optimizes the *gap* Δ = λ_1 − λ_2,
    not λ_1 itself. The gap is not a risk functional and reduces to
    something different at β → 0 (Δ → spectral gap of γ P^π, which
    is the mixing rate, not the return).

(b) **Lyapunov-exponent edge-of-chaos training** (Cohen et al. 2021
    on edge-of-stability, Pennington-Worah-Schoenholz 2017 for deep
    nets): tracks Lyapunov exponents of the *parameter-update Jacobian
    cocycle* to characterize training dynamics. Different cocycle
    (parameter-space, not state-space) and different objective (set
    λ_max ≈ 0, not maximize gap).

(c) **Oja's rule / streaming PCA on rollouts** (Oja 1982): online
    leading-eigenvector tracking via gradient on Rayleigh quotient.
    LYRA tracks a 2-frame on a non-symmetric reward-tilted cocycle,
    which is QR-based not Rayleigh-based, and the objective is the
    *gap*, not the leading direction.

(d) **Spectral RL / proto-value functions** (Mahadevan-Maggioni 2007):
    eigendecomposition of the *symmetrized* graph Laplacian
    `(P + P^T)/2`. LYRA uses the *non-symmetric* reward-tilted
    operator γ P^π · diag(e^{β r}) and the non-symmetric Lyapunov
    spectrum, which exists for all (not just reversible) chains.

(e) **Risk-sensitive policy gradient** (Borkar 2001; Mihatsch-Neuneier
    2002): optimizes E[exp(β · return)] via score-function. This is
    optimization of λ_1, not Δ.

The structural difference is uniform: LYRA's optimization target is
the *spectral gap* of a non-symmetric cocycle, which is neither a
return functional, nor a risk functional, nor a mixing-rate functional
in any of the above named ways. The gap is a policy-indexed measure
of "reward-flow coherence."

Nearest dead family from `prior_attempts.md`: none directly. The
primitive is not a bucketed tensor (Family A), not a pairwise-
trajectory comparison (B), not a within-trajectory geometric
statistic (C — Lyapunov is a spectral *limit* of an operator
cocycle, not a geometric path statistic; though if the empirical
estimator turns out to behave like a path statistic, this could be
re-classified). It is not a value-vocabulary swap (E) — there is no
Q or V variable updated; only the policy and the frame. It is not a
mechanism stack (G) — one primitive, the 2-frame.

## Proof debt

The following theorem statement is the open question to pursue if
empirical signal appears:

**Conjecture (LYRA improvement).** Let `π_β^* := argmax_π Δ(π, β)` on
the simplex of stationary policies for a finite ergodic MDP with
reward `r : S × A → ℝ`, discount γ ∈ (0,1), and tilt β > 0. Then:

(a) `π_β^*` is unique on the open dense subset of MDPs where
    `λ_1(π_β^*) > λ_2(π_β^*)` strictly;
(b) (β → ∞ limit) `lim_{β → ∞} π_β^* = π_*` whenever the optimal
    policy π_* is unique-greedy at every state;
(c) (Oseledec direction = value gradient) `u_1(π_β^*)` is parallel,
    in the L²(μ_π) sense, to the value-gradient direction `∇_π V_β^π`
    at π = π_β^*;
(d) (Gap-monotone improvement) Define the LYRA flow
    `dπ/dt = ∇_π Δ(π, β)`. Along this flow, the risk-sensitive return
    `J_β(π) := (1/β) log E_π[exp(β · return)]` is non-decreasing.

(d) is the load-bearing improvement theorem. (a)-(c) are structural
prerequisites. The proof strategy would draw on (i) Furstenberg-Kifer
1983 for the differentiability of Lyapunov exponents under stationary
cocycle perturbations; (ii) Kato perturbation theory for the
spectral-gap derivative; (iii) Donsker-Varadhan variational principle
to relate λ_1 to risk-sensitive return; and (iv) a novel lemma
(unproven) connecting `∇Δ` to a positive multiple of the score-function
risk-sensitive policy gradient on the support of the leading
Oseledec direction.

A separate proof obligation is to bound the bias of the rank-1
sample-path estimator
`(M_t v)(s) ≈ γ exp(β r_t) ⟨Φ(s_{t+1}), v⟩ Φ(s_t)`
relative to the true operator action; this is analogous to but
distinct from the standard TD-style single-sample bias analysis,
since the operator is non-self-adjoint and the QR step couples
across the frame.

The key open *negative* possibility: there may exist MDPs where the
gap-maximizing policy is *strictly suboptimal* in the risk-sensitive
sense (counterexample to (d)), in which case LYRA optimizes the wrong
target and would need β-annealing to recover π_*. The empirical probe
will hint at whether this counterexample lives in the panel substrate
or only at the abstract MDP level.
