# 20260606-06-auto — Krein-Spectral Bellman (KSB) [seed]

(Open seed `20260606-05-auto` (CBI, sheaf-Hodge) was read in full;
not closeable this turn — its open question asks for a precise
spectral-gap inequality on the joint operator
`M = [I − α_V δ₀ᵀWδ₀, α_V δ₀ᵀδ₁ᵀ; −α_ψ δ₁Wδ₀, I − α_ψ L₂]`,
which requires a careful bilinear-form analysis under non-trivial
edge weights `W ≠ I` (the off-diagonal blocks are
`δ₀ᵀδ₁ᵀ` and `δ₁ W δ₀`, which are not adjoints of each other in
the natural μ ⊕ ν inner product unless the metric on `C²` is
chosen specifically). I tried but could not produce a
counterexample-free contraction proof or a sharp counterexample
in one turn, so I am not closing it as a fresh seed in a
distinct region.)

## Principle

Place the action-value function in a reproducing-kernel **Krein**
space whose indefinite inner product is the polar phase of the
policy-induced Markov operator, so that the Bellman optimality
operator becomes Krein-self-adjoint and Q\* admits a real
spectral decomposition over generalized eigenvectors of `γP^π`
without requiring reversibility.

## Derivation

Fix a stationary distribution `μ` and let `P^π` be the
policy-induced Markov kernel viewed as a bounded operator on
`L²(S; μ)`. Write its polar decomposition
`P^π = U |P^π|` where `|P^π| = ((P^π)*P^π)^{1/2} ≥ 0` and `U`
is the partial isometry on `(ker P^π)^⊥` [Kato 1980 §VI.7]. For
reversible chains `U = I`; in general `U ≠ I` precisely measures
the chain's time-asymmetry.

Define the Krein form `[f, g]_J := ⟨f, J g⟩_{L²(μ)}` with
`J := (U + U*)/2 · sgn`, the self-adjoint sign of the polar
phase (well-defined on the spectral subspaces of `(U+U*)/2`)
[Bognar 1974 §IV; Azizov–Iokhvidov 1989]. Let `K_J` denote the
reproducing-kernel Krein space (RKKS) on `S × A` whose kernel
satisfies
  `k((s,a), (s',a')) = ⟨φ(s,a), J φ(s',a')⟩`
for a feature map `φ` consistent with `J`'s signature
[Oglic–Gärtner 2018, "Learning in RKKS"; Loosli–Canu–Ong 2016].

The Bellman optimality operator `(T*Q)(s,a) = r(s,a) + γ E_{s'~p(·|s,a)}[max_{a'} Q(s',a')]`
admits the linearization-around-greedy form `T_π Q = r + γ P̂^π Q`
where `P̂^π(s,a; s',a') = p(s'|s,a) · 𝟙[a' = π(s')]` and π is the
greedy policy w.r.t. Q. Compute the Krein-adjoint `T_π^{[*]}`
defined by `[T_π f, g]_J = [f, T_π^{[*]} g]_J`. This requires
`(γ P̂^π)^{[*]} = J^{-1} (γ P̂^π)* J = γ P̂^π`, i.e.,
  `J P̂^π = (P̂^π)* J`.   (★)

Equation (★) holds when `J = sgn(U+U*)/2` is precisely the
polar phase of `P̂^π` because then `J P̂^π = J U|P̂^π| = |P̂^π|`
(self-adjoint) and `(P̂^π)* J = |P̂^π| U* J = |P̂^π|`
(also self-adjoint, using `U* J = J U` for unitary `U`
commuting with its real part). Under (★), `T_π` is Krein-
self-adjoint, and by the **Krein spectral theorem for
definitizable operators** [Langer 1982; Behrndt–Philipp 2020]
admits a real spectrum and a Krein-orthogonal eigenbasis
`{ψ_k}` with `T_π ψ_k = λ_k ψ_k`, `λ_k ∈ ℝ`. Bellman fixed
points then satisfy
  `Q* = Σ_k [r, ψ_k]_J / (1 − λ_k) · ψ_k`   (when `1 ∉ σ(T_π)`).

This expansion is *not* a Banach contraction iteration: it is a
direct spectral solve in the indefinite Krein metric, with real
eigenvalues guaranteed by Krein-self-adjointness despite
`P^π` being non-self-adjoint in `L²(μ)`.

## Primitive

The action-value function `Q ∈ K_J(S × A)` viewed as a vector
in the reproducing-kernel **Krein** space `K_J` whose indefinite
metric is the polar phase of the policy-induced Markov operator.
A single mathematical object: `Q` together with the implicit
Krein form `J` that is *determined by the MDP* (not a free
hyperparameter). The Bellman fixed-point equation
`(I − γ P̂^π) Q = r` is solved as a Krein-spectral problem.

## Update rule

```
input: MDP (S,A,P,r,γ), policy π, Krein feature map φ with signature J
init:  Q_0 ∈ K_J, eigenestimates {(λ_k, ψ_k)}_{k=1..K} = ∅
loop k = 0, 1, 2, ...:
    # 1. (Re)estimate polar phase J of policy-induced operator
    P̂_k    ← empirical P̂^{π_k} on (S×A) basis from rollouts
    U_k|P̂_k| ← polar decomposition         # closed-form on basis
    J_k    ← sgn((U_k + U_k*)/2)
    # 2. Krein-spectral solve: top-K Krein eigenpairs of γ P̂_k
    {(λ_j, ψ_j)} ← K-largest |[· , ·]_{J_k}|-Krein eigenpairs of γ P̂_k
                   (definitizable-operator Lanczos in Krein metric)
    # 3. Reconstruct Q from spectral expansion
    Q_{k+1}(s,a) ← Σ_j [r, ψ_j]_{J_k} / (1 − λ_j) · ψ_j(s,a)
    # 4. Greedy improvement
    π_{k+1}(s) ← argmax_a Q_{k+1}(s,a)
return Q_∞, π_∞
```

## Open question

**Is `(γ P̂^π)` definitizable as a Krein-symmetric operator
under the polar-phase Krein form `J = sgn((U+U*)/2)` on
`L²(S × A; μ ⊗ unif)`, and if so, is the Krein spectrum real
with finite Krein critical points (so that the spectral
expansion in §Derivation converges in Krein norm)?**

Concretely two sub-questions:

1. *Self-adjointness condition (★)*: For which classes of
   policy-induced operators `P̂^π` does the polar phase
   `J = sgn((U+U*)/2)` satisfy `J P̂^π = (P̂^π)* J`? The
   computation in §Derivation assumed `U* J = J U` and
   commutativity of `U` with its real part — does this hold
   for all finite-state ergodic chains, only reversible-up-to-
   gauge ones, or does it fail generically?

2. *Definitizability*: Even when (★) holds, is `T_π` definitizable
   in the Krein sense — i.e., does there exist a real polynomial
   `p` such that `[p(T_π) f, f]_J ≥ 0` for all `f`? If yes, the
   Langer spectral theorem gives the expansion. If no
   (e.g., infinitely many Krein critical points), the expansion
   is at best formal and the principle collapses to a
   non-convergent infinite series.

If both hold for a non-trivial class of MDPs (e.g., all
ergodic finite chains, or all chains with bounded
non-reversibility `‖P − P*‖ ≤ c < 1`), KSB gives a *direct
spectral solve* of the Bellman equation that bypasses iterative
contraction and inherits the convergence rate of a generalized
eigenvalue problem rather than `γ^k`. If neither holds —
i.e., generic non-reversible MDPs are not definitizable under
the polar-phase Krein form — then the Krein-spectral approach
collapses to standard `L²(μ)` spectral theory of `(P + P*)/2`,
which is just the reversible-symmetrization approach (Diaconis–
Saloff-Coste) and is not new.

## Why this is not [closest published method]

Closest published methods are: (a) **kernel-based RL with RKHS**
[Ormoneit–Sen 2002; Taylor–Parr 2009; Nguyen-Tuong–Peters 2011;
Bhat et al. 2024 nonparametric Bellman mappings, arXiv 2403.20020],
which place Q in a *positive-definite* RKHS — these methods
must symmetrize the Bellman operator (residual-gradient or
projected fitted-Q) and lose the asymmetric information of `P^π`;
(b) **proto-value functions / spectral RL** [Mahadevan–Maggioni
2007; Stachenfeld et al. 2017], which use eigenvectors of the
*reversibilized* graph Laplacian `(P + P*)/2`, again discarding
asymmetry; (c) **RKKS / indefinite kernel learning**
[Oglic–Gärtner 2018; Loosli–Canu–Ong 2016], which is a general
machine-learning framework with no RL-specific application or
Bellman content; (d) **stochastic-duality and eigenfunctions of
non-reversible Markov chains** [Carinci–Giardinà–Redig 2018,
arXiv 1805.01318], which studies eigenfunctions of `P` for
duality and intertwining purposes, not for value iteration. KSB's
structural difference is that the *Krein form `J` is determined
by the MDP itself* (the polar phase of `P^π`, not a free design
choice) and the resulting `T*` becomes Krein-self-adjoint —
making the Bellman fixed-point problem a generalized eigenvalue
problem rather than a Banach-iterative one. None of (a)–(d)
identify or use this MDP-intrinsic Krein structure.
