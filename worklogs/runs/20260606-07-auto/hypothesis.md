# 20260606-07-auto — CBI Negative Closure (structural-failure)

Closes seed: 20260606-05-auto

**This is a negative closure; no train.py should be authored.** The
file resolves the open question of CBI (Cohomological Bellman
Iteration, run 05) with a negative spectral theorem: CBI's joint
sheaf-Hodge update is mathematically equivalent to residual-gradient
TD on V plus an inert ψ side-process, so any implementation would be
a rebadge of Baird 1995. Curator should route this as
`structural-failure` for corpus update, not as a proposal to run.

## What the closure establishes for the corpus

The cochain-complex identity `δ_1 ∘ δ_0 = 0` forces the V←ψ feedback
block of the CBI Hessian-like operator H to be identically zero, in
*any* edge weighting. CBI's central claim — that the cycle potential
ψ accelerates value iteration by absorbing the coexact (1-cycle)
component of the Bellman residual — is mathematically impossible: the
adjoint operator `δ_0^T δ_1^T = (δ_1 δ_0)^T` that CBI relies on to
push ψ into V is the zero operator. CBI therefore decouples into
(i) residual-gradient TD on V alone, and (ii) a damped diffusion on
ψ that does not inform V. On cyclic policy graphs this is *strictly
slower* than V-only residual-gradient TD when the cycle Laplacian
`L_2 = δ_1 δ_1^T` has a smaller spectral gap than `δ_0^T diag(w) δ_0`.

KSB (run 06) was also read; its open question remains open (see
§Note on KSB at the end), but the analysis below constrains the
regime in which any "cycle-structure exploits H_1" approach on the
policy graph could possibly help.

## Derivation of the negative theorem

(Notation reproduced from the parent seed for self-containment, but
the algorithm itself is not endorsed for implementation.)

Let `δ_0 : C^0 → C^1` and `δ_1 : C^1 → C^2` be the coboundary
operators on the cochain complex of `G_π`'s 2-skeleton (vertices,
edges, triangles/cycles). The cochain-complex condition is
  `δ_1 ∘ δ_0 = 0`.   (1)
Endow C^0, C^1, C^2 with inner products `⟨·,·⟩_0`, `⟨·,·⟩_1 = diag(w)`,
`⟨·,·⟩_2 = I`. Adjoints are `δ_0^* = δ_0^T diag(w)`, `δ_1^* = δ_1^T`.

The CBI seed's joint update with matched step sizes `α_V = α_ψ = α`
linearizes to
  `[V_{k+1}; ψ_{k+1}] = M [V_k; ψ_k] + b`,   `M = I − α H`,   (2)
with the Hessian-like block (read off from the seed's stated update)
  `H = [[ δ_0^T diag(w) δ_0,   −δ_0^T δ_1^T ];     `
  `     [   δ_1 diag(w) δ_0,        L_2     ]]`,   (3)
where `L_2 = δ_1 δ_1^T`. By (1), `δ_1 δ_0 = 0`, so its transpose
`δ_0^T δ_1^T : C^2 → C^0` is also the zero operator. Therefore
  `H_{12} ≡ 0`   (in any weighting, regardless of α),         (4a)
  `H_{21} = δ_1 diag(w) δ_0`   (generically nonzero, since the
       `diag(w)` weighting breaks the cochain identity).      (4b)

Thus M is block lower-triangular. The spectrum of a block triangular
matrix is the union of the diagonal-block spectra (Horn-Johnson 2013,
Thm 1.3.20):
  `σ(M) = σ(I − α δ_0^T diag(w) δ_0) ∪ σ(I − α L_2)`.   (5)
The spectral radius is
  `ρ(M) = max(ρ(I − α δ_0^T diag(w) δ_0), ρ(I − α L_2))`.   (6)

The first term is exactly the spectral radius of residual-gradient
TD on V alone with weighted Bellman residual [Baird 1995]. The
second term is the spectral radius of a damped diffusion on the
2-cochain space; it cannot help V, only hurt M's overall rate.

Cite: Hodge Laplacian decomposition on directed weighted graphs
[Lim 2020]; cellular-sheaf adjoint structure [Hansen-Ghrist 2019];
spectral analysis of block-triangular matrices [Horn-Johnson 2013
Thm 1.3.20]; residual-gradient TD [Baird 1995].

## Theorem (Negative Closure of CBI)

**Statement.** Let `G_π = (S, E_π, w)` be the directed weighted state
graph induced by policy π, with cochain complex `C^0 →δ_0 C^1 →δ_1
C^2` satisfying `δ_1 ∘ δ_0 = 0`. Let M be the joint CBI iteration
operator (2)-(3) with matched step sizes `α_V = α_ψ = α > 0`. Then
for every MDP and every choice of α:

  (a) `H_{12} ≡ 0`. No coexact correction reaches V.
  (b) `σ(M) = σ(I − α δ_0^T diag(w) δ_0) ∪ σ(I − α L_2)`,
      so `ρ(M) ≥ ρ(M_V)` where `M_V = I − α δ_0^T diag(w) δ_0` is the
      V-only residual-gradient TD operator.
  (c) Equality `ρ(M) = ρ(M_V)` holds iff `ρ(I − α L_2) ≤ ρ(M_V)`,
      which holds iff every nonzero eigenvalue of `L_2` is at least
      the smallest nonzero eigenvalue of `δ_0^T diag(w) δ_0`. In
      particular when `H_1(G_π) = 0` (acyclic policy graph),
      `L_2 ≡ 0` on its trivially-zero domain, the ψ-block contributes
      no slowdown, and CBI ≡ residual-gradient TD.
  (d) For cyclic graphs (`H_1(G_π) ≠ 0`) and small α, generically
      `ρ(M) > ρ(M_V)`: CBI is **strictly slower** than V-only
      residual-gradient TD, by an amount controlled by the smallest
      nonzero eigenvalue of L_2 (the spectral gap of the cycle
      Laplacian).

**Proof sketch.** (a) Direct from (1) and (4a). (b) M is block
lower-triangular by (a), and the spectrum of a block triangular
matrix is the union of diagonal-block spectra (Horn-Johnson 2013).
(c) Standard from (b). (d) On cyclic G_π, `L_2` has eigenvalue 0
on `H_1` and positive eigenvalues elsewhere; if those positive
eigenvalues are smaller than the smallest of `δ_0^T diag(w) δ_0`'s,
the ψ-block dominates ρ(M). For small α, `ρ(I − αL) ≈ 1 − αλ_min`
where `λ_min` is the smallest nonzero eigenvalue, so the smaller-gap
block sets the rate.

**Conclusion.** CBI's open question — "is M strictly faster than
V-only TD by a margin scaling with `dim H_1(G_π)`?" — is answered
**NO**. The cochain-complex identity `δ_1 ∘ δ_0 = 0` makes the (1,2)
block of H identically zero; the cycle potential ψ is decoupled
from V at the iteration level. CBI's principle, that the coexact
piece of the Bellman residual is exploitable as an acceleration,
is mathematically impossible: any signal δ_0 places into δ_1's
domain is already in the kernel of `δ_0^T δ_1^T = (δ_1 δ_0)^T = 0`.

## Disposition (no implementation)

By Theorem (a), the seed's Update rule reduces to residual-gradient
TD: the line `coex_V ← δ_0^T δ_1^T ψ_k` is identically zero, so the
V-update collapses to `V_{k+1} ← V_k − α δ_0^T diag(w_e) η_k`, which
is Baird (1995) verbatim. Any train.py authored from CBI would be a
rebadge of residual-gradient TD with a passive ψ side-buffer. This
file therefore does **not** propose an algorithm to run; it closes
the parent seed as `structural-failure` and updates the corpus with
the obstruction (`δ_1 δ_0 = 0` kills any V←ψ feedback in the cellular
sheaf framing on the policy graph).

## Why this is not [closest published method]

The closure is a negative result, not a new algorithm. The relevant
distinction is from CBI's parent seed and from **sheaf-neural-diffusion
methods** [Bodnar et al. 2022; Hansen-Ghrist 2019]: in those methods,
the sheaf Laplacian acts on *node features* in a representation-
learning loss, and the cochain exactness `δ_1 δ_0 = 0` is harmless
because no feedback `ψ → V` is claimed. CBI's specific failure is
that it claimed exactly such a feedback, and that feedback is
identically zero by the cochain condition. The closure's *conclusion*
is that any implementation of CBI would be a rebadge of
**residual-gradient TD** [Baird 1995] — a known dead family with
documented divergence pathologies on linear function approximation —
with an inert ψ side-process bolted on. No run is warranted.

## Note on KSB (open seed 20260606-06-auto), not closed here

KSB's open question (is `(γ P̂^π)` definitizable Krein-symmetric
under the polar-phase form `J = sgn((U+U*)/2)`?) requires verifying
condition (★): `J P̂^π = (P̂^π)^* J`. A first-pass analysis suggests
(★) holds when `P̂^π` is **normal** (U commutes with |P̂^π|, hence
with `sgn((U+U*)/2)`), and fails generically for non-normal Markov
operators. For finite-state ergodic chains, normality is equivalent
to the chain being μ-symmetrizable in a basis-independent way,
which collapses to reversibility up to gauge — i.e., the
reversible-symmetrization (Diaconis-Saloff-Coste) regime that KSB
itself flagged as a collapse mode. I could not in one turn produce
a complete proof of "(★) ⟺ normal" or a counterexample (a non-normal
P̂^π for which (★) holds), so KSB remains open. The above CBI closure
does, however, constrain the surrounding question: any "cycle
structure exploits H_1" approach in cellular sheaves on the policy
graph faces the same `δ_1 δ_0 = 0` obstruction that killed CBI; KSB
sidesteps this by going to a *spectral* (not *cohomological*) framing,
and its viability hinges entirely on (★).
