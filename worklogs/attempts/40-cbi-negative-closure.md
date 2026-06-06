# 40 — CBI Negative Closure (structural-failure)

**Run:** 20260606-07-auto
**Closes seed:** 20260606-05-auto (CBI — Cohomological Bellman Iteration)
**Family:** H (new — algebraic-topology / cochain-complex value iteration)
**Verdict:** structural-failure

## What was ruled out

The CBI seed proposed to accelerate Bellman iteration by augmenting the value function V with a 1-cochain potential ψ on the policy graph's cochain complex, using the sheaf-Hodge operator to push cycle-structure information from ψ into V. The negative closure proves this is mathematically impossible.

**The obstruction:** the cochain identity `δ_1 ∘ δ_0 = 0` forces the (1,2) block `H_{12} = δ_0^T δ_1^T = (δ_1 δ_0)^T = 0` of the joint iteration operator H. The V-update receives zero contribution from ψ in every weighting and for every MDP. The joint operator M is block lower-triangular; its spectrum is the union of the V-only residual-gradient TD spectrum and the ψ-only cycle-Laplacian spectrum. On cyclic graphs, M is generically strictly *slower* than V-only TD, controlled by the spectral gap of `L_2 = δ_1 δ_1^T`.

## Theorem (as stated in the closure)

For any MDP and any step size α > 0:
- (a) `H_{12} ≡ 0`; no coexact correction reaches V.
- (b) `σ(M) = σ(I − α δ_0^T diag(w) δ_0) ∪ σ(I − α L_2)`.
- (c) `ρ(M) ≥ ρ(M_V)` always; CBI cannot improve the V-only convergence rate.
- (d) For cyclic policy graphs, CBI is generically strictly slower than V-only TD.

## Conclusion for the corpus

Any implementation of CBI would be residual-gradient TD (Baird 1995) plus an inert ψ side-buffer. No run is warranted. The cellular-sheaf / cochain-complex framing for Bellman-operator acceleration is dead by a fundamental cochain identity. This obstruction applies to any proposed approach that exploits `H_k(G_π)` (the k-th homology of the policy graph) via a cochain-complex coupling to accelerate V iteration.
