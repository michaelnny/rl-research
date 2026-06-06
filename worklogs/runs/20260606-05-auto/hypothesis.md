# 20260606-05-auto — Cohomological Bellman Iteration (CBI) [seed]

## Principle

Treat Bellman optimality as the requirement that the value function be
a global section of a cellular sheaf on the state-transition graph,
and iterate by jointly evolving the value 0-cochain and a *circulation
1-cochain* drawn from the Hodge decomposition of the Bellman residual,
so that convergence is driven by both the symmetric (gradient-residual)
and antisymmetric (cycle-circulation) parts of `(I − γ P^π)`.

## Derivation

Fix a policy π and let `G = (S, E_π)` be the directed state graph with
edge weights `w_{ss'} = π(a|s) · p(s'|s,a) · μ(s)` summed over actions
`a`. Build a cellular sheaf `B` on G: 0-stalks `B(s) = ℝ`, 1-stalks
`B(e) = ℝ`, with restriction maps for `e = (s → s')`:

  `ρ_s^{(e)}(V) = V(s) − r̄(s,s')`,    `ρ_{s'}^{(e)}(V) = γ V(s')`,

where `r̄(s,s')` is the policy-and-transition-averaged reward on edge
e. The coboundary
  `(δV)(e) = ρ_{s'}^{(e)} V − ρ_s^{(e)} V = γ V(s') − V(s) + r̄(s,s')`
is exactly the per-edge TD residual. A 0-cochain V is a global section
iff `δV = 0`, i.e. iff V satisfies the policy-π Bellman equation [Hansen
& Ghrist 2019, sheaf Laplacians; Curry 2014].

Inner products `⟨·,·⟩_0 = Σ_s μ(s) (·)(s)²` and
`⟨·,·⟩_1 = Σ_e w_e (·)(e)²` give an adjoint `δ* : C^1 → C^0`. By Hodge
on a finite directed weighted graph [Lim 2020 "Hodge Laplacians on
graphs"], every 1-cochain decomposes orthogonally:

  `η = δ φ + δ* ψ + h`,    `h ∈ ker(L_1) = ker(δ) ∩ ker(δ*)`,

where `L_1 = δ δ* + δ* δ`, `φ ∈ C^0`, `ψ ∈ C^2` (a 2-cochain on the
triangle/cycle complex of G). The Bellman residual `η = δV − (−r̄)`
viewed as a 1-cochain admits this decomposition. Standard residual-
gradient TD [Baird 1995] descends only `‖δ*η‖_0 = ‖δ* δ V + δ* r̄‖`,
which projects out `δ* ψ` and `h`. The coexact piece `δ* ψ` is the
*directed circulation* of value around cycles of G — the antisymmetric
part of `(I − γ P^π)` that couples to `H_1(G)`.

CBI's update is the joint sheaf-flow

  `dV/dt    = −δ* (δV − (−r̄)) + δ* δ* ψ`     (V drives down exact part,
                                                  receives coexact push)
  `dψ/dt    = −L_2 ψ + δ_1 (δV − (−r̄))`       (ψ accumulates curl)

with `δ_1 : C^1 → C^2` and `L_2 = δ_1 δ_1*`. Steady states satisfy
`δV = −r̄` and `δ_1(δV − (−r̄)) = 0`, i.e. V is the Bellman fixed point
and ψ is the harmonic representative of the circulation absent at
optimum.

## Primitive

The pair `(V, ψ) ∈ C^0(G; B) × C^2(G; B)` — a value 0-cochain and a
cycle potential 2-cochain on the Bellman sheaf — jointly evolving
under the sheaf Hodge flow. ψ is determined by V via the Hodge
condition at steady state, but is an independent dynamical variable
during iteration; it is the single new mathematical object the
algorithm tracks beyond V.

## Update rule

```
input: MDP (S,A,P,r,γ), policy π, step sizes α_V, α_ψ
init:  V_0 ∈ ℝ^S,  ψ_0 ∈ ℝ^{triangles(G)}
build: edge weights w_e = π(a|s) p(s'|s,a) μ(s); coboundaries δ_0, δ_1
loop k = 0, 1, 2, ...:
    η_k        ← δ_0 V_k + r̄                 # signed Bellman residual
    grad_V     ← δ_0^T diag(w_e) η_k          # symmetric (exact) drive
    coex_V     ← δ_0^T δ_1^T ψ_k              # antisym (coexact) push
    grad_ψ     ← L_2 ψ_k − δ_1 (diag(w_e) η_k)
    V_{k+1}    ← V_k − α_V (grad_V − coex_V)
    ψ_{k+1}    ← ψ_k − α_ψ grad_ψ
return V_∞, ψ_∞
```
At convergence `η_∞ = 0` (Bellman) and `δ_1 η_∞ = 0` is automatic.
Outer loop wraps standard greedy improvement on V_∞.

## Open question

Is the joint linear operator
  `M(α_V, α_ψ) := [[ I − α_V δ_0^T diag(w) δ_0,   α_V δ_0^T δ_1^T ],
                    [   −α_ψ δ_1 diag(w) δ_0,         I − α_ψ L_2  ]]`
on `C^0 ⊕ C^2` a strict contraction in the
`(⟨·,·⟩_0 ⊕ ⟨·,·⟩_2)`-norm under the same step-size conditions for
which residual-gradient TD on `V` alone is a contraction, and is its
contraction rate strictly bounded above by the rate of residual-
gradient TD by an amount that scales with `dim H_1(G_π)` (the cycle
rank of the policy-induced state graph)? Concretely: does the
spectral gap of M dominate that of the V-only block by a margin
controlled by the smallest nonzero eigenvalue of `L_2`, and does this
margin vanish when `H_1(G_π) = 0` (acyclic policy graphs), recovering
the V-only rate in that limit?

If yes, CBI is provably faster than residual-gradient TD on cyclic
MDPs and equivalent on acyclic ones, with the speedup magnitude tied
to a topological invariant of the policy graph. If no — i.e. M can be
worse-conditioned than the V-only block for some MDP — then the
coexact correction is at best a heuristic and the principle collapses
to residual-gradient TD plus a side process.

## Why this is not [closest published method]

Closest published methods are (a) **residual-gradient TD** [Baird
1995], which descends `‖δV − (−r̄)‖²` in V alone — projecting out the
coexact and harmonic components of the residual; (b) **sheaf neural
networks** [Hansen-Ghrist 2019; Bodnar et al. 2022 "Neural Sheaf
Diffusion"], which use sheaf Laplacians for *graph representation
learning* with no Bellman / RL operator content; (c) the recent
**sheaf-theory-for-multi-agent-RL prospectus** [Schmid 2025, arXiv
2504.17700], which is explicitly a roadmap with no concrete algorithm,
no defined operator, and no theorem; (d) **Hodge decomposition of
Markov chains** [Jiang-Lim-Yao-Ye 2011 "Statistical ranking and
combinatorial Hodge theory"], which decomposes pairwise comparison
flows but is not coupled to a Bellman update or applied to value
iteration. CBI's structural difference is that the *primitive itself*
is the sheaf cochain pair `(V, ψ)` — the cycle potential ψ enters the
fixed-point equation as a co-evolving variable, not as a post-hoc
diagnostic of V's residual. This changes the iteration operator's
spectral structure on cyclic MDPs in a way none of (a)-(d) attempt.
