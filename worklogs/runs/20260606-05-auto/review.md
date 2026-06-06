---
verdict: reject
reviewer_run: 20260606-05-auto
hypothesis_type: seed
---

## Summary

CBI proposes a sheaf-cohomology-based joint flow on (V, ψ) for policy
evaluation; the derivation contains two load-bearing math errors that
invalidate the principle as stated.

## Math check

**Step 1 — sheaf construction.** Restrictions
`ρ_s^{(e)}(V) = V(s) − r̄(s,s')` and `ρ_{s'}^{(e)}(V) = γ V(s')` give
coboundary `(δV)(e) = γV(s') − V(s) + r̄(s,s')`. This algebraic step is
fine.

**Step 2 — "δV = 0 iff V satisfies the policy-π Bellman equation."**
This is **false**. δV = 0 says
  `V(s) = r̄(s,s') + γV(s')`  for every edge e = (s→s')
i.e. per-edge equality. The policy-π Bellman expectation equation is
  `V(s) = E_{a~π, s'~p}[r(s,a,s') + γV(s')]`
i.e. the *expectation* over s' equals V(s). For any stochastic MDP
where state s has two successors s'_1 ≠ s'_2 with non-degenerate r̄ or
V values, the per-edge system is overdetermined and has no solution,
yet the Bellman equation has a unique solution. The two conditions
coincide only for deterministic transitions. The whole framing of "V
is a global section of the Bellman sheaf" therefore does not recover
policy evaluation in the stochastic case — it recovers the
*residual-gradient TD* fixed point of weighted least-squares
`min_V Σ_e w_e (γV(s') − V(s) + r̄)^2`, which is a known different
operator from the true Bellman expectation operator (this is exactly
the Baird-1995 distinction the proposal cites without noticing it
applies here).

**Step 3 — Hodge decomposition `η = δφ + δ*ψ + h`.** Standard on a
2-complex once the 2-cells are specified. The proposal hand-waves
"triangle/cycle complex of G" without specifying which triangles, but
this is a fixable specification gap, not a math error.

**Step 4 — joint flow steady state.** The proposal claims steady states
satisfy `δV = −r̄` (i.e. η = 0) and `δ_1 η = 0` automatic. Working it
out: at equilibrium of
  `dV/dt = −δ_0*(δ_0 V + r̄) + δ_0* δ_1* ψ`
  `dψ/dt = −L_2 ψ + δ_1 (δ_0 V + r̄)`
let η = δ_0 V + r̄. The V-equation gives `δ_0*(η − δ_1*ψ) = 0`. The
ψ-equation gives `δ_1 δ_1* ψ = δ_1 η`, hence
`δ_1(η − δ_1*ψ) = δ_1 η − δ_1 δ_1* ψ = 0`. So at equilibrium,
`η − δ_1*ψ` is in `ker(δ_0*) ∩ ker(δ_1) = harmonic`. Therefore
  `η_∞ = δ_1* ψ_∞ + h_∞`,
**not** `η_∞ = 0`. The Bellman residual at convergence is coexact +
harmonic, not zero. The claim "At convergence η_∞ = 0 (Bellman)" in
the update-rule block is **wrong**. ψ absorbs the coexact part *into
itself* but does not eliminate it from the dynamics in a way that
makes V satisfy any stronger fixed-point equation than residual-
gradient TD already does. V_∞ is still the residual-gradient TD
solution (specifically: it satisfies `δ_0*(δ_0 V_∞ + r̄ − δ_1* ψ_∞) = 0`,
which projects ψ-shifted residual onto the orthogonal of exact forms;
without the ψ shift this is exactly RG-TD's normal equation).

**Step 5 — open question.** The open question asks whether the joint
operator M is a strict contraction faster than RG-TD on V alone, with
margin scaling in dim H_1(G_π). This is well-formed and checkable
*in principle*, but it is premised on the erroneous claim that the
joint flow's V-fixed-point coincides with the policy-π Bellman fixed
point. Given Step 4, even if M contracts faster than the V-only block,
the limit V_∞ is still the RG-TD limit (or a ψ-shifted variant of
it), not the Bellman expectation fixed point. So a positive answer to
the open question would not establish what the seed claims it would
establish.

**Failing steps:** 2 and 4. Each independently breaks the principle
as stated.

## Novelty check

Searched: "cellular sheaf Bellman equation reinforcement learning",
"Hodge decomposition Bellman residual", "sheaf value iteration
cochain RL", "discrete Hodge MDP cycle decomposition policy
evaluation". No direct prior on a sheaf-Bellman cohomological
algorithm; the cited prior (Hansen-Ghrist 2019, Bodnar et al. 2022,
Schmid 2025, Jiang-Lim-Yao-Ye 2011, Baird 1995) is correctly
characterized — none couples Hodge to a Bellman update. If the math
worked, there would be a real novelty kernel here. The novelty check
itself does not trigger rejection.

## Decision

Reject. The seed's slots 1–3 must clear the same math bar as a full
proposal, and they don't:

1. **Step 2 is wrong.** "δV = 0 iff V satisfies the policy-π Bellman
   equation" conflates per-edge equality with expectation equality.
   They coincide only for deterministic MDPs. Quote: "A 0-cochain V
   is a global section iff δV = 0, i.e. iff V satisfies the policy-π
   Bellman equation." This is the load-bearing identification of the
   sheaf machinery with the RL problem; without it, the principle
   ("Bellman optimality as a global section") is wrong as stated.

2. **Step 4 is wrong.** The proposal asserts "At convergence η_∞ = 0
   (Bellman)" but the joint-flow equilibrium is η_∞ = δ_1* ψ_∞ + h_∞,
   not zero. The principle that ψ "absorbs the cycle circulation and
   lets V satisfy the Bellman equation" is not what the dynamics
   actually do. ψ ends up parameterizing the coexact piece of the
   residual; V's limit is still in the RG-TD family.

The two errors compound: even if the open question's contraction
analysis went through positively, it would prove a faster route to
the wrong fixed point, not to the Bellman fixed point. The seed's
optimization principle ("Bellman optimality as global section /
Hodge-coupled flow drives V to Bellman fixed point") is therefore
not realized by the proposed algorithm.

The novelty is real (no prior couples sheaf Hodge to Bellman in this
way) and the mathematical machinery is interesting, but a seed whose
derivation has two errors at the load-bearing identification steps
fails the math check, regardless of how clean the open question is
framed. A revision would need to (a) replace "global section iff
Bellman" with the correct statement (the section equation is the
deterministic-edge condition; the stochastic Bellman equation is its
projection or expectation, with derivation), and (b) re-derive the
joint-flow equilibrium honestly, accepting that V_∞ is the RG-TD
limit and reframing the principle around what ψ-coupling actually
delivers (which may still be interesting — e.g. faster transient
contraction toward the RG-TD limit on cyclic graphs — but is a
different principle than the one stated).
