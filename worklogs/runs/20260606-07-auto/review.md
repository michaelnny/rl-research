---
verdict: pass
reviewer_run: 20260606-07-auto
hypothesis_type: proposal
---

## Summary

Negative closure of seed CBI (run 05) proving the cochain identity
δ₁∘δ₀ = 0 forces the V←ψ coupling to be identically zero, decoupling
CBI into residual-gradient TD plus an inert ψ side-process. The prior
`revise` items are addressed: the runnable Update-rule pseudocode is
removed, an explicit "no train.py should be authored" line is on top,
and the closure now routes to `structural-failure` rather than
presenting an algorithm to run.

## Math check

The math content is unchanged from the prior submission; I re-verified
it against the seed's stated M operator (seed lines 84-86).

Step (1): cochain identity `δ₁ ∘ δ₀ = 0` on the 2-skeleton's
cellular cochain complex (Hansen-Ghrist 2019; Lim 2020). Standard.

Step (2)-(3): linearization `[V; ψ]_{k+1} = M [V; ψ]_k + b` with
`M = I − αH`. Block decomposition of H reads off from the seed's
update:
- (1,1): `δ₀^T diag(w) δ₀` — symmetric, residual-gradient TD operator.
- (1,2): scalar multiple of `δ₀^T δ₁^T`. Closure writes `−δ₀^T δ₁^T`
  (sign from the gradient convention); regardless of sign, this block
  is `(δ₁ δ₀)^T = 0` by the cochain condition. ✓
- (2,1): `δ₁ diag(w) δ₀`. The `diag(w)` weight breaks the cochain
  identity, so this block is generically nonzero. ✓
- (2,2): `L_2 = δ₁ δ₁^T`. ✓

Step (4a)-(4b): `H_{12} ≡ 0` always; `H_{21}` generically nonzero. ✓

Step (5)-(6): block lower-triangular spectrum is the union of diagonal
block spectra (Horn-Johnson 2013, Thm 1.3.20). Correctly applied.

Theorem (a): direct from cochain condition. ✓
Theorem (b): block-triangular spectrum union. ✓
Theorem (c): max-of-block-spectral-radii argument; acyclic edge case
correctly handled (`L_2 ≡ 0` on trivial domain when `H_1(G_π) = 0`). ✓
Theorem (d): for cyclic graphs and small α, smaller-gap block sets
the rate; standard `ρ(I − αL) ≈ 1 − αλ_min` linearization. ✓

The Disposition section (lines 116-126) makes the algorithmic
collapse explicit: `coex_V ← δ₀^T δ₁^T ψ_k = 0`, V-update reduces
literally to `V_{k+1} ← V_k − α δ₀^T diag(w_e) η_k` (Baird 1995
verbatim), ψ becomes a passive damped diffusion. Internally consistent.

Closure-vs-seed match (per the Reviewer prompt's closure edge case):
- Principle/primitive of seed: joint `(V, ψ)` cochain pair under sheaf
  Hodge flow with operator M(α_V, α_ψ). The closure's M (eq 2-3)
  matches the seed's stated operator (seed lines 84-86). ✓
- Open question of seed: "is M a strict contraction strictly faster
  than V-only TD by a margin scaling with `dim H_1(G_π)`?" The
  closure's Theorem (a)-(d) directly answers NO with a structural
  reason (cochain identity zeroes the V←ψ block; on cyclic graphs M
  is strictly slower than V-only TD, controlled by the spectral gap
  of `L_2`, opposite the direction the seed asked about). ✓

Derivation checked: each step follows.

## Novelty check

Negative result, so the relevant check is whether the negative theorem
is itself novel and not folkloric. Searches on prior pass:
- "sheaf Laplacian Bellman equation reinforcement learning" — no
  prior joint V-ψ sheaf flow on policy graph; Schmid 2025 prospectus
  has no operator.
- "cellular sheaf reinforcement learning value iteration" — no hits.
- "Hodge decomposition Bellman residual" — Jiang-Lim-Yao-Ye 2011 is
  on statistical ranking, not Bellman residuals.
- "block triangular Hodge Laplacian spectrum" — Horn-Johnson
  block-triangular spectrum is standard; the cochain-identity-zeroes-
  the-coupling argument is folkloric in algebraic topology but
  applied here specifically to a proposed value-iteration operator.

The closure's negative theorem is not a rederivation of any published
result I can find; it's a clean application of `δ₁ ∘ δ₀ = 0` to the
specific block-matrix M proposed by the parent seed. Conclusion that
"CBI ≡ Baird 1995 + inert ψ" is a *negative* novelty claim about the
seed, not a positive algorithm claim.

## Decision

The prior `revise` items are addressed, point by point:

1. **Update-rule pseudocode removed.** The file no longer has a
   `## Update rule` section. The Disposition section (lines 116-126)
   explicitly states the seed's V-update collapses to Baird 1995
   verbatim and ψ is inert; no algorithm is presented for the
   Engineer to implement.

2. **Explicit no-train.py statement at the top.** Lines 5-7:
   "**This is a negative closure; no train.py should be authored.**
   ... Curator should route this as `structural-failure` for corpus
   update, not as a proposal to run."

3. **"Why this is not [closest published method]" section makes
   the rebadge-of-Baird conclusion explicit.** Lines 138-141:
   "any implementation of CBI would be a rebadge of
   **residual-gradient TD** [Baird 1995] — a known dead family with
   documented divergence pathologies on linear function approximation
   — with an inert ψ side-process bolted on. No run is warranted."

The file is now structured as a closure note, not a runnable proposal:
no Principle/Primitive/Update-rule slots that could mis-route the
Engineer, but the Theorem (which is the load-bearing slot for a
closure) remains in place with its math intact. The closure's M
operator matches the seed's, and the Theorem directly answers the
seed's open question with a definitive NO and a structural reason.

The KSB note at the end correctly leaves seed 06 open and does not
attempt closure; it is a forward-looking observation that constrains
the surrounding question, which is appropriate for a closure file.

Pass. Curator should route this as `structural-failure`, update the
corpus to record that the cellular-sheaf-on-policy-graph cycle-
exploitation family is killed by `δ₁ ∘ δ₀ = 0`, and not spawn the
Engineer.
