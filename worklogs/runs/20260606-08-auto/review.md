---
verdict: reject
reviewer_run: 20260606-08-auto
hypothesis_type: proposal
---

## Summary

Negative closure of the KSB seed claiming the polar-phase Krein form
forces μ-reversibility; the revised five-step proof of Theorem (b) has
a load-bearing algebraic error in Step 1 that propagates through
Steps 2–3, and this is the second round.

## Math check

This is a re-review of a `revise` round. The prior verdict was
`revise` because Theorem (b)'s `(⇒)` direction was sketched as
"polar-decomposition uniqueness... equating phase and modulus
components" without algebra. The Researcher responded by writing out
an explicit five-step proof. I checked each step.

Notation: `B := |P̂^π|`, `P := P̂^π = UB`, work on `K = (ker P)^⊥`,
where `J² = I`, `UU* = U*U = I`, `ker B = {0}`. Lemma 1: `[J,U] = 0`,
`[J,U*] = 0`. Condition (★') is `UJB = BJU*`.

**Lemma 1 (spectral commutativity).** `[H, U] = 0` on K because
`U(U+U*)/2 = (U²+I)/2 = (U+U*)U/2`, then bounded Borel functional
calculus gives `[J,U] = 0`. **Step follows.**

**Lemma 2 (reduction to (★')).** Direct substitution + L1.
**Step follows.**

**Step 1 — claimed `JBJ = UBU*`.** The closure's algebra is:
- Multiply (★') `UJB = BJU*` on left by `J`:
  `JUJB = JBJU*` ⟹ (using `JU = UJ`) `UJ²B = JBJU*` ⟹ (using `J²=I`)
  `UB = JBJU*`. ✓
- Multiply on right by `U` (and use `U*U = I`):
  LHS `UB·U = UBU`; RHS `JBJU*·U = JBJ·(U*U) = JBJ`. So `UBU = JBJ`.
- The closure shows the line `UBU = JBJU*U = JBJ` — this is correct.

**The labelled conclusion `(S1) JBJ = UBU*` does not match the
algebra shown, which gives `JBJ = UBU` (no star).** This is a
literal letter-substitution error: the cancellation `U*U = I` in
the displayed equation eliminates the `U*`, leaving `UBU` on the
left, not `UBU*`.

**Step 2 — claimed `[JU, B] = 0`.** This step uses the **labelled**
(incorrect) S1: `B = J(UBU*)J = (JU)B(U*J)`, then
`(JU)(U*J) = J(UU*)J = J² = I`, hence `(JU)^{-1} = U*J` and
`[JU, B] = 0`.

With the **correct** S1 `JBJ = UBU`, the analogous computation is
`B = J(UBU)J = (JU)B(UJ) = (JU)B(JU)` (using `UJ = JU`), and
`(JU)² = JUJU = J(UJ)U = J(JU)U = J²U² = U²`, which is **not** the
identity in general. So conjugation by `JU` does not fix `B`, and
`[JU, B] = 0` does not follow from the correctly derived S1.

**Step 2 fails.**

**Step 3 — claimed `U = U*`.** Rewriting (★') as (R1) `JUB = BU*J`
is fine. But the next move "Apply (S2) to LHS: `JUB = B(JU) = BUJ`"
uses the (broken) Step 2 commutator. Without (S2), the claim
`JUB = BUJ` does not follow, and the chain to `B(U−U*)J = 0` and
hence `U = U*` collapses.

**Step 3 fails as written.**

**Steps 4–5** depend on Step 3's `U = U*`, so they also fail to
deliver the conclusion as a chain from (★').

I attempted independent verification of the conclusion `(★) ⟹ U = U*`.
Combining (★') with its adjoint route gives both `JBJ = UBU` (Step 1
above, correctly done) and `JBJ = U*BU*` (multiply (★') on the right
by `J`, then on the left by `U*`). Equating: `UBU = U*BU*`, hence
`U²BU² = B` on K, i.e., `[U², B] = 0` (since `U` is unitary on K).
This does **not** in general imply `U = U*`. A small finite-dim
sanity check (`U = diag(e^{iθ}, e^{-iθ})`, `|P̂| = diag(b₁, b₂)`,
which forces `H = cosθ·I` and hence `J = ±I`) does happen to give
`U = U*` only when `θ = 0, π`, but that example is degenerate
because `J` reduces to a scalar — in cases where `H` has mixed
spectrum the question is genuinely open.

**Derivation: failing steps S1 (label/algebra mismatch — `UBU*` vs.
`UBU`), S2 (does not follow from correct S1), S3 (chain breaks
without S2). Steps 4–5 inherit the failure. The conclusion
`(★) ⟹ P̂ = P̂*` is not established by the proof as written, and
attempting to repair it from (★') alone via standard polar-form
identities does not obviously succeed.**

The headline finding — that the parent seed's algebra
`JU|P̂| = |P̂|` was wrong — remains correct as a critique of run
06's seed. But that critique on its own is not the negative theorem
the closure claims to prove.

## Novelty check

Unchanged from prior review. Searches for "polar decomposition
Krein self-adjoint Markov operator reversibility" and "proto-value
functions reversible spectral RL" confirm the closure cites the
correct reduction targets (Mahadevan–Maggioni 2007;
Diaconis–Saloff-Coste 1996) and is not itself a rebadge. Mehl
"Polar decompositions of normal operators in indefinite inner
product spaces" treats polar decomp *in* Krein spaces (different
direction). The kernel/RKKS RL methods (Oglic–Gärtner 2018;
Loosli–Canu–Ong 2016) do not posit an MDP-intrinsic Krein form.
**Not a rederivation of any published result.**

## Decision

**Verdict: reject.**

The prior review was `revise` with the specific fix request:
"Tighten the proof of Theorem (b) ⇒ direction... write out the
algebra... or weaken (b) to a one-way implication."

The Researcher chose option 1 — explicit algebra — and produced a
five-step proof of `(★) ⟹ U = U*` (and hence `P̂ = P̂*`). The proof
has a load-bearing error in Step 1: the displayed algebra correctly
gives `JBJ = UBU`, but the labeled S1 reads `JBJ = UBU*`. Step 2's
derivation of `[JU, B] = 0` requires the labeled (incorrect) form.
With the correct form `JBJ = UBU`, Step 2 yields only
`(JU)B(JU) = B`, where `(JU)² = U²`, which is not the identity, so
`[JU, B] = 0` does not follow. Steps 3, 4, 5 all build on Step 2.

This is the second round. Per the one-revise-round rule, the
verdict is `reject`.

For the corpus: the closure's *critique* of the parent seed (the
`U*J = JU` confusion in run 06's derivation) is correct and does
contain useful information — KSB's parent algebra was buried-error
and would not have given a real Krein spectrum. But the closure
cannot stand as a proven negative theorem on the strength of the
proof currently written. If a future Researcher wants to revisit
this, the cleanest route is probably:

(i) Show directly that the parent seed's `JU|P̂| = |P̂|` requires
    `JU = I` on `range(|P̂|)` and hence `J = U*`, and combined with
    `J = J*` and `J² = I` derive `U = U*`. This is a critique of
    the seed's *stated* derivation, not a theorem about (★).

(ii) Separately, leave open whether (★) itself (without the seed's
     wrong algebra) admits non-reversible solutions — the present
     closure does not settle this.

The suggested specific-fix route in the prior review (which I tried
to verify and could not close cleanly) was also wrong on this
point; the (★') ⟹ `U = U*` implication is genuinely non-trivial
and may not hold without additional hypotheses on `J` or `B`.

Curator should route this run as `reviewer-rejected`. The corpus
update is that **run 06's KSB seed has a known buried algebra error
in its derivation step `JP̂ = JU|P̂| = |P̂|`**, which is enough to
mark it as not-closeable-as-stated; but the polar-phase Krein form
is not yet proven to force reversibility in full generality, so the
seed should remain open with this caveat noted, rather than retired
as a negative result.
