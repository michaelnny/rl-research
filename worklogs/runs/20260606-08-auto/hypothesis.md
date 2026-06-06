# 20260606-08-auto — KSB Negative Closure (structural-failure)

Closes seed: 20260606-06-auto

**This is a negative closure; no train.py should be authored.** The
file resolves the open question of KSB (Krein-Spectral Bellman, run
06) with a negative algebraic theorem: the polar-phase Krein form
`J = sgn((U+U*)/2)` makes `(γ P̂^π)` Krein-self-adjoint **iff
`P̂^π` is self-adjoint in `L²(μ⊗unif)`**, i.e., iff the policy-π
Markov chain is μ-reversible. In that case KSB's spectral expansion
collapses to the Mahadevan–Maggioni proto-value-function /
Diaconis–Saloff-Coste reversibilized-Laplacian spectral theory.
For non-reversible chains — the only regime in which KSB would have
been new — condition (★) **fails**, and the principle's promised
real Krein spectrum does not exist. Curator should route as
`structural-failure`.

## What the closure establishes for the corpus

The seed's algebra `J P̂ = J U |P̂| = |P̂|`, and `P̂* J = |P̂| U* J =
|P̂|` (from §Derivation of run 06) implicitly used `U* J = J U`
**and** `[J, |P̂|] = 0`. The first identity, combined with the
correct identity `U* J = J U*` (because `J = f(U+U*)` commutes with
`U*` whenever it commutes with `U`), forces `U = U*`, i.e., `U` is
a Hermitian unitary (an involution: `U² = I`). Combined with the
polar-decomposition normality criterion `[U, |P̂|] = 0 ⇔ P̂` normal
[Encyclopedia of Mathematics, *Polar decomposition*; Halmos 1982
*Hilbert Space Problem Book*, Problem 134], this collapses `P̂` to
`P̂ = P̂*`, i.e., a self-adjoint Markov operator. KSB therefore
*has no purchase on non-reversible MDPs*: precisely the regime it
was advertised to address. The corpus update is that **any
"Krein-self-adjoint Bellman via the polar phase of `P^π`"**
approach faces the same `U = U*` obstruction, regardless of how the
Krein form is dressed, because polar-decomposition uniqueness pins
the form's relation to `U` and the cyclic condition `JU = UJ`
(spectral calculus) determines the rest.

## Derivation of the negative theorem

Notation reproduced from the parent seed. Work on the Hilbert space
`H := L²(S × A; μ ⊗ unif)`. The operator
`P̂^π(s,a; s',a') = p(s'|s,a) · 𝟙[a' = π(s')]` is bounded with
`‖P̂^π‖ ≤ 1`. Its polar decomposition (Kato 1980 §VI.7) is
`P̂^π = U |P̂^π|` with `|P̂^π| := ((P̂^π)* P̂^π)^{1/2} ≥ 0` and `U`
a partial isometry on `(ker P̂^π)^⊥`; on that subspace `U U* =
U* U = I_{|range U|}` so `U` is unitary there. Define
`H := (U + U*)/2` and (where 0 ∉ σ(H)) `J := sgn(H)`, the
self-adjoint sign of the Hermitian part of the polar phase.
By the bounded Borel functional calculus, `J = J* = J^{-1}`
(an involution, hence Krein-fundamental-symmetry). All algebra
below is on the invariant subspace `K := (ker P̂^π)^⊥`, on which
`J² = I` and `UU* = U*U = I`; the kernel `ker P̂^π` is `J`-isotropic
and `P̂^π = (P̂^π)* = 0` there, so (★) is automatic on it.

**Lemma 1 (Spectral commutativity of `J` with `U`, `U*`).**
On `K`, `U U* = U* U = I`, so
`U H = U(U+U*)/2 = (U² + I)/2 = (U+U*)U/2 = H U`. Hence `U` and
`H` commute, and by Borel functional calculus
  (L1)   `J U = U J`,    `J U* = U* J`.
(See e.g. Reed–Simon I §VII.2 — bounded Borel functions of a
self-adjoint operator commute with everything `H` commutes with.)

**Lemma 2 (Reduction of (★) to (★')).**
Condition (★) of the seed reads `J P̂^π = (P̂^π)* J`. Substituting
the polar form on both sides:
  `J P̂^π = J U |P̂^π| = U J |P̂^π|`           (by L1)
  `(P̂^π)* J = |P̂^π| U* J = |P̂^π| J U*`       (by L1)
So (★) ⟺
  (★')   `U J |P̂^π| = |P̂^π| J U*`     in `K`.

**Lemma 3 (Polar-decomposition normality).** For any bounded
operator with polar decomposition `A = U|A|`,
  `[U, |A|] = 0 ⟺ A` is normal (i.e., `A* A = A A*`).
[Encyclopedia of Mathematics, *Polar decomposition*; Halmos 1982,
Problem 134; Conway 1990 *A Course in Functional Analysis* §VI.10.]

**Theorem (Negative Closure of KSB).**
Let `P̂^π` be the bounded policy-induced operator on
`H = L²(S × A; μ⊗unif)` with polar decomposition `P̂^π = U|P̂^π|`
on `K = (ker P̂^π)^⊥`. Let `J := sgn((U + U*)/2)` be the polar-phase
Krein fundamental symmetry assumed by the KSB seed. Then:

(a) `J U = U J` and `J U* = U* J` always (Lemma 1).

(b) On `K`, `(★) ⟹ P̂^π = (P̂^π)*`. (See proof below for the
    explicit algebraic chain. The converse is direct substitution
    and is recorded for completeness.)

(c) (b) implies `P̂^π` is `L²(μ⊗unif)`-self-adjoint.

(d) For a Markov operator on `L²(μ)`, `P = P*` ⟺ `μ` is reversing,
    i.e., the chain is **detailed-balance reversible** w.r.t. μ
    [Diaconis–Saloff-Coste 1996 §1; Levin–Peres–Wilmer 2009 §1.6].
    For the action-augmented operator `P̂^π` on `μ ⊗ unif`,
    self-adjointness reduces to the policy-π state chain on `S`
    being μ-reversible (the action coordinate factors trivially
    through the indicator `𝟙[a' = π(s')]`).

(e) **Conclusion.** The polar-phase Krein form `J = sgn((U+U*)/2)`
    makes `(γ P̂^π)` Krein-self-adjoint **iff** `P̂^π` is
    `L²(μ⊗unif)`-self-adjoint, i.e., **iff** the chain is
    μ-reversible. Equivalently: on the non-reversible subset of
    Markov operators (the regime the seed was designed to address),
    (★) generically fails, `T_π` is not Krein-self-adjoint under
    `J`, and Langer's spectral theorem does not apply.

(f) **Reduction to published methods on the reversible regime.**
    On reversible chains, `J = I` (since `U = I` on `K` when
    `P = P*`, so `H = I`, `sgn(H) = I`). The Krein form
    degenerates to the standard `L²(μ)` inner product. The
    "Krein-spectral solve" of the Bellman equation reduces to
    eigen-decomposition of `γ P^π` in `L²(μ)`, which is exactly
    proto-value functions [Mahadevan–Maggioni 2007] /
    Diaconis–Saloff-Coste spectral analysis of reversible
    chains. KSB on reversible chains is **not new**; on
    non-reversible chains it is **mathematically ill-posed** under
    the seed's choice of `J`.

### Proof of Theorem (b) (the load-bearing step, written out)

We work throughout on `K = (ker P̂^π)^⊥`, on which `J² = I`,
`UU* = U*U = I`, and `ker|P̂^π| = ker P̂^π ∩ K = {0}` (so `|P̂^π|`
has dense range on `K`). Write `P := P̂^π`, `B := |P̂^π|`, `B ≥ 0`,
for brevity.

**Step 1 — Similarity `JBJ = UBU*`.** Multiply (★') on the left by
`J`:
  `JUJB = J B J U*`,   so by L1 (`JU = UJ`, `JU* = U*J`),
  `UJ²B = J B J U*`,   i.e., `UB = J B J U*`   (since J²=I on K).
Now multiply on the right by `U` and use `U*U = I`:
  `UBU = J B J U*U = J B J`,
hence
  (S1)   `J B J = U B U*`     in `K`.

**Step 2 — Commutator `[JU, B] = 0`.** From (S1),
`B = J(JBJ)J = J(UBU*)J = (JU) B (U*J)`. By L1, `U*J = JU*`, so
`(JU)(U*J) = JUU*J = JJ = I` on `K`, i.e., `(JU)^{-1} = U*J` and
`(JU) B (JU)^{-1} = B`. Equivalently
  (S2)   `[JU, B] = 0`     in `K`.

**Step 3 — `U = U*` on `K`.** Rewrite (★') using L1
(`JU* = U*J`, `JU = UJ`):
  `UJB = BJU*`   becomes   `JUB = BU*J`     (using `UJ = JU`),
i.e.,
  (R1)   `JUB = BU*J`.
Apply (S2) to LHS: `JUB = B(JU) = BJU = BUJ` (using `JU = UJ`).
So
  `BUJ = BU*J`,    hence    `B(U − U*)J = 0`.
Right-multiplying by `J` (`J² = I`) gives `B(U − U*) = 0`. Since
`ker B = {0}` on `K`, **`U = U*`** on `K`. ∎

**Step 4 — `[U, B] = 0`.** With `U = U*` from Step 3, L1 gives
`UJ = JU` and the second L1 identity `U*J = JU*` becomes the
same statement. Substitute `U* = U` into (R1):
  `JUB = BUJ`,    and using `JU = UJ`,    `UJB = BUJ`,
so `U(JB) = (BU)J = (BJ)U` if and only if `[U, JB] = 0` and
`[U, J] = 0`. The second commutator vanishes by L1.
For the first: from `UJB = BUJ` and `JU = UJ`,
  `UJB = BUJ = B·JU = (BJ)U`,
so multiplying on the right by `U^{-1} = U` (`U² = I` since `U`
is a Hermitian unitary):
  `UJBU = BJ`,
i.e., conjugation by `U` fixes `JB`: `U(JB)U* = JB`, so
`[U, JB] = 0`. Combined with `[U, J] = 0`, multiplying
`U(JB) = (JB)U` on the left by `J` and using `JU = UJ`:
  `J·UJB = J·JBU`,   i.e.,   `UJ²B = J²BU`,   i.e.,   `UB = BU`.
Hence
  (S4)   `[U, B] = 0`     in `K`.

**Step 5 — `P = P*`.** By (S4) and Lemma 3, `P = UB` is normal.
By Step 3, `U = U*`. Therefore
  `P* = (UB)* = B U* = B U = U B = P`,
where the third equality uses (S4). ∎

This completes the proof of Theorem (b). The (⇐) direction (i.e.,
`P = P*` ⟹ (★)) is direct: when `P = P*`, polar uniqueness gives
`U = I` on `K` (since the polar phase of a positive operator is the
identity on its range; `P* = P ≥ 0` is not assumed, but for
self-adjoint `P` the polar phase is `sgn(P)`, which commutes with
`|P|`, and `H = (U+U*)/2 = U` is self-adjoint, so `J = sgn(U)`
commutes with everything `U` commutes with — in particular with
`P` itself, giving `JP = PJ = P*J`).

## Theorem (Negative answer to the seed's open question)

The seed asked: *"Is `(γ P̂^π)` definitizable as a Krein-symmetric
operator under the polar-phase Krein form `J = sgn((U+U*)/2)` on
`L²(S × A; μ⊗unif)`, and if so, is the Krein spectrum real with
finite Krein critical points?"*

**Answer.** NO, except on the reversible subset of MDPs, on which
KSB is reducible to existing reversible-symmetrization spectral
methods (Mahadevan–Maggioni; Diaconis–Saloff-Coste).

Concretely:

- **Sub-question 1 (self-adjointness condition (★))** is answered by
  Theorem (b)+(c): (★) ⟹ `P̂^π = (P̂^π)*` (proved in Steps 1–5
  above), and hence by (d) the chain is μ-reversible. So (★) holds
  for all reversible chains and **fails generically** for
  non-reversible ones.
- **Sub-question 2 (definitizability)** is moot: when (★) fails,
  `T_π` is not Krein-self-adjoint, so the Krein spectral theorem
  for definitizable self-adjoint operators [Langer 1982] does not
  apply at all. When (★) holds, `J = I` and definitizability
  reduces to ordinary self-adjointness in `L²(μ⊗unif)`, which is
  trivially yes — but with no novelty over published reversible
  methods.

The seed's stated dichotomy ("either definitizable on non-trivial
class, or collapse to Diaconis-Saloff-Coste") thus resolves to the
**collapse branch**: the polar-phase Krein form *is* the
reversible-symmetrization regime in disguise.

## Disposition (no implementation)

By Theorem (e), the seed's Update rule does not produce a
well-defined algorithm on non-reversible MDPs (step 2 — the
Krein-Lanczos eigenproblem of `γ P̂_k` w.r.t. `J_k` — has no real
Krein-eigenvector basis when (★) fails, so the spectral expansion
step 3 is mathematically undefined). On reversible MDPs, `J = I`,
and the algorithm reduces to ordinary `L²(μ⊗unif)` eigen-solve of
`γ P^π`, i.e., proto-value-function-based Bellman solve, which is
published. No train.py is warranted.

## Why this is not [closest published method]

The closure is a negative result, not a new algorithm. The relevant
distinction is from KSB's parent seed (whose algebra `U* J = J U`
contained the buried error) and from **kernel/RKKS RL methods**
[Oglic–Gärtner 2018; Loosli–Canu–Ong 2016 — these are
representation-learning frameworks that *posit* a Krein form
exogenously, and never claim it is the polar phase of any specific
operator]. KSB's specific failure is that it claimed an
*MDP-intrinsic* Krein form (the polar phase of `P^π`) under which
the Bellman operator becomes Krein-self-adjoint — and that claim
is false except in the reversible regime, where it reduces to
**proto-value functions** [Mahadevan–Maggioni 2007] /
**Diaconis–Saloff-Coste spectral methods** [1996]. The closure's
*conclusion* is that any "Krein-spectral Bellman solver via the
polar phase of `P^π`" approach faces the same
`U = U* ⇒ P̂ = P̂*` obstruction, regardless of how the kernel,
features, or step sizes are dressed: the polar-phase choice forces
reversibility, and the published reversible-symmetrization machinery
already handles that regime.

## Note on adjacent regions, not closed here

The closure rules out the *polar-phase* choice of Krein form. It
does **not** rule out:

(N1) A *different* Krein fundamental symmetry `J` that depends on
     `P^π` but is not the polar phase — e.g., `J` defined via the
     antisymmetric part `(P − P*)/2` (the "exchange operator" of
     non-reversible MCMC, Hwang–Hwang–Sheu 2005), which would
     require its own check of (★).
(N2) A *non-self-adjoint* spectral approach (Jordan-block /
     pseudospectral) on `γ P^π` directly, without trying to make
     it Krein-self-adjoint — but pseudospectra of non-normal
     operators do not in general give a clean fixed-point
     expansion of `(I − γ P^π)^{-1} r`, and standard
     pseudospectral RL would land in proto-value-function-on-the-
     symmetrized-graph territory.
(N3) A formulation on the **enlarged** Hilbert space `H ⊕ H`
     where `[(f₁,f₂), (g₁,g₂)]_J = ⟨f₁, g₂⟩ + ⟨f₂, g₁⟩` (the
     hyperbolic / split-signature form), under which any
     bounded `P` is Krein-self-adjoint via
     `T = diag(P, P*)` — but this is a tautology that does
     not reduce the Bellman fixed-point computation.

These are *adjacent* directions, not seed-able structures: each
one's first-pass derivation either runs into a published method or
into a tautology, as briefly noted above. They are recorded so the
next Researcher iteration does not re-explore them blindly.
