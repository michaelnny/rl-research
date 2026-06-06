---
verdict: reject
reviewer_run: 20260606-06-auto
hypothesis_type: seed
---

## Summary

KSB seed places Q in a Krein RKKS whose indefinite metric is the
polar phase of `P^π`; rejected because the derivation contains an
algebraically incorrect step that the open question is then asked to
repair.

## Math check

Step-by-step walk through the derivation.

1. Polar decomposition `P^π = U |P^π|` on `L²(S; μ)`. Standard
   (Kato VI.7). Note: for a non-self-adjoint Markov operator that is
   typically a contraction (not invertible), `U` is a *partial*
   isometry on `(ker P^π)^⊥`, not a unitary. The seed itself notes
   this in the derivation but then later silently treats `U` as
   unitary.

2. Definition `J := sgn((U+U*)/2)`. As a function of the
   self-adjoint operator `(U+U*)/2`, this is a well-defined bounded
   self-adjoint operator with spectrum `⊆ {−1, 0, +1}` via the
   Borel functional calculus. Fine, modulo the zero set of
   `(U+U*)/2` (where `sgn` is conventionally set to 0 or ±1; the
   seed does not specify).

3. Krein form `[f,g]_J := ⟨f, J g⟩`. Standard construction
   (Bognar 1974). Fine — provided `J` is bounded self-adjoint with
   bounded inverse on its range, which here would require `0 ∉
   σ((U+U*)/2)`, an unstated assumption.

4. **Failing step.** Claim: `J P̂^π = J U |P̂^π| = |P̂^π|`. This
   requires `JU = I` (on the range of `|P̂^π|`). The argument given
   is "J = sgn((U+U*)/2)" plus "`U* J = J U` for unitary `U`
   commuting with its real part."

   - Even granting `U` unitary and `J` commuting with `U` (which
     does *not* follow from `J` being a function of `(U+U*)/2`
     unless one assumes `U` is normal *and* its spectral
     projections align with those of its real part — a special
     class), the identity `JU = I` does not hold.
   - Counterexample at the spectral level: take `U` unitary with
     eigenvalue `e^{iθ}` for `θ ∉ {0, π}` on some eigenvector `v`.
     Then `(U+U*)/2 · v = cos(θ) v`, so `J v = sgn(cos θ) · v =
     ±v`, while `U v = e^{iθ} v`. Hence `J U v = ±e^{iθ} v ≠ v`,
     so `J U ≠ I` on `v`. Consequently `J U |P̂^π| ≠ |P̂^π|` in
     general.
   - The parallel claim `(P̂^π)* J = |P̂^π| U* J = |P̂^π|` fails
     for the same reason.

   So the seed's derivation that (★) `J P̂^π = (P̂^π)* J` holds is
   mathematically incorrect.

5. Krein-self-adjointness of `T_π` and the Langer/Behrndt–Philipp
   spectral expansion. *Conditional* on (★), the spectral expansion
   would follow once definitizability is also established. But (★)
   itself is unestablished by the derivation in step 4.

6. Internal inconsistency. The seed's open question (sub-question 1)
   asks: "For which classes of policy-induced operators `P̂^π` does
   the polar phase `J = sgn((U+U*)/2)` satisfy `J P̂^π = (P̂^π)* J`?"
   This is exactly (★). If the derivation actually proved (★), the
   open question would be answered ("for all `P̂^π` with the stated
   polar decomposition"). The fact that the seed asks (★) as an
   open question is an admission that the derivation step in (4) is
   not in fact a derivation. The seed is therefore internally
   inconsistent: it cannot both *derive* (★) and *open-question* (★).

   The well-formed version of this seed would have to either (a)
   downgrade slot 2 to "we assume (★) and derive the spectral
   expansion conditionally" — but then slot 2 is no longer a
   derivation of the algorithm, just of the consequence — or (b)
   actually prove (★), at which point the spectral question becomes
   only the definitizability sub-question.

   Failing steps: 4 (algebraic identity does not hold), 6
   (derivation contradicts open question).

Open-question quality: sub-question 1 is well-posed (a class of
operators is asked for) and checkable. Sub-question 2
(definitizability) is also well-posed. Both are at exemplar
quality *as questions*. The problem is upstream: the derivation
that motivates them contains a wrong step.

## Novelty check

Searched for:
- "Krein space reinforcement learning Bellman operator non-reversible
  Markov" — no hits combining Krein space with Bellman/RL.
- "indefinite kernel reproducing Krein space value function Bellman"
  — RKKS work (Oglic–Gärtner 2018; Loosli–Canu–Ong 2016; Ong et al.
  2004; scalable RKKS Nyström 2018) is general kernel learning,
  not RL-specific. The seed cites these correctly.
- "polar decomposition Markov chain non-reversible spectral theorem"
  — additive/multiplicative reversibilization of `P^π`
  (Diaconis–Saloff-Coste; Choi 2020 similarity orbit) and `L²`
  spectral gaps for non-reversible chains exist, but none impose a
  Krein structure indexed by the polar phase.
- Proto-value functions / spectral RL (Mahadevan–Maggioni 2007;
  Stachenfeld 2017) work on the reversibilized Laplacian.

The specific construction — Krein form determined by the polar
phase of the policy-induced Markov operator, applied to make the
Bellman operator Krein-self-adjoint — does not appear in the
literature I can find. So the seed is not a rebadge of a published
method, and is not a member of dead families A–G in
`prior_attempts.md` (it is operator-theoretic, not a bucketed
tensor, not a within-trajectory geometry, not a stack).

If the math worked, this would be a genuinely new direction.

## Decision

**reject**, on math grounds.

Specific failure: the derivation step
"`J P̂^π = J U|P̂^π| = |P̂^π|` (self-adjoint), and `(P̂^π)* J =
|P̂^π| U* J = |P̂^π|` (also self-adjoint, using `U* J = J U` for
unitary `U` commuting with its real part)" is algebraically
incorrect. `J = sgn((U+U*)/2)` does not satisfy `J U = I` on the
range of `|P̂^π|`; the spectral counterexample
`U v = e^{iθ} v, J v = sgn(cos θ) v` gives `JU v = ±e^{iθ} v ≠ v`.

This makes (★) `J P̂^π = (P̂^π)* J` an unestablished assumption
rather than a derived identity. The seed then asks (★) as its open
question, contradicting the derivation. A seed cannot
simultaneously claim to derive identity X and ask "for which
operators does X hold?" — one of the two has to go. As written,
slot 2 (Derivation) does not survive the math check.

The principle (slot 1) and primitive (slot 3) are at exemplar
quality and the novelty check passed; the rejection is solely on
the failed derivation step. A revised version that either (a)
proves (★) for a stated operator class as part of the derivation,
or (b) recasts slot 2 honestly as "we *assume* (★) and derive the
spectral expansion" — and accordingly removes the now-redundant
sub-question 1 and sharpens sub-question 2 — could be a clean seed
on a future turn. Not a `revise` because the fix changes which
question is open and which is settled, which is a structural
rewrite of slots 2 and 4 rather than a one-or-two-slot patch.
