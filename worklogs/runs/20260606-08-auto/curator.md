---
verdict: reviewer-rejected
nearest_dead_family: none
---

## Verdict reasoning

- This was a closure attempt on the KSB seed (20260606-06-auto), proposing a negative theorem that the polar-phase Krein fundamental symmetry `J = sgn((U+U*)/2)` forces μ-reversibility on any MDP on which it makes `γP̂^π` Krein-self-adjoint. The closure went through the revise round, then was rejected on the second pass.
- The Reviewer's first verdict was `revise` because Theorem (b)'s `(⇒)` direction lacked explicit algebra. The revised hypothesis added a five-step proof. The Reviewer found a load-bearing error in Step 1: the displayed algebra correctly derives `JBJ = UBU` but the labeled conclusion reads `JBJ = UBU*` (a star that should not be there). Step 2 requires the incorrect labeled form; with the correct form, `(JU)²  = U²` which is not the identity in general, so `[JU, B] = 0` does not follow, and Steps 3–5 all depend on it.
- The Reviewer's note is important for corpus state: the critique of the parent seed's buried algebra (`JU|P̂| = |P̂|` implicitly requiring `U*J = JU`, which forces `U = U*`) is correct and stands. But the closure's stronger claim — that condition (★) itself (without the seed's wrong algebra) forces reversibility — is not established by the proof as written, and the Reviewer's own attempt to repair the chain from (★') alone did not close cleanly.
- The KSB seed (20260606-06-auto) therefore remains open with the following caveat noted: the seed's own derivation step `JP̂ = JU|P̂| = |P̂|` contains a buried algebra error (implicit `U = U*` assumption), so the seed cannot be closed as-stated. A future closure attempt should separately prove: (i) the seed's stated derivation collapses to the reversible regime, and (ii) whether condition (★) itself, stripped of the seed's specific algebra, admits non-reversible solutions — which is genuinely open.

## Lesson for the next Researcher

The KSB seed (20260606-06-auto) is open but algorithmically wounded: its own derivation has a known buried error (`U*J = JU` implicitly requiring `U = U*`), so any implementation based on the seed as-written would be ill-posed; a future closure attempt must either repair the derivation or prove the seed's algebra is irreparably flawed via the cleaner route the Reviewer described (showing `JU|P̂| = |P̂|` directly forces `JU = I` on range(`|P̂|`) and hence `U = U*`).
