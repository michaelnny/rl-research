---
verdict: reviewer-rejected
nearest_dead_family: none
---

## Verdict reasoning

- The KSB seed proposed placing the action-value function Q in a reproducing-kernel Krein space whose indefinite metric is the polar phase of the policy-induced Markov operator, aiming to make the Bellman operator Krein-self-adjoint and directly solvable via a spectral expansion — a genuinely novel direction with no match in dead families A–G and no published precedent found by the Reviewer.
- The Reviewer identified a specific algebraic error in the derivation: the claim `J U = I` on the range of `|P̂^π|` (needed to establish equation ★, `J P̂^π = (P̂^π)* J`) does not hold; a spectral counterexample (`U v = e^{iθ} v`, `J v = sgn(cos θ) v` gives `JU v = ±e^{iθ} v ≠ v`) kills the step.
- The seed is internally inconsistent: it simultaneously claims to derive ★ in the derivation slot and asks "for which operators does ★ hold?" in the open question — a seed cannot open-question an identity it claims to have proved.
- The principle (slot 1) and primitive (slot 3) survived at exemplar quality; the rejection is confined to the derivation (slot 2). A corrected version that either genuinely proves ★ for a stated operator class or honestly recasts slot 2 as conditional on ★ (and removes the now-answered sub-question 1) could be a clean seed on a future turn.

## Lesson for the next Researcher

The Krein-spectral / polar-phase direction for Bellman operators is not ruled out as a mechanism family — the derivation step ★ remains an open mathematical question (`J P̂^π = (P̂^π)* J` for which operator classes?); a future iteration that resolves ★ before building the spectral expansion could close this into a valid seed or full proposal.
