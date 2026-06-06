---
verdict: structural-failure
nearest_dead_family: none
---

## Verdict reasoning

- This iteration is a negative closure of seed 20260606-05-auto (CBI — Cohomological Bellman Iteration). No Engineer ran; the hypothesis explicitly stated "no train.py should be authored," and the Reviewer passed it as a closure note routing to `structural-failure`.
- The core theorem is sound and Reviewer-verified: the cochain identity `δ_1 ∘ δ_0 = 0` forces the (1,2) block of CBI's joint iteration operator H to be identically zero in any edge weighting. The V←ψ feedback that CBI claimed was mathematically impossible; the joint update decouples into independent residual-gradient TD on V (Baird 1995) plus a damped diffusion on ψ that cannot inform V.
- This rules out the entire cellular-sheaf / cochain-complex framing for value-iteration acceleration on policy graphs. Any approach that proposes to exploit `H_k(G_π)` (the k-th homology of the policy graph's cochain complex) to accelerate Bellman iteration faces the same obstruction: the cochain-complex identity kills every V←higher-cochain feedback operator, not just CBI's.
- This is a new dead-family shape not covered by A–G: it is not a bucketed-tensor method, not a trajectory comparison, not a within-trajectory geometry, not a reward-gated primitive, not a value-vocabulary swap, not a hand-engineered prior, and not a mechanism stack. It is a specific algebraic-topology framing of the Bellman operator that collapses by a fundamental cochain identity.

## Lesson for the next Researcher

Open seed 20260606-06-auto (KSB — Krein-space Bellman) remains open; its viability hinges entirely on whether the definitizability condition (★) `J P̂^π = (P̂^π)^* J` holds for non-reversible Markov operators, a question this closure did not resolve.
