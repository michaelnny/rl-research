---
verdict: reviewer-rejected
nearest_dead_family: none
---

## Verdict reasoning

- The hypothesis proposed Cohomological Bellman Iteration (CBI), a seed using a cellular sheaf on the state-transition graph where a value 0-cochain and a cycle-potential 2-cochain co-evolve under a Hodge-decomposed Bellman flow.
- The Reviewer found two independent load-bearing math errors: (1) "global section iff Bellman equation" is only true for deterministic MDPs — in the stochastic case the per-edge equality system is overdetermined and the sheaf section condition collapses to the residual-gradient TD (Baird 1995) fixed point, not the Bellman expectation equation; (2) the joint-flow equilibrium does not yield η_∞ = 0 but η_∞ = δ_1* ψ_∞ + h_∞, meaning V_∞ is still the RG-TD limit, not the Bellman limit.
- The novelty kernel is real (no prior work couples sheaf Hodge machinery to a Bellman update), but the derivation as stated does not realize the stated principle. A revision would need to correctly identify what the section equation actually encodes in stochastic MDPs and re-derive the equilibrium honestly, accepting that V_∞ is the RG-TD solution and reframing around what ψ-coupling actually delivers.
- This rejection does not map to any existing dead family (A–G covers discrete-signal channels, trajectory comparison, geometric statistics, gated primitives, value renaming, hand-engineered vocabularies, and mechanism stacks). The topological/sheaf shape is novel and may be revisitable with corrected math; no new family is warranted since the shape itself was not proven dead — the specific derivation was wrong.

## Lesson for the next Researcher

The sheaf-cohomology / Hodge-decomposition shape is not ruled out as a mechanism family, but any future attempt must correctly handle stochastic transitions (the section equation encodes a per-edge equality, not the Bellman expectation, so the identification "global section = Bellman fixed point" requires a weighted-expectation / projection argument to be valid in the stochastic case).
