# 20260606-10-auto — empty-handed

reason: (a) **No open seeds** in the recent corpus: seed 05 (CBI,
sheaf-Hodge cochain) was negatively closed by run 07 via the
`δ_1 ∘ δ_0 = 0` obstruction; seed 06 (KSB, polar-phase Krein) was
negatively closed by run 08 via the `U = U* ⇒ P̂ = P̂*` (reversibility-
collapse) theorem; runs 01, 03, 09 are empty-hand. The "adjacent
regions not closed" notes from run 08 (N1: antisymmetric-exchange
Krein form; N2: non-self-adjoint pseudospectral; N3: hyperbolic
split-signature on `H ⊕ H`) were re-checked on independent pass and
each first-pass derivation still collapses to a published method
(non-reversible MCMC reweighting, Trefethen pseudospectra of
Markov operators, or a tautological doubling that does not reduce
Bellman computation), so none was honestly seed-able this turn.

(b) **Fresh regions attempted, distinct from the cumulative
empty-hand log of runs 01/03/09 and from the closed seeds 05/06.**
This turn explored seven further structurally different regions
and each collapsed to published machinery or to a heuristic
without a clean theorem slot:

(i) **Replica-symmetric / Parisi-type Bellman.** Define the policy
partition function `Z(β) = Σ_π exp(β J(π)) μ_0(π)` and characterize
V* through the β → ∞ free-energy saddle. Without random disorder
over the MDP (rewards/dynamics i.i.d. from a quenched law), the
replica trick has nothing to average; the deterministic-MDP version
collapses to the Gibbs variational identity
`−F(β) = sup_ρ E_ρ[J] − (1/β) KL(ρ‖μ_0)`, which is
**KL-regularized RL with reference policy** [Galashov et al. 2019
"Information asymmetry in KL-regularized RL"; Levine 2018 RL-as-
inference]. With quenched random MDPs the framework gives an SK-like
free-energy bound but the algorithm extracted is **annealed
importance sampling over policies** [Neal 2001], not a Bellman
replacement.

(ii) **Quantum-walk / Szegedy-walk Bellman.** Build the Szegedy walk
`W = (2|π⟩⟨π| − I) · SWAP` on the bipartite policy-graph Hilbert
space and find V* via amplitude amplification on the reward-tilted
stationary state. Quadratic hitting-time speedup is real
[Szegedy 2004; Magniez-Nayak-Roland-Santha 2011] and the framing has
been worked out for RL [Dunjko-Briegel 2018 review; Wang-Zhang 2023
"Quantum policy iteration"]. **Published**, and on classical hardware
the substrate sees no algorithmic primitive — the Szegedy walk is a
unitary on `H_S ⊗ H_S` whose simulation cost on classical hardware is
the same as the underlying Markov chain. Slot 3 (typed primitive) is
abstract; slot 4 reduces to the Szegedy hitting-time theorem. No new
classical algorithm.

(iii) **Free-probability / R-transform composition of policy
operators.** Voiculescu's free additive convolution gives the
spectral law of `A + B` for free `A, B` via the R-transform
`R_{A+B} = R_A + R_B` [Voiculescu-Dykema-Nica 1992]. Apply to a
random ensemble of policy operators `P^{π_i}` with the goal of
getting the spectrum of an "averaged" policy. The free-probability
limit assumes asymptotic freeness, which holds for independent random
matrices [Voiculescu 1991] but **fails for the policy operators of
a fixed MDP** — `P^π` and `P^{π'}` share row-stochastic structure
and the same state-space basis, so they are emphatically not free.
The framework therefore does not specialize to a single MDP; on
random-MDP ensembles it gives a spectral-density characterization
of the policy-averaged operator that is not a Bellman fixed point.

(iv) **CLR (Cwikel-Lieb-Rozenblum) bound on the number of bound
states of a Schrödinger-type Bellman Hamiltonian.** Treat
`H = −L_π + V_r` with `L_π` the policy-graph Laplacian and `V_r`
the reward potential. CLR bounds `#{negative eigenvalues}` by
`C_d · ∫ V_r^{d/2}` [Cwikel 1977; Lieb 1980; Rozenblum 1972].
This is a *counting bound*, not a fixed-point characterization. It
gives information about the dimension of the "good" subspace but not
a Bellman update. Slot 4 has a real theorem (the CLR bound itself)
but slot 1 (optimization principle) cannot be stated as a sentence —
"count bound states" is not what value iteration optimizes. The
mismatch between principle and primitive is unrecoverable; not
seed-able.

(v) **Tarski / Kleene fixed-point with abstract-interpretation
widening.** The Bellman operator is monotone on the complete lattice
`(ℝ^S, ≤)`; Tarski's theorem gives existence of a least fixed point
without the contraction property; Cousot-Cousot widening accelerates
fixed-point iteration on monotone operators by *over-approximating*
fixed points and then *narrowing*. Apply to RL: get a "value-iteration
with widening." This is **abstract-interpretation RL** [Pinosky et al.
2024 "Abstract interpretation for RL"; ANI-style methods] and earlier
**verification-of-RL-policies via abstract interpretation**
[Bastani-Pu-Solar-Lezama 2018]. Published, and the widening operator
is a heuristic in the AI literature without a sharp convergence-rate
theorem on stochastic operators.

(vi) **Stein operator / Stein discrepancy zero as the optimality
condition.** A Stein operator `A_p f = (∇ log p)·f + ∇·f` characterizes
the distribution `p` via `E_p[A_p f] = 0` for all test `f` [Stein 1972;
Gorham-Mackey 2017]. For RL, build a Stein operator on (s,a,r)
trajectories under the optimal occupancy `d^*` and minimize the
kernelized Stein discrepancy of the empirical occupancy to `d^*`.
This is **Stein variational policy gradient (SVPG)** [Liu-Ramachandran-
Liu-Peng 2017] and the broader **score-based RL via Stein operators**
line. Published.

(vii) **Occupational-measure martingale / Doob-Martin boundary
representation of value functions.** Every `γP^π`-superharmonic
function admits a Choquet integral representation
`V(s) = ∫_∂ K(s, ξ) m(dξ)` over the Martin boundary `∂` with kernel
`K(s, ξ) = lim_{s_n → ξ} G_γ(s, s_n)/G_γ(s_0, s_n)` where `G_γ` is
the discounted Green function [Doob 1959; Martin 1941; Sawyer 1997
"Martin boundaries and random walks"]. The optimal V* would be a
Choquet integral over an *optimal* boundary measure m*. On finite
ergodic MDPs the Martin boundary is a finite set (the recurrent
classes), and the representation collapses to the standard finite
linear combination of stationary-distribution components — i.e., a
rebadge of the fundamental matrix `(I − γP^π)^{-1}`. On infinite
state spaces the boundary becomes nontrivial, but the substrate is
finite/tabular by construction; the principle has no purchase on
the substrate. Slot-1 is fine but slot-3 (typed primitive) reduces
to "linear combination of recurrent-class projectors" which is just
spectral decomposition of P^π (covered by proto-value functions
[Mahadevan-Maggioni 2007]).

(c) **Why partial structure is not seed-able.** Each of (i)–(vii)
fails the seed contract at slot 3 (typed primitive collapses to a
named published object) or slot 1 (the principle cannot be stated
as a one-sentence optimization target distinct from a published
one). For (iv), slot 4 (theorem) has a real statement but slots 1
and 3 do not align — counting bound states is not an optimization
principle for return-maximization, and the bound is a one-shot
inequality, not a fixed-point that an iteration converges to. Posting
a vague seed in any of these regions would fail the Reviewer's
math/novelty check at slots 1–3 with the same rigor as a full
proposal, and the prompt is explicit that slot-1/2/3 standards do
not relax for seeds. The corpus is saturated against the principles
I can productively reach; the calibrated honest output is an
empty-hand documenting the seven regions explored and the specific
collapse mode of each.
