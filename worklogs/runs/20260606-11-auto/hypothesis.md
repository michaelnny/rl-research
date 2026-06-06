# 20260606-11-auto — empty-handed

reason:

(a) **No open seeds in the recent corpus.** Read the file headers
of runs 01, 03, 05, 06, 07, 08, 09, 10. Seed 05 (CBI, sheaf-Hodge
cochain) was negatively closed by run 07 via the `δ_1 ∘ δ_0 = 0`
obstruction; seed 06 (KSB, polar-phase Krein) was negatively closed
by run 08 via the `U = U* ⇒ P̂ = P̂*` reversibility-collapse theorem.
Runs 01, 03, 09, 10 are empty-hand. Re-read the full closures of
runs 07 and 08 to confirm there is no residual partial structure
to seed forward; the negative theorems pin down the obstruction at
a level (cochain identity, polar-decomposition uniqueness) that
forecloses the surrounding directions (N1/N2/N3 in run 08; KSB-on-
non-reversible in run 07's note) without leaving a checkable open
question to seed.

(b) **Fresh regions attempted this turn, distinct from the cumulative
empty-hand log of runs 01/03/09/10 and from the closed seeds 05/06.**
Eleven structurally different regions explored; each collapsed to
published machinery or to a structure that does not yield a typed
primitive at exemplar quality:

(i) **Halpern's anchored iteration on the Bellman optimality
operator** with the goal of strong convergence under non-expansiveness
without contraction (e.g., average-reward / multichain / discount=1
or projected-Bellman with function approximation where contraction
fails). Published as **Anc-VI** [Park-Ryu, "Accelerating Value
Iteration with Anchoring," arXiv 2305.16569 / NeurIPS 2023] with
matching upper-and-lower-bound optimality (factor 4); also
**Robust Halpern Iteration** [arXiv 2505.12462] for robust average-
reward; and **"Faster Fixed-Point Methods for Multichain MDPs"**
[arXiv 2504.09913, OpenReview vrbUfvcNZ6] for the multichain case.
No new principle: the principle ("anchor a non-expansive Bellman
iterate to its initial point with weight 1/(k+1)") is exactly
Halpern (1967) applied to the Bellman operator, and the rate
analysis is the published Lieder/Kim/Sabach-Shtern O(1/k).

(ii) **Choquet integral / non-additive recursive Bellman.**
`V(s) = sup_a Choquet[r(s,a, ·) + γ V(s') | s,a]` with a state-
dependent capacity `ν(·|s,a)` replacing the conditional probability
`p(·|s,a)`. Decision-theoretic origin in **Schmeidler (1989)** and
the dynamic version in **Chateauneuf-Kast-Lapied (2001)**,
**Nishimura-Ozaki ("Search and Knightian uncertainty," JET 2004)**,
**Eichberger-Grant-Kelsey** for dynamic ambiguity. For RL the
specialization either (α) keeps a non-trivial capacity and reduces
to **risk-sensitive RL** with a distortion-risk Bellman
[Petrik-Subramanian; Chow-Tamar-Mannor-Pavone 2015 "Risk-sensitive
and robust decision-making"; Dabney et al. IQN 2018 already covers
distortional Bellman], or (β) marginalizes the capacity to a
probability and recovers ordinary Bellman. No new operator.

(iii) **Wiener-Hopf factorization of `(I − γP^π)`.** In ladder-
height theory of random walks and **Markov-additive processes**
[Asmussen 2003; Kemperman 1961; Bertoin 1996] one factors the
resolvent `(I − γP)^{-1} = K_+ K_-` where K_± isolate "ascending"
and "descending" parts. For general MDPs without a half-line
ordering on the state space, the factorization is trivial
(`K_+ = (I − γP)^{-1}, K_- = I` or vice versa) because there
is no ladder structure. On half-line MDPs (e.g., queueing /
inventory) the published Wiener-Hopf decomposition gives the
Green function in closed form via ladder epochs but is not a
new RL algorithm — it is the standard transient-analysis
machinery from queueing theory.

(iv) **Newton-Schulz iteration on the resolvent.**
`X_{k+1} = X_k(2I − A X_k)` for `A = I − γP^π` converges quadratically
to `(I − γP^π)^{-1}`. The Newton-iterate-on-Bellman-optimality
view of policy iteration is exactly **Howard's policy iteration
= Newton's method on the Bellman optimality equation**
[Puterman-Brumelle 1979 "On the convergence of policy iteration in
stationary dynamic programming"]. Newton-Schulz on the policy
evaluation step is matrix-form policy evaluation and provides no
new mechanism.

(v) **Schur-complement value iteration via state partition
`S = S_1 ⊔ S_2`.** Maintain V on S_1, eliminate S_2 by Schur
complement of `(I − γP^π)`. This is **Bertsekas state-aggregation
/ Gauss-Seidel value iteration** [Bertsekas-Tsitsiklis 1996, §6.3]
and **action elimination methods** for MDPs. No new principle.

(vi) **Toric/Newton-polytope enumeration of Bellman fixed points
via Bernstein's theorem.** The Bellman optimality system is a
piecewise-polynomial system (max replaced by argmax); the mixed-
volume enumeration would bound the number of solutions. Bellman
optimality has a unique fixed point under contraction, so the
enumeration question is moot. The polyhedral-method angle on MDPs
collapses to the **LP formulation** [Manne 1960; d'Epenoux 1963;
de Farias-Van Roy 2003 ALP] and gives no new operator.

(vii) **Hamiltonian / symplectic-flow HJB via method of
characteristics.** The continuous-time HJB
`V_t + max_u {r + ∇V · f} = 0` admits a symplectic structure on
`T^*S` with Hamiltonian `H(x,p) = max_u{r + p·f}`; characteristics
are Pontryagin's costate equations. Collapses to **Pontryagin's
maximum principle** / **stochastic value gradient (SVG)** [Heess
et al. 2015], already noted as published in run 01.

(viii) **Diffusion-wavelet / multiresolution Bellman solver.**
A wavelet basis adapted to the policy operator P^π (built from
spectra of P^π via Coifman-Maggioni dyadic powers) gives a
multiscale solve of `(I − γP^π) V = r`. Published as **diffusion
wavelets in MDPs / proto-value functions** [Mahadevan-Maggioni
2007; Coifman-Maggioni 2006 "Diffusion wavelets"]. No new
principle.

(ix) **Renormalization-group / coarse-graining fixed point on
state-space hierarchy.** Iteratively coarse-grain S into blocks
B_k, with the Bellman operator on B_k+1 obtained by a Kadanoff-
type restriction from B_k. The RG fixed point gives the "scaling
limit" of V*. Collapses to **state aggregation** [Bertsekas-
Castanon 1989; Singh-Jaakkola-Jordan 1995] and bisimulation-based
state abstraction [Givan-Dean-Greig 2003; Ferns-Panangaden-Precup
2004 "Metrics for finite Markov decision processes"]. No new
operator.

(x) **Galois connection on the (policy lattice, value lattice).**
The Bellman policy-evaluation map G: π ↦ V^π and improvement
F: V ↦ greedy(V) form an alternating descent on a lattice; the
Galois-connection / abstract-interpretation view gives convergence
under monotonicity without contraction. Published as **abstract-
interpretation RL** [Pinosky et al. 2024; Bastani-Pu-Solar-Lezama
2018], already noted in run 10. The widening operator that would
make this an algorithm is heuristic.

(xi) **Submartingale / Riesz decomposition of V into harmonic +
potential parts.** Every `γP^π`-superharmonic function decomposes
uniquely as `V = h + Π r` with `h` harmonic (`(I − γP^π) h = 0`)
and `Π r` the potential (Green-function image of reward). For
finite ergodic chains, h is constant on each recurrent class.
This is the **Riesz decomposition for transient Markov chains**
[Doob 1984; Revuz 1984] and underpins the **average-reward
Hordijk-Kallenberg policy-evaluation decomposition**. No new
operator.

(c) **Why partial structure was not seed-able.** Each of (i)–(xi)
collapses at slot 3 (typed primitive is a named published object —
Halpern iterate, Choquet integral, ladder operator, Newton-Schulz
factor, Schur complement, mixed volume, Hamiltonian flow, diffusion
wavelet basis, RG block-spin, Galois adjoint, Riesz harmonic +
potential) or at slot 1 (the principle is a one-sentence
restatement of a published optimization target — "optimal-rate
fixed-point iteration of a non-expansive operator," "ambiguity-
averse expected utility," "ladder-height factorization," "Newton's
method on Bellman," "Gauss-Seidel block elimination," "polynomial-
system solving," "characteristic flow on HJB," "wavelet basis
adapted to the operator," "block-spin coarse-graining," "monotone
descent on a lattice," "harmonic-plus-potential decomposition").
Posting a vague seed in any of these regions would fail the
Reviewer's slot-3 typed-primitive check or slot-1 novelty check
with the same severity as a full proposal. The corpus is saturated
against the principles I can productively reach this turn from a
fresh-region start, and the open seeds posted earlier in the run
stream have all been negatively closed by their own follow-up
iterations, leaving no within-reach seed to upgrade. The honest
output is an empty-hand documenting the eleven distinct regions
explored and their respective collapse modes, additive to the
empty-hand log of runs 01/03/09/10.
