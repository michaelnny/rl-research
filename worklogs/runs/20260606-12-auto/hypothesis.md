# 20260606-12-auto -- empty-handed

reason:

(a) **No open seeds in the corpus.** Re-read the headers/tails of runs
01, 03, 05, 06, 07, 08, 09, 10, 11. Seed 05 (CBI, sheaf-Hodge cochain)
was negatively closed by run 07 via the `δ_1 ∘ δ_0 = 0` obstruction;
seed 06 (KSB, polar-phase Krein) was closure-attempted in run 08 but
the curator flagged a load-bearing algebra error in Step 1 of the
proof (the labeled identity `JBJ = UBU*` should read `JBJ = UBU` from
the displayed algebra). The KSB seed therefore remains *open but
algorithmically wounded*: the seed's own derivation has a buried
`U*J = JU` ⇒ `U = U*` step that forces the polar phase to coincide
with the reversibility regime. Curator note (run 08) recommends a
future closure must either (i) prove the seed's stated derivation
collapses to reversibility via the cleaner `JU|P̂| = |P̂| ⇒ JU = I on
range(|P̂|)` route, or (ii) prove condition (★) stripped of the
seed's specific algebra admits non-reversible solutions. I tried
both this turn and could not produce a clean proof in either
direction in one session — the (i) direction needs a careful argument
that `JU = I` extends from `range(|P̂|)` to all of `K`, which
requires controlling `J` on `ker|P̂|` (it is determined there by the
spectral calculus of `(U+U*)/2` but the algebra is delicate); the
(ii) direction requires constructing an explicit non-reversible `P̂`
for which (★) holds, and the obvious candidates (cyclic permutation
on a 3-cycle; non-reversible random-walk on a directed graph) all
fail (★) on hand-check. Neither closure nor seed forward is honest.

Runs 01, 03, 09, 10, 11 are empty-hand; their cumulative covered
regions (re-read in full) include:

- 01: Pontryagin/SVG; spectrum-reshaping/proto-value; correlated-eq
  policies; α-divergence/Tsallis-NPG.
- 03: deep-BSDE/Z-process; passivity/dissipativity; reward-shaping
  cohomology (Ng-Harada-Russell); homotopy continuation;
  cavity/BP-on-factored-MDP; persistent homology of sublevel sets;
  SOS/LMI Lyapunov; abstract operator-distillation T:Π→Π.
- 09: characteristic-function distributional Bellman; tropical/max-
  plus eigenvalue Bellman; Doob h-transform / large-deviation
  conditioning; Itakura-Saito β-divergence trust region;
  set-valued viability kernel; em-alternation on dually flat
  manifold; Yoneda category-theoretic policy iteration.
- 10: Parisi/replica-symmetric Bellman; Szegedy quantum walk;
  free-probability R-transform; CLR bound on Schrödinger-Bellman;
  Tarski/Kleene with abstract-interpretation widening; Stein
  discrepancy as optimality condition; Doob-Martin boundary.
- 11: Halpern/Anc-VI anchoring; Choquet/risk-sensitive non-additive
  Bellman; Wiener-Hopf ladder factorization; Newton-Schulz on
  resolvent (=Howard PI); Schur complement state-aggregation;
  toric/Bernstein polynomial-system on Bellman; symplectic HJB /
  PMP; diffusion wavelets / proto-value functions; RG block-spin
  coarse-graining; Galois connection abstract-interpretation;
  Riesz harmonic+potential decomposition.

(b) **Fresh regions attempted this turn, structurally distinct from
the cumulative empty-hand log of runs 01/03/09/10/11 and from the
closed/wounded seeds 05/06.** Eight further mathematically distinct
regions explored; each collapsed at slot 3 (typed primitive) or slot
1 (one-sentence principle distinct from a published method).

(i) **Hilbert projective metric / Birkhoff coefficient on the
positive-value cone.** For positive-reward MDPs the discounted
expected operator `γ P^π` is positive linear; Birkhoff's theorem
gives a contraction rate `tanh(diam(γP^π)/4)` in Hilbert's
projective metric on the cone of positive functions, often tighter
than `γ` in sup-norm [Birkhoff 1957; Eveson-Nussbaum 1995;
arXiv:2312.11147 *On the contraction properties of a pseudo-Hilbert
projective metric* 2023; arXiv:2309.02413 *Hyperbolic contractivity
and the Hilbert metric on probability measures* 2023]. The
projective-metric framing is published — it is exactly the standard
machinery used to prove exponential mixing of positive integral
operators and Sinkhorn convergence (arXiv:2311.04041 / Springer
2025 PTRF). For RL specifically, the rate `tanh(diam/4)` is at
best a tighter analysis of value iteration, not a new operator;
moreover, on sparse-reward MDPs the cone-positivity assumption
itself fails (zero-reward states have `r = 0` and `V` is not
strictly positive on the cone interior), so the Birkhoff
coefficient degenerates. No new principle, no new primitive.

(ii) **Davis-Yin three-operator splitting on the Bellman fixed
point.** Decompose `(I - γP_{greedy(Q)}) Q - r = A(Q) + B(Q) +
C(Q)` with A monotone (linear part `(I-γP_π)` for fixed reference
π), B Lipschitz (the `γ(P_{greedy(Q)} - P_π)Q` correction), and C
the negative reward shift. Davis-Yin would alternate proximal
maps. The obstruction: `B(Q) = γ(P_{greedy(Q)} - P_π)Q` is *not*
cocoercive (it is discontinuous in Q at policy-switching
boundaries), so the Davis-Yin convergence theorem [Davis-Yin
2017] does not apply. Forcing a Moreau-Yosida regularization on
B turns it into soft-Bellman / MaxEnt RL, which is published
(SAC). Slot 4 (theorem) collapses to "softmax Bellman is
Lipschitz, hence Davis-Yin applies," which is a renaming of soft
policy iteration.

(iii) **Anderson-acceleration / Krylov-subspace Bellman.** Maintain
the last `m` iterates `Q_{k-m}, …, Q_k` and the residuals `T*Q -
Q`; solve a small least-squares for the optimal linear
combination that minimizes the residual; iterate. Geist-Pietquin
[2013 "Anderson Acceleration for Reinforcement Learning"] and
recent follow-ups [Sun et al. 2021 "Damped Anderson mixing for
deep RL"; Shi et al. 2019] cover this. The principle is "least-
squares mixing of past Bellman iterates," which is a published
quasi-Newton method; the primitive collapses to the QR factor of
the residual matrix. No new operator.

(iv) **Proximal-point on occupancy in Hilbert projective metric.**
The primal LP-MDP has a closed convex feasible set on the
occupancy simplex; proximal-point with Hilbert metric as the
prox term gives a multiplicative-weights-style update on
occupancy. With KL prox this is mirror descent / NPG (published);
with Hilbert prox the closed form involves the projective
distance, which on the simplex is `log(max(d_i/d'_i)) -
log(min(d_i/d'_i))` and the prox-step has no closed form (numerical
inner solve required). The framework is **interior-point method
on LP-MDP** [Ye 2011 "The simplex and policy-iteration methods
are strongly polynomial for the Markov decision problem with a
fixed discount rate"], which is published, and the Hilbert-prox
variant is a known slow-down because the projective diameter on
the simplex blows up at the boundary.

(v) **Self-consistent invariant measure on (s, a, return).** Define
`ρ(s, a, g)` = joint density of being at `(s,a)` and receiving
total discounted return `g` thereafter. Self-consistency:
`ρ(s,a,g) = ∫ p(s'|s,a) π(a'|s') ρ(s',a',g') 𝟙[g = r + γg']`. This
is exactly the **distributional Bellman equation** [Bellemare-
Dabney-Munos 2017 "A distributional perspective on RL"; Dabney et
al. 2018 IQN, QR-DQN; Nguyen-Tang et al. 2021 MMD-DRL]. The
primitive `ρ` is the published return-distribution; framing it
as an invariant measure is a re-statement, not a new primitive.

(vi) **Halpern with trajectory-conditioned anchor.** The Halpern
anchored iteration `Q_{k+1} = (1/(k+2)) Q_anchor + (1 - 1/(k+2))
T*Q_k` gives `O(1/k)` strong convergence for non-expansive
operators [Halpern 1967; Lieder 2021; Park-Ryu 2023]. Replace the
fixed anchor with `Q_anchor = MC-return-on-trajectory τ_k`. The
convergence theorem requires the anchor to be a fixed point of
T* (or a zero of `I - T*`); the MC return is an unbiased estimator
of `Q^π` for the *current* π, not of `Q*`, so the anchor wanders.
Either (a) average the anchor over many trajectories (the
average converges to `Q^π_k` by ergodicity, not to `Q*`, so the
Halpern iteration drifts toward a moving target — this is just
Q-learning with Polyak averaging, published), or (b) use the
optimal trajectory's MC return — but identifying optimal
trajectories pre-convergence requires `Q*` itself, circular. Slot
4 collapses to "Polyak-averaged TD-with-MC-anchor," which is
published.

(vii) **Reflected Bellman with learned lower envelope.** Maintain
`Q^L ≤ Q* ≤ Q^U` and shrink the gap. The standard *interval value
iteration* [Pineau et al. PBVI 2003; Smith-Simmons HSVI 2005]
already does this for POMDPs; for fully observed MDPs, **bounded
real-time DP** [Barto-Bradtke-Singh 1995; McMahan-Likhachev-Gordon
2005 BRTDP] is published. The "reflected" framing in Skorohod
sense is a continuous-time analog [Lions-Sznitman 1984] that for
MDPs reduces to interval VI on the discrete grid. No new principle.

(viii) **Fenchel conjugate of V as primitive.** Let `V*(p) :=
sup_s [<p,s> - V(s)]` for `p ∈ ℝ^S`. The Bellman optimality
condition for `V` translates to a fixed-point equation on `V*` via
Fenchel-conjugate calculus. The translated equation is `V*(p) =
sup_s [<p,s> - r(s, π*(s)) - γ Σ_{s'} P(s'|s,π*(s)) V(s')]`. This
*does not* reduce to a clean equation in `V*` alone (the inner
expression involves `V`, not `V*`), because the Bellman operator
is not self-conjugate-friendly: it is a max of affine maps, whose
conjugate is a piecewise-linear convex function but the iteration
on the conjugate side is *not* a fixed-point of a convex operator.
The Fenchel-dual representation of value functions appears in
**convex duality for HJB** [Fleming-Soner 2006 §III; Lions
1982 viscosity-solution theory] and in **occupancy-LP duality**
[Wang 2017; ALP de Farias-Van Roy 2003]; the dual variables
there are the *occupancy* `d`, not the conjugate of `V`. The
"conjugate of V" framing has no native fixed-point structure
distinct from the primal. Slot 1 has no one-sentence principle,
slot 3 has no typed primitive that closes under iteration.

(c) **Why partial structure is not seed-able.** Each of (i)–(viii)
fails the seed contract at slot 1 (the optimization principle
restates a published target — Birkhoff cone contraction, three-
operator splitting, Anderson mixing, interior-point on LP, return-
distribution self-consistency, Halpern anchoring, interval value
iteration, Fenchel duality on V) or at slot 3 (the typed primitive
is a named published object — Hilbert projective metric, proximal
operator, Krylov subspace, occupancy measure, return distribution,
trajectory-anchored iterate, value envelope, Legendre conjugate of
V). Posting any of these as a probe would fail the Reviewer's
nearest-disqualifier check (Bellman backup / scalarization /
distributional-RL rebadge / dead family A or E).

(d) **Why not force a probe with proof debt.** The Researcher
prompt says "If you can write [the seven sections] honestly, write
a probe." For each of (i)–(viii), the *one-sentence principle*
section cannot be written honestly without naming a published
method (Birkhoff, Davis-Yin, Anderson, Halpern, Bellemare et al.,
Fleming-Soner, etc.) or restating a known dead family (E: value-
vocabulary swap; A: bucketed-tensor on (s,a,channel)). The slot-1
honesty bar fails before slot-8 (proof debt) is even reached.
Forcing a probe by writing a stitched principle that hides the
collapse mode would be a Reviewer-rejection turn, which the loop
already has many of (08 reviewer-rejected, 06 reviewer-rejected,
05 reviewer-rejected). The honest output is empty-hand.

(e) **Diversity note for next turn.** The next Researcher iteration
should orient toward regions not yet touched in the cumulative
empty-hand log. Specific suggestions (for the seed-diversity
mechanism, not as proposals here): (α) variational-inequality /
generalized-equation framings of Bellman (Facchinei-Pang 2003)
distinct from the LP-MDP and Davis-Yin angles; (β) ergodic-control
/ multiplicative ergodic theory (Oseledec 1968) applied to the
Lyapunov spectrum of `γP^π` — distinct from the polar-phase /
Krein angle of the wounded KSB seed; (γ) optimal-stopping /
Snell-envelope embeddings of MDP optimality, distinct from the
reflected/interval-VI angle of (vii); (δ) the **reverse Bellman
equation** for the *time-reversed* optimal policy under a known
target distribution (Mira-Geffner 2008 inverse RL is published, but
the *reverse Bellman fixed-point structure as a primitive*, not as
an inverse-RL objective, may be unexplored). None of these were
seed-able this turn but they delineate fresh region for future
iterations.
