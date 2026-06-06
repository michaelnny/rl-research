# prior_attempts.md — dead mechanism families

This file records the *shape* of mechanisms that have already been ruled
out, not the individual attempts that ruled them out. The 39 individual
attempts are sealed in `worklogs/attempts/01–39-*.md` and indexed at the
bottom of this file. The Researcher reads this to know what shapes are
dead at the level of mechanism, not to study individual attempts for
inspiration on near-variations.

**The dominant lesson from 39 attempts: variation within these families
does not help.** Switching the index axis (state hash → cluster → exit
hash → policy regime), changing the statistic (mean → variance → JSD →
TV → Lévy area → spectral), changing the aggregation (Pareto → Kemeny →
sup-norm → strict-superset) — all stayed inside the same dead families.
A proposal whose central idea is "switch one knob inside one of these
families" is a known failure, not a new direction.

---

## Family A — Bucketed-tensor + partial-order vote

**Shape.** Maintain a tensor `T[bucket, action, channel]` indexed by
some bucketing of observations, accumulate a per-channel statistic
into it, and select actions by a partial-order vote (Pareto-non-
dominance, strict-superset, Kemeny consensus, sup-norm) over the
channel dimension. Optionally apply the result as a logit nudge.

**Why it dies.** The bucket space is starved on the substrate. Either
buckets never collide enough to fill the tensor (FED, CEC, TPP, TRAC,
CSD, ARP), or the channel dimension collapses because terminal-only
reward channels contribute zero to all entries during the bootstrap
window (PICAV, CHX, CRP, TCP, ATP, PRAR, PCR, ACFC, ACCD, BLIC), or
the partial-order saturates as channel count grows (ACS).

**Verdict.** This whole family is dead on this substrate. The fix is
not "a better bucketing axis" or "a better channel statistic" or "a
better partial order." Those have all been tried. A new candidate
that fits this shape will fail in the same way.

**Attempts in this family:** 15 FED, 16 PICAV, 17 CHX, 18 CEC, 19
CWTP, 20 LRA, 21 TPP, 22 CRP, 23 TCP, 24 PCR, 25 ACS, 26 TRAC, 28 ARP,
29 PFA, 30 CPR, 31 ATP, 32 CSD, 33 ACFC, 34 ACCD, 36 PRAR, 37 HRC, 39
BLIC.

## Family B — Pairwise trajectory comparison

**Shape.** Maintain pairs of trajectories from a replay buffer, find
something matched between them (shared start, shared end, shared
intermediate hash, shared cluster), compare the diverging segments
along some dimension (sign, rank, channel difference), and use the
comparison to drive policy updates.

**Why it dies.** Either the matching condition is too rare (RSD's
shared start+end, CWTP's confluence, TPP's terminal-hash collisions),
or the divergence statistic collapses to scalar step-penalty when one
side terminates early.

**Attempts in this family:** SIT, RSD (sprint-3 derivation),
19 CWTP, 21 TPP, 22 KTAC, 31 KSV (alive-weak — kernel-weighted
relaxation, but still in family).

## Family C — Within-trajectory signal geometry

**Shape.** Compute a within-episode geometric statistic of the
observation/cumulant trace (convex hull, Lévy area, rank position,
spectral coefficient, signature) per (state, action) and use it to
drive logit updates.

**Why it dies.** When any vector channel is terminal-only, the
geometric statistic collapses to a near-line in the always-firing
direction; the geometric "structure" reduces to "shorter is better,"
which is shortest-path-to-terminal, which actively harms long-horizon
problems where the long path is the rewarding one.

**Attempts in this family:** 17 CHX (hull), 22 CRP (rank), 23 TCP
(precedence DAG), 25 ACS (spectrum), 35 PHI (Lévy area), 36 PRAR
(antisymmetric residual).

## Family D — Reward-independent primitive + reward-gated operator

**Shape.** Compute a primitive that fires from step 1 without needing
reward (cosine alignment, action-frequency concordance, persistence
horizon, imminence shift). Then gate its application on a terminal
or rewarded outcome.

**Why it dies.** The gate inherits the bootstrap wall. The primitive
is reward-free in form but reward-dependent in effect, because the
gate is silent until the first rewarded trajectory.

**Attempts in this family:** 24 PCR, 27 PCGA, 31 ATP, 34 ACCD, 35
PHI, 39 BLIC.

## Family E — Avoid value vocabulary, keep value structure

**Shape.** Replace the words Q / V / advantage / return-to-go with
something else, but the central learned object still does
future-compression of return into a scalar (or vector with a fixed
weight collapse).

**Why it dies.** "Avoid value vocabulary" is not a research direction.
The mission is to replace what value *does* (future compression,
temporal composition, local improvement), not its name.

**Attempts in this family:** 1 FROST, 11 TOP, 14 Primal Behavior Flow,
many sprint-4 entries on inspection.

## Family F — Hand-engineered structural priors

**Shape.** Hand-design a vocabulary (event types, edit grammar,
clause symbols, segment boundaries) that lets a passive correlation
miner produce reasonable-looking certificates.

**Why it dies.** The hand-engineering is the algorithm. The
optimizer over the hand-engineered basis is generic. The basis
encodes a person's belief about what structure exists, which is not
a research contribution and does not transfer.

**Attempts in this family:** 1 FROST, 7 OPP, 8 EOP/COP, 12 PEO.

## Family G — Mechanism stack

**Shape.** Three or more named components stitched together. No
single composition law. The hypothesis fills the "core primitive"
slot with whichever component the author finds most distinctive that
day, but the mechanism's behavior is determined by the joint
operation of all three.

**Why it dies.** Not an algorithm. Cannot have a derivation, cannot
have a theorem, cannot be implemented faithfully because the
"primitive" is fictional.

**Attempts in this family:** 6 T-CTBP (canonical example), various
sprint-4 entries that filed extra components under "side
information."

## Family H — Algebraic-topology / cochain-complex value iteration

**Shape.** Frame the Bellman operator on the cochain complex of the policy graph `G_π = (S, E_π, w)`, augmenting the value function V (a 0-cochain) with higher cochains (e.g., a 1-cochain ψ representing cycle potentials) and using sheaf-Hodge or cellular-homology operators to couple ψ into V, claiming that exploiting `H_k(G_π)` (the k-th homology group of the policy graph) accelerates value iteration.

**Why it dies.** The cochain-complex identity `δ_1 ∘ δ_0 = 0` kills every V←higher-cochain feedback operator. The (1,2) block of the joint iteration operator H equals `(δ_1 δ_0)^T = 0` in any weighting, making the joint update block lower-triangular. The spectrum of the joint operator is the union of the V-only and ψ-only spectra; the ψ block cannot improve the V-only convergence rate, and on cyclic policy graphs it generically slows convergence by an amount controlled by the spectral gap of `L_2 = δ_1 δ_1^T`. Any implementation collapses to residual-gradient TD (Baird 1995) plus an inert side-buffer.

**Verdict.** Dead by fundamental cochain identity. Any approach that proposes to exploit `H_k(G_π)` via a cochain-complex coupling to the Bellman operator faces the same obstruction, regardless of weighting, step-size matching, or which homology degree is targeted.

**Attempts in this family:** 40 CBI (negative closure of seed 20260606-05-auto).

---

## Disqualifier families (the standard negative space)

Independently of the specific dead families above, the central
improvement operator must not reduce under variable renaming to:

- Bellman backup (Q-learning, DQN, SAC, TD3).
- Scalar-weighted log-prob update (PPO, REINFORCE, GRPO, A2C).
- Actor-critic — a critic supplying the actor's weight.
- Reward-model optimization (RLHF, DPO).
- Scalarized vector reward `wᵀr` for any fixed or learned `w`.
- CEM / ES / CMA-ES elite refitting.
- Top-k trajectory cloning.
- Go-Explore / count-based / RND with renamed counts or novelty.
- Options / hierarchical RL with renamed skills.
- Model-based planning with renamed states.
- Verifier-guided search (best-of-N, MCTS, ReAct) with renamed verifier.
- GVFs / successor features with renamed cumulants.
- Distributional RL with renamed return distribution.
- Hindsight Experience Replay with renamed virtual goals.
- Decision Transformer with renamed conditioning.
- Reward machines with renamed automaton states.

Existing methods may appear as **components** (a torch network, an
optimizer, a replay buffer, a sequence model). They cannot be the
*explanation* for why the method works.

## Substrate budget constraint — vector-stage floor clamping

**Pattern.** Across runs 15–18 (and earlier in runs 13–17 cited in
those hypotheses), every score-function / policy-gradient probe at the
vector stage (DST-concave + RG, 120s budget) lands at DST=99.0,
RG=0.011 for **both candidate and ablation**. `ablation_delta` is 0.0
on every env. The ablation comparison is vacuous because neither arm
departs the random floor within budget.

**Why this happens.** DST-concave and RG under the vector-stage 120s
budget provide too little compute for a policy-gradient method to find
even one non-nearest-treasure trajectory. The mechanism under test (e.g.
Pareto-frontier KDE, coverage growth, etc.) cannot fire until the policy
first discovers a non-trivial trajectory. At 120s, that discovery rarely
happens. The result is that candidate and ablation are both stuck at the
initial random-walk floor, and any mechanism difference is invisible.

**Constraint for the Researcher.** Do not claim a vector-stage win for a
mechanism whose novelty fires only *after* the policy departs the random
floor, unless a simpler warm-start (quick or sparse stage first) confirms
that some learning occurs within budget. If the quick/sparse stage shows
nonzero lift, the vector stage can be trusted. If quick/sparse also
yields floor scores, the mechanism is not learning on this substrate at
all and a different probe direction is needed.

---

## Appendix — attempt-to-family map

| Attempt | Name | Family | Sealed record |
| --- | --- | --- | --- |
| 1 | FROST | E, F | `worklogs/attempts/01-frost-vector-repair-certificate.md` |
| 2 | BRIC | (counterfactual cost) | `worklogs/attempts/02-bric-bracketed-reward-intervention.md` |
| 3 | KERNEL | (passive correlation) | `worklogs/attempts/03-kernel-rl-rsk.md` |
| 4 | CARL | (Go-Explore rebadge) | `worklogs/attempts/04-carl-frontier-graph.md` |
| 5 | BCE-v0 | (count rebadge on ablation) | `worklogs/attempts/05-bce-v0.md` |
| 6 | T-CTBP | G | `worklogs/attempts/06-t-ctbp.md` |
| 7 | OPP | F | `worklogs/attempts/07-opp-order-projection.md` |
| 8 | EOP/COP | F | `worklogs/attempts/08-eop-cop-minigrid.md` |
| 9 | Causal Dominance | (no clean primitive) | `worklogs/attempts/09-causal-dominance-certificates.md` |
| 10 | Value-first reset | (conceptual) | `worklogs/attempts/10-value-function-first-reset.md` |
| 11 | TOP | E | `worklogs/attempts/11-top-temporal-outcome-profiles.md` |
| 12 | PEO | F | `worklogs/attempts/12-peo-policy-edit-optimization.md` |
| 13 | ETB/HPC | (GCSL/options rebadge) | `worklogs/attempts/13-etb-hpc.md` |
| 14 | Primal Flow | E | `worklogs/attempts/14-primal-behavior-flow.md` |
| 15 | FED | A | `worklogs/attempts/15-fed-frontier-expanding-dispersion.md` |
| 16 | PICAV | A | `worklogs/attempts/16-picav-path-integrated-channel-asymmetry-voting.md` |
| 17 | CHX | A, C | `worklogs/attempts/17-chx-cumulant-hull-extremality.md` |
| 18 | CEC | A | `worklogs/attempts/18-cec-continuation-endpoint-concordance.md` |
| 19 | CWTP | A, B | `worklogs/attempts/19-cwtp-confluence-witness-trajectory-pairs.md` |
| 20 | LRA | A | `worklogs/attempts/20-lra-loop-return-aversion.md` |
| 21 | TPP | A, B | `worklogs/attempts/21-tpp-terminal-postfix-pairing.md` |
| 22 | CRP | A, C | `worklogs/attempts/22-crp-channel-rank-position-concordance.md` |
| 23 | TCP | A, C | `worklogs/attempts/23-tcp-temporal-channel-precedence.md` |
| 24 | PCR | A, D | `worklogs/attempts/24-pcr-policy-commitment-recovery.md` |
| 25 | ACS | A, C | `worklogs/attempts/25-acs-action-conditional-suffix-spectrum.md` |
| 26 | TRAC | A | `worklogs/attempts/26-trac-transition-refractive-action-channels.md` |
| 27 | PCGA | A, D | `worklogs/attempts/27-pcga-per-channel-gradient-alignment.md` |
| 28 | ARP | A | `worklogs/attempts/28-arp-action-reachable-pattern-lattice.md` |
| 29 | PFA | A | `worklogs/attempts/29-pfa-per-channel-phase-flow-asymmetry.md` |
| 30 | CPR | A | `worklogs/attempts/30-cpr-channel-posterior-chebyshev-reweight.md` |
| 31 | ATP | A, C, D | `worklogs/attempts/31-atp-action-tangent-persistence.md` |
| 32 | CSD | A | `worklogs/attempts/32-csd-channel-conditional-successor-disagreement.md` |
| 33 | ACFC | A | `worklogs/attempts/33-acfc-action-frequency-channel-frequency-concordance.md` |
| 34 | ACCD | A, D | `worklogs/attempts/34-accd-action-conditional-channel-dissociation.md` |
| 35 | PHI | C, D | `worklogs/attempts/35-phi-path-homology-invariants.md` |
| 36 | PRAR | A, C | `worklogs/attempts/36-prar-policy-regime-antisymmetric-residual.md` |
| 37 | HRC | A | `worklogs/attempts/37-hrc-horizon-recursive-concordance.md` |
| 38 | CSA | A | `worklogs/attempts/38-csa-channel-spectral-action-influence.md` |
| 39 | BLIC | A, D | `worklogs/attempts/39-blic-block-lookback-imminence-concordance.md` |
| 40 | CBI (negative closure) | H | `worklogs/attempts/40-cbi-negative-closure.md` |
