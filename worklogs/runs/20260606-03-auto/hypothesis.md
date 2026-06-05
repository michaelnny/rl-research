# 20260606-03-auto — CWTP (Confluence-Witness Trajectory Pairs)

## Research Gate

primitive: per-(state-hash, action, action') sign-vote tensor of per-channel
segment-cumulant differences between two trajectories that *diverged* at
that state with different actions and *reconverged* at a later shared
observation-hash, taken at the **latest divergence preceding the
confluence**.
improvement_operator: at decision time, nudge policy logits at state s
toward actions a whose accumulated per-channel sign-vote profile across
all (s, a, a') confluence-witness pairs is Pareto-non-dominated by the
reverse a' → a profile — no scalar collapse, no reward weighting, no
critic.
side_information: transition geometry, vector diagnostics.
nearest_prior_or_disqualifier: 18-CEC (exit-hash conditioning) and the
candidates `rsd-reconvergent-segment-dominance` (parallel-edge multigraph
aggregation) and `dpc-divergent-prefix-concordance` (first-divergence
sign votes on terminal cumulants).
falsifier: if, after a 30 s seeding phase of uniform-random rollouts on
the panel's vector envs, the rate of confluence-witness pairs (pairs of
buffered trajectories sharing any non-terminal observation-hash and having
distinct preceding actions at the latest such confluence) is below 1 per
buffer-thousand-steps, the operator is silent and the candidate is dead.
A second falsifier: if the operator fires at non-trivial rate but vector
envs (Deep Sea Treasure, Resource Gathering) score below random, the
sign-vote primitive does not carry the right structural information.

## Mechanism

Maintain a buffer of completed trajectories with their full action and
per-step vector-cumulant traces and observation-hash sequence. For each
unordered pair of trajectories `(τ_i, τ_j)` in the buffer, scan their
observation-hash sequences and identify all *confluence pairs*
`(t_i, t_j)` where `obs_hash(τ_i, t_i) == obs_hash(τ_j, t_j)`. For each
such confluence, walk *backwards in both trajectories simultaneously* to
find the **latest divergence** — the largest `(d_i, d_j)` with
`d_i ≤ t_i, d_j ≤ t_j`, `obs_hash(τ_i, d_i) == obs_hash(τ_j, d_j) = s_div`,
and `action(τ_i, d_i) ≠ action(τ_j, d_j)`. (If no divergence exists
within the segment back to the trajectory starts, skip the confluence.)
The confluence witness is the tuple `(s_div, a_i, a_j, Δv)` where
`Δv = (Σ_{u=d_i+1..t_i} v_u^{τ_i}) − (Σ_{u=d_j+1..t_j} v_u^{τ_j}) ∈ R^k` —
the per-channel difference of segment cumulants between divergence and
reconvergence. Maintain a sign-vote tensor `V[s_div, a, a', m] ∈ Z` that
accumulates `sign(Δv[m])` (with the segment ordered so `a = a_i`,
`a' = a_j`). At training time, for each state s with two or more witnesses
involving distinct actions, compute for each pair `(a, a')` the
sign-vote vector `v(a,a') = V[s,a,a',:] / |V[s,a,a',:]|_1`. Action `a`
*dominates* `a'` at s iff `v(a,a')` is coordinate-wise ≥ 0 with at least
one strict inequality. Apply a logit nudge `Δlogit(s,a) = +α · #{a' :
a dominates a'} − α · #{a' : a' dominates a}`. This is the entire
primitive.

## Required candidate shape

1. **Experience object:** completed trajectories with per-step
   observation-hash, action, and vector signal `v_t ∈ R^k` from
   `info["vector"]`. Buffer size bounded by memory; trajectories evicted
   FIFO.
2. **Core primitive:** sign-vote tensor `V[s, a, a', m]` over per-channel
   segment-cumulant difference signs at latest-divergence-before-confluence
   events between trajectory pairs sharing an observation-hash.
3. **Improvement operator:** logit nudge by Pareto-non-dominance count of
   normalized sign-vote rows; no scalar weighting, no reward weighting, no
   critic, no value backup.
4. **Execution rule:** sample actions from softmax over (logits + nudge)
   with temperature `τ` annealed from 1.0 to 0.5 over training. The base
   logits come from a small policy network trained with a maximum-entropy
   regularizer on top of the nudges.
5. **Vector feedback rule:** per-channel sign votes stay separated through
   the entire pipeline. Pareto-non-dominance on the sign-vote vector
   replaces any scalar collapse. Channels with degenerate (always-zero)
   votes contribute nothing and are ignored automatically — no manual
   weighting.
6. **Rollout-cost discipline:** one rollout per training trajectory; no
   counterfactual rollouts, no simulator branching, no verifier calls.
   Confluence-witness extraction is offline post-processing on the buffer
   at update time. Per-update cost is `O(B^2 · L^2)` in worst case where
   `B` is buffer size and `L` is trajectory length, but practically
   `O(B^2 · H)` where `H` is the number of observation-hash collisions
   per pair (small on compact envs).
7. **Nearest-neighbor novelty audit:**
   - vs **RSD (rsd-reconvergent-segment-dominance):** RSD aggregates *all*
     parallel segments between the same (u,v) endpoint pair into one
     Pareto comparison and requires multigraph edge density between
     reward-bearing endpoints. CWTP does *not* aggregate — each
     individual confluence witness contributes one sign-vote per channel,
     indexed only by the divergence state, and explicitly takes the
     **latest** divergence so the comparison segment is local to the
     action choice itself. CWTP also does not require (u,v) pairs to be
     reward-bearing; per-step channels (step-penalty, energy, safety) fire
     continuously and produce non-zero `Δv` even on confluence pairs in
     unrewarded regions.
   - vs **DPC (dpc-divergent-prefix-concordance):** DPC uses the
     *first-divergence* of trajectory pairs and compares **terminal**
     cumulants — the comparison is contaminated by all downstream
     stochasticity. CWTP uses the *latest divergence before a confluence*,
     so by construction the downstream of the confluence is shared and
     drops out of the difference; only the **bracketed segment**
     `[d_i+1..t_i]` vs `[d_j+1..t_j]` contributes to `Δv`.
   - vs **CEC (#18, exit-hash conditioning):** CEC indexes by the
     *terminal* exit-hash of the entire trajectory and required
     coverage of (state, action, exit-hash) triples that never accumulated.
     CWTP indexes by intermediate observation-hash *as confluence*, not
     by exit-hash, and the comparison is between two trajectories rather
     than between a trajectory and a bucket distribution.
8. **Predicted failure modes:**
   - On envs with very large or near-injective observation-hash (Craftax
     pixel-style), confluence pairs are rare; expect operator silence.
   - On stochastic-transition envs (Resource Gathering with stochastic
     dynamics), the `obs_hash(τ, t) == obs_hash(τ', t')` condition may
     hold while the *latent* state differs; sign votes will be noisier
     than on deterministic envs.
   - On envs where the only meaningful channel is terminal-only and there
     is no per-step vector signal at all, `Δv` is zero on every
     non-terminal segment and the operator collapses to silence — same
     bootstrap wall as FED/PICAV. The substrate's vector envs (DST has
     step penalty + terminal treasure; RG has continuous resource counts)
     do have per-step structure, so this is a survivable design point on
     the substrate but a real concern at scale.
   - If the policy collapses to deterministic before confluences form
     (because nudges get too strong), the sign-vote tensor stops growing
     and the operator stagnates. Mitigation: maintain a minimum entropy
     floor on the policy.
9. **Side-information channel:** transition geometry (the
   observation-hash equivalence we use to detect confluences is itself a
   geometric property of the env's transition structure) and vector
   diagnostics (per-channel sign votes are the load-bearing primitive,
   not scalar reward).
10. **Monotonic improvement claim:** under the assumption that the
    transition map is deterministic and the observation-hash is
    state-injective on visited states, any single confluence-witness
    pair `(s_div, a, a', Δv)` is a *valid local certificate* that taking
    `a` instead of `a'` at `s_div` yields a vector outcome difference of
    exactly `Δv` *conditional on the policy's downstream behavior being
    the rest of `τ_i` and `τ_j`*. Under uniform mixing over downstream
    behaviors (i.e., over the empirical distribution induced by the
    buffer), the *expected* per-channel sign-vote at `(s, a, a')`
    converges to the sign of the conditional advantage of `a` over `a'`
    on each channel, *as a per-channel quantity*. The improvement
    operator therefore monotonically improves the policy's
    Pareto-non-domination count at each visited state under a
    sufficiently-mixing buffer and a deterministic transition map. The
    operator does *not* claim monotonic improvement under stochastic
    transitions — that is an explicit predicted failure mode.

## Why it is not RSD / DPC / CEC

CWTP is not RSD because RSD aggregates *all* parallel (u,v) segments
into a single multigraph edge and Pareto-compares the aggregate; under
variable renaming, RSD's primitive is `argmax_a Pareto-domination over
edge_means_aggregated_per_endpoint_pair`, which is bucket-dense.
CWTP's primitive is `sign-vote over individual (s_div, a, a', Δv)
witnesses keyed by divergence state alone`, so it does not require
bucket density at the endpoint pair — every shared-hash event between
any two buffered trajectories produces a witness. CWTP is not DPC
because DPC's `Δ` is the **terminal** cumulant difference, contaminated
by every downstream choice; CWTP's `Δv` is the bracketed
segment-between-divergence-and-confluence cumulant difference, with
downstream identical by construction. CWTP is not CEC because CEC
indexes by exit-hash buckets that need coverage; CWTP indexes by
divergence-state and the comparison primitive is pairwise sign votes,
not bucket-conditional Pareto fronts.

## Why it scales beyond the substrate

At long horizons (10k–20k steps) and large action spaces, two
trajectories generated by the same policy will share many intermediate
observation-equivalence-class events — recurring tool calls, recurring
intermediate states in a multi-step task, recurring high-level subgoals.
The "latest divergence before a confluence" primitive does not depend
on the action space being enumerable: it only requires *recognizing that
two trajectories visit the same intermediate state*. For an LLM agent
with action = "generate a paragraph", the relevant "observation-hash" is
a learned semantic equivalence (e.g., embedding-cluster of the
working-memory state), and the vector channel structure (cost, latency,
correctness, safety, validity) accumulates per-step regardless of
whether the terminal reward fires. The per-channel sign-vote
accumulation continues to fire even at 20k-step horizons because *every
confluence pair contributes one witness regardless of trajectory
length*; the primitive's signal-rate scales with `B^2 · H̄` (buffer-size
squared times average confluences per pair), independent of horizon.
For vector feedback at scale, the sign-vote tensor naturally preserves
all `k` channels in their native ordinal form — there is no fixed weight
vector to tune, and the Pareto-non-dominance count automatically adapts
to which channels are informative on this state.
