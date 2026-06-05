# 20260606-02-auto — DPC: Divergent-Prefix Concordance

## Research Gate

primitive: per-divergence-event signed channel-vote tensor
  `V[s_div, a, a', k] ∈ Z^d`, indexed by the **divergence event**
  (last-shared-context `s_div`, two first-divergent actions `a, a'`,
  shared-prefix-length `k`) and accumulating the per-channel sign of the
  suffix vector-cumulant difference between the two trajectories of the
  pair.
improvement_operator: policy logit nudge at `s_div` proportional to the
  componentwise majority-channel margin of `V[s_div, a, a', ·]` summed
  over `a'`, gated by a Pareto-non-dominance condition on the underlying
  channel sums (no scalar collapse).
side_information: vector diagnostics, transition geometry.
nearest_prior_or_disqualifier: candidates/rsd-reconvergent-segment-dominance
  (most similar; structurally inverted — RSD compares **convergent**
  segments sharing both endpoints, DPC compares **divergent** segments
  sharing only the start), with secondary proximity to candidates/sit-
  suffix-inheritance-trie and disqualifier "MC-advantage."
falsifier: if after a 120 s panel run the per-decision invocation rate of
  the operator (fraction of decision steps where a non-zero
  Pareto-non-dominated channel-margin nudge fires) is below 5 % on the
  vector envs, OR the operator fires but produces below-random scores on
  Deep Sea Treasure / Resource Gathering, the primitive's
  bootstrap-bypass claim is wrong and the family is dead in the same way
  FED/SIT/RSD/VCC/PTW were.

## Mechanism

DPC's central object is the **divergence-event tensor**. For every pair
of trajectories `(τ_i, τ_j)` in the on-policy buffer of the current
training cycle, find their longest common action-prefix; let `k` be its
length, `s_div = s_k` the last shared context, `a_i, a_j` the two first
divergent actions, and `Δ_ij = c_T(τ_i) − c_T(τ_j) ∈ R^d` the
componentwise difference of their **terminal vector cumulants** (the
sum over per-step vector signals from step `k` onward, computed from
`info["vector"]` directly — not from scalar reward). The primitive
accumulates `sign(Δ_ij)[m] ∈ {-1, 0, +1}` into the slot
`V[s_div, a_i, a_j, m]` for each channel `m`. By construction this slot
is non-empty for *every* trajectory pair that ever differs, including
the very first pair where any two trajectories took different first
actions — so DPC has a non-trivial signal from rollout #2 onward,
without requiring obs-hash bucket coverage. The improvement operator
nudges policy logits at `s_div`: `logit(a) ←
logit(a) + α · ψ(V[s_div, a, ·, ·])` where `ψ` is the count of channels
on which `a`'s aggregated channel-margin against any alternative `a'` is
strictly positive **and** the underlying raw channel-sum vector
`Σ_pairs Δ` for that `(a, a')` is Pareto-non-dominated by the reverse
sum (so a single channel-flip in a multi-objective env cannot trigger
the nudge). No critic, no Bellman, no scalar reward weight — the only
inputs are vector diagnostics and the divergence-event index from
transition geometry.

## Required candidate shape

1. **Experience object:** Trajectories with per-step vector observations
   `v_t ∈ R^d` (read from `info["vector"]`, never from scalar reward),
   action sequences, and an observation-hash *only* used to locate the
   first-divergent step `k` in a trajectory pair (a cheap, near-free
   prefix-matching key, not a bucketing scheme).

2. **Core primitive:** The divergence-event tensor `V[s_div, a, a', m]`
   over all trajectory pairs in the recent buffer, where each cell holds
   the empirical sum of `sign((c_T(τ_i) − c_T(τ_j))[m])` over pairs whose
   first divergence is at `s_div` taking `a` vs. `a'`. The companion
   tensor `S[s_div, a, a', m] = Σ_pairs Δ_ij[m]` (raw cumulant-sum
   difference) is kept for the Pareto gate.

3. **Improvement operator:** At each `s_div`, for each candidate next
   action `a`, compute
   `μ(a) = Σ_{a'≠a} 1[ V[s_div, a, a', ·] componentwise-dominates 0
                       ∧ S[s_div, a, a', ·] is Pareto-non-dominated by
                          S[s_div, a', a, ·] ]`.
   Update the policy via a small SGD step on the cross-entropy between
   the current `π(·|s_div)` and the softmax of `α · μ(·)`. There is no
   reward in this expression and no scalar collapse of the channel
   vector.

4. **Execution rule:** Sample actions from the current policy with a
   fixed temperature throughout (no separate exploration policy). The
   policy itself is the only trajectory generator. Divergence events are
   created naturally because the policy is stochastic.

5. **Vector feedback rule:** Each channel `m` is treated independently
   throughout: the sign tensor is per-channel, the Pareto-non-dominance
   gate is per-channel, the channel-margin count `μ(a)` is the number of
   channels on which `a` beats some `a'`. There is no single point in the
   pipeline where the channels are summed with weights `w` — substituting
   `wᵀv` for any fixed or learned `w` would change the operator's
   output, which is the test that this is not a scalarization rebadge.

6. **Rollout-cost discipline:** One trajectory per training step, as in
   on-policy PG. No counterfactual rollouts, no simulator branches, no
   verifier calls. The divergence-event tensor is built from natural
   trajectory pairs already in the buffer; pair count grows quadratically
   with buffer size but the operator only consults the tensor, not the
   raw pairs, at decision time. Per accepted improvement: zero extra env
   interactions over a vanilla rollout.

7. **Nearest-neighbor novelty audit:** Closest prior is
   `candidates/rsd-reconvergent-segment-dominance.md`. RSD aggregates
   over **convergent** segments — pairs sharing both a start hash *and*
   an end hash — which is why it stalled (closed witness pairs are
   exponentially rare on long-horizon sparse envs). DPC inverts this:
   it aggregates over **divergent** segments sharing only the start hash,
   which are abundant because every distinct trajectory pair has at
   least one divergence event. SIT (suffix-inheritance trie) is also
   nearby: SIT *grafts* a Pareto-dominant suffix as a behavior-clone
   replacement — DPC never grafts trajectories; it only emits a logit
   nudge at the divergence point. The MC-advantage disqualifier: MC
   advantage is `Σ_τ (R(τ) − b) ∇log π(a_t|s_t)` for *all* steps in
   *all* trajectories with a *scalar* return; DPC's update fires only at
   first-divergence steps of trajectory *pairs* and uses per-channel
   sign-votes that are not algebraic returns.

8. **Predicted failure modes:**
   - **Stochastic transitions:** if the same `(s_div, a)` produces
     varied immediate next-states across pairs, the channel-sign vote
     is dominated by transition noise rather than action choice; the
     operator becomes a policy-shrink toward marginal action frequency.
   - **Channel collinearity:** if all `d` channels are positive monotone
     transformations of one underlying signal (true on Deep Sea Treasure
     where the only non-step-penalty channel is terminal treasure), the
     per-channel sign-vote collapses to a single sign and the
     Pareto-non-dominance gate is trivially satisfied — DPC degrades to
     a pairwise-signed-MC-advantage rebadge on collinear-channel envs.
   - **Long shared prefix:** on envs where the policy quickly converges
     to a near-deterministic prefix, divergence events concentrate at
     small `k` (early in the trajectory) and the operator never fires
     on late-trajectory decisions — the long-horizon credit assignment
     it claims to deliver doesn't actually fire late in the episode.
   - **Observation-hash collision at `s_div`:** if hash collisions are
     systematic (two semantically different states map to the same
     `s_div`), the tensor cells aggregate over inconsistent decision
     points; the Pareto-non-dominance gate may still hold but the nudge
     direction is wrong. Diagnostic: report
     `mean #pairs / unique-(s_div, a, a')` after each rollout phase.

9. **Side-information channel:** {vector diagnostics, transition
   geometry}. Vector diagnostics: per-step `info["vector"]` channel
   values are the only signal driving the sign votes. Transition
   geometry: the divergence-event index requires identifying the
   first-divergent step in a trajectory pair, which is a transition-
   structural property of the rollout (action prefix matching). No
   demonstrations, no language, no learned dynamics, no verifier.

10. **Monotonic improvement claim:** Under the assumption that the env's
    transitions are deterministic and that the per-step vector signal
    `v_t` is independent of trajectory identity beyond the action
    sequence (i.e., a fixed-MDP setting), the DPC operator monotonically
    increases the **expected per-channel sign-margin** of the policy's
    decision distribution against itself: the policy's distribution at
    `s_div` puts mass on `a` such that the expected sign of
    `(c_T(τ | a, π) − c_T(τ' | a', π))[m]` is positive for the majority
    of channels `m`, jointly across all `a' ≠ a`. This is a per-channel
    stochastic-dominance refinement — distinct from value monotonicity
    because no scalar combination of channels is being maximized.

## Why it is not RSD / SIT / MC-advantage

RSD (`candidates/rsd-reconvergent-segment-dominance.md`) requires the
two compared segments to share **both** start and end hashes; on
long-horizon sparse envs closed witness pairs are exponentially rare and
the operator was a no-op (scored 0 across all envs). DPC drops the
shared-end-hash condition entirely — the pairs only share a start, and
their *full terminal cumulants* are compared, so every distinct pair in
the buffer contributes a non-zero update. SIT
(`candidates/sit-suffix-inheritance-trie.md`) grafts entire suffixes
between trajectories and verifies with extra rollouts, paying
counterfactual-rollout cost; DPC never substitutes a behavior, only
nudges logits at one decision point per pair and pays zero extra
rollout cost. Versus MC-advantage / scalar-weighted log-prob (the
disqualifier): scalar PG would compute `Σ_τ R(τ) ∇log π(a_t|s_t)` on
every step of every trajectory with a single scalar return; DPC fires
only at *pairwise first-divergence steps* and uses per-channel sign
votes plus a Pareto-non-dominance gate — substituting any `wᵀv` for
the channel vector demonstrably changes the output (the gate's
satisfaction depends on per-channel signs, not on a weighted sum).

## Why it scales beyond the substrate

At 20 k-action horizons with vector feedback, the divergence-event
tensor's signal density is governed by **how often the policy makes
distinguishable decisions**, not by how often it sees reward. A
transformer policy choosing among "generate a paragraph" actions
naturally produces a fan-out tree of stochastic divergences at every
context; pairs of rollouts in the on-policy buffer always have a
first-divergence point, and the per-channel sign-vote mechanism
operates identically whether the channel is a verifier verdict
(correctness), a tool-use latency, a token-budget cost, or a user
preference. The primitive does not depend on hashing the LLM context;
it only depends on locating the first action-token where two rollouts
diverged, which is a `O(min(|τ_i|, |τ_j|))` prefix-match — well-defined
on token sequences. The per-channel structure means a 6-component
feedback vector (success, cost, safety, latency, validity, preference)
drives 6 independent sign-votes whose Pareto-aggregation never
collapses to a hand-tuned weighting; the engineering team can
re-prioritize channels at deployment by re-aggregating the same tensor
without retraining. The bootstrap behavior is the same at scale: even
the very first two rollouts on a fresh task, if they differ at all,
produce one divergence event and one DPC update — there is no
"sufficient bucket sample mass" precondition that long horizons can
defeat.
