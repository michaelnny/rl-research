# 20260606-14-auto — TPM (Trajectory-Pair Mass-Transport)

## Research Gate

primitive: per-(episode-position `t`, action `a`) sliced per-channel 1-D
Wasserstein gap `M[t,a] ∈ R^k` between two empirical distributions of
terminal vector cumulants — episodes that took `a` at position `t` and
episodes that took some other action at position `t`.
improvement_operator: at decision step `t`, nudge policy logits toward
actions whose row `M[t,a,:]` is Pareto-non-dominated (coordinate-wise)
in the partial order on `R^k`, with margin equal to the count of
dominated rows minus dominating rows. No scalar collapse.
side_information: vector diagnostics (`info["vector"]`); episode-position
index (a substrate-free temporal coordinate, not a state hash).
nearest_prior_or_disqualifier: candidate dpc-divergent-prefix-concordance
(closest), and indirectly the scalar-weighted log-prob update
disqualifier family (PPO/REINFORCE).
falsifier: if on the vector envs (Deep Sea Treasure, Resource Gathering)
the fraction of decision steps at which the Pareto gate produces a
non-zero margin is below ~5 % during the 120-s budget, OR the panel
score on both vector envs is below random, the primitive is rejected
as non-informative under episode-position indexing.

## Mechanism

Maintain a buffer `B` of recent completed episodes, each storing the
action sequence `(a_0, a_1, …, a_{T-1})` and the terminal vector
cumulant `c_T ∈ R^k`. For every position index `t` and every action
`a`, partition `B` into `B^+_{t,a}` (episodes whose action at step `t`
equaled `a`) and `B^-_{t,a}` (episodes whose action at step `t` differed
from `a`, restricted to episodes of length > `t`). For each channel
`m ∈ {1..k}`, compute the signed 1-D Wasserstein distance between the
two empirical 1-D distributions `{c_T(τ)[m] : τ ∈ B^+_{t,a}}` and
`{c_T(τ)[m] : τ ∈ B^-_{t,a}}`, using the median-difference shortcut so
the sign carries the direction of channel-shift. Stack across `m` to
obtain `M[t,a,:] ∈ R^k`. The improvement operator is a logit nudge:
`logit(π_θ(a | s_t, t)) ← logit(π_θ(a | s_t, t)) + α · margin(M[t,a,:])`
where `margin(v) = |{a' ≠ a : v Pareto-dominates M[t,a',:]}| −
|{a' ≠ a : M[t,a',:] Pareto-dominates v}|`. The policy is conditioned on
both observation `s_t` and the position index `t` (or a coarse bucket
of it) so that the position-keyed nudge is delivered to the right
decision context. No critic, no Bellman, no scalar reward weighting, no
elite cloning, no graft+verify.

## Required candidate shape

1. **Experience object:** completed episodes — action sequences plus
   terminal vector cumulants `c_T ∈ R^k` from `info["vector"]`. No
   state-hash, no event tags, no observation buckets, no per-step
   diagnostics required.
2. **Core primitive:** the per-(episode-position `t`, action `a`) signed
   per-channel 1-D Wasserstein vector
   `M[t,a,:] ∈ R^k = (W_1^{signed}(B^+_{t,a}, B^-_{t,a}, m))_{m=1..k}`,
   where each component is the signed difference of channel-`m`
   medians between the two terminal-cumulant subpopulations (a
   bias-robust 1-D Wasserstein surrogate). Indexed by **episode position
   only** — never by state-hash, observation-hash, cumulant-bucket, or
   exit-hash.
3. **Improvement operator:** logit nudge by Pareto-non-dominance margin
   `α · (n_a − m_a)` where `n_a / m_a` are counts of actions `a'` that
   `M[t,a,:]` dominates / is dominated by, computed coordinate-wise in
   the partial order on `R^k`. The nudge is delivered to the policy
   head conditioned on `(s_t, t)`. No scalar collapse, no weighting by
   reward magnitude, no critic.
4. **Execution rule:** sample `a_t ∼ softmax(logit_θ(s_t, t))` during
   data collection. Episode position `t` is part of the policy's
   conditioning input (a small position embedding). The position
   coordinate is a substrate-free coordinate that always exists; it
   does not require any environment instrumentation.
5. **Vector feedback rule:** the primitive operates per-channel from
   end to end. `M[t,a,:]` keeps each of the `k` channels separate; the
   Pareto gate uses the coordinate-wise partial order on `R^k`. There
   is never a step where the channels are summed, weighted, or
   projected onto a scalar. On vector envs the operator's directional
   nudge is exactly the multi-channel non-domination structure; on
   scalar envs `k = 1` and the operator degenerates to "prefer actions
   whose terminal-return median exceeds the others' at this position",
   which it must be flagged as scalar in that limit.
6. **Rollout-cost discipline:** one environment interaction per data
   point; one improvement update per `N`-episode buffer refresh
   (concrete `N` is an engineering knob, not a counterfactual). Zero
   counterfactual rollouts, zero verifier calls, zero simulator
   branches. Cost = wall-clock of vanilla on-policy collection + a
   small constant per update for sorting per-position cumulant
   subpopulations.
7. **Nearest-neighbor novelty audit:** Closest prior is
   `dpc-divergent-prefix-concordance` (alive-weak): DPC indexes the
   sign-vote tensor by *first-divergence-state* `s_div`, requiring
   trajectory pairs to share an observation-prefix and split there.
   TPM indexes by *episode-position* `t` alone — pairs need only have
   length ≥ `t`, never need to share any observation. This dissolves
   DPC's hash-collision dependence and reaches further into the
   episode. Distinction from REINFORCE/MC-advantage (the
   scalar-weighted log-prob disqualifier): MC-advantage weights
   `∇log π(a|s)` by a *scalar* `(R(τ) − b(s))` summed across channels;
   TPM never weights by a scalar — it nudges the logit by a *count* of
   coordinate-wise vector dominations, which depends on the partial
   order, not on any (signed-or-unsigned) scalar magnitude. Distinction
   from return-to-go conditioning: there is no conditioning on a
   target return; `M` is a *between-population* distributional shift,
   not a *within-trajectory* return projection.
8. **Predicted failure modes:**
   (a) when episode lengths vary widely, late-position cells `M[t,·,:]`
   for large `t` get sparse and the Pareto margin saturates at zero —
   so on environments where successful episodes are dramatically
   shorter or longer than failed ones the late-position nudge is
   silenced;
   (b) when terminal-cumulant variance within `B^+_{t,a}` is large
   relative to the mean shift between `B^+_{t,a}` and `B^-_{t,a}`
   (high stochasticity, e.g. Resource Gathering with random ore
   spawns), the median-difference signal is dominated by sampling
   noise and the partial order on `R^k` becomes near-random;
   (c) on scalar-reward envs (`k=1`) the operator degenerates to a
   median-shift weighted MC update — must be flagged as scalar in this
   limit and not claimed as a vector-method success there;
   (d) when the policy's exploration is highly concentrated, `B^+_{t,a}`
   and `B^-_{t,a}` may not both be non-empty for many `(t,a)` — early
   training will need either an entropy floor or a small action-dither
   to keep both subpopulations populated;
   (e) cold-start: until at least one episode hits a terminal-only
   channel above its baseline, that channel contributes only its
   default value and the partial order on `R^k` has effective rank
   `k − k_terminal_silent` — diagnostic logging of per-channel
   between-subpopulation median spread is required to verify the
   primitive is engaging.
9. **Side-information channel:** {vector diagnostics, transition
   geometry}. Vector diagnostics is the explicit channel: every
   `info["vector"]` reading is consumed verbatim, never collapsed.
   Transition geometry enters only implicitly via the episode-position
   coordinate — a coordinate that always exists, requires no
   environment instrumentation, and is not a hand-engineered event
   lens.
10. **Monotonic improvement claim:** under stationary action-position
    distributions and assuming the buffered episodes are i.i.d. from
    the current policy, the operator monotonically increases the
    expected per-channel terminal cumulant `E[c_T[m]]` for every
    channel `m` simultaneously, in expectation over `(t, a)` cells
    where `M[t,a,:]` is *strictly* Pareto-non-dominated. On cells
    where no strict non-domination holds the operator is the
    identity; on cells where it does hold, the logit-nudge points
    along a per-channel non-dominated improvement direction in `R^k`,
    so no channel decreases in expectation while at least one strictly
    increases. Stochasticity weakens "monotonic" to "monotonic in
    median terminal cumulant" (median-Wasserstein surrogate).

## Why it is not dpc-divergent-prefix-concordance / scalar-weighted log-prob

DPC's sign-vote tensor `V[s_div, a, a', m]` is keyed by *observation
state at first divergence*: a pair contributes only if the two
trajectories share a prefix and split at a hashable observation state.
That is a state-hash dependency that becomes silent on long-horizon
sparse envs whenever observation-hash collisions are rare (the
underlying reason DPC tied random on Resource Gathering). TPM's index
is `(episode-position, action)` — every trajectory pair of length
≥ `t` contributes a sample to `M[t,·,:]`, with no requirement that the
trajectories ever share an observation. The coordinate is the agent's
own clock, not the environment's state. Distinction from
PPO/REINFORCE/GRPO (the scalar-weighted log-prob disqualifier): those
multiply `∇log π(a|s)` by a scalar derived from `(R(τ) − b)` or a
clipped advantage; under variable renaming they are linear-in-scalar.
TPM's update coefficient is `count_dominated − count_dominating`, an
integer derived from the *partial order* on `R^k` — it is invariant
to per-channel monotone reparameterizations of `c_T` and explicitly
reflects which channels improve, so collapsing `c_T` to a single scalar
weight `wᵀc_T` would change the dominance graph and hence the operator,
demonstrating it is not a scalarized rebadge.

## Why it scales beyond the substrate

Episode-position indexing extends to long-horizon settings without
modification — the primitive's only requirement is that two episodes
share a common length prefix at some position `t`. For 20 k-action
horizons this position grid can be discretized into log-spaced buckets
so storage stays sub-linear in horizon. For action spaces of "generate a
paragraph" or "call this tool", actions are not enumerable; in that
regime `a` is replaced by a learned action-class embedding (e.g. a
clustering of action-token-sequence embeddings produced by the base
LLM) and `M[t, c, :]` is computed per-class — the logit nudge becomes a
per-class energy adjustment on the policy's softmax over emitted
actions, identical in form to the discrete case. For 6-component vector
feedback (success / cost / safety / latency / energy / correctness),
the partial order on `R^6` is exactly what the primitive needs; the
Pareto-non-dominance margin is well-defined and never invokes any
fixed weighting `w`. The bottleneck — that on vector envs with
terminal-only channels we need *some* successful trajectories before
those channels' median-shifts become informative — is exactly the
cold-start condition specified in failure mode (e); it is the same
bottleneck any reward-honest method faces, and it is *not* compounded
by an additional state-hash bottleneck the way DPC, FED, CEC, TPP, and
RSD were.
