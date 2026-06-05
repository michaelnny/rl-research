# 20260606-13-auto — TRAC (Transition-Refractive Action Channels)

## Research Gate

primitive: per-(state-cluster, action, channel) Jensen-Shannon divergence
between two empirical successor-cluster distributions, partitioned by
whether channel `m` fired in the window after taking action `a` at
state-cluster `c`.
improvement_operator: at each decision step, Pareto-non-dominance logit
nudge over rows of `R[c, :, a] ∈ R^k` (the per-channel JSD vector for
each action), no scalar collapse, no reward weighting.
side_information: vector diagnostics, transition geometry.
nearest_prior_or_disqualifier: GVFs / successor features (disqualifier
family), CWAI (#cwai-channel-wise-action-influence, alive-promising).
falsifier: TRAC fails if (a) within the 120 s panel budget, fewer than
20 % of (cluster, action, channel) cells accumulate the minimum
sample-mass needed to estimate JSD reliably (≥ 8 fired-vs-not-fired
splits per cell with non-empty successor support on both sides), or (b)
on Resource Gathering the per-channel JSD vectors collapse to coordinate
collinearity (rank-1 across actions), reducing the Pareto-non-dominance
gate to a singleton scalar argmax.

## Mechanism

Each trajectory is a sequence of (observation, action, vector-feedback)
triples. We hash observations into a discrete cluster index `c` (BYOL or
random-projection LSH; specifics are an implementation choice). For each
(c, a, m) cell we maintain two empirical histograms over successor
clusters `c'` reached one window-step later (where the window `W` is a
small fixed integer, e.g. 4): `H_fire[c,a,m]` collects `c'` values from
trajectories where channel `m` fired anywhere in `[t+1, t+W]` after the
(c, a) event, and `H_nofire[c,a,m]` collects `c'` values from
trajectories where it did not. The primitive is
`R[c,a,m] = JSD(H_fire[c,a,m] || H_nofire[c,a,m])` — a non-negative
scalar per cell. The improvement operator at decision time computes, for
each candidate action `a`, the row vector `R[c,a,:] ∈ R^k_{≥0}`, then
performs a coordinate-wise Pareto-non-dominance count `n_a` (number of
actions whose row is Pareto-dominated by `R[c,a,:]`) minus `m_a`
(number that dominate it), and nudges policy logits by `α(n_a − m_a)`.
There is no critic, no Bellman backup, no scalarization across channels,
and no reward-weighted update; the only learning signal flowing into the
policy is the partial-order count over information-gain coordinates.

## Required candidate shape

1. **Experience object:** Ordinary trajectories — sequences of
   (observation, action, info["vector"]) triples — collected by the
   current stochastic policy. No counterfactuals, no resets, no
   demonstrations.

2. **Core primitive:** The non-negative tensor `R[c, a, m] ∈ R_{≥0}`
   defined as
   `R[c,a,m] = JSD(p(c'_{t+W} | c_t = c, a_t = a, fired_m_{[t+1,t+W]}) ||
   p(c'_{t+W} | c_t = c, a_t = a, ¬fired_m_{[t+1,t+W]}))`.
   This is a `k`-vector per (c, a) cell of conditional mutual-information
   surrogates between channel-firing events and future-state cluster.
   It is *not* an expected cumulant, *not* a value, *not* a return-to-go
   distribution — it is a divergence between two empirical conditional
   successor distributions partitioned by a binary channel-firing event.

3. **Improvement operator:** At decision step with cluster `c`, for each
   action `a` compute the per-channel row `R[c, a, :]`. Compute
   `n_a = Σ_{a'≠a} 1[R[c,a,:] ≻ R[c,a',:]]` (Pareto-strict-dominance
   count) and `m_a` (the reverse). Logit update:
   `logit(a | c) ← logit(a | c) + α (n_a − m_a)`. No scalar reward
   appears in this update.

4. **Execution rule:** Sample actions from softmax(logits + α(n−m)).
   When `R` is empty for a cell (cold start), the nudge is zero and the
   policy is whatever supervised initialization or prior gives
   (uniform). No greedy argmax; sampling is required because the
   primitive provides a partial-order direction, not a full ranking.

5. **Vector feedback rule:** The `k` channels of `info["vector"]` are
   *never* collapsed. Each channel induces an independent coordinate of
   `R`, and the improvement operator is partial-order Pareto on those
   coordinates. A channel that fires only at termination still
   contributes a non-degenerate coordinate, because the partition
   (fired anywhere in window vs. not) is still a Bernoulli event with
   non-trivial support — the JSD does not require the channel to fire
   *every* step, only that *some* trajectories fire it within the window
   and others do not.

6. **Rollout-cost discipline:** One environment step per environment
   step. Updating `R` is O(W) per step (incrementing two histograms per
   channel). No counterfactual rollouts, no verifier calls, no simulator
   branches. At deployment, decisions are O(|A| · k) for the
   Pareto-count over rows.

7. **Nearest-neighbor novelty audit:** Closest are (i) successor
   features / GVFs (disqualifier family) and (ii) CWAI
   (alive-promising). Against successor features: SF predicts an
   *expected* discounted cumulant feature-vector under the policy, used
   for value-like extraction; TRAC computes a *divergence* between two
   *conditional empirical distributions* over discrete successor
   clusters, never a discounted expectation, never extracted by inner
   product with a weight vector. Against CWAI: CWAI uses a learned
   forward-model Jacobian (a local linearization of *predicted next
   observation* w.r.t. action embedding); TRAC uses an *empirical
   conditional distribution* over already-visited successor clusters
   partitioned by a binary channel-firing event — there is no forward
   model, no gradient, no embedding, and the partition source is the
   feedback channel rather than the action argument.

8. **Predicted failure modes:**
   - On long-horizon sparse envs where, within 120 s, fewer than 20 %
     of (c, a, m) cells reach minimum-mass for reliable JSD (≥ 8
     fired-vs-not-fired splits with nonempty successor support each
     side), the primitive is silent and the operator is a no-op — same
     bootstrap wall that killed FED/CEC/TPP/SIT.
   - On Resource Gathering, where two of the channels (resource pickup,
     goal reach) fire only at terminal steps within rare successful
     trajectories, the fired-side histogram for those channels stays
     empty until the first success — predicted to underperform until at
     least one rewarded trajectory is in the buffer.
   - On stochastic-transition envs, if next-state cluster transitions
     are dominated by environment noise rather than action choice,
     `R[c, a, :]` will look identical across `a` for the same `c`,
     collapsing the Pareto-count to zero (curiosity-rebadge risk
     similar to CWAI's RG collapse).
   - If the cluster index `c` is too coarse, distinct states get fused
     and the JSD measures noise; if too fine, every cell is singleton
     and JSD is undefined. There is a sweet spot that may not exist on
     all panel envs simultaneously.
   - On envs where channels are perfectly anti-correlated with each
     other (e.g. a hard tradeoff is enforced), the Pareto front
     contains every action and the operator becomes vacuous — same
     saturation risk as ACS in 8-d.

9. **Side-information channel:** {vector diagnostics, transition
   geometry}. Vector diagnostics provides the per-channel firing
   indicators that partition the histograms. Transition geometry
   (state-cluster successor relation) provides the discrete `c'` random
   variable. No demonstrations, no language, no instrumentation, no
   pretrained priors are needed.

10. **Monotonic improvement claim:** The operator monotonically
    increases the policy's bias toward actions that are
    Pareto-non-dominated in their channel-conditional successor
    information. Equivalently, in the limit of infinite samples and
    perfect cluster identification, TRAC selects actions `a*` at each
    `c` such that for every other action `a'`, there exists at least
    one channel `m` with
    `I(c'; fired_m | c, a*) ≥ I(c'; fired_m | c, a')`,
    where `I` is the JSD-based MI surrogate, with strict inequality on
    at least one channel. Under stationarity of the cluster-transition
    kernel, repeated application strictly increases the
    coordinate-wise vector of channel-conditioned successor
    information at each `c` until a Pareto-stationary policy is
    reached.

## Why it is not GVFs / successor features / CWAI

Successor features and GVFs (disqualifier family): both predict
*expected* discounted cumulant trajectories under a policy and extract
behavior by linear scalarization with a task-weight vector. TRAC
computes a *divergence between two empirical conditional successor
distributions*, partitioned by a Bernoulli event on the channel — it
is not an expectation of a cumulant, it is not a TD-bootstrapped
target, and it is never combined with a task-weight vector. Against
CWAI (`cwai-channel-wise-action-influence`): CWAI's primitive is the
column-norm of a learned forward-model Jacobian — a *local
differential* of predicted observation in action-embedding space.
TRAC's primitive is an empirical *integral* (a JSD between two
histograms) over the realized future-state distribution conditioned on
a channel event. The two cannot be reduced to each other under
variable renaming: one is a learned local gradient magnitude, the
other is a model-free conditional-distribution divergence.

## Why it scales beyond the substrate

When the horizon grows to 20k actions and the action space is
"generate a paragraph" or "call an API", the per-action JSD primitive
remains well-defined: state-cluster `c` is replaced by a learned
embedding cluster from the policy's own representation, and the
Bernoulli channel-firing event on each vector-feedback dimension still
partitions trajectories into fired/not-fired groups. The successor
distribution is computed over a learned cluster index of future
representations, computable in O(1) per step regardless of action
cardinality. The primitive's cost scales with the number of vector
feedback channels `k`, which is constant and small (typically 3–8 for
agentic LLM systems: success, latency, cost, validity, safety, user
preference, refusal). The Pareto-non-dominance count operates on these
`k` coordinates and is index-free over actions — at decision time we
score only the candidate actions the policy is considering, never the
full action set. Critically, the *partition by channel event* is
exactly the structure that LLM-with-tools agents already produce:
"this tool call returned an error," "this code passed the test," "the
verifier rejected." Each is a binary channel-firing event whose
information about future trajectory is precisely what TRAC measures.
