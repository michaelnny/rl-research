# 20260606-18-auto — PFA (Per-Channel Phase-Flow Asymmetry)

## Research Gate

primitive: per-(observation, action, channel) signed phase-area swept in
a 2-D channel-imminence plane during one observed transition, where the
plane's coordinates are learned short-horizon and long-horizon
firing-probability scalars `(p_m(o), q_m(o))` for channel `m`.

improvement_operator: per-action k-vector of running-mean signed
phase-areas across channels; Pareto-non-dominance logit nudge in the
coordinate-wise partial order on R^k. No scalar collapse.

side_information: vector diagnostics, learned dynamics (the two
firing-probability heads `p_m, q_m`).

nearest_prior_or_disqualifier: CID-canonical
(`worklogs/candidates/cid-channel-imminence-differential.md`, parked
`failed-implementation`) — a per-channel one-step log-likelihood-ratio of
firing on the next observation. Also adjacent: CWAI (Jacobian magnitude),
GVF / successor features (disqualifier family).

falsifier: if, after a 120 s vector-stage run, the per-channel signed
phase-area mean is collinear within 1 e-2 across actions on Deep Sea
Treasure (so the Pareto vote is symmetric and the operator never fires
informatively), or if the signed area on Resource Gathering stays at the
sign-noise floor (median |area| within 5% of the random-data baseline)
because terminal-only channels make `p_m, q_m` indistinguishable, then
PFA collapses to the same bootstrap-wall pattern as the FED family and
the rotational-invariant claim is empirically void.

## Mechanism

For each vector channel m we train two small probability heads on the
replay buffer: `p_m(o) ≈ P(fire_m at next step | o)` and `q_m(o) ≈ P(fire_m
within next H steps | o)`, with H a fixed horizon (e.g., 16). At each
buffer transition `(o_t, a_t, o_{t+1})`, define the 2-D imminence vectors
`φ_t = (p_m(o_t), q_m(o_t))` and `φ_{t+1} = (p_m(o_{t+1}), q_m(o_{t+1}))`.
The primitive is the **signed phase-area swept** on the transition:
`A_m(o_t, a_t) = φ_t × φ_{t+1} = p_m(o_t)·q_m(o_{t+1}) − p_m(o_{t+1})·q_m(o_t)`
— the z-component of the cross product of the two imminence vectors. We
maintain a per-(observation-cluster c, action a) running mean
`Ā[c, a, m]` over all observed transitions in that cell. The improvement
operator at decision time looks up the row `Ā[c(o), a, :] ∈ R^k` for each
action a, and applies a logit nudge `α · (n_a^{dom} − m_a^{dom})` where
`n_a^{dom}` is the count of other actions whose row is coordinate-wise
dominated by a's row, and `m_a^{dom}` is the count of actions dominating
a's row, both under the partial order on R^k. No critic, no Bellman, no
scalar reward, no scalar weighting of channels. Cluster `c(o)` is a
fixed-radius hash of the policy-trunk activation; cluster choice is
mechanically separate from the primitive.

## Required candidate shape

1. **Experience object:** standard replay of transitions
   `(o_t, a_t, o_{t+1}, v_{t+1}, done)` where `v_{t+1} ∈ R^k` is the
   `info["vector"]` channel reading. No counterfactual rollouts, no
   verifier, no demonstrations.

2. **Core primitive:** per-(cluster, action, channel) running-mean signed
   2-D phase-area `Ā[c, a, m]`, where the 2-D coordinates per channel are
   learned firing probabilities at two horizons. The signed area is a
   *rotational* quantity: it captures whether the transition curves the
   imminence trajectory (e.g., short-horizon up while long-horizon
   stable, indicating "imminent firing") rather than translates it
   uniformly.

3. **Improvement operator:** at each decision step, build the per-action
   matrix `M ∈ R^{|A| × k}` from `Ā[c(o), :, :]`. Compute the
   coordinate-wise Pareto-non-dominance count vector. Add
   `α · (n_a^{dom} − m_a^{dom})` to action a's pre-softmax logit. No
   weight, no scalar collapse. Sample from the resulting categorical for
   data collection.

4. **Execution rule:** sample `a ∼ softmax(z(o) + α · (n^{dom} − m^{dom}))`
   where `z(o)` is the policy-network logit and the nudge is recomputed
   from the current `Ā` table. No greedy override, no temperature
   schedule beyond a single fixed entropy floor.

5. **Vector feedback rule:** the k channels never get scalarized. Each
   channel m has its own `(p_m, q_m)` head, its own signed-area row in
   `Ā`, and the Pareto vote operates over the k-dimensional signed-area
   row directly under coordinate-wise partial order. No `wᵀr` step
   anywhere in the pipeline — including no implicit weighting through a
   shared trunk (the heads are channel-specific).

6. **Rollout-cost discipline:** one transition = one update contribution;
   no counterfactual replays, no simulator branches, no verifier calls.
   Per gradient step the cost is one minibatch forward/backward through
   the heads + one decision-time lookup against `Ā`. Per accepted
   improvement: zero extra environment cost beyond the on-policy
   trajectory that produced the transition. At deployment: just the
   logit-nudge lookup, O(|A| · k) per step.

7. **Nearest-neighbor novelty audit:** CID-canonical
   (`cid-channel-imminence-differential.md`) computes a 1-D per-channel
   log-likelihood ratio `LR_m = log q_m(o_{t+1}) − log q_m(o_t)`. PFA
   computes a 2-D rotational invariant — the cross-product of two
   imminence vectors at two horizons. The structural distinction is
   *rotation vs translation*: CID fires when the imminence magnitude
   changes; PFA fires when the imminence direction curves (e.g.,
   short-horizon imminence rises while long-horizon imminence falls,
   meaning "this action commits to a near-firing trajectory at the
   expense of long-horizon options" — a signed quantity invariant to
   any per-channel scalar rescaling of probabilities). PFA also differs
   from CWAI (Jacobian column-norm magnitude under stochastic
   transitions; PFA does not use the model's parameter gradient) and
   from GVF / successor-feature methods (PFA's per-channel head is a
   probability of firing, not a discounted cumulant; the action update
   does not consume the head's expected-value output, only the signed
   2-D area swept on the actually-observed transition).

8. **Predicted failure modes:**
   - (a) If `p_m` and `q_m` collapse to near-equal scalar functions of o
     (the two horizons are not resolved by the head architecture), the
     signed area is identically zero per transition and the operator is
     silent. Falsifier: log per-channel `Var(p_m − q_m)` over a sample of
     buffer states; if it is below 1 e-3 the heads have not separated
     the horizons.
   - (b) On stochastic-transition envs (Resource Gathering), the heads
     may average over outcome variability so that `p_m, q_m` become
     nearly action-invariant; in that case `Ā[c, a, :]` rows are
     near-equal across a and the Pareto vote is symmetric.
   - (c) On terminal-only channels with no shaping, `p_m` and `q_m` are
     near-zero everywhere except the final two steps, so signed area is
     concentrated at episode end and `Ā[c, a, m]` for non-terminal
     clusters never accumulates non-trivial mass — same bootstrap
     symptom as the FED family.
   - (d) If observation-cluster collisions are too rare on long-horizon
     sparse envs (the SIT / RSD failure mode), `Ā[c, a, :]` cells
     never accumulate enough samples to form a stable Pareto vote and
     the operator is dominated by per-cell noise.
   - (e) If action a is rare under the policy in cluster c, `Ā[c, a, :]`
     is estimated from few samples; this is the standard rare-action
     bias and may force a Laplace-style smoothing prior on the row,
     which in the limit reduces PFA to a uniform nudge (no signal).

9. **Side-information channel:** {vector diagnostics, learned dynamics}.
   Vector diagnostics provides the per-channel firing-indicator
   supervision for `(p_m, q_m)`. Learned dynamics manifests as the two
   probability heads themselves — a small forward classifier per channel
   per horizon, trained from buffer data, no auxiliary cumulant loss
   beyond the firing-indicator BCE objective.

10. **Monotonic improvement claim:** under the assumption that for
    fixed `(p_m, q_m)` heads the per-(cluster, action) signed-area
    population mean `Ā*[c, a, m]` is a stable estimator (large-sample
    limit), the Pareto-non-dominance logit nudge induces a monotone
    increase in the population fraction of executed actions whose row
    `Ā*[c, a, :]` is Pareto-non-dominated within `Ā*[c, :, :]`,
    measured per cluster c. This is a behavioral monotonicity (the
    distribution over actions concentrates on the Pareto-front of the
    signed-area matrix per cluster) — not a return monotonicity. The
    *return* monotonicity story only follows if the signed-area
    direction in some channel correlates with that channel's terminal
    contribution, which is an empirical claim the substrate run tests.

## Why it is not CID-canonical / GVF / CWAI

CID-canonical (parked `failed-implementation`,
`worklogs/candidates/cid-channel-imminence-differential.md`) is a
**translational** per-channel signal — it fires when next-step firing
probability differs from this-step probability. PFA is a **rotational**
per-channel signal: signed cross-product of two imminence vectors at
two horizons. The two are distinguishable under variable renaming — the
cross-product cannot be expressed as a difference of any two scalar
functions of o because it inherently couples the two horizons
multiplicatively (`p_m(o_t)·q_m(o_{t+1})` is bilinear in the head
outputs). GVFs / successor features (disqualifier family) accumulate a
discounted cumulant whose expected value drives the policy via linear
extraction; PFA's heads output bounded firing probabilities and the
operator consumes only the signed area on the *observed* transition,
never the head's expected-value prediction at decision time. CWAI uses
the parameter-space Jacobian of a forward dynamics model; PFA's heads
are scalar classifiers and their parameter gradients are not consumed by
the operator.

## Why it scales beyond the substrate

At horizon 20k actions and pixel/text observations, the substrate
per-channel firing-indicator supervision (e.g., `did_validity_pass`,
`did_safety_violate`, `did_user_satisfy`, `did_tool_call_succeed`) is
exactly the kind of dense channel signal that long-horizon agentic
systems already produce as side diagnostics. The per-channel `(p_m, q_m)`
heads are small classifiers that consume a transformer encoder's
embedding of the observation; they do not require a critic or a
preference model. The Pareto vote on the signed-area matrix scales as
O(|A| · k) at decision time, with |A| controlled by sampling a small
candidate set from the policy itself when the action space is "generate
a paragraph" — at which point each candidate's `Ā[c, a, :]` is
estimated by hashing the candidate text into the cluster index. The
rotational primitive is invariant to monotone rescalings of the head
probabilities (cross-product preserves sign under independent
positive-monotone reparameterizations of `p_m, q_m`), which is the
property that allows the same operator to apply across channels of very
different rate scales (e.g., a high-frequency latency channel and a
low-frequency user-satisfaction channel) without any hand-tuned
weighting.
