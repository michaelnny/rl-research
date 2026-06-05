# 20260605-04-auto — FED (Frontier-Expanding Dispersion)

## Research Gate

primitive: per-channel empirical dispersion of the vector-outcome signature
distribution conditional on a state-context and an action-edit, evaluated on
the running data (no scalar collapse, no learned critic, no goal label)
improvement_operator: action-edit acceptance via Pareto-dominant outward
shift of the empirical channel-wise outcome frontier — accept the edit at a
context iff its post-edit conditional dispersion *strictly extends* the
empirical attainment set on at least one channel without contracting it on
any other channel above a small tolerance
side_information: vector diagnostics (mandatory `info["vector"]` per step,
already present on the panel's vector envs) + transition geometry
(observation-hash buckets used to define the context grouping)
nearest_prior_or_disqualifier: SIT (sit-suffix-inheritance-trie, alive-weak)
and Pareto-front / multi-objective scalarization (disqualifier family).
falsifier: if the Pareto-frontier of empirically observed vector outcomes
fails to expand outward over training on DoorKey-6x6 / KeyCorridor /
Craftax-Symbolic, or if the resulting policy collapses to scalar-reward
parity with PPO (i.e., reduces under inspection to `wᵀr` for any extracted
`w`), the primitive is dead.

## Mechanism

Maintain, per observation-hash bucket `u`, the empirical multiset
`F(u) = {v_1, v_2, ...}` of vector-outcome signatures `v_i ∈ ℝ^k`
*accumulated over the suffix following the i-th visit to `u`*, where the
signature is the per-channel sum (or other channel-respecting aggregator) of
`info["vector"]` from that visit until episode end. The Pareto frontier
`P(u) ⊆ F(u)` is the set of non-dominated points under the standard
component-wise partial order. The candidate primitive at decision time is
the vector quantity `D(u, a) = `the empirical conditional dispersion across
the *next k channel-coordinates* observed when action `a` is selected from
`u` — concretely, the multiset of vector-outcome deltas `{v_j - v_i : v_j
follows action a from u, v_i is the running median outcome at u}`. The
improvement operator is: for each context `u` with sufficient samples,
identify the action `a*` whose conditional outcome multiset adds at least
one new Pareto-non-dominated point to `P(u)` while not removing any, and
nudge the policy logits at `u` toward `a*` by a fixed step. No scalar
weight, no critic, no return-to-go: the update is gated by *set-extension
of an empirical Pareto front*, which is a partial order on sets of vectors
and is *not* expressible as `wᵀr` for any fixed `w`. Composition over time
is by inheritance: the Pareto frontier `P(u)` of ancestor contexts is
augmented as descendant outcomes accrue, so improvement at deep `u` carries
forward the structural information from shallow `u`.

## Required candidate shape

1. **Experience object:** Standard `(s, a, r, s', info["vector"], done)`
   trajectories from on-policy rollouts. The vector signal is the load-
   bearing piece — it is required for the primitive to be non-degenerate on
   vector envs. On DoorKey (which is a sparse-reward env, not a vector env)
   the primitive degenerates to a single channel and the operator becomes
   "extend the empirical max" — still a partial-order test, not a scalar
   gradient. We expect FED to be weak on DoorKey by design and strong on
   KeyCorridor / Craftax-Symbolic.
2. **Core primitive:** The empirical per-channel outcome-multiset
   `F(u, a)` = vector signatures of suffixes that follow `(u, a)`, plus its
   Pareto-front `P(u, a)`, indexed by observation-hash. This is *not* Q (no
   scalar), *not* return-to-go (no conditioning, no scalar collapse), *not*
   GVF/successor features (we never project to a chosen cumulant; we keep
   the raw vector multiset and operate on its set-extension order).
3. **Improvement operator:** `π(a|u) ← π(a|u) · exp(η · I[F(u,a) extends
   P(u) outward])`, with `η` a small fixed step. The indicator is binary
   per action, not a real-valued advantage; the policy update has no
   scalar weight beyond `η` itself. Re-normalize after each update.
4. **Execution rule:** Sample action from `π(·|u)` softmax. Use the same
   policy network used by panel baselines (transformer / MLP head); FED
   modifies the logits via the indicator-driven nudge above. Greedy at
   eval.
5. **Vector feedback rule:** The vector `info["vector"]` is *never*
   collapsed to scalar inside the algorithm. Channel-wise empirical Pareto
   dominance is the only way the channels interact. If two channels are
   redundant, the front collapses to one of them automatically (no harm).
   If they are conflicting (success vs. cost), both are tracked
   independently; the operator finds actions that extend the front along
   either axis.
6. **Rollout-cost discipline:** One on-policy rollout per update. No
   counterfactual replays. No verifier calls. No simulator branches. Per
   update, the primitive is computed from already-collected trajectories
   in the current rollout buffer plus a bounded-size running per-bucket
   multiset. Memory cost is `O(B · k)` for `B` buckets and `k` channels.
7. **Nearest-neighbor novelty audit:** Closest priors are SIT
   (`sit-suffix-inheritance-trie`) and the multi-objective / Pareto
   scalarization disqualifier. Distinction from SIT: SIT grafts whole
   suffixes between observation-equivalent nodes and verifies the graft
   with a rollout — a 2-stage edit-and-verify procedure operating on
   *trajectories*. FED is a logit nudge driven by per-action set-extension
   — no graft, no verification, no trajectory copying; just an indicator
   over Pareto-frontier extension at a single action. Distinction from
   multi-objective scalarization: FED never assigns `w ∈ ℝ^k` (fixed,
   adaptive, or learned) to the channels. The improvement criterion is
   *set extension under a partial order*, which dominates any fixed
   linear scalarization (Pareto-front extension implies `wᵀr` improvement
   for *some* `w`, but the converse is false — FED accepts edits that no
   linear scalarization would).
8. **Predicted failure modes:** (a) Fails when `info["vector"]` is
   degenerate (single-channel, e.g., raw DoorKey without vector envs) —
   the partial order collapses to scalar max and the operator approaches
   reward-greedy hill-climbing. (b) Fails when observation-hash collision
   rate is too low (<1% in the same bucket within a rollout) — there is
   never enough sample mass at any `u` for an empirical Pareto front to
   stabilize. (c) Fails on stochastic transitions where two visits to the
   same `u` with the same `a` produce wildly different vector signatures
   — Pareto-front extension becomes noise-driven. (d) Fails if the panel
   baselines' vector envs report channels that are nearly perfectly
   correlated (then FED reduces to scalar greedy on the dominant
   correlate). (e) Fails if the channel directionality is unsigned — Pareto
   dominance requires a consistent "more is better" semantics on each
   channel, which the harness-supplied vector envs already provide.
9. **Side-information channel:** vector diagnostics + transition geometry.
   Both are explicitly available on the panel's vector envs by design.
   Without vector diagnostics, FED is on-mission only as a degenerate
   sanity-check; with vector diagnostics, FED is *the* candidate.
10. **Monotonic improvement claim:** Under stationary vector-outcome
    distributions and a sufficiently large per-bucket sample budget, the
    sequence of empirical Pareto fronts `{P_t(u)}` is monotonically
    non-contracting under the operator (an accepted edit only adds non-
    dominated points; no removal of existing front points by construction).
    Therefore the *attainment set* `A_t = ⋃_u P_t(u)` is monotonically
    expanding in the set-inclusion sense in expectation. This is a
    monotonic claim on a partial-order-valued quantity — *not* on a scalar
    value or scalar return — and it is precisely what scalar value-based
    monotonic improvement claims (policy improvement theorem) cannot say
    about vector-feedback environments.

## Why it is not SIT or multi-objective scalarization

Closest prior `sit-suffix-inheritance-trie` is a graft-and-verify procedure
on whole suffixes; FED never copies a suffix and never verifies with an
extra rollout — it modifies action logits at a single context using a
partial-order test on accumulated vector signatures. The disqualifier
"scalarized vector-reward maximization" requires a (possibly learned)
weight vector `w` collapsing `r ∈ ℝ^k` to `wᵀr`; FED's update gate is the
*existence* of a Pareto-non-dominated extension of the empirical frontier,
which is a relation on subsets of `ℝ^k` that no `wᵀr` rule can express
(a `w`-induced ordering has at most one maximum per channel-direction;
Pareto extension can accept arbitrarily many incomparable maxima
simultaneously). FED is also not GVF/successor-features, because no
cumulant projection is chosen — the raw vector multiset is preserved
through the entire pipeline and the operator's domain is sets, not
cumulants.

## Why it scales beyond the substrate

When the horizon is 20k actions and feedback is a 6-component vector
(success, cost, latency, safety, validity, preference), per-bucket
empirical Pareto fronts become *the* compact summary: each bucket stores
a small set of incomparable outcome vectors — typically `O(k)` points,
not `O(N)` — yielding a sublinear-in-trajectory-length memory footprint.
At LLM-tool-use scale, the action "generate a paragraph" is replaced by
a token-prefix or tool-invocation hash, the observation-hash by a
compressed dialog-state hash (a transformer encoder fingerprint already
exists in the policy), and the vector signal by the multi-channel
verifier output that LLM agentic systems already emit (correctness,
cost, latency, harm probability, user-preference proxy). The improvement
operator — "nudge logits toward actions that extend the empirical
attainment frontier on at least one channel without contracting any" —
remains well-defined token-by-token because Pareto extension is a local,
context-conditional test. Crucially, FED does not require any
reward-bearing trajectory to be present in the data before it can act:
the moment the first vector signal is emitted (from step 1, on every
vector env), the empirical front begins forming, and the operator has
something to extend. This is the precise property that killed SIT and
RSD on bootstrap.
