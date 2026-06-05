# 20260605-07-auto — PTW (Prefix-Witness Floor)

## Research Gate

primitive: per-trajectory **prefix-witness staircase** — for each completed
  trajectory `τ` and each prefix length `k`, the coordinate-wise minimum of
  the vector terminal outcomes across all trajectories in experience that
  share `τ`'s first `k` actions. This is a vector-valued, monotone,
  prefix-indexed *floor over outcomes the agent has already empirically
  guaranteed* by committing to that prefix.
improvement_operator: nudge the policy at state `s_k` toward action
  `a_k = τ_k` proportional to the *positive cone projection* of the
  staircase increment `w(τ, k+1) − w(τ, k)`; do nothing on prefixes whose
  increment lies outside the non-negative cone.
side_information: vector diagnostics; transition geometry (used only as
  prefix-equality on action sequences, not on observations).
nearest_prior_or_disqualifier: candidates/vcc-vector-cumulant-confluence
  (closest live), and disqualifier *advantage / return-to-go* (nearest
  named family).
falsifier: on the two vector envs (Deep Sea Treasure, Resource Gathering)
  the staircase increment `w(τ, k+1) − w(τ, k)` is non-zero on a
  non-trivial fraction of (τ, k) pairs from step one (because both envs
  emit per-step or near-terminal channel signals other than the headline
  reward); if despite that the panel score does not exceed
  `panel_n_beat_random` on at least one vector env, the primitive is
  disconfirmed — not as a bootstrap artifact but as a signal-quality
  artifact, since the bootstrap problem that killed FED/SIT/RSD/VCC is
  structurally absent here.

## Mechanism

Maintain a buffer of completed trajectories with their full per-channel
vector outcomes `v(τ) ∈ ℝᵏ` (the panel-supplied `info["vector"]`,
aggregated to a terminal vector by sum or by terminal-state read; the
choice is uniform across envs). For any prefix `(a_0, …, a_{k−1})` define
the **witness floor**

    w(prefix) = min_{coord} { v(τ′) : τ′ in buffer, τ′_{0:k} = prefix }

where `min_{coord}` is coordinate-wise minimum (the Pareto floor of the
set). For each trajectory `τ` in the most recent batch and each prefix
length `k`, compute the **staircase increment**

    Δ(τ, k) = w(τ_{0:k+1}) − w(τ_{0:k})    ∈ ℝᵏ.

`Δ(τ, k)` is the vector amount by which committing to action `τ_k`
raised the empirical guaranteed-outcome floor over what the prefix-`k`
floor already guaranteed. It is non-negative coordinate-wise *by
construction* (extending a prefix can only restrict the comparison set,
which can only raise a coordinate-wise min). The improvement operator
is

    ∇log π(a_k | s_k)  ←  ‖Δ(τ, k)‖_{1, +} · ∇log π(a_k | s_k)

where `‖·‖_{1, +}` is the L1 norm restricted to the strictly-positive
coordinates. There is no scalar reward weight, no critic, no Bellman
target, no advantage subtraction; the update weight is the *vector L1
of the empirical floor lift along channel dimensions that strictly
increased*. Mass on a channel that did not lift is zero, regardless of
whether some other channel collapsed (it cannot collapse — `Δ` is
always coordinate-wise non-negative). When prefix-`k+1` has only one
trajectory matching, `w` equals `v(τ)` itself, so the operator
naturally interpolates from a "soft floor" near the root to a
"hard outcome" near the leaf.

## Required candidate shape

1. **Experience object:** completed trajectories `τ = (a_0, s_0, …, a_{T-1},
   s_{T-1})` paired with the full per-channel terminal vector outcome
   `v(τ) ∈ ℝᵏ`. The action sequence is the indexing key; the
   observation sequence is *not* used by the primitive (and so does not
   incur observation-hash collision risk that killed FED/SIT/RSD/VCC).
2. **Core primitive:** the **prefix-witness floor**
   `w(prefix) = coord-min { v(τ′) : τ′_{0:k} = prefix }`. This is a
   vector-valued function on the trie of executed action prefixes,
   defined entirely from observed terminations, monotone non-decreasing
   in the prefix-extension partial order, and not equal to any of
   {Q, V, advantage, return-to-go, return-distribution, successor
   feature, cumulant, GVF target, occupancy flow}. Critically it is
   *not* a backup target — there is no Bellman-style recursion; the
   floor at depth `k` is computed from observations at depth `T`,
   not from the floor at depth `k+1`.
3. **Improvement operator:** at each (τ, k), set the policy update at
   `(s_k, a_k)` to the L1 norm of the strictly-positive coordinates of
   `Δ(τ, k) = w(τ_{0:k+1}) − w(τ_{0:k})`, multiplied by the policy
   log-likelihood gradient of `a_k` at `s_k`. No baseline subtraction;
   no critic; the weight is intrinsically non-negative and channel-aware.
4. **Execution rule:** during data collection, sample `a ~ π(·|s)`
   (softmax with optional temperature). No observation-hash lookup, no
   bucket retrieval at decision time — the primitive is computed at
   *update time*, from completed trajectories, against the current
   buffer.
5. **Vector feedback rule:** vectorial throughout. The floor is a vector;
   the staircase increment is a vector; the update weight is the L1 of
   strictly-positive coordinates (no fixed channel weight, no learned
   channel weight, no scalar projection at any point). On envs with `k=1`
   the operator degenerates to a scalar floor-lift weight, but it does
   *not* equal advantage or return-to-go — it equals the empirical-min
   improvement of one prefix-extension over the prior. The k=1
   degenerate case must be flagged explicitly in the audit.
6. **Rollout-cost discipline:** one trajectory consumed = one trajectory.
   No counterfactual rollouts, no simulator branches, no verifier calls.
   Update cost per trajectory is `O(T·B)` where `B` is the number of
   buffered trajectories sharing some non-trivial prefix with `τ`
   (bounded by `|buffer|`); a prefix-trie keeps this near linear.
7. **Nearest-neighbor novelty audit:** Closest live candidate is VCC
   (`vcc-vector-cumulant-confluence`), which indexes by *cumulant
   bucket* and uses Pareto-front *extension*. PTW indexes by *action
   prefix* (no observation involvement) and uses Pareto-*floor lift*
   (coordinate-wise min, not coordinate-wise non-domination). Closest
   disqualifier family is *advantage / return-to-go*: PTW's update
   weight is *not* `R(τ) − b(s)` for any baseline `b`; it is the
   coordinate-wise floor *of a multiset of futures sharing a prefix*,
   which is a different functional of experience and is non-negative by
   construction (no centering, no sign flip). PTW also avoids the SIT
   trap (no observation-hash collisions) and the RSD trap (no closed
   witness pairs needed).
8. **Predicted failure modes:**
   - On terminal-only single-channel sparse envs (DoorKey, KeyCorridor)
     the floor is uniformly zero until first success, after which the
     staircase has at most one non-zero step (the leaf). The operator
     should produce signal *only after the first success* on these envs
     and signal will be sparse; performance may not exceed strong on
     pure single-scalar sparse envs.
   - When the buffered trajectories all share short prefixes (early
     training, narrow policy), the floor saturates to a single
     trajectory's outcome and `Δ` collapses to the all-or-nothing
     pattern; the operator behaves like vanilla policy-gradient on
     successful trajectories. Diagnostics must report the average
     prefix-sharing depth and the fraction of (τ, k) pairs with
     non-zero `Δ`.
   - When per-channel scales differ by orders of magnitude (e.g.,
     treasure value 100 vs step cost 1), the L1 of positive coords is
     dominated by the largest-magnitude channel and the operator
     collapses to scalar maximization on that channel (vector-channel
     rebadge risk). Channel-wise standardization (running median /
     median absolute deviation per channel) is required before L1.
   - On stochastic-transition envs the floor is *too pessimistic*: a
     single unlucky outcome among many trajectories with the same
     prefix pulls the floor down to the unlucky value. A robust-floor
     variant (e.g., 10th-percentile per channel instead of min) should
     be considered if min-floor is too pessimistic — but this must
     not be tuned per env.
9. **Side-information channel:** `vector diagnostics` (the per-step
   `info["vector"]` channel), and `transition geometry` (only as
   action-sequence prefix equality — not as state equivalence). No
   observation hashing, no event lens, no goal labeling, no
   demonstrations.
10. **Monotonic improvement claim:** the prefix-witness floor `w` is
    *monotone non-decreasing* in the prefix-extension partial order
    (i.e., `w(prefix · a) ≥ w(prefix)` coordinate-wise) by definition,
    because extending the prefix shrinks the comparison set and the
    coord-min of a subset is coord-greater than or equal to the
    coord-min of its superset. The improvement operator therefore moves
    policy mass toward actions whose *empirical floor lift over the
    parent prefix* is strictly positive on at least one channel —
    monotonically increasing the expected coordinate-wise floor of the
    policy's induced trajectory distribution at every prefix depth, in
    expectation, to the extent the buffer supports the comparison.
    Formally: `E_{τ ~ π_{t+1}}[w(τ_{0:k})] ≥ E_{τ ~ π_t}[w(τ_{0:k})]`
    coordinate-wise for all `k`, in the limit of small step size and
    large buffer, because every nudge has non-negative weight on the
    channel that lifted.

## Why it is not VCC or advantage / return-to-go

VCC (#vcc-vector-cumulant-confluence) indexes by quantized cumulant
bucket and uses *Pareto-frontier dominance* (cross-action dominance
margin); PTW indexes by *action-sequence prefix* and uses *Pareto-floor
lift along the agent's own trajectory* (`Δ(τ, k)`). These are different
functionals of experience: dominance margin requires comparing two
buckets' frontiers; floor lift requires extending one trajectory's
prefix by one action and measuring how the comparison set shrinks. PTW
also has no quantization knob and so cannot fail by bucket explosion
or bucket collapse, the diagnosed VCC failure.

PTW is not advantage / return-to-go (disqualifier family). Advantage is
a scalar `Q(s,a) − V(s)` involving two learned (or bootstrapped) scalar
estimators; PTW's weight is the L1 of strictly-positive coordinates of
a coordinate-wise *empirical-min* difference between two sets of
observed terminations, with *no learned scalar estimator anywhere* and
*no negative weight* (no centering, no sign flip). Return-to-go is the
sum of remaining rewards along a single trajectory; PTW's floor is the
coord-min over a *multiset of trajectories sharing a prefix*, a
different functional that depends on the buffer composition rather
than on a single trajectory's tail.

## Why it scales beyond the substrate

The primitive's update is computed at trajectory completion from action
sequences and terminal vectors only — no per-step bootstrap, no value
function, no observation-hash bucketing. At horizon 20k actions the
prefix-trie's branching density is what matters, not horizon length:
the staircase is well-defined regardless of `T`. For
"generate a paragraph" actions, the *action prefix* generalizes to a
*prefix of textual outputs* (or of tool calls); two trajectories share
a prefix iff their first k generations match (or hash-match under a
chosen text-equivalence — this is observation-hashing's risk, but on
*outputs* rather than *states*, where exact match is more tractable).
For a 6-component vector feedback (success, cost, safety, latency,
energy, validity), the floor is naturally 6-dimensional and the L1 of
strictly-positive coordinates is exactly the metric the algorithm
wants — improvements on cost and safety simultaneously count even when
success has not changed. Sample efficiency does not come from
counterfactual rollouts but from the *combinatorial sharing* of
prefixes across trajectories: every new trajectory updates the floor
of every prefix it shares with prior trajectories, so the per-update
information yield grows roughly linearly with buffer size for small
buffers and tapers as the trie saturates — the opposite of value
methods, where information yield is bounded by per-step bootstrap
quality.
