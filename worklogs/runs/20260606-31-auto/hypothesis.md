# 20260606-31-auto — KSV (Kernel-Smoothed Suffix-Pair Vote)

## Research Gate

primitive: per-(action, channel) kernel-weighted signed pair-vote
`SV[a, m] = Σ_{(i,j) | a_t^i = a, a_t^j ≠ a} κ(o, o_t^i)·κ(o, o_t^j)·sign((c_T^i − c_t^i)[m] − (c_T^j − c_t^j)[m])`
where κ is a Gaussian kernel on a learned observation embedding and the sum
runs over trajectory pairs in a sliding replay window.
improvement_operator: per-decision logit nudge `α·(D⁻(a) − D⁺(a))` where D⁻(a)
counts actions whose SV row is strictly Pareto-dominated by SV[a, :] in the
coordinate-wise partial order on R^k, and D⁺(a) counts actions that
strictly dominate SV[a, :]. No scalar collapse, no critic, no Bellman, no
verification rollout.
side_information: vector diagnostics + transition geometry (kernel on a
self-supervised observation embedding from a small forward model used as
an embedding-only auxiliary, not a planner).
nearest_prior_or_disqualifier: dpc-divergent-prefix-concordance (alive-weak,
20260606-02-auto) and #15 FED.
falsifier: on the vector stage (DST + RG, 120 s), with diagnostic logging of
(a) mean number of pairs contributing weight > 1e-3 per decision step, (b)
fraction of decision steps where Pareto-non-dominance margin is non-zero,
and (c) per-channel histogram of `sign((c_T^i − c_t^i)[m] − (c_T^j − c_t^j)[m])`,
KSV is falsified if either: the Pareto margin is non-zero on < 5 % of decision
steps within budget (kernel bandwidth or pool too sparse to fire), or the
per-channel sign histogram for terminal-only reward channels remains > 99 %
zero throughout training (collinearity collapse to step-penalty — same
mechanism that killed PRAR/CHX/CRP). Sparse stage acts as a sanity gate
(should at least match random on DoorKey/KeyCorridor since the operator is
state-conditional and uses per-channel signs, not magnitudes).

## Mechanism

KSV indexes experience continuously (not by hash, cluster, or bucket) and
votes pairwise (not per-cell). Every trajectory pair (i, j) in the sliding
buffer contributes to *every* decision step, with weight given by a kernel
between the current observation's embedding and each pair-member's
matched-step observation embedding; the matched step is the position in
trajectory i (resp. j) whose embedding is closest in cosine to e(o). The
per-(action, channel) primitive `SV[a, m]` is the kernel-weighted sum of
*signs* of pairwise suffix-cumulant differences for pairs where one
member took action a and the other did not. The improvement operator is
strict Pareto-non-dominance over the k-vector `SV[a, :]`: nudge logits
toward actions whose sign-vote vector is dominated by no other action's
sign-vote vector and dominates as many alternatives as possible. The
embedding is supplied by a small auxiliary self-supervised forward model
(predict next observation from (o, a)) trained alongside the policy — used
only as an embedding map, not as a planner or value head. Composition law:
one primitive (kernel-weighted sign-vote tensor over pairs) plus one
operator (Pareto-non-dominance margin to logit nudge). The forward model is
a component, not the explanation.

## Required candidate shape

1. **Experience object:** sliding replay buffer of full trajectories
   `(o_0, a_0, v_0, …, o_T, a_T, v_T)` with per-step vector signal `v_t ∈ R^k`
   (consumed from `info["vector"]` on vector envs, terminal-only on sparse
   envs). No edit grammar, no event lens, no verifier.
2. **Core primitive:** the per-(action, channel) sign-vote tensor
   `SV[a, m] = Σ_{(i,j): a_t^i = a, a_t^j ≠ a} w_ij(o) · sign(δ_m^{ij})` where
   `w_ij(o) = κ(e(o), e(o_t^i)) · κ(e(o), e(o_t^j))` is a Gaussian kernel
   product on the auxiliary embedding, the matched step `t` is the step in
   each trajectory whose embedding is closest to `e(o)`, and
   `δ_m^{ij} = (c_T^i − c_t^i)[m] − (c_T^j − c_t^j)[m]`.
3. **Improvement operator:** at each decision step, after computing
   `SV[a, :]` for all actions in the legal action set, compute
   `D⁻(a) = #{a' : SV[a, :] ≻ SV[a', :]}` and
   `D⁺(a) = #{a' : SV[a', :] ≻ SV[a, :]}` where `≻` is strict coordinate-wise
   dominance. Add a logit nudge `α · (D⁻(a) − D⁺(a))` to the policy's
   pre-softmax logit at this state. The base policy update is REINFORCE-style
   *only on the entropy regularizer* (no scalar reward weighting); the
   directional learning signal comes solely from the Pareto nudge applied
   on-policy at each step. The auxiliary forward model is updated by MSE on
   replay batches.
4. **Execution rule:** sample `a_t ~ softmax(logits_θ(o_t) + α · (D⁻ − D⁺))`
   where `(D⁻ − D⁺)` is computed online per decision step. No greedy argmax
   over a value head; no conditioning on goals or returns.
5. **Vector feedback rule:** vector signals enter only through (a) the
   per-channel suffix-cumulant difference `δ_m^{ij}` whose *sign* (not
   magnitude) is the only quantity used, and (b) the dimension of the Pareto
   comparison. There is no `w^T r` step at any point. Per-channel signs are
   never aggregated; the operator's only cross-channel rule is the
   coordinate-wise partial order.
6. **Rollout-cost discipline:** one environment step per decision; one
   policy update per `B` env steps where `B` is the replay batch period. The
   pairwise sum is approximated by random subsampling: at each decision
   step, draw `M` pairs uniformly from the buffer where M ≤ 256, evaluate
   the kernel weights, and accumulate. No counterfactual rollouts, no
   verifier calls, no tool-use loops. At deployment the forward-model
   embedding can be precomputed offline; only the kernel-weighted pairwise
   lookup runs at decision time.
7. **Nearest-neighbor novelty audit:** closest prior is
   dpc-divergent-prefix-concordance (alive-weak after 20260606-02-auto;
   above-strong on DST, random on RG). DPC indexes its sign-vote tensor by
   *first-divergence state hash* — pairs only contribute at the exact
   decision-state where they first chose different actions, requiring
   collision in the divergence-state hash, and a single pair influences only
   one decision-state's vote. KSV indexes by *kernel similarity in
   continuous embedding space* — every pair contributes to every decision
   step with continuous weight, eliminating the hash-collision bottleneck
   that killed FED/CEC/TPP/KTAC and limited DPC to envs with strong
   divergence-state structure (DST). Second-nearest is KTAC, which uses
   k-means clusters; KSV's kernel removes the discrete-cluster
   bootstrap wall (KTAC scored 0.0/0.011 on DST/RG due to cluster
   under-population). The structural distinction is that KSV's kernel
   provides a continuous credit-assignment substrate where DPC/KTAC require
   discrete coincidences.
8. **Predicted failure modes:**
   (a) **Collinearity collapse** — on terminal-only-reward envs, before the
       first rewarded trajectory enters the buffer, every reward channel's
       per-pair `δ_m^{ij}` is identically zero, so SV[a, m] = 0 for every
       reward channel and the Pareto vote degenerates to scalar comparison
       on the step-penalty channel (curiosity-rebadge / fastest-termination
       failure mode shared with PRAR/ATP/TCP). Falsifier: per-channel sign
       histogram > 99 % zero on reward channels.
   (b) **Kernel-bandwidth pathology** — bandwidth too narrow → only a few
       pairs contribute to each decision (matching KTAC's cluster-population
       failure); bandwidth too wide → SV becomes state-independent (matches
       no-state baseline). Falsifier: ablation across bandwidths shows no
       interior optimum.
   (c) **Stochastic-transition shrinkage** — on Resource Gathering (stochastic),
       per-channel sign deltas may flip with high probability across pairs
       sharing similar matched-step embeddings, driving SV[a, :] toward zero
       and the Pareto margin to silence. This is the same mechanism that
       killed CWAI and DPC on RG.
   (d) **Embedding underfitting** — if the auxiliary forward model
       underfits, `e(o)` carries little structure, the kernel collapses to a
       global average, and SV becomes action-marginal (no state
       discrimination). Falsifier: forward-model MSE plateaus above a
       baseline value within budget.
   (e) **Pairwise-O(N²) approximation noise** — sub-sampling M pairs
       introduces variance in SV; if M is too small, the sign-vote tensor is
       too noisy for the Pareto margin to be reliable. Falsifier: ablation
       on M shows no plateau as M grows.
9. **Side-information channel:** {vector diagnostics, transition geometry}.
   Vector diagnostics enter as the per-channel suffix-cumulant difference
   sign. Transition geometry enters as the auxiliary forward-model
   embedding (a learned dynamics map used as an embedding only, not as a
   planner). Reward signals enter only through the cumulant differences;
   no goal labels, no event lens, no verifier feedback, no demonstrations.
10. **Monotonic improvement claim:** in the limit of an infinite buffer
    sampled from the current policy and an unbiased kernel
    (bandwidth → 0 with adequate density), SV[a, :] converges to the
    per-channel Pr-dominance signature of action a's suffix cumulant
    distribution at o vs. the marginal-over-other-actions distribution at o,
    in the (0, 1) sense. The policy update increases the policy's
    probability mass on coordinate-wise non-dominated actions in this
    Pareto ordering; thus the operator monotonically reduces (in
    expectation) the cardinality of the Pareto-dominated support of π in
    the per-channel sign-of-cumulant-difference partial order — a strict
    refinement of the policy that does not collapse multi-channel
    structure into a scalar. The claim breaks under (a)–(e) above.

## Why it is not dpc-divergent-prefix-concordance / FED / KTAC

DPC (alive-weak): votes only at the *first-divergence state hash* of each
trajectory pair, so each pair contributes to one decision state and that
state must be re-encountered exactly. KSV's kernel makes every pair
contribute continuous weight to every decision, severing the
hash-coincidence bootstrap dependency that limited DPC's RG performance.
KTAC (alive-weak): indexes by k-means cluster identity, requiring
cluster-population mass before the Kemeny consensus fires; KSV replaces
discrete cluster identity with a continuous Gaussian kernel, so credit
flows even when no two trajectories match exactly. FED (#15, dead): uses
empirical Pareto fronts indexed by observation hash with no smoothing —
the entire FED family was killed by hash-collision starvation, which the
kernel formulation directly addresses without re-introducing scalarization,
elite cloning, or value bootstrap.

## Why it scales beyond the substrate

The kernel-smoothed pair-vote primitive has three properties that scale
to long-horizon, large-action, vector-feedback settings: (1) The kernel
operates on a learned observation embedding rather than discrete state
identity, so it transfers directly to pixel observations, transformer
hidden states, or LLM activations — these are the natural embeddings
already produced by modern policy networks. (2) The improvement operator
is per-channel-sign Pareto, not magnitude — so a 6-component vector
feedback {success, cost, safety, latency, validity, preference} from an
agentic LLM-with-tools system is consumed natively without a hand-tuned
weighting; the operator strictly refines the policy's support against
each channel's pairwise sign-of-difference simultaneously. (3) Pairwise
sub-sampling makes the per-decision cost O(M·d) rather than O(|A|·d) —
critical when the action space is "generate a paragraph" or "call API
with arguments," because the operator never enumerates the action space:
it scores only the actions actually emitted by sampling pairs from the
buffer, and the kernel re-weights them by current-state similarity.
At 20 k-step horizon, the suffix cumulant `c_T − c_t` is well-defined
because v_t is per-step (not terminal-bootstrapped); the only
horizon-scaling concern is buffer size, which is a hardware constraint
shared by every trajectory-replay method.
