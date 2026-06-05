# 20260606-10-auto — PCR (Policy Commitment Recovery)

## Research Gate

primitive: per-(context-cluster, action) **commitment-recovery vector**
`R[c,a] ∈ R^L` whose component `R[c,a]_ℓ` is the expected number of steps,
along the realized trajectory immediately following step t, until the
realized action `a_{t+ℓ'}` again equals the snapshot policy's argmax action
at the realized observation `o_{t+ℓ'}` — measured at L different
"alignment thresholds" (top-1 match, top-2 set match, top-K set match,
plus a logit-cosine threshold). Computed on-trajectory by querying the
frozen behavior-snapshot policy at every realized state — no extra
environment rollouts.
improvement_operator: at each context c, push policy logits toward actions
whose **commitment-recovery vector `R[c,a]` is Pareto-non-dominated** by
the recovery vectors of other actions seen at c, with the sign of the
nudge **gated** by whether the terminal vector outcome `v_T(τ)` of the
trajectories that took a at c is Pareto-non-dominated by the terminal
outcomes of trajectories that took other actions at c. No critic, no
TD, no scalar weighting, no advantage.
side_information: transition geometry (the policy's own action-logit
sequence along realized trajectories) + vector diagnostics (terminal
vector outcome only, used as a sign gate, not as a magnitude weight).
nearest_prior_or_disqualifier: closest prior is **#12 PEO** (policy-edit
optimization) and the actor-critic disqualifier family; closest alive
candidate is **CWAI** (channel-wise action influence via forward-model
Jacobians).
falsifier: PCR fails iff (a) commitment-recovery vectors collapse to a
near-constant column across actions within the same context cluster
(measured: per-cluster row-rank of R below 2 on > 80% of clusters), OR
(b) the Pareto-non-dominance test over R produces a non-singleton dominant
set on < 5% of decision steps (operator silent), OR (c) the terminal-
outcome gate inverts the sign of the operator on > 30% of clusters
relative to its ungated form (gate is destructive, primitive carries no
new signal). All three are checked in a single 120 s panel run.

## Mechanism

PCR posits that the policy's own action-logit trajectory along realized
experience is a dense, reward-independent signal that encodes how
*disruptive* each action is to the policy's intended plan — and that this
disruption signal is structurally orthogonal to reward correlation, so
it can drive an improvement operator on long-horizon sparse-reward envs
where reward-bearing signal is too rare to bootstrap. Concretely: along
each realized trajectory, the snapshot policy `π_θ` (frozen at the start
of the buffer) is queried at every observation `o_t` to produce a
distribution `π_θ(·|o_t)`. The action `a_t` actually taken either
"confirms" the policy (matches the modal action under one of L
alignment thresholds) or "deviates" from it. After each deviation we
measure how many steps it takes for confirmation to re-establish at each
threshold ℓ — yielding a vector `r_t ∈ R^L`. We accumulate `r_t` into a
running per-(cluster, action) mean `R[c,a]`, where c is a coarse cluster
of the snapshot policy's logit vector at o_t (a cheap quantization, e.g.
sign-pattern of the top-K logits). The improvement operator at update
time computes, per cluster c, the Pareto-non-dominated set
`P(c) = {a : R[c,a] is Pareto-non-dominated in R^L over a' ≠ a}` — and
pushes logits toward `a ∈ P(c)` only when, restricted to the same
cluster, the terminal vector outcomes of trajectories that visited c
*and* took a are Pareto-non-dominated by terminal outcomes of trajectories
that visited c and took some `a' ∉ P(c)`. The terminal vector outcome
acts as a sign gate (Pareto-better → push toward, Pareto-worse → push
away, Pareto-incomparable → no update); it is never used as a magnitude.
The primitive is the recovery vector R; the gate is a binary sign chosen
from terminal vector dominance; together they form one composition law
(Pareto-meet of recovery non-dominance and outcome non-dominance).

## Required candidate shape

1. **Experience object:** ordinary on-policy trajectories with
   per-step `(o_t, a_t, π_θ(·|o_t))` tuples (logits already in memory
   from the forward pass) and a single terminal vector outcome
   `v_T ∈ R^k` per episode. No counterfactual rollouts, no additional
   forward passes per step beyond the one already used to act, plus a
   single buffer-time pass to query the snapshot policy.
2. **Core primitive:** per-(cluster, action) **commitment-recovery
   vector** `R[c,a] ∈ R^L`, where each component is the expected step-lag
   until alignment-threshold ℓ is re-satisfied along the realized
   trajectory after the agent acted at cluster c with action a.
3. **Improvement operator:** per-cluster Pareto-meet — at cluster c,
   nudge logits of actions in the recovery-Pareto-non-dominated set
   `P(c)` upward (and recovery-Pareto-dominated actions downward) by a
   small fixed step, but **only** for clusters where the terminal vector
   outcomes likewise exhibit a Pareto-non-dominance separation between
   `P(c)` and its complement. No reward-magnitude weight.
4. **Execution rule:** sample actions from
   `softmax(logits + α · pcr_nudge)`. The nudge `pcr_nudge` is a sparse
   per-cluster correction (0 wherever the gate did not fire) added to
   the base logits at each forward pass. No greedy argmax, no MCTS, no
   verifier.
5. **Vector feedback rule:** terminal vector outcome `v_T ∈ R^k` is
   never scalarized. It enters the algorithm exclusively as a binary
   per-cluster sign — Pareto-non-dominance of the cluster-conditional
   terminal-outcome multiset between two action sets. The recovery
   primitive R is itself a vector in R^L with no scalarization.
6. **Rollout-cost discipline:** **one** environment interaction per
   step. **One** snapshot-policy forward pass per realized step at
   buffer-flush time (vectorized over the buffer; ~B forward passes per
   update for buffer size B, identical to one PPO update's forward
   pass cost). No counterfactual env rollouts, no simulator branches,
   no verifier calls. At deployment, a single policy forward pass per
   step — no PCR machinery is needed, since the operator is folded into
   the trained logits.
7. **Nearest-neighbor novelty audit:** the closest prior failure is
   **#12 PEO (Policy-Edit Optimization)** — both center the policy as
   primary and use vector-cone tests on policy-edit responses. PEO's
   edits were a hand-designed semantic basis whose optimization reduced
   to scalar edit-ES; PCR's "edits" are not edits at all but rather
   a *passive measurement* of how the agent's realized actions perturbed
   its own modal-action plan, with no edit basis to scalarize over. The
   closest disqualifier family is **actor-critic** — but PCR's nudge
   weight is not a critic-supplied magnitude (it is a binary
   Pareto-meet sign), and the central object R is a step-lag-dispersion
   vector, not an expected return. The closest alive candidate is
   **CWAI** (forward-model Jacobian column norms): CWAI uses a learned
   transition model's local linearization; PCR uses no transition model
   at all, only the policy's own logit trajectory.
8. **Predicted failure modes:** (a) **collinearity of L thresholds**:
   if the L alignment thresholds turn out to be near-comonotonic
   (top-1 match implies top-2 match implies cosine-near-1 with high
   probability), then R rows in R^L lie on a near-line and Pareto-non-
   dominance reduces to scalar argmin — the operator collapses to a
   "reward minimal disruption" rebadge. Mitigation: include thresholds
   that are designed to disagree (top-1 vs entropy-quantile match).
   (b) **terminal-outcome gate dominance**: on Resource Gathering where
   precious channels fire only terminally, the gate may fire only on
   trajectories that reached terminal goals — so the operator inherits
   the same bootstrap wall as FED, CEC, TPP. (c) **cluster collapse**:
   if the logit-sign-pattern cluster ID degenerates to a single bucket
   on simple envs (e.g. early DoorKey where logits are near-uniform),
   R[c,a] is computed over too few clusters and the operator over-
   regularizes the policy to a global recovery preference. (d)
   **stochastic transitions** (RG): step-lag `r_t` becomes high-
   variance and dominates the mean R[c,a] estimate — same RG-killer that
   broke CWAI, DPC, and the sprint-4 family.
9. **Side-information channel:** **transition geometry** (specifically
   the action-logit sequence the policy itself produces over realized
   trajectories — visible to the algorithm at zero environment cost
   because the logits were already computed during action selection)
   plus **vector diagnostics** (terminal vector outcome only, used as a
   sign gate). No event lens, no demonstrations, no learned dynamics,
   no verifier, no environment instrumentation.
10. **Monotonic improvement claim:** under the assumption that, at each
    cluster c, the conditional distribution of terminal vector outcome
    given (c, a) is stationary across the buffer window, the operator
    monotonically increases the policy's expected probability mass on
    actions whose `(R[c,a], v_T|c,a)` pair is Pareto-non-dominated in
    the product partial order on R^L × R^k, and decreases mass on
    Pareto-dominated actions, leaving Pareto-incomparable actions
    unchanged. This is monotone improvement in the **product partial
    order** on (recovery-vector, terminal-outcome-vector) space — a
    proxy for "shorter commitment recovery and Pareto-better outcome,"
    which is the property we ultimately care about.

## Why it is not <nearest prior or disqualifier>

PEO (#12) reduced to scalar edit-ES because its edit basis was
hand-designed and the optimizer collapsed to a scalar acceptance test;
PCR has no edit basis — it passively measures the policy's own
self-disruption along realized trajectories, and the "weight" applied is
a binary Pareto-meet sign, not a scalar reward delta. Compared to the
actor-critic disqualifier family, the central learned object is *not* an
expected return or expected advantage; it is a step-lag-dispersion
vector in R^L computed from action-logit sequences, with no Bellman
recursion, no temporal-difference target, and no scalar magnitude
mediating the policy update. Compared to CWAI (the closest alive
candidate), CWAI requires a learned forward model and computes Jacobians
through it; PCR uses no transition model and computes only logit
self-comparisons along realized trajectories — an information channel
that is free at training time and completely orthogonal to forward-model
quality (which is what kills CWAI on stochastic RG).

## Why it scales beyond the substrate

The commitment-recovery primitive is defined entirely from the policy's
own per-step action-logit output, which exists at any scale — including
for transformer policies producing token-level or paragraph-level action
distributions. At a 20k-step horizon the recovery vector is no harder to
compute than at a 100-step horizon (it is a per-step lag measurement,
not a horizon-spanning bootstrap). For LLM-policy "generate a paragraph"
actions, the natural alignment thresholds are perplexity-quantile
matches and beam-set overlaps — all computable from the policy's own
logits at zero extra rollout cost. For tool-using LLM agents, the
recovery primitive captures "how disruptive is calling this tool to the
agent's intended plan-of-action" purely from the agent's own next-token
distribution evolution, with no need for a verifier, world model, or
preference signal. Because the primitive is dense, reward-free, and
internally computed, it survives the bootstrap wall that killed every
sprint-4 candidate that needed reward correlation or observation-hash
collisions to fire.
