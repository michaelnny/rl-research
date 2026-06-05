# 20260606-32-auto — BLIC: Block-Lookback Imminence Concordance

## Research Gate

primitive: per-(observation-cluster s, action a, channel m) running empirical mean
`IC[s, a, m] = E[ q_m(o_{t+1}) − q_m(o_t) | cluster(o_t)=s, a_t=a ]`,
where `q_m: o → [0, 1]` is a small supervised classifier trained on the replay
buffer to predict whether channel m fires somewhere in the next H steps (binary
windowed label).
improvement_operator: Pareto-non-dominance margin on the k-vector IC[s, a, :].
At decision time the policy logit for action a is nudged by
`α · (n_a^dom − n_a^sub)` where `n_a^dom = #{a' : IC[s,a,:] coord-wise ≥ IC[s,a',:],
strict on ≥ 1 channel}` and `n_a^sub` is the reverse count. No scalarization,
no Bellman, no critic, no decision-time forward model query.
side_information: [vector diagnostics, learned dynamics]. q_m is a component
trained on observed channel firings; the primitive is the per-cell IC tensor,
not q_m itself.
nearest_prior_or_disqualifier: graduation of the parked-failed-implementation
candidate `cid-channel-imminence-differential`, with structural advances
(observation-anchored offline computation; probability-difference replaces
log-ratio; explicit cluster-conditioned aggregation; Pareto operator on the
empirical IC tensor rather than on raw per-transition LR rows).
falsifier: BLIC is falsified if (a) on Resource Gathering the per-channel
IC magnitudes shrink to within 2x of channel-wise empirical noise floor for the
terminal-only treasure/gold/gem channels (stochastic-transition collapse of the
classifier signal), or (b) the Pareto-non-dominance gate fires on < 5% of decision
steps within the 120 s vector-stage budget across both DST and RG (operator
silence — the bootstrap wall returning at the cluster level), or (c) the
learned q_m signal is collinear across channels (Pearson |ρ| > 0.9 between any
two q_m heads on held-out replay samples), reducing the k-dim Pareto vote to
effective scalar comparison.

## Mechanism

BLIC's primitive is an **action-conditional channel-firing imminence shift
tensor** computed entirely offline on replay-buffer transitions. For each
channel m we maintain a small supervised binary classifier q_m(o) trained on
windowed labels: `y_m(t) = 1` iff channel m fires somewhere in {t+1, …, t+H},
0 otherwise. After every replay-buffer episode insertion we update q_m by a few
gradient steps. The primitive is then the per-(cluster, action, channel)
running empirical mean of the per-transition imminence delta
`δ_m = q_m(o_{t+1}) − q_m(o_t)`. Cluster identity comes from online k-means on
the policy's penultimate-layer activation, with K self-tuned to keep cell
occupancy above a small floor. The improvement operator is a coordinate-wise
Pareto-non-dominance vote on the k-vector IC[s, a, :]: at each decision step
the policy's logit for action a is shifted by α times the signed dominance
margin computed across all observed action alternatives in cluster s. There is
no scalarization, no bootstrapping of q_m off itself, and no decision-time
forward-model query — the only inference at decision time is the cluster lookup
and the Pareto count, both O(|A|·k) in cluster s. q_m is a component (a
supervised classifier with a bounded-output sigmoid head), the policy network is
a component (a torch MLP/conv net with categorical action head), online k-means
is a component; the *primitive* doing the explanatory load is the empirical
imminence-shift tensor IC and its induced action partial order via Pareto
non-dominance.

## Required candidate shape

1. **Experience object:** per-step transitions `(o_t, a_t, v_t, o_{t+1})` with
   `v_t ∈ R^k` from `info["vector"]`, accumulated in a replay buffer along with
   episode-position index for windowed-label construction.
2. **Core primitive:** the per-(cluster, action) running empirical k-vector
   `IC[s, a, :]` where each entry is the conditional mean of the
   imminence-shift `δ_m = q_m(o_{t+1}) − q_m(o_t)` for that (cluster, action,
   channel) cell. q_m are k binary-classifier components supervised on
   windowed channel-firing labels.
3. **Improvement operator:** at each decision step in cluster s, compute for
   each action a the signed Pareto-non-dominance margin
   `m_a = n_a^dom − n_a^sub` over the k-vector IC[s, a, :] vs.
   IC[s, a', :] for all a' ≠ a; apply logit nudge `Δlogit(a) = α · m_a`.
   Then sample from softmax(logits + Δlogit) using the categorical policy head.
4. **Execution rule:** sample actions from the nudged-softmax categorical
   policy; the base policy network is trained on standard supervised behavior
   (cross-entropy of selected action vs. the nudged categorical, with a small
   stop-gradient on the nudge term to prevent the policy from chasing its own
   shadow); no value head, no advantage head, no critic.
5. **Vector feedback rule:** vector outcomes enter only as binary windowed
   labels for q_m supervision (one classifier per channel) and as the source
   of `δ_m` per-channel. The k channels are never collapsed by a fixed or
   learned weight — Pareto non-dominance on the k-vector is the only
   aggregation. On a single-channel env (k = 1) the primitive degenerates to
   "imminence-shift maximization on the one channel," which is logged
   explicitly as the rebadge boundary; this case is irrelevant to the vector
   panel where k ≥ 2.
6. **Rollout-cost discipline:** one environment step per primitive update.
   q_m supervision: fixed mini-batch of replay transitions per env-step (no
   extra rollouts). Decision-time cost: O(|A|·k) Pareto-margin computation
   plus one cluster lookup. No counterfactual rollouts, no simulator branches,
   no verifier calls, no best-of-N. Total interactions per accepted policy
   gradient step: 1 env step.
7. **Nearest-neighbor novelty audit:** nearest prior is
   `cid-channel-imminence-differential` (parked-failed-implementation). CID
   used a decision-time forward model `f_φ(o, a) → ô_{t+1}` and computed
   log-ratios on this counterfactual prediction; the failure was that early in
   training f_φ could not distinguish actions, making LR rows
   action-invariant. BLIC eliminates the decision-time forward model entirely:
   it computes `δ_m` on **actually-observed** post-action observations
   `o_{t+1}` from replay, accumulates the cell-conditioned mean offline, and
   reads it back at decision time without invoking q_m on counterfactuals.
   Action discrimination comes from the cell-conditioning structure
   (different actions populate different (s, a) cells) rather than from
   counterfactual queries on a noisy forward model. Probability-difference
   replaces log-ratio for numerical stability when q_m ≈ 0 everywhere (early
   training on terminal-only reward channels).
8. **Predicted failure modes:**
   - On Resource Gathering, stochastic transitions may shrink `δ_m` to noise
     floor for terminal-only treasure/gold/gem channels (q_m signal averages
     out before any reward fires), in which case Pareto vote is dominated by
     the step-penalty channel and reduces to a step-penalty-min preference
     — the same single-channel collapse that killed ATP, PRAR, and FFTV-on-RG.
   - On DoorKey-8x8 / KeyCorridor, if cluster cells stay sparse (online
     k-means hasn't converged within 120 s), Pareto vote fires on a small
     fraction of decision steps and the operator reduces to base-policy
     exploration. Diagnostic: log fraction of steps where dominance margin
     is non-zero per env per minute.
   - q_m heads collinearity across channels: if all four channels' classifiers
     learn essentially the same function (because all channels correlate
     strongly with episode position), the k-vector IC[s, a, :] is
     near-rank-1 and the Pareto vote is structurally equivalent to scalar
     comparison. Check Pearson correlation of q_m predictions on a held-out
     replay batch as a structural-collapse diagnostic.
   - Self-confirmation pathology: if α is too large the policy chases its own
     cluster boundary (cluster identity changes as policy features drift,
     IC entries become stale). Stop-gradient on the nudge term and a slow
     cluster-update schedule mitigate but do not eliminate this risk.
9. **Side-information channel:** `[vector diagnostics, learned dynamics]`.
   Vector diagnostics: per-step `info["vector"]` provides the binary labels
   for q_m supervision. Learned dynamics: q_m is a learned forward predictor
   restricted to channel-firing imminence within a fixed window — it does
   not predict the full next observation, only k binary firing flags, and it
   is never queried counterfactually. No demonstrations, no pretrained
   priors, no language description, no verifier feedback, no environment
   instrumentation beyond `info["vector"]`.
10. **Monotonic improvement claim:** in the asymptotic regime where (a)
    cluster cells have accumulated sufficient samples for IC entries to
    converge to their true conditional means, (b) q_m has converged to the
    Bayes-optimal next-H-step firing probability for each channel, and (c)
    the policy is approximately stationary, the BLIC nudge α · m_a strictly
    increases the probability of any action a* that Pareto-dominates the
    current policy's action distribution at observation o on the
    cell-conditional imminence-shift IC[cluster(o), :, :]. Specifically:
    every nudge step weakly improves the policy's expected per-channel
    imminence-shift at o under the Pareto partial order on R^k, and the
    improvement is strict whenever the dominance margin is non-zero.
    No scalar quantity is monotonically improved (deliberately: the Pareto
    partial order has no scalar ordering compatible with all multi-channel
    decisions), but the *partial-order improvement claim* is well-defined
    and falsifiable via the dominance-margin diagnostic.

## Why it is not cid-channel-imminence-differential

CID's parked-failed-implementation log explicitly identified the binding
failure: a decision-time forward model `f_φ(o, a) → ô_{t+1}` that could not
distinguish actions early in training, causing LR rows to be action-invariant
and the Pareto nudge silently zero throughout the run. BLIC structurally
eliminates this bottleneck: there is **no decision-time forward model**.
Imminence shifts `δ_m` are computed on actually-observed `o_{t+1}` from
replay, then aggregated per-(cluster, action) into the IC tensor offline;
decision-time uses only the cached IC tensor and a cluster lookup, so the
operator's discriminative power does not depend on f_φ's quality. Under
variable renaming the two are not equivalent: CID's primitive was a
*counterfactual log-ratio at the decision boundary*, while BLIC's is an
*empirical conditional-mean shift on observed transitions*. The parked
candidate's own canonical-fix note proposed essentially this offline-form
fix; BLIC realizes it with the explicit cluster-conditioning and Pareto-vote
operator structure made precise.

## Why it is not GVFs / successor features (disqualifier)

q_m on its own is a fixed-horizon GVF on a binary cumulant — that is the
component-level family. The structural distinction is at the operator level:
the central learned object is **not** q_m, and the policy is **not** trained
by maximizing any function of q_m or any linear combination thereof. BLIC's
primitive is the **partial order on actions induced by the conditional-mean
imminence-shift tensor IC[s, a, :]** under coordinate-wise dominance on R^k.
There is no scalar GVF maximization, no fixed weight vector w applied to
δ_m, no learned channel weighting. A GVF-with-Pareto rebadge would sum
discounted future channel firings into an infinite-horizon prediction and
pick actions maximizing each component — BLIC instead computes a *one-step
empirical imminence shift averaged over the cell*, and uses *dominance
counts* (not magnitudes) as the operator. Under stochasticity, q_m absorbs
the noise into a calibrated probability, while the Pareto vote uses only
the sign-and-coordinate structure of `δ_m`'s mean — the magnitude of any
individual q_m head does not enter the operator.

## Why it scales beyond the substrate

At 20k-action horizons or LLM-tool-use settings, the q_m classifier
generalizes to predicting whether each named feedback channel (success,
cost, safety, latency, validity, preference) fires within the next H tokens
or tool-calls. The IC[s, a, :] tensor scales naturally because actions can
be replaced by *action embeddings* (paragraph encodings, tool-call
embeddings) — Pareto comparison is over a discrete sample of recently-tried
embeddings near the current decision context, not over a full enumeration of
A. The cluster s comes from the policy's own representation, which a
pretrained transformer already provides. The operator's per-decision cost
is O(C · k) where C is the number of cached neighbor embeddings (a
hyperparameter, e.g. 16) and k is the number of feedback channels (e.g. 6
for {success, cost, safety, latency, validity, preference}) — independent
of the long horizon. The vector feedback is preserved end-to-end without
collapse: the only operation is coordinate-wise dominance counting, which
extends from k=2 to k=6 with no structural change. The bootstrap-wall
problem of many prior attempts is mitigated because q_m on at least one
dense channel (latency or cost in the LLM setting; step-penalty on the
substrate) is non-degenerate from the first interaction — providing a
non-trivial nudge from episode 1 even before any rare-event channel fires.
