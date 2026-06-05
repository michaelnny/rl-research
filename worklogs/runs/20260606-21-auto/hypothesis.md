# 20260606-21-auto — CSD: Channel-Conditional Successor Disagreement

## Research Gate

primitive: per-(observation cluster `c`, action `a`, channel `m`) total-variation
distance between two empirical next-cluster distributions: trajectories that
fired channel `m` in the K-step post-action window vs. trajectories that did not
— `S[c,a,m] = TV(p(c'|c,a, m fires in [t+1..t+K]), p(c'|c,a, m does not fire))`.
improvement_operator: Pareto-non-dominance count `n_a − m_a` on the k-vector
`S[c,a,:]`; logit nudge `α(n_a − m_a)` at cluster `c` at decision time.
side_information: vector diagnostics, transition geometry.
nearest_prior_or_disqualifier: TRAC (#26 — JSD over post-action successor
clusters partitioned by channel firing), CWAI (alive-promising — forward-model
Jacobian per channel).
falsifier: if on Resource Gathering (within 120 s budget, vector stage) the
fraction of decision steps where the Pareto vote is non-zero is below ~10%, OR
if `S[c,a,step-penalty]` dominates `S[c,a,m≠step-penalty]` by more than 5x in
mean magnitude across the buffer (indicating step-penalty alone drives the
operator and the primitive has reduced to "shortest-path-to-terminal"), the
candidate is dead. Pre-commit: log per-channel mean-S histogram and Pareto-vote
firing rate.

## Mechanism

Cluster observations with a small online clusterer (mini-batch k-means or
running-mean LSH on a learned embedding; the clusterer is a component, not the
primitive). Each transition `(o_t → c_t, a_t, o_{t+1} → c_{t+1}, v_{t+1})` is
appended to a per-(c,a) episode-position-tagged ring buffer. For each
(c,a,m), partition the buffer's K-step post-action windows into two
sub-populations indexed by whether channel m fired at least once in the window.
Compute the empirical next-cluster (cluster of `o_{t+1}`) distributions
`p_m^+[c,a]` and `p_m^-[c,a]` over the cluster vocabulary; the primitive is the
total-variation distance `S[c,a,m] = ½ Σ_{c'} |p_m^+[c,a](c') − p_m^-[c,a](c')|`.
At decision time at cluster `c`, the per-action k-vector `S[c,:,:]` is read out
and the improvement operator nudges policy logits toward actions whose row is
Pareto-non-dominated in the coordinate-wise partial order on `R_+^k`. No scalar
collapse, no Bellman backup, no critic, no return-to-go, no parameter gradients
flow into the primitive (it is a sufficient statistic of the buffer).

The primitive is reward-independent in the same sense as CWAI: it is computable
before any reward is observed. Its connection to reward is structural: actions
that strongly couple "which next-cluster the agent enters" with "whether channel
m subsequently fires" are *channel-m-discriminating* actions — they are exactly
the actions whose downstream trajectory identity carries information about
channel m. Because reward channels are members of the channel set, an action
whose next-cluster identity is informative about reward-channel firing is by
construction an action that controls (positively or negatively) reward-channel
firing.

The robustness-to-stochasticity story is the load-bearing structural claim.
Magnitude-based primitives (CWAI Jacobian norm, JFP gradient peak time, ATP
threshold horizon) shrink toward a noise floor when transitions are stochastic
because the signal *is* a magnitude. CSD measures distributional *separation*
between two conditional distributions; transition stochasticity adds noise to
both `p_m^+` and `p_m^-` and the TV gap shrinks but does not reverse — the
ordering of which action is most channel-m-discriminating is preserved as long
as the noise is approximately exchangeable across the conditioning event.

## Required candidate shape

1. **Experience object:** ordinary on-policy rollouts; each transition recorded
   as `(o_t, c_t = cluster(o_t), a_t, o_{t+1}, c_{t+1}, v_{t+1} ∈ R^k,
   episode_step t)`. K-step post-action firing indicators
   `f_m(t) = [Σ_{τ=t+1}^{t+K} 1{v_τ[m] fires} ≥ 1]` are computed lazily on
   buffer scan. No counterfactual rollouts, no edit grammar, no event lens.
2. **Core primitive:** the (|C| × |A| × k) tensor `S[c,a,m] = TV(p_m^+, p_m^-)`
   where `p_m^±[c,a]` are the empirical next-cluster distributions over
   `c_{t+1}` conditional on whether channel m fired in `[t+1..t+K]` after
   taking action `a` at cluster `c`. Computed periodically (every N
   transitions) by buffer scan; smoothed with a Laplace prior (add ε to each
   cluster bin) to avoid degenerate small-sample TVs.
3. **Improvement operator:** at cluster `c`, the logit update is
   `Δlogit[c,a] = α (n_a − m_a)` where `n_a = #{a' ≠ a : S[c,a,:] ≻ S[c,a',:]}`
   and `m_a = #{a' ≠ a : S[c,a',:] ≻ S[c,a,:]}` in the strict coordinate-wise
   partial order on `R_+^k` (one strict inequality, no reversed). When all
   actions are mutually incomparable the nudge is zero. The policy is the only
   parameterized object.
4. **Execution rule:** sample from the softmax of `(logits + Δlogits)`. No
   verifier call, no best-of-N, no rollout simulation. Default policy entropy
   floor (Dirichlet prior on logits) to keep all (c,a) cells receiving
   exploration mass.
5. **Vector feedback rule:** never collapsed. Each channel `m` participates as
   an independent partition variable producing its own conditional next-cluster
   distributions; Pareto-non-dominance is the sole aggregator. Channels with a
   degenerate partition (one side empty) contribute zero on both sides of the
   Pareto comparison and are silently dropped from that (c,a) row.
6. **Rollout-cost discipline:** zero counterfactual rollouts. Zero verifier
   calls. Zero simulator branches. The primitive is a sufficient statistic of
   the on-policy buffer; recomputation is `O(|buffer|)` per refresh and
   amortized to one pass every N transitions. Each environment interaction
   contributes to all `(c, a, m)` cells it touches and counts once. At
   deployment, decision-time cost is one read of `S[c,:,:]` and one Pareto vote
   over `|A|` k-vectors — no model unroll, no gradient.
7. **Nearest-neighbor novelty audit:** closest prior is **TRAC (#26)** — also
   computes a divergence (JSD) between successor-cluster distributions
   partitioned by channel firing, also Pareto-non-dominance over the k-vector,
   also no scalar collapse. The structural distinction is the **cell-collision
   bottleneck**: TRAC's failure was that `(state-cluster, action)` cells
   accumulated insufficient sample mass under uniform exploration on long-horizon
   sparse envs; the JSD primitive stayed silent. CSD attacks this directly by
   (a) using the *next-cluster* identity as the partition variable's *target*
   (which fills with every transition, not just channel-firing transitions),
   (b) applying Laplace-smoothed TV instead of plug-in JSD (TV is well-defined
   with one sample per side; JSD requires both empirical distributions to have
   support overlap or it diverges), and (c) computing the partition over a
   K-step *window* of channel firing, so a channel that fires at all in the
   window contributes — relaxing TRAC's binary "fired post-action vs not" gate
   that required the channel to fire in the immediate next step. This is a
   structural relaxation of the cell-mass requirement, not a renaming.
   Secondary nearest is **CWAI (alive-promising)**: CWAI computes a per-(action,
   channel) Jacobian column-norm `‖∂f_θ(o,a)[m]/∂e(a)‖`; CSD computes a
   per-(cluster, action, channel) empirical TV-distance over post-action
   *cluster identities*. CWAI's signal is a parameter-space gradient norm that
   shrinks under stochastic transitions; CSD's signal is a discrete-distribution
   TV that is bounded in [0,1] and noise-robust by construction.
8. **Predicted failure modes:**
   - **Step-penalty domination.** If `S[c,a,step-penalty]` is the only
     non-degenerate column (because the step-penalty channel fires every step
     and partitions windows only by whether they contain a terminal step), the
     Pareto vote reduces to single-channel maximization on
     `S[c,a,step-penalty]`, which is a "control time-to-termination" signal —
     a scalarization of a single channel. Pre-commit: log
     `mean_a S[c,a,m]` per channel m; if step-penalty's mean is more than 5x
     the next channel's, the candidate has collapsed.
   - **Cluster-vocabulary explosion.** If the clusterer produces too many
     clusters, every transition lands in a fresh cluster and `p_m^±` are
     supported on disjoint singletons; TV is uniformly 1 and uninformative.
     Pre-commit: keep cluster vocabulary small (≤ 256) and log mean
     bin-occupancy.
   - **Cluster-vocabulary collapse.** If clusters collapse to ≤ 2, `p_m^±` lose
     resolution and TV is uniformly small. Same diagnostic.
   - **Bootstrap silence on terminal-only channels.** Same family as FED/CEC:
     if no rewarded trajectory exists yet, the partition `f_m = 1` is empty for
     terminal-only channel m, and S is degenerate on that column. CSD is
     non-degenerate on the step-penalty column from episode 1, so the operator
     fires from episode 1 — but the channels we care most about are silent
     until first success. Mitigation: rely on the dense channels' partial
     ordering to drive exploration toward states with high
     channel-discrimination diversity, which is a structural prior toward
     "interesting" states (states whose action choices most discriminate
     post-action behavior).
   - **Stochastic-transition collapse.** If transitions are uniform-random over
     next clusters regardless of action and channel, `p_m^+` ≈ `p_m^-` ≈
     uniform, S → 0 across all channels, Pareto vote degenerates. CSD is dead
     in fully-stochastic environments by design; the substrate's RG has finite
     stochasticity (5% slip), so this is a magnitude question, not a sign one.
   - **Partial-observability collisions.** If two semantically distinct states
     collapse to the same cluster, p^+ and p^- mix incompatible distributions
     and TV is artificially inflated. Mitigation: use a learned embedding for
     clustering rather than a hash on raw obs.
9. **Side-information channel:** vector diagnostics (the per-step k-channel
   `info["vector"]` provides the partition variable for the K-step firing
   indicator) + transition geometry (the cluster-vocabulary on observations
   provides the categorical variable for the conditional distribution). Both
   are surfaced by the substrate; neither is hand-engineered as an event lens.
10. **Monotonic improvement claim:** under a fixed clusterer, fixed K, and
    on-policy data collection, the operator monotonically increases the
    expected per-channel *causal-effect lower bound*: `E_π[ I(C_{t+1}; F_m | c, a) ]`
    where `F_m = f_m(t)` is the K-step firing indicator and `C_{t+1}` is the
    next-cluster identity. TV-distance lower-bounds this mutual information up
    to a constant (Pinsker-type); Pareto-non-dominance over the k-vector
    monotonically improves the per-channel MI lower bound on at least one
    channel without decreasing it on any other. This is the meaningful
    improvement claim: the policy preferentially executes actions whose
    next-cluster identity is most informative about each channel's firing,
    coordinate-wise — i.e., the policy navigates toward "channel-controlling"
    states. The claim holds in expectation under stationary on-policy data,
    fails under non-stationary data.

## Why it is not TRAC (#26) or CWAI (alive-promising)

vs **TRAC (#26)** — TRAC partitioned successor clusters by channel firing in
the *immediate* post-action window and used JSD as the divergence; the cell
that broke it was `(state-cluster, action)` cell-mass undersaturation under
uniform exploration. CSD relaxes the partition window to K steps (so a channel
firing anywhere in the window contributes) and uses Laplace-smoothed TV (which
remains well-defined with one sample per side); the cell-mass requirement is
reduced because the partition fills more reliably and TV is meaningful at
smaller sample sizes than JSD. This is a structural relaxation of the bottleneck
that killed TRAC, not a notational shift.

vs **CWAI (alive-promising)** — CWAI's signal is a parameter-space Jacobian
column-norm of a learned forward model; CSD's signal is an empirical TV
between two cluster-distribution columns. CWAI shares model parameters across
actions and channels, which causes the rank-1 collapse predicted as a falsifier;
CSD has no shared parameters in the primitive itself (clustering is a
disjoint preprocessing step) and the per-channel statistics are independent
empirical estimates. CWAI fails on RG because Jacobian norms shrink under
stochastic transitions; CSD's TV is bounded in [0,1] and degrades smoothly
rather than collapsing to noise floor.

## Why it scales beyond the substrate

At a 20k-step horizon with paragraph-scale actions and a 6-component vector
feedback signal: the primitive is `O(|C| × |A| × k)` storage where `|C|` is the
clusterer vocabulary (independent of horizon, controlled by the clusterer), the
"action" is hashed by intent-cluster (e.g., embedding bucket of generated
paragraph) so `|A|` does not blow up at decision time, and `k` is the number
of feedback channels. The K-step window is a hyperparameter that scales
sub-linearly with horizon (K=64 or K=256 is enough; the channels need only fire
*somewhere* in the window for the partition to be non-degenerate). At paragraph
scale, channels like {validity, safety, latency, cost, correctness, preference}
are exactly the targets the primitive was designed to discriminate over, and
the next-cluster distribution becomes the embedded next-state of the
LLM-with-tool environment. The fundamental claim — that next-state-identity is
informative about subsequent feedback-channel firing, and an action that
controls this informativeness controls the channel — is horizon-agnostic and
becomes more valuable as the channel count grows, because Pareto-non-dominance
gets sharper in higher-dimensional `R_+^k`. The primitive does not require
recurrent state, does not require terminal-cumulant matching, and does not
require trajectory pairs — it is a single-pass buffer statistic that scales
with experience, not with horizon.
