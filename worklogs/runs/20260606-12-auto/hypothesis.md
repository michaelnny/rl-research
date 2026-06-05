# 20260606-12-auto — JFP: Jacobian Firing-Phase

## Research Gate

primitive: per-(state, action, channel) **timestep-of-peak-sensitivity** of the
learned forward model along a counterfactual H-step rollout under the current
policy with a one-shot action intervention at t=0.
improvement_operator: Pareto-non-dominance logit nudge over the per-action
firing-phase vector `J(s,a) ∈ R^k`; actions whose channel-phase profile is
non-dominated (effect peaks earliest across channels) receive a positive
logit shift; dominated rows receive a negative shift; incomparable rows are
left alone.
side_information: learned dynamics (forward model trained on observed
transitions) + vector diagnostics (per-channel cumulant signal v_t).
nearest_prior_or_disqualifier: alive candidate **CWAI**
(channel-wise-action-influence — 1-step Jacobian column-norm magnitude) and
disqualifier family **RND/curiosity** (forward-model error magnitude).
falsifier: if on the vector panel the per-channel argmax-time `t_m^*` is the
same across actions for >50% of decision steps (the forward model's
sensitivity peak is determined by the policy's rollout schedule, not by the
intervened action), JFP collapses to a constant logit nudge; or if the
forward model fails to encode terminal channels (treasure on DST, gold/gem
on RG), `J[·,·,m]` is undefined for those channels and JFP degenerates to
single-channel timing on the step-penalty channel — a CWAI/FFTV-style
single-channel collapse.

## Mechanism

We train a self-supervised forward model `f_θ: (o_t, a_t) → ô_{t+1}` on
observed transitions, and we additionally let the model output a predicted
per-channel cumulant projection `ĉ_t ∈ R^k` (a small linear head on top of
the latent — supervised from observed `v_t`, never used as a reward). At any
decision state `o_t`, for each candidate action `a`, we roll the forward
model H steps with the *intervention* that the first action is `a` and
subsequent actions are sampled from the current policy `π_θ`. Along this
rollout we compute, for each channel `m`, the timestep
`t_m^*(s,a) = argmax_{1≤τ≤H} ‖∂ĉ_τ[m] / ∂e(a)‖_2`
where `e(a)` is the one-hot action embedding fed at step 0, and the
gradient is taken through the unrolled forward dynamics. The primitive is
the per-channel **firing-phase vector** `J(s,a) = (t_1^*, …, t_k^*)`. The
improvement operator computes, at every decision state, the row-wise
Pareto-non-dominance status of `J(s,·)` (lexicographic minimization is
forbidden — only coordinate-wise dominance counts), and applies a logit
nudge `Δ_a = α · (#dominated_by_a − #dominates_a)`. No critic, no Bellman
backup, no scalar advantage, no return-to-go, no scalarized vector reward.
The forward model and its cumulant head are trained by ordinary supervised
prediction losses; the policy's only learning signal is the Pareto nudge.

## Required candidate shape

1. **Experience object:** transitions `(o_t, a_t, o_{t+1}, v_t)` collected
   on-policy (where `v_t ∈ R^k` is the per-step vector signal from
   `info["vector"]`); a small replay buffer for the forward-model SL
   loss. No trajectory-pair indexing, no observation-hash bucketing, no
   reward-bearing-suffix gating.
2. **Core primitive:** `J(s,a) ∈ R^k` — the per-channel argmax-timestep
   along an H-step counterfactual rollout of the learned forward model
   `f_θ`, with a one-shot intervention of action `a` at the head and the
   current policy `π_θ` sampling subsequent actions. The quantity
   measured per channel is the timestep where the *gradient norm* of the
   predicted cumulant w.r.t. the head-action embedding peaks.
3. **Improvement operator:** at each decision state `o_t`, compute
   `J(o_t, a)` for every action `a` (cheap if the forward model is small
   and H is moderate); compute the Pareto-non-dominance count
   `n_a = #{a' : J(o_t, a) dominates J(o_t, a') coordinate-wise}` and
   `m_a = #{a' : J(o_t, a') dominates J(o_t, a)}`; nudge logits by
   `α(n_a − m_a)`. "Dominates" means coordinate-wise ≤ with at least one
   strict <. No scalar collapse.
4. **Execution rule:** sample actions from the policy `π_θ(·|o_t)` whose
   logits have been nudged by the JFP signal at the current step. The
   forward model's rollouts are used only to compute J — never to plan,
   never to substitute for environment interaction.
5. **Vector feedback rule:** Pareto-non-dominance is the only operation
   over channels. The cumulant head `ĉ_t` is supervised per-channel
   (one MSE loss per dimension), preserving channel identity. There is
   no point at which a vector becomes a scalar.
6. **Rollout-cost discipline:** per environment step, the JFP primitive
   costs `|A|` H-step forward-model rollouts (no environment
   interaction). On the substrate with `|A| ≤ 7` and a small forward
   model, H = 8 gives ~56 forward-model passes per decision — entirely
   on the GPU, zero environment cost. Forward-model SL loss is
   computed once per rollout batch from on-policy transitions. The
   primitive consumes exactly one environment step per accepted policy
   update on average; no counterfactual environment rollouts, no
   verifier calls.
7. **Nearest-neighbor novelty audit:** the closest live candidate is
   **CWAI** (alive-promising; Jacobian column-norm magnitude, 1-step,
   nudge by Pareto over magnitudes). JFP differs structurally in two
   ways: (a) it measures *timing* of peak sensitivity rather than its
   *magnitude*, addressing CWAI's failure on Resource Gathering where
   stochastic transitions shrink Jacobian magnitudes toward the noise
   floor (timing of the argmax is more robust to magnitude noise than
   the magnitude itself); (b) it uses an H-step unrolled trajectory,
   exposing the policy-induced channel propagation pattern, where CWAI
   collapses to a single-step view that cannot resolve "action a affects
   channel m two steps later" vs "action a affects channel m
   immediately." The closest disqualifier is **RND/curiosity** — JFP
   does not use prediction error at all; it uses the *gradient
   structure* of a perfectly-fit forward model, which stays informative
   even when prediction error is zero (the structural distinction CWAI
   articulated and that JFP inherits).
8. **Predicted failure modes:**
   (a) If the forward model under-fits the cumulant head on terminal-only
   channels (treasure, gold/gem) within the 120 s budget, the gradient
   `∂ĉ_τ[m]/∂e(a)` is near-zero for those channels and `t_m^*` is
   determined by argmax over noise, collapsing JFP to single-channel
   timing on the step-penalty channel — same shape as FFTV's RG
   collapse.
   (b) If the policy entropy is high enough that all actions produce
   nearly-identical rollout trajectories under the current `π_θ`, the
   per-action argmax-timestep is identical across `a` and the
   Pareto-non-dominance test produces a constant zero nudge — the
   stated falsifier above.
   (c) On extremely stochastic environments (RG's slip noise) the
   forward model may converge to a low-information mean predictor,
   making `t_m^*` an artifact of the model's inductive bias rather
   than the dynamics — a known forward-model-pathology shared with
   CWAI.
   (d) If `H` is too small relative to the channel-firing horizon, the
   argmax saturates at the rollout boundary `t_m^* = H` for terminal
   channels, providing no discrimination across actions on those
   channels.
9. **Side-information channel:** *learned dynamics* (the forward model
   `f_θ` trained on transitions) + *vector diagnostics* (per-channel
   cumulant signal `v_t` used as supervised target for the cumulant
   head, never as a scalar reward). No environment instrumentation
   beyond the standard `info["vector"]` exposure.
10. **Monotonic improvement claim:** under the assumption that the
    forward model's cumulant-head Jacobian timing-pattern is a faithful
    proxy for *true* per-channel time-of-effect (i.e.,
    `t_m^*(s,a) ≈ argmax_τ ∂E[c_τ[m]]/∂e(a)`), the JFP nudge improves a
    Pareto-frontier-minimizing surrogate over per-channel response
    delay: in expectation, the policy concentrates probability on
    actions whose channel-response timing vector is non-dominated in
    the partial order on `R^k`. This is monotonic in the
    Pareto-dominance count, not in any scalar return. Under stochastic
    transitions the surrogate degrades smoothly: the
    argmax-of-noisy-gradient is still consistent in expectation as
    long as the gradient signal exceeds the noise variance integrated
    over `H` steps.

## Why it is not CWAI

CWAI's central improvement object is the per-(action, channel) **gradient
column-norm magnitude** at a single state, with the Pareto comparison run
over magnitudes. JFP's central object is the per-(action, channel)
**argmax-timestep** of the gradient norm along an unrolled rollout, with
the Pareto comparison run over timesteps (in `R^k`-with-natural-numbers
codomain). The structural distinction holds under variable renaming
because CWAI is silent on *when* an action's channel-effect peaks; it
ranks actions by *how loud* the effect is at one fixed time. JFP is
silent on *how loud* the effect is and ranks actions by *when* the effect
peaks across channels. The two primitives are orthogonal: CWAI ranks by
amplitude, JFP ranks by phase, and they respond differently to noise
(magnitude-noise hurts CWAI; phase noise hurts JFP, but phase is
typically more stable than amplitude under stochastic dynamics — this is
the Resource-Gathering theory of failure for CWAI that JFP is designed
to address). It is also not GVFs/successor-features because nothing is
predicted as a cumulant target for the policy's argmax — the cumulant
head exists only to define the gradient against which the timing is
measured.

## Why it scales beyond the substrate

At long horizon (10k–20k actions), per-channel propagation timing is
exactly the structure value-bootstrap fails on: a sparse-terminal-reward
chain has Q ≈ 0 everywhere because the bootstrap is silent; but a
forward-model rollout of length H ≪ horizon still exposes
*phase-structure* between action choices and channel-firing predictions
on the dense channels (step cost, validity, latency, energy, safety), and
those dense channels carry the bulk of the partial-order signal
needed to navigate toward the rare terminal reward. For agentic LLM
settings, "generate a paragraph" admits an action embedding `e(a)` (token
sequence, latent code, or tool-call signature) over which the
forward-model gradient is well-defined; the H-step rollout is a small
internal simulator, not a tree search; the per-channel cumulants
(correctness, cost, safety, latency, validity, preference) are exactly
what the deployed system already exposes. Pareto-non-dominance over
phase scales with `O(|A|² · k)` per decision, which is tractable when `k`
is the natural feedback-channel count (≤ 8 in deployed settings) and
`|A|` is the small set of structured tool-calls or candidate
continuations being considered. The primitive does not require enumerating
the full action space; it requires only the local action set being
considered at the decision step.
