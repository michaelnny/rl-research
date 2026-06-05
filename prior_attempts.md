# prior_attempts.md

Fourteen directions have been tried and failed across three prior research
sprints. This list is a **hint**, not a gate. Read it before proposing a new
direction so you don't re-derive a known dead end. If your candidate is
structurally identical to one of these, it will not produce a new result.

Each entry: name, one-line mechanism, one-line reason it failed. These
summaries are intended to be self-sufficient — you should not need to open
the worklogs to act on them.

Each entry below has its full record at `worklogs/attempts/NN-<slug>.md`
(template: `worklogs/TEMPLATE.md`). Open the per-attempt file when you
need the math, the prototype detail, or the cross-attempt comparison.

Alive-but-not-yet-tested directions live under `worklogs/candidates/` and
do not appear in this list — they are not failed attempts. A candidate
graduates into `attempts/` (and gets an entry here) when a `train.py`
commit runs it against the panel.

## Sprint 1 (2026-05-24)

1. **FROST** — vector repair certificate over structured per-channel feedback;
   solve a constrained projection of behavior toward feasible region.
   *Failed:* not reward-native; needs dense per-channel diagnostics + local
   counterfactual influence; collapses on terminal-only reward.

2. **BRIC** — bracketed reward-intervention control: swap segment of anchor
   trajectory with donor segment; keep if terminal reward improved.
   *Failed:* requires many counterfactual env rollouts per accepted bracket;
   depends on knowing the right edit grammar in advance.

3. **KERNEL-RL / RSK** — passively mine compact behavior atoms whose presence
   statistically separates reward-bearing from non-reward-bearing experience.
   *Failed:* DeepSea-style: when reward correlations don't exist until a deep
   path is intentionally traversed, there is nothing to mine. Pure association.

4. **CARL / Frontier-Graph** — controllability-first: learn reproducibly
   reachable cells, expand frontier, attach reward events post-hoc.
   *Failed:* structurally indistinguishable from Go-Explore + count-based
   exploration unless a genuinely new abstraction-learning principle is added.

## Sprint 2 (2026-05-26)

5. **BCE-v0** — branch-certificate editing: local successor support + vector
   outcome cones; expand around current trajectory.
   *Failed:* discovery mechanism collapsed to novelty/count exploration on
   ablation.

6. **T-CTBP** — transported causal/event-transform boundary projection: stack
   of event transforms + separators + cones + transported logit head.
   *Failed:* mechanism stack with > 3 named components; not a primitive. No
   single composition law; the math was ugly.

7. **OPP / Order Projection** — learn partial order over decisions from passive
   data, KL-project policy onto it.
   *Failed:* passive action order is not causal prerequisite structure.
   DoorKey requires "key before door"; passive correlation does not give that.

8. **EOP/COP on MiniGrid** — symbolic clause orders over observed event-effects.
   *Failed:* manufactured thousands of plausible-looking certificates;
   none of them were the right ones for DoorKey. Volume ≠ understanding.

9. **Causal Dominance Certificates** — compare local interventions instead of
   passive trajectory correlations.
   *Failed conceptually:* more principled than OPP but still inelegant; not a
   clean primitive; not carried to implementation.

10. **Value-function-first reset** — not an algorithm; the conceptual turning
    point that surfaced *why* the prior nine all failed: they avoided value
    *terminology* without replacing what value *does* (compress future, compose
    temporally, support local improvement).

11. **TOP — Temporal Outcome Profiles** — replace scalar value with first-hit
    event/outcome profiles as the local future-compression object.
    *Alive but weak:* solves DoorKey-5x5 (eval success 0.98 but worse than Q);
    DoorKey-6x6 needed more episodes; KeyCorridor unstable across seeds. Close
    to GVFs / successor features / multi-objective RL / reward machines under
    inspection. Not promoted; structural distinction not yet established.

## Sprint 3 (offline exploration, 2026-05-29)

These three came from an offline exploration batch — derivation only, no
substrate `train.py` commits. Logged here to keep the negative space
canonical.

12. **Policy-Edit Optimization (PEO)** — make the policy primary by estimating
    the response of the outcome law to each policy edit; apply edits whose
    response points into a desirable vector cone.
    *Failed:* matched by a scalar edit-ES on the same hand-designed semantic
    edit basis. The basis was the algorithm; the optimizer was an ES rebadge.

13. **ETB / HPC** — Event-Time Behavioral Basis (first-action on shortest
    suffix to event `g` from context `c`) plus Hindsight Policy Compression
    (MDL-compress event-reaching suffixes into a conditional program).
    *Failed:* ETB is goal-conditioned hindsight + event-options; HPC is
    GCSL-like supervised hindsight imitation. Useful as components, not as a
    family.

14. **Primal Behavior Flow Pivot** — pivot the central object from value to
    occupancy flow `μ_π(s,a)` and recover the policy by normalization.
    *Failed:* mathematically clean but does not expose any new side-information
    advantage for sparse long-horizon discovery. Collapses to occupancy-measure
    LPs / max-ent RL / GAIL / GFlowNets / mirror-descent PI under inspection.

## Sprint 4 (2026-06-05)

15. **FED — Frontier-Expanding Dispersion** — per-action empirical Pareto-front over
    vector-outcome signatures indexed by observation-hash bucket; accept action logit
    nudge iff the action's conditional outcome multiset extends the bucket's attainment
    set in the partial-order sense without contracting it.
    *Failed:* same bootstrap wall as SIT: observation-hash buckets never accumulate
    sufficient sample mass under uniform exploration on long-horizon sparse envs, so
    the Pareto-front extension indicator never fires; scored 0.0 on all envs including
    the vector envs (Deep Sea Treasure, Resource Gathering) where it was predicted to
    be strong. Rules out the "empirical Pareto front / outcome-multiset indexed by
    obs-hash" family unless paired with an explicit exploration primitive.

16. **PICAV — Path-Integrated Channel-Asymmetry Voting** — per-(obs-hash, action) empirical mean of signed antisymmetric pair-contribution vectors `δ_{jk,t} = v_t[j]·Δn_t[k] − v_t[k]·Δn_t[j]`; nudge policy logits toward upper-orthant Pareto-frontier actions within each bucket.
    *Failed:* claimed to bypass FED's bootstrap wall because pair-contributions are nonzero on every step — but on Deep Sea Treasure the treasure channel fires only at the terminal step, making all pair entries zero throughout the episode (same bootstrap collapse as FED). Scored 0.0 / 0.011 vs random 1.331 on both vector envs. Rules out the "signed cross-channel temporal-ordering moment" family whenever any vector channel is terminal-only.

17. **CHX — Cumulant-Hull Extremality** — per-step L2 distance-to-convex-hull of within-trajectory cumulant trace in `R^k`; weight log-prob update by centered hull-contribution `(h_t − 1/T)` — no critic, no cross-trajectory comparison, no scalarization.
    *Failed:* when any vector channel is terminal-only (as in Deep Sea Treasure and Resource Gathering), the cumulant trace is effectively k_eff=1 (a near-line in the step-penalty direction with the reward dimension firing only at termination); the hull's extremes reduce to episode start/end, collapsing CHX to a return-to-go rebadge. Scored 99.0 / 0.011 vs random 194.0 / 1.331 on both vector envs (below random). Extends PICAV's ruling: any within-trajectory signal-geometry primitive collapses on the substrate's terminal-only vector channels.

18. **CEC — Continuation-Endpoint Concordance** — per-(state-hash, action) multiset of vector cumulants indexed by *exit-observation-hash bucket* (the terminal observation of the episode); logit update driven by signed Pareto-dominance count across buckets where action a's bucket-conditional mean cumulant dominates action a'.
    *Failed:* same bootstrap wall as FED (#15) despite switching from mid-trajectory obs-hash to terminal exit-hash bucketing. The concordance signal never fired within the 120 s budget because the seeding phase could not accumulate ≥ 2 samples per (state, action, exit-hash) bucket — the hypothesis's own stated falsifier. Scored 0.0 / 0.011 vs random 194.0 / 1.331 on both vector envs. Extends the FED family ruling to cover exit-hash variants: the entire "empirical Pareto-front / cumulant-multiset indexed by any observation hash" family fails without an explicit exploration primitive providing sufficient coverage before concordance comparisons are made.

19. **CWTP — Confluence-Witness Trajectory Pairs** — sign-vote tensor over per-channel segment-cumulant differences between trajectory pairs that diverged at a state and reconverged at a later shared observation-hash; logit nudge by Pareto-non-dominance count of normalized sign-vote rows, no scalar collapse.
    *Failed:* same bootstrap wall as FED/CEC, compounded: requires both cross-trajectory observation-hash collisions at non-terminal states AND non-trivial per-step vector signal in the bracketed segment; neither condition is reliably met on sparse long-horizon envs with terminal-only reward channels. Scored 0.0 / 0.011 vs random 194.0 / 1.331 on both vector envs. Extends the sprint-4 ruling to cover the "pairwise trajectory comparison indexed by intermediate shared state" sub-family.

20. **LRA — Loop-Return Aversion** — per-(obs-hash, action) empirical mean of within-episode closed-loop vector cumulant deltas `Δc = c_{t'} − c_t`; suppress action logit iff loop-signature mean is Pareto-dominated by the zero vector (all channels ≤ 0, at least one < 0).
    *Failed:* on both vector panel envs (DST and RG) the universal step-penalty channel is strictly negative on every step, so every intra-trajectory loop accumulates a negative entry in that channel and every looping action is unconditionally suppressed — the operator reduces to count-based exploration suppression, a named disqualifier. On DoorKey the partial-observable state changes on most steps (carrying the key changes obs), so hash collisions are rare and the operator almost never fires. Scored 0.0 / 0.121 vs random 0.137 / 1.331. Hypothesis's own falsifier confirmed: the family is dead when every vector env requires excluding the step-penalty channel from the dominance test.

21. **TPP — Terminal-Postfix Pairing** — for trajectory pairs whose terminal observation-hashes match, walk both backward in lockstep to find the postfix-divergence anchor; accumulate a Pareto-vote count W[s,a] and nudge policy logits toward the Pareto-non-dominated action at each anchor.
    *Failed:* same bootstrap wall as CEC (#18) and the FED family — the primitive is silent until terminal-observation-hash collisions accumulate, which does not happen within 120 s on long-horizon sparse envs; W stayed effectively empty, operator never fired. Scored 0.0 / 0.011 vs random 194.0 / 1.331. Extends the FED/CEC ruling to the "terminal-observation-matched pair + backward lockstep walk" sub-family; any hash-collision-gated pair primitive fails without a paired exploration primitive providing coverage.

22. **CRP — Channel Rank-Position Concordance** — per-(state-cluster, action, channel) running mean of within-trajectory rank-percentile of channel firing magnitude, restricted to firing steps; Pareto logit nudge toward actions whose trend-corrected rank vector R̃[s,a,:] is non-dominated across channels.
    *Failed:* when a vector channel fires only at the terminal step (as in Deep Sea Treasure and Resource Gathering), within-trajectory rank-percentile is constant 1.0 for every trajectory, making R cells degenerate — the claimed magnitude-invariant rank statistic carries zero discriminating information. The hypothesis's own falsifier (a) was confirmed. Scored 0.0 / 0.011 vs random 194.0 / 1.331. Extends the CHX/PICAV/LRA ruling to rank-based within-trajectory signal-geometry primitives: temporal rank position is no more informative than magnitude when a channel fires only once per episode.

23. **TCP — Temporal Channel Precedence** — cross-trajectory cross-channel lag-asymmetry tensor `Λ[j,k] = E[sign(t_first(j) − t_first(k))]` builds a precedence DAG P; at each decision step the residual channel set R (channels not yet fired whose DAG predecessors have fired) restricts a Pareto-vote logit nudge over per-(cluster,action) empirical firing means.
    *Failed:* on Deep Sea Treasure the terminal-only treasure channel forces a trivially asymmetric DAG edge (step-penalty always precedes treasure), collapsing R to the singleton {step-penalty} at every non-terminal step; Pareto comparison over a single channel reduces to scalar maximization of that channel, which is the scalarized-vector-reward disqualifier. Scored 99.0 / 0.121 vs random 194.0 / 1.331 (below random on both). Extends the ruling to DAG/temporal-ordering primitives: any precedence-gating primitive that produces a singleton residual channel set on terminal-only-reward substrates collapses to scalar channel optimization.

24. **PCR — Policy Commitment Recovery** — per-(context-cluster, action) commitment-recovery vector `R[c,a] ∈ R^L` measuring expected step-lag until the snapshot policy's modal action is re-confirmed at L alignment thresholds; improvement operator is a Pareto-meet of recovery-non-dominance and terminal-outcome-non-dominance, with terminal vector outcome used only as a binary sign gate.
    *Failed:* despite the reward-independent primitive (R fires on every step from action-logit self-comparisons), the improvement operator requires a terminal-outcome sign gate that is universally silent before any rewarded trajectory is collected — same bootstrap wall as FED/CEC/TPP. Scored 0.0 / 0.011 vs random 0.137 / 1.331 on sparse envs, 99.0 vs random 194.0 on DST. Extends the bootstrap-wall ruling: a reward-independent primitive paired with a terminal-outcome-gated operator inherits the full FED-family collapse; the gate must be eliminated or replaced with a direction signal that fires before reward appears.

25. **ACS — Action-Conditional Suffix-Spectrum** — per-(state-cluster, action, channel, frequency-band) empirical spectral-variance tensor `S[s,a,m,f]` measuring within-band firing-indicator variance at F log-spaced temporal scales; Pareto-non-dominance logit nudge over the flattened (k·F)-dimensional spectral matrix per action.
    *Failed:* expanding FFTV's k-dimensional Pareto comparison to k·F dimensions (k=2 channels × F=4 bands = 8 dims) caused Pareto-front saturation — virtually all actions become non-dominated in 8-dimensional coordinate-wise partial order, rendering the logit nudge symmetric (random). FFTV scored 1382 on DST; ACS scored 0.0 vs random 194.0 on the same env — a complete collapse, not degradation. Rules out multi-band spectrum extensions of FFTV unless a front-compression mechanism (lexicographic ordering by frequency tier, strict-margin dominance, or band aggregation before the Pareto test) prevents dimension-induced saturation.

26. **TRAC — Transition-Refractive Action Channels** — per-(state-cluster, action, channel) JSD between two empirical successor-cluster histograms partitioned by whether channel `m` fired in a post-action window; Pareto-non-dominance logit nudge over the k-dimensional JSD row, no scalar collapse.
    *Failed:* same bootstrap wall as FED/CEC/TPP/PCR — despite the step-penalty channel seeding H_fire from early trajectories, the (cluster, action) cell-collision bottleneck still applies: repeated revisitation of the same cluster-action pair doesn't happen reliably under uniform exploration on long-horizon sparse envs within 120 s. JSD primitive stayed silent on DoorKey-8x8/KeyCorridor (0.0), below random on DST (98.0 vs 194.0) and RG (0.011 vs 1.331). Extends the FED-family ruling to cluster-indexed conditional-distribution primitives: the bottleneck is state-cluster revisitation frequency, not channel-firing frequency.

27. **PCGA — Per-Channel Gradient Alignment** — per-(state, action, channel) cosine similarity in parameter space between the policy's log-prob gradient `∇_θ log π(a|s)` and an auxiliary cumulant-prediction head's gradient `∇_θ ĉ_m(s)`; Pareto-dominance logit nudge over the `|A|×k` alignment matrix, no scalar collapse and no use of the head's output magnitude.
    *Failed:* the shared trunk between policy head and auxiliary head causes both gradient vectors to be dominated by the same trunk-parameter activations, making A[s,a,m] nearly action-invariant — the Pareto vote sees near-uniform rows and produces near-symmetric nudges equivalent to random perturbation. Scored 99.0 / 0.011 vs random 194.0 / 1.331 on both vector envs (below random). Rules out parameter-space gradient alignment via shared-trunk architecture; future variants must operate on action-specific parameter subsets or zero out shared trunk components before computing cosines.

28. **ARP — Action-Reachable Pattern Lattice** — per-(state-cluster, action) empirical set `S(s,a) ⊆ {0,1}^k` of distinct binary channel-firing patterns observed in completed suffixes; improvement operator is strict-superset existence on the lattice `({0,1}^k, ≤)` — logit nudge `α(n_a^{dom} − n_a^{sub})` where `n_a^{dom}` counts actions whose entire support is subsumed by some element of `S(s,a)`.
    *Failed:* same bootstrap wall as FED/CEC/TPP/PCR/TRAC. The strict-superset operator requires at least one rewarded trajectory to populate `S` cells with a rich pattern; before that all cells hold only the step-penalty singleton and the operator is universally silent. Scored 99.0 / 0.011 vs random 194.0 / 1.331 (below random on both vector envs). Extends the bootstrap-wall ruling to "empirical set of binary suffix patterns": the bootstrap wall is a property of when the first rewarding suffix appears, not of whether downstream channel information is stored as magnitude vectors, Pareto fronts, or binary-pattern sets.

29. **PFA — Per-Channel Phase-Flow Asymmetry** — per-(cluster, action, channel) running-mean signed 2-D phase-area `Ā[c,a,m] = E[p_m(o_t)·q_m(o_{t+1}) − p_m(o_{t+1})·q_m(o_t)]` where `p_m, q_m` are learned short- and long-horizon firing-probability heads per channel; Pareto-non-dominance logit nudge over the k-vector of signed areas per action.
    *Failed:* the rotational primitive (signed cross-product of two imminence vectors) is near-zero on all substrate channels: always-firing channels (step-penalty) produce `p_m ≈ q_m ≈ 1` so the area `≈ 1·1 − 1·1 = 0`; terminal-only channels produce `p_m, q_m ≈ 0` throughout the episode so the area is also ≈ 0. The signed-area primitive requires non-trivial divergence between short- and long-horizon probabilities, which does not occur on channels that are either always-on or terminal-only. Scored 0.0 / 0.011 vs random 194.0 / 1.331 — identical to the FED/CRP bootstrap-wall family. Extends the CRP (#22) ruling to two-horizon probability heads: any primitive depending on short-vs-long-horizon divergence is structurally silent on the current substrate.

30. **CPR — Channel-Posterior Chebyshev Reweight** — k-tuple of per-channel replay-reweighted empirical action posteriors `π̂_m(a|o)`; improvement operator minimizes `sup_m KL(π̂_m || π_θ)` (Chebyshev center projection) via per-step argmax over channels, with no fixed scalar channel weight.
    *Failed:* on terminal-only-reward envs (DST, RG), the reward channel posterior is degenerate (no mass) throughout the bootstrap window; `sup_m KL` reduces to the single non-degenerate step-penalty channel, collapsing the Chebyshev operator to single-channel weighted cloning — the hypothesis's own stated falsifier confirmed. Scored 99.0 / 0.011 vs random 194.0 / 1.331 (below random on both vector envs). Extends the bootstrap-wall ruling to the "per-channel posterior + sup-norm aggregation" family: the aggregation level does not rescue posterior degeneration; all k channel posteriors must be simultaneously non-degenerate for the Chebyshev center to differ from scalar behavior.

31. **ATP — Action-Tangent Persistence** — learned forward model with per-channel firing-probability heads; action-conditional k-vector of persistence horizons `h*[o,a,m]` (first predicted step at which channel m crosses a self-tuned 75th-percentile threshold); Pareto-non-dominance logit nudge over the integer horizon vectors, no scalar collapse, no critic, no Bellman.
    *Failed:* the universal step-penalty channel (fires every step) dominates the Pareto vote: `h*[o,a,step-penalty]` encodes time-to-termination, making "shorter persistence horizon" equivalent to "reach any terminal state fastest." On DST the operator actively chose nearby low-value treasure over far high-value treasure, scoring 99.0 vs random 194.0 — below-random harm. Reward channels (treasure/gold/gem) remained at h* = H_max for all actions throughout the bootstrap window, contributing nothing. Extends CHX/CRP ruling to forward-model-predicted horizons: any "faster channel onset dominates" primitive reduces to shortest-path-to-terminal planning on envs with a universal step-penalty channel.

32. **CSD — Channel-Conditional Successor Disagreement** — per-(cluster, action, channel) total-variation distance between two empirical next-cluster distributions partitioned by whether channel m fired in a K-step post-action window; Pareto-non-dominance logit nudge over the k-dimensional TV row, no scalar collapse.
    *Failed:* same (cluster, action) cell-revisitation bottleneck as TRAC (#26) — reducing the per-cell sample-size requirement (Laplace-smoothed TV instead of plug-in JSD, K-step window instead of immediate post-action window) does not increase revisitation frequency under uniform exploration on long-horizon sparse envs. The primitive stayed silent on DoorKey-8x8/KeyCorridor/DST (0.0) and below random on RG (0.011 vs 1.331). Confirms that the binding constraint for cluster-indexed conditional-distribution primitives is revisitation frequency, not sample-size per visited cell.

33. **ACFC — Action-Frequency / Channel-Frequency Concordance** — cross-episode sign-concordance matrix `C[a,m]` formed by averaging `sign(Δfreq_a)·sign(Δcount_m)` over trajectory pairs in a rolling buffer; Pareto-non-dominance logit bias over C rows, no state hash, no clustering, no critic.
    *Failed:* before any rewarded trajectory is collected, goal/treasure/gem channels contribute zero firing counts to every episode, so C[a,m] contains only step-penalty signal; the Pareto-non-dominance operator over a rank-1 concordance matrix reduces to scalar step-penalty minimization (fastest-termination preference) — same bootstrap-wall collapse as FED/PCR. Scored 0.0/0.0/99.0/0.011 vs random 0.137/0.0/194.0/1.331 on DoorKey/KeyCorridor/DST/RG (below or at random everywhere). Extends the bootstrap-wall ruling to whole-episode frequency-histogram concordance primitives: eliminating the state-hash index does not cure the pre-reward bootstrap collapse when terminal-only reward channels dominate the vector structure.

## Cross-attempt failure modes

Patterns that appeared more than once. If your candidate exhibits any of them,
expect the same outcome:

- **The primitive needs reward correlation to bootstrap, but reward
  correlation does not exist on long-horizon sparse tasks until a deep
  unrewarded path is traversed.** Pure association mining (KERNEL, OPP, EOP)
  has nothing to operate on.
- **The mechanism is a stack of named components, not a primitive.** T-CTBP is
  the canonical example: more than 3 components and no single composition law
  stitching them together.
- **Passing DeepSea/Chain/Tree is not strong evidence.** Monotone-progress
  benchmarks are passed by count-bonus, novelty, RND, progress heuristics, and
  crude profile dominance. Do not promote on those alone.
- **Custom-built toy benchmarks are insufficient.** All four sprint-1
  candidates "succeeded" on a benchmark designed around the method; all
  failed on standard tasks (DoorKey, KeyCorridor).
- **"Avoid value vocabulary" is not a research direction.** Value's *role*
  (future compression, temporal composition, local improvement) is what
  needs replacing, not its name.
- **Hand-engineered event lenses are side information.** Counts against the
  side-information channel declaration; not free.
- **Abstract mathematical pivot without an exposed side-information
  advantage is a notational shift, not a new family.** A pivot from one
  central object (value, flow, distribution, etc.) to another must say
  *what new side information* the new center makes usable, and *how* that
  side information drives the discovery of new informative trajectories
  before reward correlation exists. (#14 Primal Behavior Flow.)

## Disqualifier families (the negative space)

If your candidate's central improvement operator reduces to any of these under
variable renaming, it is a rebadge — not a new family — even if it beats panel
baselines:

- Bellman backup of any flavor (Q-learning, DQN, Rainbow, SAC, TD3).
- TD-error minimization as the primary update.
- Q / V / advantage / return-to-go as the central learned object.
- Scalar-weighted log-prob update (PPO, TRPO, GRPO, REINFORCE, A2C, IMPALA).
- Actor-critic — any variant where a critic supplies the actor's weight.
- Reward-model optimization (RLHF, DPO, preference optimization).
- Scalarized vector-reward maximization — collapsing `r ∈ ℝᵏ` to `wᵀr` and
  optimizing as if scalar. (The vector envs in our panel are designed to
  detect this directly.)
- CEM / ES / CMA-ES elite refitting.
- Top-k trajectory cloning.
- Go-Explore with renamed cells.
- Count-based exploration with renamed counts.
- RND / curiosity with renamed novelty.
- Options / hierarchical RL with renamed skills.
- Model-based planning with renamed states.
- Verifier-guided search (best-of-N, MCTS, ReAct, reflection) with renamed
  verifier.
- GVFs / successor features with renamed cumulants.
- Distributional RL with renamed return distribution.
- Hindsight Experience Replay with renamed virtual goals.
- Decision Transformer / Trajectory Transformer with renamed conditioning.
- Reward machines with renamed automaton states.

Existing methods may appear as **components** (a torch network, an Adam
optimizer, a replay buffer, a sequence model). They cannot be the
*explanation* for why the method works.

## What "good evidence" looks like

A candidate is interesting only when **all** hold:

1. Mechanism is one primitive + one improvement operator. Not three.
2. There is a one-paragraph monotonic improvement claim — what does the
   operator improve, under what condition.
3. Side-information channel is named explicitly, from the list:
   {transition geometry, event traces, object state, reachability/reset
   structure, demonstrations, pretrained priors, language/task description,
   verifier feedback, learned dynamics, vector diagnostics, environment
   instrumentation}. "None — pure terminal black-box" is rejected as
   information-theoretically impossible at long horizon.
4. The candidate beats `panel_n_beat_strong` on ≥ 2 of the 6 panel envs,
   with at least one win on a vector env.
5. The candidate's structural distinction from the named nearest item in this
   list (1–14 above, or one of the disqualifier families) is articulated in
   2–3 sentences in the commit message.
