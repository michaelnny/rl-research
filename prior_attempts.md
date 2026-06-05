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
