# 20260606-04-auto — LRA: Loop-Return Aversion

## Research Gate

primitive: per-(state-hash, action) running mean of the **vector
cumulant signature of the closed intra-trajectory loop entered by taking
that action** — `Δc(loop) = c_{t'} − c_t` over all witnessed within-episode
returns where `obs_hash(s_t) == obs_hash(s_{t'})` and `s_t.action = a`.
improvement_operator: at each state, suppress the logit of any action
whose loop-signature mean is **Pareto-dominated by the zero vector**
(strictly negative in at least one channel and non-positive in all
others), and retain (no nudge) otherwise. No critic, no Bellman backup,
no scalar reward weight, no cross-trajectory matching.
side_information: transition geometry (intra-trajectory return events)
+ vector diagnostics (per-channel cumulant signature of the loop).
nearest_prior_or_disqualifier: 05-bce-v0 (local successor support /
returnability) and rsd-reconvergent-segment-dominance (multigraph
parallel-edge dominance). LRA differs from both: BCE-v0 used returnability
to define a frontier-expansion novelty signal that ablated to a count
bonus; RSD compares parallel segments **across** trajectories. LRA
operates only on **intra-trajectory loops** (the agent has literally
returned to the same observation in the same episode) and uses the
zero-vector Pareto comparison as the operator — nothing cross-trajectory.
falsifier: on the vector envs (DST, RG) and DoorKey, log fraction of
decision steps that participate in a closed loop, and fraction of those
loops with Pareto-dominated-by-zero signatures. If either is below
≈ 5 % within the 120 s budget, the operator fires too rarely to drive
learning and the family is dead. If both fire frequently but panel
score is ≤ random, the per-channel Pareto-vs-zero comparison is the
wrong improvement principle and the family is dead.

## Mechanism

LRA's primitive is the **loop-signature aggregator** `L[s, a] ∈ ℝ^k`,
the empirical mean of all vector-cumulant deltas accumulated along
**closed within-episode loops** entered by action `a` from state `s`:
each time the agent takes action `a` at state `s`, follows some path,
and later in the same episode returns to a state `s'` with
`obs_hash(s') = obs_hash(s)`, the segment cumulant `Δc = c_{t'} − c_t`
is added to `L[s, a]`'s running mean. The improvement operator is a
single logit-suppression rule: at decision time, for each action `a`
whose `L[s, a]` is Pareto-dominated by the zero vector (every channel
non-positive and at least one strictly negative — i.e. the action is
a *loop-suicide* in vector terms), subtract `α` from `logit(a)`; for
all other actions (including those with no observed loops), do
nothing. The operator is a within-trajectory dominance test against
a fixed reference (zero) — no cross-trajectory comparison, no
scalar collapse of the channels, no bootstrapping target. Loops are
the agent's own counterfactual control: the start and end states have
identical observation hash by construction, so the per-channel
cumulant difference is the **provable cost of the detour the action
caused** in vector space, not a return-to-go estimate.

## Required candidate shape

1. **Experience object:** ordinary on-policy rollouts with per-step
   vector signal `v_t` from `info["vector"]` and observation hashes
   `h_t`. No demonstrations, no resets, no oracles.
2. **Core primitive:** `L[s, a] ∈ ℝ^k` — empirical mean per-channel
   cumulant of all closed within-episode loops entered by action `a`
   from observation-hash bucket `s`, plus a count `n[s, a]`. Updated
   online by scanning each new trajectory for hash recurrences.
3. **Improvement operator:** at training-step decision time and at
   gradient time, for each `(s, a)` with `n[s,a] ≥ n_min`, define the
   suppression mask `m[s,a] = 1` iff `L[s,a]` is Pareto-dominated by
   the zero vector (all `≤ 0`, at least one `< 0`). The policy logits
   are updated by `logit(a | s) ← logit(a | s) − α · m[s,a]`. No
   reward-weighted gradient on the un-suppressed actions; the policy
   on those is shaped only by entropy + the suppression of
   loop-suicidal alternatives.
4. **Execution rule:** sample `a ∼ softmax(logit(· | s))` from the
   shaped logits during rollouts. No greedy argmax; no value-based
   exploration bonus.
5. **Vector feedback rule:** the operator is a Pareto comparison
   against the fixed reference (zero), not a scalarization. An action
   that loses one channel and gains another is **not** suppressed;
   only actions that lose without gaining anywhere are. The decision
   rule is sign-of-coordinate; no fixed `w` is ever applied.
6. **Rollout-cost discipline:** **zero counterfactual rollouts**. The
   loop signatures are extracted from the actual on-policy trajectory
   the agent already took. Each rollout contributes O(T) loop-update
   events (one per recurrent observation-hash). At deployment, no
   extra calls — the suppressed-action mask is consulted once per
   step from a static lookup.
7. **Nearest-neighbor novelty audit:** closest priors are
   05-bce-v0 (local-successor returnability used as a frontier
   signal) and rsd-reconvergent-segment-dominance (cross-trajectory
   parallel-edge dominance over vector cumulants). The structural
   distinction is that LRA does not use returnability as an
   exploration novelty bonus (BCE-v0) and does not match parallel
   segments across distinct trajectories (RSD). LRA's primitive is
   the **intra-trajectory return event itself** — the agent's own
   path from `s` back to `s` — and the Pareto comparison is against
   the fixed zero vector (the dominance test the loop must satisfy
   to *not be wasted motion*), not against another action's
   bucket-mean.
8. **Predicted failure modes:**
   - Fails when intra-trajectory hash collisions are rare (DoorKey
     after key collection — the partial-observable state changes
     every step). Falsifier: collision-rate diagnostic.
   - Fails when every loop has a strictly positive "step-cost"
     channel (e.g. step-penalty channel in DST/RG): the
     Pareto-vs-zero test then suppresses **every** action that ever
     loops, which is too aggressive and reduces to "never repeat
     observations" — a count-suppression rebadge. Mitigation
     diagnostic: report per-channel sign histograms of `L[s,a]`; if
     one channel is always negative across all `(s,a)`, that channel
     should be excluded from the dominance test. Excluding it must
     be a stated **structural invariant of the env's vector spec**,
     not a tuned hyperparameter; if every vector env requires this
     exclusion, the family is dead.
   - Fails on stochastic transitions where two visits to the same
     observation hash come from genuinely different latent states
     and the loop signature is meaningless noise.
   - Fails when the action that *enters* a productive loop is
     identical to the action that *enters* a wasteful loop from the
     same state — the (s,a) aggregation cannot separate them.
9. **Side-information channel:** `{transition geometry, vector
   diagnostics}`. Transition geometry shows up as observation-hash
   recurrence within a single rollout; vector diagnostics is the
   per-step `info["vector"]` cumulant signal that defines the loop
   signature. No event lens, no demonstrations, no learned dynamics.
10. **Monotonic improvement claim:** on deterministic transitions and
    a stationary policy, every action `a` from state `s` whose
    accumulated loop signature `L[s,a]` is strictly Pareto-dominated
    by zero is a **certified Pareto-suboptimal action** with respect
    to the do-nothing alternative at that state — taking it strictly
    worsens at least one cumulant channel without compensating any
    other, **conditional on returning to s before episode end**.
    Suppressing such actions monotonically reduces the expected
    fraction of decision steps spent on certified-Pareto-suboptimal
    actions, in the limit of `n_min` → ∞ samples per `(s,a)`. This
    is a within-conditional monotonicity (improvement on the
    sub-policy of states with `n[s,a] ≥ n_min`), not global
    optimality; states that never participate in loops receive no
    update.

## Why it is not 05-bce-v0 / rsd-reconvergent-segment-dominance / count-based exploration

BCE-v0 used local successor returnability to define a *novelty/frontier*
signal that ablated to a count bonus; LRA uses returnability as a
**dominance certificate condition** (the start-end state equality is
the precondition for the per-channel cumulant difference to be a
provable cost), not as an exploration bonus. RSD compares parallel
segments **across trajectories** that share endpoints — a 2-source
comparison that requires populated multigraph edges to bootstrap, the
exact wall RSD hit; LRA compares a single intra-trajectory loop
against the **fixed reference (zero vector)**, requiring only one loop
witness per `(s,a)` and no cross-trajectory matching. Count-based
exploration suppresses repeated observations regardless of vector
content; LRA suppresses only when the **vector structure of the loop
is Pareto-dominated by zero**, leaving productive loops (which gain in
some channel) untouched. Under variable renaming, the count of
suppressed actions is not the count of repeats; it is the count of
sign-coordinate-negative aggregates.

## Why it scales beyond the substrate

In long-horizon settings (20k action episodes), intra-trajectory
recurrence is the dominant structural signal: an agent operating in a
large environment will revisit observations many times within a single
long rollout, and each such revisit is a self-contained
provable-cost-or-benefit experiment that requires no other rollout to
adjudicate. For LLM-tool-use, the analog of a "closed loop" is a
**state hash on the conversation/working-memory representation** that
recurs after a sequence of tool calls; the per-channel vector signal
is the suite of {tokens used, tool latency, validity, correctness,
preference} accumulated over the loop, and the dominance-by-zero test
asks "did this detour cost something in every channel and gain
nothing anywhere?" — a natural waste-detector at the paragraph or
tool-call level. The primitive's cost is O(T) per rollout regardless
of horizon length, and the dominance-by-zero test is a fixed
reference that does not require any other trajectory or any reward
correlation to bootstrap. Vector feedback is preserved as a partial
order throughout — the operator never collapses to a scalar.
