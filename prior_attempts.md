# prior_attempts.md

Eleven directions have been tried and failed across two prior research sprints.
This list is a **hint**, not a gate. Read it before proposing a new direction
so you don't re-derive a known dead end. If your candidate is structurally
identical to one of these, it will not produce a new result.

Each entry: name, one-line mechanism, one-line reason it failed. These
summaries are intended to be self-sufficient — you should not need to open
the worklogs to act on them.

The verbatim sprint reports live under `worklogs/` for the rare case where
you need the full derivation, the math, or the discarded sub-attempts:
`worklogs/research_attempts_20260524.md` covers entries 1–4 below;
`worklogs/research_attemps_20260526.md` covers entries 5–11 plus the
process post-mortem reflected in **Cross-attempt failure modes** later in
this file.

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
   list (1–11 above, or one of the disqualifier families) is articulated in
   2–3 sentences in the commit message.
