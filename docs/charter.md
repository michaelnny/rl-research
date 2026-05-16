# rl-research charter

This is the project's north-star document. Every other doc and every role prompt
references this. Changes here ripple through the loop — make them deliberately.

## Mission

**Discover a third family of reinforcement-learning algorithms** — distinct from the
value-based and policy-based families — through autonomous research by coding agents
on classic (non-LLM) RL problems.

The two existing families both ultimately propagate a *scalar* signal (TD error or
advantage-weighted log-prob) along single trajectories. A third family must differ
at the **optimization-principle** level, not the loss function. PPO/GRPO/TRPO are
patches on REINFORCE; Rainbow is a patch on Q-learning. We are not in the patch
business.

## The three pillars (pain points, not metrics)

A third-family algorithm should be evaluated on its ability to handle modern RL
pain points that scalar-bottlenecked methods struggle with:

1. **Sparse rewards over long horizons** — credit assignment when meaningful signal
   is rare.
2. **Long-horizon dense control** — credit propagation over thousands of steps in
   high-DoF action spaces.
3. **Multi-signal feedback** — natively consuming a *vector* of reward channels
   (not summed into a scalar).

The benchmark suite ([benchmarks.md](benchmarks.md)) is one task per pillar.

These are pain points, **not optimization targets**. A candidate's hypothesis names
*which* pillar(s) it claims to address; the corpus accumulates evidence; the Curator
weighs it. There is no "beat PPO at pillar X by Y%" objective.

## Hard rules (non-negotiable)

1. **No imported RL algorithm libraries.** SB3 / CleanRL / Tianshou / RLlib / Acme /
   Coax / garage are forbidden as dependencies. The single allowed exception is the
   own-authored PPO baseline at `src/rl_research/baselines/ppo.py`.
2. **Symbolic search only.** Each iteration produces a self-contained `train.py`.
   No fill-in-the-blank parametric meta-learners.
3. **2-hour wallclock cap per run** in early-phase exploration. Hard kill at the OS
   level on budget exceeded.
4. **Promotion is curatorial, never numerical.** No automated "beat PPO at X%" gate.
   The Curator weighs evidence across performance, novelty, generality, and
   implementation hardness.
5. **Pin every dependency to an exact version** in `pyproject.toml`. Reproducibility
   is non-negotiable.
6. **Single-GPU assumption.** One RTX 3090 Ti, 24 GB. No distributed training.

## Disqualifiers

A candidate is "still REINFORCE / Q-learning in disguise" if its update step contains
*any* of:

- `∇ log π(a|s) · A(s,a)` as the learning signal — the REINFORCE / actor-critic /
  PPO / GRPO / TRPO / A2C / IMPALA family.
- `r + γ Q(s', a') - Q(s, a)` as the primary update target — the Q-learning / DQN /
  Rainbow / SAC / TD3 family.
- The Bellman fixed-point as the optimization target — any DP-family method.
- Cross-entropy of policy vs an *expert* policy — that's imitation, not the third
  family we are looking for.

The Reviewer's pre-code text check enforces this. Catching a rebadge in 30 seconds
of review beats discovering it 90 minutes into a benchmark run.

## What "good evidence" looks like

A run produces *valuable evidence* if any of the following hold:

- The implementation runs end-to-end on the sanity gate and produces interpretable
  training curves on a primary benchmark.
- It fails to learn but the failure mode is informative — e.g., "vector-credit
  method diverges on dense reward; only stable when channels are decorrelated."
- It demonstrates a structurally different mechanism, even if it is weaker than
  PPO so far.

A run is *not* valuable just because it beats PPO on one benchmark. The corpus
rewards diversity and clarity, not leaderboard position.

## Anti-patterns

Things the loop must not produce — Reviewer should reject these on sight:

- "PPO + X" where X is a hyperparameter trick or a bonus reward shaping term.
- "DQN with a different replay sampling scheme."
- A hyperparameter sweep dressed as an algorithm.
- Anything that loads a pretrained model and continues training (BC + RL hybrid).
- Anything that uses the disqualified update equations under a renamed variable.

## Two phases

- **Exploration** (current): 2h cap per run, broad latitude on hypotheses, single
  primary benchmark per candidate. Goal: cover ground, build the corpus.
- **Mass run** (Curator-promoted only): extended wallclock, more seeds, all three
  primary benchmarks. Goal: validate a curator-promoted candidate at scale.

The transition is curatorial. There is no automatic gate.
