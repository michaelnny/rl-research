---
verdict: reject
reviewer_run: 20260606-23-auto
hypothesis_type: probe
---

## Summary

TRACE proposes a coherent novel REINFORCE reweighting (squared
consecutive-state log-likelihood gap) but applies it on the `quick`
(DST-concave, vector) stage by training on a single channel of
`info["vector"]`, which is the explicitly listed scalarized-vector-reward
disqualifier and is the exact failure mode that killed run 23 (TCP).

## Schema check

`candidate.json` is internally consistent with the prose:

- `principle`, `primitive_name`, `primitive_type`, `update_family =
  direct_policy_update`, `memory = none` all match the hypothesis text.
- `claimed_stage = quick`, `nearest_disqualifier = policy_gradient`, and
  the empirical-claim/falsifier/ablation triple match the hypothesis
  sections faithfully.
- `uses_vector_reward = false` is internally consistent with the prose
  ("we use SCALAR return"). The validator at
  `scripts/validate_candidate.py` only flags the case
  `uses_vector_reward = true` on a non-vector stage, so it will pass —
  but the structural problem is exactly the inverse: this is a vector
  stage with `uses_vector_reward = false`, which on this substrate is
  the scalarization disqualifier (see below).

So the JSON itself validates; the schema-vs-prose match is fine. The
reject reason is substantive, not schema-mechanical.

## Coherence check

The derivation is coherent at the level required for a probe:

- Step 1 (definition of `δ_t`) is well-typed for any softmax policy.
- Step 2 (zero locus) follows directly.
- Step 3 (definition of `J̃`) is a well-defined functional.
- Step 4 (gradient identity under stop-gradient on `δ_t²`) is honestly
  marked as proof debt; the alternative full-gradient form is also named.
- Step 5 (random-init non-zero claim) is plausible — for any softmax
  policy with non-zero logit variance, `δ_t` is non-zero a.s. on
  visited transitions.
- Steps 6–12 (distinguishing TRACE from baselines/curiosity/Fisher
  PG/COPDEV/PARGRAD/Family A/Family C) hold structurally.
- Proof debt is named, not hidden.

The mechanism is one typed primitive, the update rule is implementable
as a single forward-pass-at-`s_{t+1}` plus a stop-gradient weight, and
the cosine-alignment scalar is a sensible load-bearing observable that
fires from rollout 1. None of this is the rejection trigger.

## Novelty check

Searches considered: "policy gradient with policy-derived per-step
weight", "log-likelihood asymmetry policy gradient", "consecutive-state
score-function reweighting". The closest published methods are:

- REINFORCE / vanilla PG (Williams 1992) — TRACE adds a multiplicative
  positive weight `δ_t²` that is non-trivial on softmax policies.
- AWR (Peng 2019) — exponential, advantage-based; TRACE is quadratic,
  policy-only.
- Natural / Fisher-preconditioned PG — direction modification; TRACE
  preserves direction.
- Self-imitation / SVPG — different mechanism slot.

Against the dead-family list:

- Family A (bucket+vote): not present, no buckets.
- Family B (pairwise traj): not present, single rollout.
- Family C (within-trajectory geometric): the weight is on the policy
  manifold, not on the cumulant trace.
- Family D (reward-free primitive + reward gate): the application is
  multiplicative against `G_t`, not gated; the hypothesis explicitly
  argues this point.
- Family E/F/G/H: not applicable.

If the only issue were novelty, this would pass. The mechanism is a
novel REINFORCE reweighting, not a rebadge.

## Implementability and ablation check

Implementability: the Engineer can implement TRACE in `train.py` as
a single REINFORCE loop with one extra forward pass at `s_{t+1}` for
the same realized action `a_t`, and one stop-gradient on `δ_t.detach()
** 2`. The ablation `c_t ≡ 1` is exactly REINFORCE-without-baseline and
is mechanical. The cosine-alignment scalar (TRACE gradient vs.
uniform-weight gradient) is a clean load-bearing discriminator. None of
the implementability checks fail.

The ablation is genuinely load-bearing: it disables the per-step weight
without changing rollout, returns-to-go, or the score-function form.
The secondary `c_t ~ Exp(1)` ablation correctly isolates
non-uniformity-vs-policy-correlation.

**Vector-reward issue (decisive).** The `quick` stage in `harness.py`
is `["deep-sea-treasure-concave-v0"]`, which has
`ENV_TYPE = "vector"` and emits `info["vector"]` with two channels
(treasure terminal + step penalty). The substrate rule in `CLAUDE.md`
states: "For vector envs, training must consume `info["vector"]`.
Training only on scalar reward in vector envs is scalarization and is
disallowed." `prior_attempts.md` lists "Scalarized vector reward `wᵀr`
for any fixed or learned `w`" as a disqualifier.

TRACE explicitly proposes (hypothesis lines 168–172) to use
`info["vector"][0]` only — i.e., `wᵀr` with `w = [1, 0]`. Run 23
(TCP) is a directly relevant precedent: it failed because
"a Pareto comparison restricted to the single coordinate
R = {treasure} is identical to scalar maximization of
E[v[treasure] | c, a] — the 'scalarized vector-reward maximization'
disqualifier." TRACE makes the same restriction without the Pareto
machinery. This is the disqualifier in pure form.

The hypothesis attempts to defend this with "the mechanism does NOT
scalarize multi-channel reward; it operates on whichever SINGLE
channel the panel exposes," but that defense conflates "the
mechanism's primitive `δ_t²` is reward-free" (true) with "the training
signal `G_t` is not a scalarization" (false: `G_t = Σ γ^k r_k` where
`r_k = info["vector"][0]` is a single fixed channel projection, which
is exactly `wᵀr`). The reward-free primitive does not rescue a
scalarized return signal.

Note that this is fixable in two distinct ways without changing the
TRACE mechanism: (a) move to a scalar-stage env (e.g.,
MiniGrid-Empty-5x5-v0 if available, or pick a stage that the harness
treats as `scalar`) and re-state the empirical claim there; or
(b) consume `info["vector"]` as a vector return-to-go and apply
TRACE's per-step weight against a vector-typed objective (e.g., a
Pareto-comparison-based update or a per-channel score-function update
with `δ_t²` weight). Either change is more than mechanical and
materially alters either the substrate or the update rule.

## Decision

Reject. Triggered criteria:

1. Vector-reward scalarization on the proposed `quick` stage:
   `info["vector"][0]` is `wᵀr` with `w = [1, 0]`, the explicit
   disqualifier in `prior_attempts.md`, and the same shape as TCP
   (run 23) on the same env.
2. Substrate-rule violation: `CLAUDE.md` requires training on
   `info["vector"]` (not on a single channel) for vector envs.

Not triggered: rebadge of REINFORCE (the per-step `δ_t²` weight is
genuinely novel and reward-independent in form), dead-family shape,
incoherent derivation, untyped primitive, weak ablation, or missing
empirical claim.

The next Researcher turn should either pick a non-vector stage to
defend the cosine-alignment claim on, or rewrite TRACE to act on the
full vector return so that `G_t` is a vector-typed quantity consumed
without `wᵀr` collapse. The TRACE mechanism itself (squared
consecutive-state log-likelihood gap as a per-step reweighting of
score-function ascent) is worth retaining; it is the substrate choice
plus the channel-selection rule that triggers rejection.
