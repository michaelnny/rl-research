---
verdict: reject
reviewer_run: 20260606-26-auto
hypothesis_type: probe
---

## Summary

UNIRANK is a fixed `w=[1,0]` scalarization of the vector reward run on a
vector env, justified by a closure-semantic argument whose conclusion
the author has already derived algebraically; the substrate rule's
scalarization disqualifier is decisive and the closure value does not
require compute to obtain.

## Schema check

The candidate.json fields align with the prose: `principle`,
`primitive_name`/`primitive_type`, `claimed_stage`, `empirical_claim`,
`falsifier`, `ablation_plan`, `nearest_disqualifier`, and
`novelty_boundary` all match the corresponding sections of
`hypothesis.md`. `nearest_disqualifier=scalarization` is correctly named
and the prose acknowledges it explicitly. `uses_vector_reward=false` is
truthful: channel 2 is read for shadow-logging only and never enters the
update. The schema is internally consistent; the issue is not schema
mismatch.

## Coherence check

The derivation is internally coherent. Step 5 (PARGRAD's bivariate
weak-dominance count reduces to channel-1 strict-below CDF on the alive
cohort because `M^2_t` is constant given survival on DST-concave) is a
clean algebraic argument. Step 6 (COPDEV's `|F̂^1_t - F̂^2_t|` reduces
to `|F̂^1_t - 1|` because `F̂^2_t` is a unit step at the constant
alive-cohort `M^2_t`) is also correct. The update rule (step 7) follows
mechanically. The implementation pseudocode is unambiguous: ring buffer
per `t`, weak-below CDF, score-function ascent.

The coherence problem is at the meta level: the closure conclusion the
author wants to extract from the empirical run is **already in the
derivation sketch**. Steps 5 and 6 *prove* the substrate-degenerate
reduction analytically. The empirical probe is then asked to confirm
that "the analytical reduction is empirically faithful," which is
either trivially true (the buffers will indeed produce the reduced
weight up to `O(1/N)` tie corrections) or — under Outcome B — would
indicate the buffers are sample-starved rather than that the bivariate
information was load-bearing. The closure-value claim is not robust to
a panel run on a noisy substrate.

## Novelty check

Searches: per-step rank-weighted REINFORCE; quantile-weighted policy
gradient; survival-conditioned CDF policy gradient. The closest published
neighbors are correctly identified by the author: Tamar 2015 / Chow 2017
for quantile-weighted REINFORCE (per-trajectory cohort, scalar return),
and PER (Schaul 2015) for rank-weighted sampling. The per-`t`
alive-cohort granularity is a marginal variant in the same neighborhood.

The decisive novelty problem is the substrate disqualifier rather than
prior literature. Per `prior_attempts.md` disqualifier list:
*"Scalarized vector reward `wᵀr` for any fixed or learned `w`"* is
dead, and the project's substrate rule states *"Training only on scalar
reward in vector envs is scalarization and is disallowed."* UNIRANK
reads only `info["vector"][0]` for its update; this is `w=[1,0]`, the
canonical fixed scalarization. The author's "rank-invariance of the
weight" defense addresses a different invariance (under monotone
transformations of channel 1) and does not rescue the channel
selection, which is a fixed linear projection.

The probe is also adjacent to Family I (per-channel gradient
aggregation, IMTL-G) and Family A region (rank statistic over a
per-`t` index). Per Family I's "even strong empirical signal on this
shape would only show ... not a new family," even Outcome C would not
constitute a novelty result.

## Implementability and ablation check

Implementability: the update rule is straightforward. `T_max` ring
buffers, weak-below CDF query, score-function update. An Engineer can
write `train.py` and `train_ablate.py` against the existing contract
without invention.

Ablation: the primary ablation (replace `q_t` with constant 1) is the
same shape used by COPDEV and PARGRAD. It is structurally clean. The
risk — observed in COPDEV run 21 and PARGRAD run 22 — is that the
ablation collapses to T=1 episodes with vanishing gradients, making
the candidate-vs-ablation delta an artifact of ablation instability
rather than a measurement of the rank weight's contribution. The
author cites this risk but does not propose a fix.

Vector-reward issue: the central problem. On DST-concave (vector env,
m=2), training consumes only channel 1. Even though `info["vector"]`
is technically read, the channel-2 component is discarded from the
update. This is the disqualified shape. The author's framing of the
probe as "closure-semantic, not novelty" is a request to relax the
substrate rule for one run; relaxing it produces a run whose only
positive empirical outcome (Outcome C, score > 200) would be a
disqualified-shape win that cannot be promoted, and whose confirmatory
outcome (Outcome A) duplicates an analytically derivable claim.

## Decision

Reject. Triggered criteria:

1. **Vector scalarization on a vector env.** `info["vector"]` is read
   but only channel 1 enters the update; this is `w=[1,0]` fixed
   scalarization on a vector substrate, which the substrate rule
   disallows and `prior_attempts.md` lists as a dead disqualifier.

2. **The central novelty is a renamed channel selection plus REINFORCE
   variant.** Stripped of the closure framing, UNIRANK is "REINFORCE
   on channel 1 of a vector env, weighted by per-`t` cohort rank" —
   a scalar-weighted log-prob update that the disqualifier list rules
   out, plus a rank-statistic reweight in the same region as Family I.

3. **The closure value is already derivable analytically.** Steps 5
   and 6 of the author's own derivation produce the
   substrate-degenerate reduction in closed form. The empirical probe
   does not deliver corpus knowledge that the algebra has not already
   delivered, except in the negative case where buffer sample-starve
   would render the result uninterpretable.

4. **Run 22 curator note pointed away from this direction.** The
   recommendation was to test bivariate-dominance on a
   non-degenerate two-channel substrate or change family entirely.
   UNIRANK does neither: it tests a *strictly weaker* primitive on the
   *same* degenerate substrate.

A productive next probe should either (a) propose a non-scalarizing
mechanism that genuinely consumes channel 2 (e.g., on a vector env
where channel 2 is *not* deterministic given survival, perhaps by
modifying the env choice within the vector-stage panel), or (b) leave
the bivariate-rank region with the analytical closure recorded as a
corpus statement and move to a structurally different family. UNIRANK
is not the right vehicle for either.
