---
id: 15
slug: fed-frontier-expanding-dispersion
status: failed
sprint: 2026-06-05
verdict_in_one_line: "Pareto-front extension over vector-outcome multisets indexed by obs-hash never bootstraps on sparse envs — same wall as SIT"
side_information: [vector diagnostics, transition geometry]
nearest_prior: sit-suffix-inheritance-trie
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: 1fd43f1efd68e6807ff0a083f57c40d6146071b8
---

# 15 — FED (Frontier-Expanding Dispersion)

## One-sentence idea

Maintain, per observation-hash bucket, the empirical Pareto frontier of per-channel
vector-outcome signatures from past visits; nudge policy logits toward actions whose
conditional outcome multiset extends that frontier in the set-extension partial order
without contracting it on any channel.

## Core primitive

`F(u, a)` = the multiset of vector-outcome signatures (channel-wise suffix sums of
`info["vector"]`) for all observed transitions `(u, a)`, indexed by observation-hash
bucket `u`. The Pareto frontier `P(u) = {v ∈ F(u) : no v' ∈ F(u) dominates v
component-wise}` is the compact summary. This is not a scalar Q-value, not a return,
not a GVF cumulant — it is a set of incomparable vectors in `ℝ^k` on which the
standard Pareto partial order is defined.

## Improvement operator

`π(a|u) ← π(a|u) · exp(η · I[F(u,a) adds at least one Pareto-non-dominated point to
P(u) without removing any])`, renormalized. `η` is a small fixed step. The indicator
is binary; there is no scalar advantage weight. The operator is defined solely by
set-extension under the Pareto partial order, which is not expressible as `wᵀr` for
any fixed or learned `w`.

## Why it looked promising

- Structurally novel: improvement criterion is a partial-order relation on sets of
  vectors, provably not reducible to any linear scalarization.
- Side-information channel (vector diagnostics) is present from step 1 on the panel's
  vector envs — does not need to wait for reward correlation.
- Memory cost is `O(B · k)` for `B` buckets and `k` channels; sublinear in trajectory
  length.
- Monotonic non-contraction of attainment set `A_t` is provable under stationary
  distributions and sufficient sample mass.
- Reviewer verdict: `novel-direction` with no structural rebadge found.

## What was tested

Stage: core (4 envs). Budget: 120 s/env. Commit: `1fd43f1`.
- MiniGrid-DoorKey-8x8-v0: 0.000 (random 0.137, strong 0.137)
- MiniGrid-KeyCorridorS3R3-v0: 0.000 (random 0.000, strong 0.000)
- deep-sea-treasure-concave-v0: 0.000 (random 194.0, strong 285.0)
- resource-gathering-v0: 0.011 (random 1.331, strong 1.331)
beat_random=0, beat_strong=0. Wallclock: 115.5 s. No retries.

## Why it failed

Bootstrap problem identical to SIT (alive-weak, run 20260605-02-auto): the
observation-hash bucket scheme requires >1% collision rate within a rollout for the
empirical Pareto front to accumulate meaningful sample mass. Under the initial uniform
exploration policy, hash-bucket collisions on long-horizon sparse envs (DoorKey,
KeyCorridor, Deep Sea Treasure) are essentially zero — each episode visits novel
states. The Pareto-front extension indicator never fires, so the logit nudge is never
applied, and the policy remains a uniform random walk for the entire 120 s budget.

Even on Deep Sea Treasure (a vector env predicted to be FED's strength), the score
was 0.000 vs random 194.0 — the method explored less efficiently than random because
the softmax over zero-update logits has higher entropy than the random-walk baseline's
implicit policy. This matches the canonical cross-attempt failure mode: "primitive
needs reward correlation to bootstrap, but reward correlation does not exist on long-
horizon sparse tasks until a deep unrewarded path is traversed."

## Lesson / constraint added

Any method that accumulates future-compression objects (Pareto fronts, outcome
multisets, suffix tries) indexed by observation-hash buckets must pair with an
explicit exploration primitive that drives hash-bucket collisions before the
compression object can be used; without it, the improvement operator is vacuous on
long-horizon sparse envs throughout the entire training budget.

## Nearest neighbors in the literature

- **Pareto Q-learning / multi-objective RL:** FED's operator is structurally distinct
  (no scalar scalarization, set-extension criterion), but the bootstrap failure is
  analogous — both require enough observations per state-action pair.
- **SIT (attempt 15-candidate):** Closest structural sibling in this corpus; both
  index a future-compression object by observation-hash and fail to bootstrap on
  sparse envs.
- **Distributional RL (C51, QR-DQN):** FED replaces the return distribution with a
  vector Pareto front, but the bootstrap requirement is similar.
- **MORL / GPI with successor features:** FED avoids cumulant projection, but shares
  the sample-mass requirement per state.

## Artifacts

- `worklogs/runs/20260605-04-auto/train.py`
- `worklogs/runs/20260605-04-auto/result.json`
- `worklogs/runs/20260605-04-auto/panel.txt`
- `worklogs/runs/20260605-04-auto/hypothesis.md`
- `worklogs/runs/20260605-04-auto/review.md`
