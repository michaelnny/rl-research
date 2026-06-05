---
id: 18
slug: cec-continuation-endpoint-concordance
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Exit-hash bucketing still cannot bootstrap in 120 s; concordance signal was zero, same as FED's obs-hash variant."
side_information: [reachability/reset structure, vector diagnostics]
nearest_prior: "15"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: c58bc8131e1f1196b613e670404e72800133e572
---

# 18 — CEC (Continuation-Endpoint Concordance)

## One-sentence idea

Index a per-(state-hash, action) multiset of vector cumulants by the terminal *exit-observation hash* of the episode, then nudge logits by the signed count of exit-hash buckets where action a's bucket-conditional mean cumulant Pareto-dominates action a'.

## Core primitive

For each step (s_t, a_t) of a completed episode with terminal observation o_T, record `(exit_hash := H(o_T), Δc := Σ_{u=t}^{T-1} v_u)` into the bag `B(s_t, a_t)`. The central object is a labeled multiset partitioned by exit-hash bucket `b`, carrying the bucket-conditional mean cumulant vector `μ(s, a, b) ∈ R^k`. The concordance between a and a' at s is `C(s; a, a') = Σ_b 1[μ(s,a,b) ≻ μ(s,a',b)] − 1[μ(s,a',b) ≻ μ(s,a,b)]`, where `≻` is coordinate-wise strict Pareto order.

## Improvement operator

After each completed episode, for every visited state s update: `logit(s, a) ← logit(s, a) + α · Σ_{a' ≠ a} C(s; a, a') / |A|`. The magnitude of the concordance is a count of exit-hash buckets with clear Pareto dominance — never a function of cumulant magnitudes themselves. No critic, no Bellman backup, no scalar weighting.

## Why it looked promising

- Exit-hash is a reset-structural coordinate that produces exactly one sample per episode per visited state, so bucket occupancy grows as O(episodes) vs O(episodes/|reachable_states|) for FED's visited-state-hash scheme.
- On Deep Sea Treasure the treasure positions are a small finite set of exit observations, guaranteeing a bounded exit-hash space — the hypothesis predicted this would be the env most likely to succeed.
- The concordance operator is a pure partial-order vote count, structurally distinct from all disqualifier families (no scalar weighting, no Bellman backup, no GVF expectation, no HER relabeling).
- The hypothesis provided an explicit falsifier (median bucket size < 2 → abort) making it directly testable.

## What was tested

Stage: vector (Deep Sea Treasure, Resource Gathering). Time budget 120 s per env. 0 retries. Commit c58bc8131e1f1196b613e670404e72800133e572.
- deep-sea-treasure-concave-v0: score=0.0, random=194.0, strong=285.0. beat_random=0, beat_strong=0.
- resource-gathering-v0: score=0.011, random=1.331, strong=1.331. beat_random=0, beat_strong=0.

## Why it failed

The hypothesis's own falsifier triggered: after the seeding phase, the median (state, action) pair did not accumulate ≥ 2 distinct exit-hash buckets each with ≥ 2 samples. Therefore `C(s; a, a') = 0` for essentially all state-action pairs — the concordance signal was structurally zero throughout the run. On Deep Sea Treasure the exploration budget is too thin for any softmax-sampled policy (without an explicit exploration primitive) to repeatedly visit the same state-action pair under multiple distinct exits. This is the same bootstrap wall as FED (#15); switching from mid-trajectory obs-hash to terminal exit-hash did not provide enough coverage density within 120 s.

Applicable cross-attempt failure mode: "The primitive needs reward correlation to bootstrap, but reward correlation does not exist until a deep unrewarded path is traversed." Exit-hash is informative only after the agent has already reached a terminal — before that, all exit-hashes come from timeout terminations, which are a single bucket (same exit), so the concordance is uncomputable even in principle.

## Lesson / constraint added

The "empirical Pareto-front / cumulant-multiset indexed by any observation hash" family is ruled out in both mid-trajectory (FED) and terminal-exit (CEC) variants. Future candidates using a hash-indexed multiset as their core primitive must pair it with an explicit exploration primitive that achieves repeated state-action-exit coverage before concordance comparisons are computed.

## Nearest neighbors in the literature

- **FED (#15):** Same family (hash-indexed cumulant multiset + Pareto-front extension). CEC uses exit-hash; FED uses visited-state-hash. Same failure mode.
- **Multi-objective RL / Pareto-front methods (e.g., PCGrad, MORL surveys):** CEC's concordance count is a partial-order vote, not a gradient-aggregation or scalarization. Structurally different, but same bucket-accumulation bottleneck on sparse-reward envs.
- **Hindsight Experience Replay (HER):** CEC uses actual terminal observations as bucket keys, not relabeled virtual goals. The exit-hash does not generate counterfactual reward signals.
- **GVFs:** CEC never computes an unconditional expectation of cumulants; the multiset is kept as-is until a Pareto comparison fires. But the sample-accumulation requirement is similar to GVF variance reduction requirements.

## Artifacts

- `worklogs/runs/20260606-01-auto/hypothesis.md`
- `worklogs/runs/20260606-01-auto/review.md`
- `worklogs/runs/20260606-01-auto/result.json`
- `worklogs/runs/20260606-01-auto/panel.txt`
- `worklogs/runs/20260606-01-auto/curator.md`
- Commit: c58bc8131e1f1196b613e670404e72800133e572
