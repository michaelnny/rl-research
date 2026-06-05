---
id: 28
slug: arp-action-reachable-pattern-lattice
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Binary suffix-pattern set with strict-superset operator collapses to bootstrap-silent family — operator never fires until first rewarded trajectory, same as FED/CEC/TPP/PCR/TRAC."
side_information: [vector diagnostics, transition geometry]
nearest_prior: "15 (FED); also 18 CEC, 21 TPP, 24 PCR, 26 TRAC"
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random: null
  hard_n_beat_strong: null
  commit: 451d985816442fab8b0a53331b727c0ae4d07ae0
---

# 28 — ARP: Action-Reachable Pattern Lattice

## One-sentence idea

Maintain per-(state-cluster, action) the empirical set `S(s,a) ⊆ {0,1}^k` of distinct binary channel-firing patterns observed in completed suffixes, and nudge policy logits toward actions whose pattern sets strictly contain (coordinate-wise superset) every pattern of some competitor's set.

## Core primitive

For each (state-cluster `s`, action `a`) cell, keep a hash-set of binary vectors `b ∈ {0,1}^k` where `b[m] = 1` iff channel `m` fired at least once in the suffix following one observed `(s, a)` transition. Each completed episode contributes one pattern per visited cell (suffix-OR computed at episode end). With `k ≤ 4` on the panel, each cell holds at most `2^k = 16` distinct patterns; total storage is `O(N_clusters · |A| · 2^k)`.

## Improvement operator

At decision step in cluster `s`, for each action pair `(a, a')` compute `dom(a, a') = ⋁_{b ∈ S(s,a)} ⋀_{b' ∈ S(s,a')} (b ≥ b' ∧ b ≠ b')` — `S(s,a)` contains some element that strictly supersets every element of `S(s,a')`. Define `n_a^{dom} = |{a' : dom(a, a')}|` and `n_a^{sub} = |{a' : dom(a', a)}|`. Apply additive logit nudge `α(n_a^{dom} − n_a^{sub})`. No critic, no return weighting, no Bellman.

## Why it looked promising

- Combinatorial set representation is structurally distinct from GVFs/successor features (set membership vs. weighted magnitude expectation).
- Strict-superset operator is non-linear and admits no `wᵀ·` scalarization.
- Binary existence requires only one observation per channel per suffix — lower data demand than magnitude-based primitives.
- Reviewer passed it as novel-direction; the rebadge test against GVFs/VCC/FED family articulated clearly.
- Hypothesis provided explicit falsifier conditions and mapped predicted failure modes.

## What was tested

Stage: `vector` (Deep Sea Treasure concave v0, Resource Gathering v0). Time budget: 120 s per env. No retries.

- DST: 99.0 vs random=194.0, strong=285.0 (beat_random=0, beat_strong=0)
- RG: 0.011 vs random=1.331, strong=1.331 (beat_random=0, beat_strong=0)

Commit: 451d985816442fab8b0a53331b727c0ae4d07ae0

## Why it failed

The falsifier the hypothesis itself predicted was confirmed: "Lattice degeneracy" — before the first rewarded trajectory, every populated `(s,a)` cell holds only the step-penalty singleton pattern `{1,0,...,0}` and the strict-superset operator is universally silent (no action's singleton can strictly superset another singleton of the same value). The operator is identical in behavior to the FED/CEC/TPP/PCR/TRAC family during the pre-reward phase: zero nudge signal, exploration is purely uniform, and within the 120 s budget on sparse long-horizon tasks the first successful trajectory is never collected. The binary representation does not buy earlier firing than magnitude-based approaches — it merely names the waiting problem differently.

Cross-attempt failure mode: "The primitive needs reward correlation to bootstrap, but reward correlation does not exist on long-horizon sparse tasks until a deep unrewarded path is traversed."

## Lesson / constraint added

Extends the FED/CEC/TPP/PCR/TRAC bootstrap-wall ruling to the "empirical set of binary suffix patterns" family: any operator that gates logit nudges on downstream channel-firing sets (whether stored as magnitude vectors, Pareto fronts, exit-hash buckets, or binary-pattern sets) inherits the full bootstrap silence until the first rewarded suffix is observed.

## Nearest neighbors in the literature

- **GVFs / Successor Features** (Dayan 1993, Barreto 2017): ARP differs in that it stores a set of binary vectors rather than a magnitude expectation, and uses set-theoretic superset rather than linear scalarization. The rebadge test fails, but the bootstrap timing is identical.
- **Pareto-MORL / SFOLS**: multi-objective policy improvement via Pareto dominance on return vectors; closest in spirit to the operator, but ARP uses binary suffix patterns not return magnitudes.
- **FED (#15), CEC (#18), TPP (#21), PCR (#24), TRAC (#26)**: all share the bootstrap wall; ARP is the binary-set variant of this family.

## Artifacts

- `worklogs/runs/20260606-17-auto/train.py`
- `worklogs/runs/20260606-17-auto/result.json`
- `worklogs/runs/20260606-17-auto/panel.txt`
- `worklogs/runs/20260606-17-auto/hypothesis.md`
- `worklogs/runs/20260606-17-auto/review.md`
