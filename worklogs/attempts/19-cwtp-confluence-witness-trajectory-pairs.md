---
id: 19
slug: cwtp-confluence-witness-trajectory-pairs
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Sign-vote tensor from trajectory-pair confluence witnesses; bootstrap wall compounded — requires cross-trajectory state revisits AND non-terminal vector signal, both sparse on the substrate."
side_information: [transition geometry, vector diagnostics]
nearest_prior: 18-cec-continuation-endpoint-concordance
panel_evidence:
  smoke_n_beat_random: 0
  smoke_n_beat_strong: 0
  hard_n_beat_random: null
  hard_n_beat_strong: null
  commit: 263816d9fe7ec9935ea7ea4b06f2aa4bf2961278
---

# 19 — CWTP (Confluence-Witness Trajectory Pairs)

## One-sentence idea

Maintain a buffer of completed trajectories; for each pair that diverged at a state with distinct actions and later reconverged at a shared observation-hash, record the per-channel segment-cumulant difference as a sign vote; nudge policy logits by the Pareto-non-dominance count of normalized sign-vote rows, with no scalar collapse.

## Core primitive

Sign-vote tensor `V[s_div, a, a', m] ∈ Z` accumulated from confluence-witness events: for each trajectory pair `(τ_i, τ_j)` sharing an observation-hash at some `(t_i, t_j)`, find the latest divergence `(d_i, d_j)` where the two trajectories also share a hash but took distinct actions; compute `Δv = Σ_{u=d_i+1..t_i} v_u^{τ_i} − Σ_{u=d_j+1..t_j} v_u^{τ_j} ∈ R^k` and accumulate `sign(Δv[m])` into `V[s_div, a_i, a_j, m]`. The bracketed-segment design means downstream of the confluence cancels by construction.

## Improvement operator

At training time, normalize each sign-vote row to `v(a,a') = V[s,a,a',:] / |V[s,a,a',:]|_1`. Action `a` dominates `a'` at `s` iff `v(a,a')` is coordinate-wise ≥ 0 with at least one strict inequality. Apply `Δlogit(s,a) = +α · #{a' : a dominates a'} − α · #{a' : a' dominates a}`. Temperature-annealed softmax policy on top of (logits + nudge) with maximum-entropy regularizer.

## Why it looked promising

- The "latest divergence before confluence" key means the comparison segment is local to the action choice; downstream behavior is shared by construction and drops out of `Δv` — unlike DPC (first-divergence, terminal cumulants) or CEC (exit-hash buckets).
- No bucket density requirement on (s_div, a, a') triples — any shared observation-hash between any two buffered trajectories produces a witness immediately.
- Per-step vector channels (step penalty on DST, resource counts on RG) were expected to produce non-zero `Δv` even before terminal reward fires.
- Structural distinction from RSD (aggregated multigraph endpoint-pair edges vs individual witness sign votes), DPC (terminal cumulants from first divergence vs bracketed segment), and CEC (exit-hash bucket coverage vs pairwise divergence-state index) was articulated precisely.

## What was tested

Vector stage only: `deep-sea-treasure-concave-v0` and `resource-gathering-v0`, 120 s budget. One syntax fix (array bounds on cum indexing, fix-1.md). Scored 0.0 / 0.011 vs random 194.0 / 1.331. Beat random: 0/2. Beat strong: 0/2. Commit: `263816d9fe7ec9935ea7ea4b06f2aa4bf2961278`.

## Why it failed

Two simultaneous sparsity conditions must both be satisfied for the operator to produce any signal:
1. Cross-trajectory observation-hash collisions at **non-terminal** states — requires the same intermediate state to be visited by at least two buffered trajectories.
2. Non-trivial per-step vector signal in the bracketed segment `[d_i+1..t_i]` — on Deep Sea Treasure, the treasure channel fires only at the terminal step; any confluence at a terminal hash leaves the pre-confluence segment carrying only the step-penalty channel.

On sparse long-horizon envs, condition (1) is rarely met during the budget, and condition (2) further filters out the cases where (1) does hold. The result is that the sign-vote tensor remains near-zero and the logit nudge never fires — the same bootstrap wall as FED (#15) and CEC (#18), compounded. The 0.0 score on DST (exactly at the floor) is consistent with the operator being entirely silent throughout the run.

This extends the sprint-4 family ruling: the "pairwise trajectory comparison indexed by intermediate shared state" sub-family fails on the substrate regardless of whether the comparison key is obs-hash bucket (FED), exit-hash bucket (CEC), or latest-divergence-before-confluence (CWTP).

## Lesson / constraint added

Any candidate requiring two simultaneous sparsity conditions to fire (cross-trajectory state revisits AND non-terminal vector channel activity between key events) inherits the FED bootstrap wall and adds a second one; future candidates must produce signal from the first episode's first step without relying on rare geometric coincidences across the trajectory buffer.

## Nearest neighbors in the literature

- **CEC (#18):** Same indexing-by-observation-hash family; CWTP uses intermediate hashes rather than exit hashes but hits the same coverage wall.
- **Hindsight Experience Replay (HER):** Retroactive relabeling of goals in trajectory buffers shares the same geometric coincidence requirement (two trajectories sharing a state); CWTP differs in using pairwise comparison rather than goal-conditioned relabeling, but the coverage failure mode is analogous.
- **Counterfactual Multi-Agent Policy Gradients (COMA):** Compares per-agent actions against a counterfactual baseline; CWTP's per-action comparison is structurally related but uses empirical trajectory pairs rather than a learned baseline.
- **Return Decomposition (RUDDER):** Redistributes return signal to the decisive action step; CWTP's bracketed-segment `Δv` attempts something similar but via pairwise comparison rather than a learned redistribution network.

## Artifacts

- `worklogs/runs/20260606-03-auto/hypothesis.md`
- `worklogs/runs/20260606-03-auto/review.md`
- `worklogs/runs/20260606-03-auto/result.json`
- `worklogs/runs/20260606-03-auto/fix-1.md`
- `worklogs/runs/20260606-03-auto/curator.md`
- Commit: `263816d9fe7ec9935ea7ea4b06f2aa4bf2961278`
