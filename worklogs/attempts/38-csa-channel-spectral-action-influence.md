---
id: 38
slug: csa-channel-spectral-action-influence
status: failed
sprint: 2026-06-06
verdict_in_one_line: "graph-spectral Fiedler-ascent primitive collapsed via hash-bucket sparsity and terminal-only-channel Laplacian degeneracy — beat_random=0 on all 4 core envs"
side_information: [transition geometry, vector diagnostics]
nearest_prior: "15 (FED) / hash-collision-gated-primitive family"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: f608781a034313627e89f29497c1de38ab27082a
---

# 38 — CSA (Channel-Spectral Action-Influence)

## One-sentence idea

Maintain an online observation-hash-bucket transition multigraph with per-channel edge weights (from `info["vector"]`), compute the Fiedler vector of each channel's weighted Laplacian, and nudge policy logits toward actions whose per-channel Fiedler-ascent vector `Phi[a,:]` is Pareto-non-dominated in the coordinate-wise partial order.

## Core primitive

For each channel m, form the symmetrized channel-reweighted Laplacian `L_m = D_m - (W_m + W_m^T)/2` over an online-grown vocabulary of observation-hash buckets, backed by an epsilon-count backbone `W_0` for connectivity. Extract the Fiedler vector `f_m` — the eigenvector of the second-smallest eigenvalue of `L_m` — by sparse Lanczos every K=4 episodes. The per-(action, channel) primitive is `Phi[a, m] = (1/N_a) sum_{t: a_t=a} (f_m(h(o_{t+1})) - f_m(h(o_t)))`, the average Fiedler-coordinate ascent induced by action a on the channel-m-weighted graph. This is a global spectral statistic of the full empirical transition graph, not a per-state value function, and satisfies no Bellman recursion.

## Improvement operator

At each decision step, compute the Pareto-non-dominance count `n_a` (number of actions a' such that `Phi[a,:] strictly dominates Phi[a',:]` coordinate-wise) and the reverse count `m_a`, then apply additive logit nudge `delta_logit(a) = alpha * (n_a - m_a)`. The improvement operator never collapses to `argmax_a w^T Phi[a,:]` for any fixed or learned w when k >= 2; the partial order does not totally order actions.

## Why it looked promising

- The Fiedler vector is a genuinely global spectral object — it encodes the dominant bottleneck cut of the experience graph, a long-range structural property inaccessible to local TD bootstrapping.
- The channel-reweighted Laplacian construction provides one independent Laplacian per vector channel, with channels aggregated only at the Pareto vote — cleanly separates channel contributions without scalarization.
- The primitive fires from step 1 (every transition contributes edge weights), unlike FED/CEC which require completed episodes with successful reward.
- The structural distinction from GVFs/successor features is mathematically sound: the Fiedler vector satisfies `L_m f_m = lambda_2 f_m`, not a Bellman recursion, and the improvement operator is partial-order-based rather than `argmax_a Q(s,a)`.
- The hypothesis correctly identified all principal failure modes with instrumented falsifiers before the run.

## What was tested

Stage: core (4 envs: MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0, deep-sea-treasure-concave-v0, resource-gathering-v0). 120s/env budget. 1 retry (fix-1: matmul shape mismatch on small obs — pad/truncate to `rp_matrix.shape[1]` not `.shape[0]`).

Scores: DoorKey=0.0 (random=0.137), KeyCorridor=0.0 (random=0.0), DST=0.0 (random=194.0), RG=0.011 (random=1.331). beat_random=0, beat_strong=0. Commit: f608781a034313627e89f29497c1de38ab27082a.

## Why it failed

Two independent failure modes, both predicted by the hypothesis's own falsifiers:

**(a) Laplacian channel degeneracy** — on DST and RG, the reward/treasure/gem channels fire only at the terminal step. Before any rewarded trajectory, `W_m` for reward channels is near-zero, making `L_m`'s eigenvalue gap dominated by the epsilon-backbone term; `f_m` is near-collinear with the all-ones vector, `Phi[a,m] ≈ 0` for all a. The k_eff of meaningful channels drops to 1 (step-penalty only), collapsing the Pareto vote to a scalar comparison on step-penalty ascent. Same structural collapse as FED (#15), CHX (#17), CRP (#22), ATP (#31), PRAR (#36), PHI (#35).

**(b) Hash-bucket revisitation starvation** — on DoorKey and KeyCorridor (and likely DST/RG too), the mean bucket-revisit count likely never exceeded 2.0 within 120s under uniform exploration. With a tree-like sparse graph, the Fiedler partition is vacuous (Fiedler vector entries collapse to per-connected-component indicators, not continuous partition coordinates). The sparse Lanczos either produced degenerate eigenvectors or the eigenvalue gap check triggered the abort condition.

Both failure modes place CSA in the same operational family as FED/CEC (#18)/TRAC (#26)/CSD (#32): hash-bucket-gated primitives that cannot accumulate sufficient graph density before the budget expires on long-horizon sparse envs.

## Lesson / constraint added

Graph-spectral primitives (Laplacian Fiedler vectors, Cheeger cuts) on online observation-hash graphs fail without (1) a paired exploration primitive that densifies the graph before spectral computation and (2) non-terminal vector channels providing edge weights throughout the episode; this extends the FED-family ruling to the full "build online channel-weighted adjacency graph + extract spectral structure" sub-family.

## Nearest neighbors in the literature

- **Spectral RL / RKHS-based Laplacian representations** (Mahadevan & Maggioni 2007): Laplacian eigenvectors as basis functions for value function approximation — similar graph-Laplacian object but computed offline on a fixed MDP and used as a value-function basis, not as a Pareto-vote improvement operator.
- **GVFs / Successor Features** (Sutton et al., Barreto et al.): per-state expected discounted future cumulants; the nearest disqualifier family. CSA's Fiedler vector is not a per-state object and satisfies no Bellman recursion.
- **Proto-value Functions** (Mahadevan 2005): same Laplacian eigenvector idea but on offline-collected transition graphs, used as PVF basis for value approximation — CSA's online bucketed variant with per-channel reweighting and Pareto-non-dominance improvement is structurally distinct but shares the same graph-sparsity bottleneck on online data.
- **Primal Behavior Flow Pivot (#14)**: prior attempt using occupancy flow `mu_pi(s,a)` — different central object but same side-information channel (transition geometry) and same verdict (no new side-information advantage for sparse long-horizon discovery).

## Artifacts

- `worklogs/runs/20260606-30-auto/train.py`
- `worklogs/runs/20260606-30-auto/result.json`
- `worklogs/runs/20260606-30-auto/panel.txt`
- `worklogs/runs/20260606-30-auto/fix-1.md`
- `worklogs/runs/20260606-30-auto/hypothesis.md`
- `worklogs/runs/20260606-30-auto/review.md`
- Commit: f608781a034313627e89f29497c1de38ab27082a
