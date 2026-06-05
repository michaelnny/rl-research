---
verdict: failed-structural
nearest_prior_or_disqualifier: FED (#15) / hash-collision-gated-primitive family
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction:** The Fiedler vector of a channel-reweighted observation-transition Laplacian is mathematically distinct from GVFs/successor features — it is a global eigenvector satisfying `L_m f_m = lambda_2 f_m`, not a per-state Bellman recursion, and Phi[a,:] is accumulated over the whole empirical graph rather than discounted from a start state. The Pareto-non-dominance vote over R^k ascents is not reducible to `argmax_a w^T Phi[a,:]`. The structural distinction from the GVF disqualifier is sound. However, the *operational failure mode* that killed it is the same hash-collision-starvation / terminal-only-channel degeneracy that killed FED (#15), CEC (#18), TRAC (#26), and CSD (#32): the online-grown observation-hash graph is too sparse on long-horizon sparse envs within 120s, so the Laplacian is either near-degenerate (tree-like graph, vacuous Fiedler partition) or the channel weight matrix is degenerate (terminal-only reward channels yield W_m near-zero). This places CSA in the same operational failure family as FED/CEC/TRAC despite its mathematical novelty — the graph-spectral wrapper does not escape the bottleneck.

- **Primitive vs stack:** One clear primitive (per-channel Fiedler vector of channel-reweighted Laplacian) plus one improvement operator (Pareto-non-dominance logit nudge over per-action Fiedler-ascent vectors). Primitive count is clean.

- **Evidence quality:** beat_random=0, beat_strong=0 across all 4 core envs. Score of 0.0 on DoorKey, KeyCorridor, and DST; 0.011 vs random 1.331 on RG. The hypothesis's own falsifiers (channel eigenvalue gap < 1e-6, mean bucket revisit < 2.0) almost certainly triggered — the candidate collapsed under exactly the predicted failure modes (a) and (b). No env showed signal above random; no vector env beat random. Evidence is entirely negative.

- **Failure-mode informativeness:** The failure rules out the "graph-spectral primitive on an online observation-hash graph" family as a standalone approach. The bottleneck is that sparse hash graphs on long-horizon envs cannot support a non-degenerate Fiedler partition within the 120s budget, and terminal-only reward channels collapse the channel-reweighted Laplacian to near-zero signal. This is a structural ruling: no variant of "build a channel-weighted adjacency graph from online hash-bucketed transitions and extract the Fiedler vector" will work without (1) a paired exploration primitive that densifies the graph before spectral computation begins, and (2) a non-terminal reward channel providing meaningful edge weights.

## Lesson for the next iteration

Graph-spectral primitives (Laplacian eigenvectors, Cheeger cuts) on online observation-hash graphs fail for the same reason as all cluster/hash-indexed primitives: the hash graph is too sparse on long-horizon sparse envs within budget, and terminal-only reward channels cannot provide non-degenerate edge weights; future proposals using transition geometry must be paired with an explicit graph-densification or exploration primitive that fires before spectral computation begins.
