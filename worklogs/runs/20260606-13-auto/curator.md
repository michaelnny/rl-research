---
verdict: failed-structural
nearest_prior_or_disqualifier: FED (#15) / bootstrap-wall family
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** TRAC's primitive (JSD between two empirical successor-cluster histograms partitioned by binary channel-firing event) is genuinely distinct from successor features/GVFs (no discounted cumulant expectation, no TD-bootstrap, no weight-vector extraction) and from CWAI (no learned forward-model Jacobian, no gradient, no embedding). The reviewer correctly approved this as novel. However, the failure mode is structural, not implementation-specific.
- **Primitive vs stack:** One primitive (JSD over H_fire/H_nofire) + one improvement operator (Pareto-non-dominance logit nudge). Clean shape.
- **Evidence quality:** beat_random=0, beat_strong=0 on all 4 core envs. DoorKey-8x8 and KeyCorridor both scored 0.0. DST scored 98.0 vs random 194.0 (below random). RG scored 0.011 vs random 1.331. The hypothesis's own falsifier (a) was confirmed: the (cluster, action, channel) cells did not accumulate sufficient mass within 120 s on sparse long-horizon envs, rendering the JSD primitive silent.
- **Failure-mode analysis:** Despite the Researcher's argument that step-penalty channels would seed H_fire/H_nofire from trajectory 1 (avoiding the terminal-reward-only bootstrap collapse of FED/CEC), the cluster-indexed cell-collision bottleneck still applies: even if H_fire accumulates entries, the (cluster c, action a) pair must be visited repeatedly from the same cluster, which doesn't happen reliably under uniform exploration on long-horizon nav envs within 120 s. The JSD primitive is doubly gated: (1) channel firing must partition trajectories AND (2) the same (c,a) state must be revisited enough times to fill both histogram sides. This is structurally equivalent to FED's observation-hash bucket coverage failure.
- **Ruling scope:** Extends the FED/CEC/TPP/PCR bootstrap-wall family to cover cluster-indexed conditional-distribution primitives. Any primitive that requires per-(cluster, action) histogram coverage on sparse long-horizon envs faces the same wall. The step-penalty argument does not rescue TRAC because the bottleneck is state-cluster revisitation, not channel-firing frequency.

## Lesson for the next iteration

Any cluster-indexed divergence primitive requiring repeated (c, a) revisitation faces the same bootstrap wall as hash-indexed primitives on sparse long-horizon envs; future candidates must either provide an explicit coverage/exploration primitive paired with the signal, or rely on a primitive that fires from singleton observations without needing cross-visit aggregation.
