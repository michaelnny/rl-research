---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-24-PCR (cluster-revisitation bottleneck family: TRAC #26, CSD #32)
side_information: [transition geometry, vector diagnostics]
---

## Verdict reasoning

- **Structural distinction**: The hypothesis correctly identified a real structural gap from PCR (#24): the product-with-floor-1 composition `α · Δ_K · max(Δ_P, 1)` was designed to let the horizon-concordance side fire unilaterally even when the channel side is silent. However, "fires unilaterally" required the K-vector cells to have sufficient population — the cluster-revisitation bottleneck (failure mode c, explicitly predicted) bound the entire K-primitive to silence on DoorKey-8x8 and KeyCorridor. The product-with-floor composition is a genuine structural difference in the operator, but the primitive it operates on inherits TRAC's (#26) and CSD's (#32) binding constraint: the (cluster, action) cell-revisitation frequency under uniform exploration on long-horizon sparse envs within 120 s.

- **Primitive vs stack**: One paired primitive `(K[s,a,:], P[s,a,:])` with one composition law (product-with-floor). This is compliant. However the combined dependency chain — online k-means clustering, L-horizon snapshot-argmax concordance accumulation, k-channel sign-conditional propensity with variance gate, and meet-of-two-Pareto-orders — means all sub-components must be simultaneously populated for the operator to fire non-trivially. Any one bottleneck silences the whole.

- **Evidence quality**: 0 envs beat random, 0 beat strong. DST scored 99.0 vs random 194.0 — below-random harm — replicating the ATP/PRAR failure (operator preferring nearby low-value treasure when long-horizon concordance is uniformly low, effectively encoding shortest-path-to-any-terminal). RG scored 0.011 vs random 1.331. DoorKey and KeyCorridor both scored 0.0. The failure pattern is fully consistent with the hypothesis's own predicted failure modes (a) and (c), confirming the cluster-revisitation bottleneck is the binding constraint.

- **Failure informativeness**: This failure extends the TRAC/CSD ruling to the "snapshot-policy self-concordance at multiple horizons" sub-family. The key new datum is that the product-with-floor composition does NOT rescue a K-primitive from cluster-revisitation sparsity; the operator's floor-1 fallback is a no-op (additive zero logit nudge), not an alternative signal source. Rules out the "multi-scale snapshot-concordance + channel-propensity product" family without an explicit cell-seeding or exploration mechanism.

## Lesson for the next iteration

The cluster-revisitation bottleneck is the binding constraint for any per-(cluster, action) primitive indexed by online k-means; a new candidate must either eliminate the state-cluster index entirely, or pair it with an explicit cell-seeding mechanism that guarantees minimum revisitation before the main primitive fires — and the Explorer candidates (RSD, SDS) in `worklogs/candidates/` should be read first as the nearest alive attempts addressing this axis.
