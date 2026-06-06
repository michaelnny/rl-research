---
verdict: reviewer-rejected
nearest_dead_family: none
---

## Verdict reasoning

- TRACE proposes a coherent novel REINFORCE reweighting: per-step gradient weight = squared consecutive-state action log-likelihood gap `delta_t^2 = (log pi(a_t|s_t) - log pi(a_t|s_{t+1}))^2`, multiplied by scalar return-to-go. The primitive is one typed object on the policy manifold, requires only one extra forward pass per step, and is structurally distinct from baselines (multiplied not subtracted), AWR (quadratic not exponential), curiosity (gradient weight not reward), and Fisher/NPG (magnitude weight not direction change).
- Schema check passed: `candidate.json` is internally consistent with prose, `uses_vector_reward = false` is consistent with the proposal, and no schema-mechanical violations were found.
- Coherence and novelty checks passed: derivation is well-typed, proof debt is named not hidden, and the mechanism does not reduce to any dead family (A through H) or published named method.
- Rejection trigger is a clean disqualifier hit: the `quick` stage targets `deep-sea-treasure-concave-v0`, which is a vector-reward env (`ENV_TYPE = "vector"`, two channels). TRACE proposes using `info["vector"][0]` only, which is `w^T r` with `w = [1, 0]` — exactly the "scalarized vector reward" disqualifier listed in `prior_attempts.md`. The reward-free nature of the primitive `delta_t^2` does not rescue the scalarized return signal `G_t`.
- The reviewer's citation of "run 23 (TCP)" as a direct precedent is a hallucination — there is no such prior probe in the worklogs. The underlying disqualifier identification is correct regardless: scalarized-channel return on a vector env is the clean disqualifier. No family-level corpus update is warranted; the scalarization disqualifier is already listed.
- The TRACE mechanism is worth retaining for a scalar-stage env. Two fixes are available: (a) re-target to a scalar-stage env and re-state the empirical claim there; or (b) extend TRACE to consume `info["vector"]` as a vector return-to-go with a multi-channel score-function update.

## Lesson for the next Researcher

The TRACE squared-log-likelihood-gap weight is a structurally clean novel primitive; re-probe it on a scalar-stage env (e.g., MiniGrid or a non-vector harness stage) to avoid the vector-scalarization disqualifier, or extend the update rule to consume the full vector return channel.
