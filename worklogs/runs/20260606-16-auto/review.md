---
verdict: probe
reviewer_run: 20260606-16-auto
hypothesis_type: probe
---

## Summary

TEAR proposes a per-trajectory backward-linear adjoint co-state with rank-1 trajectory-empirical Jacobian, used as a Hamiltonian-weighted score-function gradient on the vector stage. The principle is coherent, the primitive is typed and load-bearing, the ablation directly tests it, and the method is not a rename of any disqualified family.

## Schema check

`candidate.json` contains all 15 required string fields and both required booleans. `claimed_stage = "vector"` is in `harness.STAGES`. `update_family = "direct_policy_update"` is valid. `memory = "episode"` matches the per-rollout backward pass. `nearest_disqualifier = "actor_critic"` is in the enum. `uses_vector_reward = true` is consistent with the `vector` stage. The scalarization defense is present in `novelty_boundary` ("not of the form w^T r_t for any t-independent or policy-independent w") because the JSON nearest_disqualifier is `actor_critic` not `scalarization`, but the prose still treats scalarization seriously (boundary item (f)).

Prose-to-JSON mapping:
- `principle` matches `## Principle` (Pontryagin/adjoint/Hamiltonian-as-score-weight).
- `primitive_name = "trajectory-empirical adjoint co-state"` matches `## Primitive`.
- `primitive_type` encodes the typed object lambda : episodes x {0..T} -> R^k with the recursion.
- `claimed_stage`, `empirical_claim`, and `falsifier` agree with `## Empirical claim` (DST/RG hypervolume vs. random baseline).
- `ablation_plan` (i.i.d. Gaussian replacement of lambda) matches `## Ablation plan` step (1)-(4).
- `nearest_disqualifier` and `novelty_boundary` agree with `## Novelty boundary`.

No contradiction.

## Coherence check

1. Step 1 (discrete PMP) is standard textbook material (Bertsekas).
2. Step 2 (stochastic extension via score function) is heuristic. The probe explicitly flags this as proof debt item (2). The substitution `H_t -> score-function weight` is the load-bearing creative move; it does not collapse to a known identity but is a defensible analog of the Hamiltonian-maximization condition under a stochastic policy. This is the kind of leap a probe is allowed to make.
3. Step 3 (vector co-state) follows componentwise from step 1 and is internally consistent: lambda is in R^k, recursion is linear in lambda, so vector-valuedness is preserved.
4. Step 4 (rank-1 trajectory-empirical Jacobian) is ad hoc. Note that `J_t · phi_t = phi_{t+1}` exactly by construction, so the estimator at least matches the realized one-step transition along the rank-1 direction phi_t. Bias in directions orthogonal to phi_t is acknowledged as proof debt item (1).
5. Step 5 (cold-start) and step 6 (order-changing) explicitly distinguish from runs 13 (LYRA) and 14 (NORMAL). I checked both prior hypotheses; the distinctions are real (LYRA used a multiplicative reward tilt; NORMAL used an active-set indicator over advantages with no backward recursion).
6. Steps 7-9 (vs. actor-critic, GAE, SVG) are correct: TEAR has no learned V/Q, no TD residual, no learned dynamics model, and no reparametrization gradient.
7. Degenerate-case sanity check: with phi = identity and J_t replaced by I (no Jacobian), the recursion becomes lambda_t = sum_{s>=t} r_vec_s, and `H_t ~ <r_vec_t, 1> + <return-to-go, action_proj>`. This is *not* exactly REINFORCE-with-return because of the action-projection inner product; the action-conditional contribution distinguishes it from a baseline pure return weight. With the non-trivial rank-1 J, the action-projection direction is itself trajectory-dependent, further departing from REINFORCE.

The derivation is heuristic but coherent, and the heuristic moves are explicitly listed as proof debt rather than smuggled in as theorems.

## Novelty check

Searches: "discrete pontryagin maximum principle reinforcement learning policy gradient adjoint score function". Closest published methods:
- Li et al. 2018 (MSA for deep learning): uses PMP for *layer-wise* training of deep nets, not for RL policy gradient on stochastic MDPs.
- Bao et al. 2023 (BAL, "stochastic maximum principle for RL with parameterized environment"): assumes a parameterized known/learned environment and solves a backward adjoint SDE. TEAR uses no learned/parameterized environment; the Jacobian is a per-step rank-1 estimator from realized features.
- SVG / SVG(infinity) (Heess 2015): learned dynamics + reparametrization gradient; TEAR has neither.
- Pontryagin-guided portfolio optimization (2024): uses PMP in continuous time on a known stochastic system. Not the same setting.

TEAR's specific construction (rank-1 trajectory-empirical Jacobian + score-function ascent on per-step Hamiltonian + vector co-state with realized terminal boundary `lambda_T = r_vec_T`) is not a documented method. It is not a Bellman backup, not a TD residual sum, not a critic-supplied advantage, not scalarization, not GAE, not options/HER/RND/successor features/distributional/decision-transformer, and not in dead families A-H of `prior_attempts.md`. Family C (within-trajectory geometric statistic) is the closest dead family but is not a backward linear recursion driven by realized transitions; the probe addresses this distinction explicitly.

Not a rebadge.

## Implementability and ablation check

The Engineer can implement `train.py` against the harness contract:
- Read `info["vector"]` per step (harness.py line 160 exposes it).
- Use a small torch policy network for `pi_theta(a|s)`.
- Compute features phi(s): identity for DST/RG (small state vectors); fixed random projection for image envs.
- Run rollout, compute features, compute rank-1 J_t lazily (apply via two inner products in O(k)).
- Compute lambda backward in O(T*k).
- Compute H_t per step, accumulate `H_t * grad log pi_theta` and step the optimizer.

`train_ablate.py` differs only in steps 3-4: skip Jacobian and backward recursion; sample lambda~_t ~ N(0, I_k) i.i.d. per step. Same downstream score-function update. This is mechanical.

Vector consumption is correct: the probe genuinely uses `r_vec_t` from `info["vector"]` (not a scalarization). For sparse scalar envs, the augmentation `r_vec_t = (r_t, 1)` is a within-probe construction, not a scalarization of an existing vector reward — it injects a constant time-marker channel to keep `lambda_T != 0`.

The ablation (i.i.d. Gaussian lambda) is load-bearing: if it matches TEAR's hypervolume, the backward adjoint structure is decorative. The secondary ablation (zero terminal boundary) further isolates the role of the realized-reward boundary condition. Both ablations test the *primitive*, not a different algorithm.

## Decision

Approve as `probe`. The principle is one sentence (Hamiltonian-as-score-weight under a backward linear adjoint with realized terminal-reward boundary). The primitive is one typed object (the vector co-state path). The update rule is fully specified pseudocode. The ablation directly disables the primitive. The empirical claim is on the appropriate stage (vector) with a falsifier tied to fixed baseline numbers and to the ablation. Proof debt is named (Jacobian consistency, Robbins-Monro convergence, hypervolume monotonicity) and is the correct reason to defer theorem work until empirical signal exists. The probe is not a rename of any baseline or dead family.
