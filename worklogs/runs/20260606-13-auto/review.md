---
verdict: probe
reviewer_run: 20260606-13-auto
hypothesis_type: probe
---

## Summary

LYRA optimizes the Lyapunov-exponent gap `Δ(π) = λ_1(π) − λ_2(π)` of a
reward-tilted policy cocycle via online QR-tracked 2-frame and a
spectral-perturbation score-function gradient; the principle is coherent,
the primitive is typed and load-bearing, the ablation is decisive, and
both fixes from the prior `revise` are addressed.

## Schema check

- `candidate.json` should pass `scripts/validate_candidate.py`: all
  required string fields present and non-empty, both required booleans
  present, `update_family=direct_policy_update` valid, `memory=none`
  valid, `nearest_disqualifier=published_method` valid, `claimed_stage=sparse`
  valid, `uses_vector_reward=false` consistent with sparse stage.
- Prose vs JSON alignment:
  - `principle` and `## Principle`: match (gap of reward-tilted cocycle,
    QR-estimated, gradient ascent).
  - `primitive_name`/`primitive_type` and `## Primitive`: match
    (Stiefel-2-frame on `ℝ^d` carrying leading two Lyapunov directions).
  - `claimed_stage=sparse` and `## Empirical claim` "stage: sparse" with
    MiniGrid DoorKey + KeyCorridor: match. The prior vector/sparse
    contradiction is resolved.
  - `falsifier` matches: gap fails to grow, return fails to track gap,
    or random-frame ablation matches LYRA.
  - `ablation_plan` matches `## Ablation plan`: freeze a random orthonormal
    2-frame, no QR update, identical gradient computation otherwise.
  - `nearest_disqualifier=published_method` and `novelty_boundary` align
    with `## Novelty boundary` (risk-sensitive RL, edge-of-stability,
    proto-value functions, Oja, risk-sensitive PG).
- `uses_reward=true` and `uses_vector_reward=false` correctly describe
  the actual mechanism (scalar reward enters as multiplicative tilt
  `exp(β r_t)`).

## Coherence check

1. Cocycle `M_t = γ exp(β r_t) P^π` as a linear operator on `ℝ^{|S|}` is
   well-typed and reduces to a real matrix in tabular MDPs. (follows)
2. Multiplicative ergodic theorem invocation under stationarity and
   `E[log+ ‖M_0‖] < ∞` is standard. (follows)
3. The gap Δ = λ_1 − λ_2 is a non-trivial functional of π distinct from
   the return functional and from the symmetric-Laplacian spectral gap;
   the distinction from risk-sensitive RL (which targets λ_1 alone via
   Donsker–Varadhan) is correctly drawn. (follows)
4. Step 5 (gap maximization principle) is openly conjectural and listed
   under proof debt; the degeneracy case `λ_1 = λ_2` is named.
5. Online QR estimation via Benettin–Galgani–Strelcyn with
   Goldsheid–Margulis 1989 convergence is the textbook estimator for
   stochastic-cocycle Lyapunov spectra. (follows)
6. Spectral-perturbation gradient `∇_θ λ_i = E[u_i^T (∂M/∂θ) u_i]`
   requires λ_i simple; the file flags this. (follows with proof debt)
7. The rank-1 sample-path estimator
   `M_t v ≈ γ exp(β r_t) ⟨Φ(s_{t+1}), v⟩ Φ(s_t)` is heuristic but is
   the natural single-sample stochastic operator estimator, analogous
   to TD's single-sample Bellman estimator. Flagged as proof debt. The
   pseudocode fix now writes the gradient coefficients explicitly per
   index `i ∈ {1,2}`:
   `c_i = tilt · (u_i^T Φ(s_t)) · (Φ(s_{t+1})^T u_i)`,
   `θ ← θ + α (c_1 − c_2) · ∇_θ log π_θ(a_t|s_t)`. The broadcasting is
   no longer ambiguous.

Internally consistent; load-bearing heuristics are flagged in
`## Proof debt`.

## Novelty check

Searches of the principle (not the name) confirm:
- "Lyapunov exponent gap policy gradient" — published Lyapunov-RL work
  treats Lyapunov exponents of parameter dynamics or policy-objective
  dynamics, not state-space reward-tilted cocycle gaps.
- "Oseledec multiplicative ergodic theorem MDP policy optimization" — no
  RL algorithm matches.
- Risk-sensitive RL (Howard–Matheson, Borkar, Mihatsch–Neuneier)
  optimizes λ_1; LYRA's gap is structurally different and reduces to
  the mixing-rate spectral gap at β → 0.
- Spectral RL / proto-value functions (Mahadevan–Maggioni 2007) uses
  the symmetrized graph Laplacian; LYRA uses the non-symmetric
  reward-tilted operator.

Family check against `prior_attempts.md`: not a bucketed tensor (A), not
a pairwise-trajectory comparison (B), not a within-trajectory geometric
path statistic (C — Lyapunov is an operator-cocycle spectral limit, not
a path geometric quantity, with the C-reclassification caveat correctly
acknowledged), not a value-vocabulary swap (E), not a mechanism stack
(G), not a cochain-complex iteration (H).

Not a rebadge.

## Implementability and ablation check

The Engineer can implement `train.py` against the existing contract:
- Φ : obs → ℝ^d as a small fixed random encoder (e.g., d=64) is the
  natural choice for non-tabular envs; the hypothesis names this option.
- The QR step is `torch.linalg.qr` on a `d × 2` matrix per step, trivial.
- The score-function gradient is REINFORCE-shaped with `(c_1 − c_2)`
  replacing the advantage; the per-step pseudocode is now concrete
  enough to implement without inventing missing pieces.
- The Lyapunov accumulators `L = [0,0]` and step counter `T` give the
  empirical gap `Δ_T = (L[0] − L[1])/T` for the falsifier check.

Ablation: replace QR-tracked frame with a fixed random orthonormal
2-frame; if random-frame matches LYRA on gap and return, the Lyapunov
primitive is decorative and the algorithm is REINFORCE projected onto
random features. This is exactly load-bearing. Implementable as a
one-line change in `train_ablate.py` (skip the QR update; freeze Q at
init).

No vector-scalarization issue: the algorithm is honestly scalar-reward
and the JSON correctly reports `uses_vector_reward=false`; the stage is
sparse not vector.

## Decision

Verdict: `probe`. Both fixes from the prior review are in:
1. The stage contradiction is resolved — sparse throughout, with
   MiniGrid DoorKey + KeyCorridor as the empirical target and no stray
   `info["vector"]` language.
2. The gradient pseudocode now writes the per-frame-index scalar
   coefficients `c_1`, `c_2` explicitly, removing the broadcasting
   ambiguity.

The principle (gap maximization on a non-symmetric reward-tilted
cocycle) is novel relative to risk-sensitive RL (λ_1 alone),
proto-value-function spectral RL (symmetrized Laplacian), and the
edge-of-stability / parameter-Jacobian Lyapunov line of work. The
typed primitive (Stiefel 2-frame) is load-bearing, the ablation
(freeze the frame at random) directly tests whether the Oseledec
direction matters, and the falsifier (Δ_t fails to grow, or grows
without return) is concrete. Proof debt — gap-monotone improvement
conjecture, simplicity of λ_1, bias of the rank-1 estimator — is
honestly stated and is exactly the kind of debt the probe-first loop
allows. Empirical signal on the sparse stage is worth the compute.
