---
verdict: probe
reviewer_run: 20260606-27-auto
hypothesis_type: probe
---

## Summary

EFLO proposes a coherent, typed, single-primitive update — the GAE-style
exponential average of policy-entropy residuals
`Â_t^H(λ) = Σ_l (γλ)^l (H(π_θ(·|s_{t+l+1})) - H(π_θ(·|s_{t+l})))` used
as the sole credit weight on the score-function gradient — that is
structurally distinct from SAC, A2C/PPO entropy bonus, GAE, RND/counts,
and state-visitation MaxEnt; and is genuinely reward-free, so the
scalarization disqualifier that killed TRACE (run 23) does not apply.

## Schema check

`scripts/validate_candidate.py` will accept this file: all required
string fields are non-empty; `uses_reward=false`,
`uses_vector_reward=false`; `claimed_stage="quick"` (a known stage);
`update_family="direct_policy_update"` and `memory="episode"` are valid;
`nearest_disqualifier="sac"` is valid. The `uses_vector_reward=true with
non-vector stage` and `nearest_disqualifier=scalarization without
"scalar" in boundary` checks are not triggered.

Schema vs. prose: `principle`, `primitive_name`/`primitive_type`,
`claimed_stage`, `empirical_claim`/`falsifier`, `ablation_plan`,
`nearest_disqualifier`/`novelty_boundary`, and `proof_debt` all match
the corresponding hypothesis sections. The `uses_reward=false` flag
matches the prose's explicit reward-free claim and step 6 of the
derivation. No material schema/prose contradiction.

## Coherence check

Step-by-step reading of the derivation:

- Step 1: `δ^H_t := H(π_θ(·|s_{t+1})) - H(π_θ(·|s_t))` is well-typed for
  any softmax policy. OK.
- Step 2: Substituting `δ^H_t` for the value-TD residual in GAE's
  exponential-averaging recursion is mechanically valid; the recursion
  doesn't depend on what the residual *means*. OK.
- Step 3: Reduction at λ=0 (one-step entropy difference) and λ=1
  (full discounted forward-entropy-flow) is correct algebra.
- Step 4: The telescoping identity at λ=1 has a small but non-load-
  bearing infelicity — `Σ_l γ^l (H_{t+l+1} - H_{t+l}) = (1-γ) Σ_{l≥1}
  γ^{l-1} H_{t+l} + lim_{L→∞} γ^L H_{t+L} - H_t` (the boundary term and
  the geometric coefficient sum need careful handling). The derivation
  states the identity without showing the algebra and labels the result
  loosely as "the entropy advantage of step t." The infelicity does not
  break the construction; the key bias-variance interpolation property
  in Step 3 is what the empirical probe tests. Acceptable as proof
  debt.
- Step 5: The claimed identity
  `∇_θ J^H(θ) = E_τ[Σ_t Â_t^H(λ) ∇_θ log π_θ(a_t|s_t)
                   + Σ_t γ^t ∇_θ H(π_θ(·|s_t))]`
  is a heuristic policy-gradient-theorem-style decomposition; the
  precise form when the per-step quantity (`H(π_θ(·|s))`) is itself
  policy-parameter-dependent is non-trivial (the chain rule has extra
  terms beyond just the score-function and direct-entropy pieces).
  The hypothesis treats `Â_t^H(λ)` as a stop-gradient and explicitly
  acknowledges this as proof debt item (2). For a probe, dropping the
  direct-entropy term and using only `g_θ^EFLO = Σ_t Â_t^H · ∇log π`
  is a coherent heuristic update, not an algebraic error.
- Step 6: Reward-independence of `Â_t^H(λ)` is correct by construction
  (the recursion reads only `H_t = -Σ_a π_θ(a|s_t) log π_θ(a|s_t)`).
- Step 7: Random-init non-zero variance of `δ^H_t` is plausible —
  Glorot/standard inits give logit-scale fluctuations across visited
  states, hence per-state entropy fluctuations.

Coherence verdict: one load-bearing primitive, deterministic update
rule, and the heuristic step (5) is openly listed as proof debt rather
than disguised as a theorem. This passes the probe coherence bar.

## Novelty check

User-flagged comparisons addressed in detail:

(1) **Vs. SAC / soft-Bellman.** SAC's per-step entropy
`α H(π(·|s_t))` enters the **soft-Bellman target** of a learned soft
Q-function: `Q_soft(s,a) ← r + γ E_{s'}[V_soft(s')]` with
`V_soft(s') = E_{a'∼π}[Q(s',a') - α log π(a'|s')]`. The policy is
then improved via softmax-of-Q. EFLO has no learned Q, no Bellman
operator, no soft target, and no entropy added to reward. Entropy
enters at the **credit-assignment-weight slot** on the score-function
gradient — a different mechanism slot from the soft-Q value-bootstrap
slot. Distinct.

(2) **Vs. A2C/PPO entropy bonus / entropy-regularized PG.** The
A2C/A3C/PPO entropy bonus adds `+β · ∇_θ H(π(·|s_t))` per visited
state — an *additive single-step* entropy-derivative term. EFLO uses
`Â_t^H(λ)` as a *multiplicative* per-step weight on
`∇_θ log π_θ(a_t|s_t)`, where the weight is a *temporal sum across
future steps* of entropy *differences*. Crucial distinction: the
A2C/PPO bonus is purely local (depends only on `H(π_θ(·|s_t))`); EFLO's
weight at step `t` depends on `H_{t+1}, H_{t+2}, …, H_T`, i.e., on
future-state entropies along the realized rollout. The reduction at
λ→0 is `Â_t^H(0) = H_{t+1} - H_t`, which is still a *forward-difference
across two states*, not the *local entropy gradient* at one state. Not
a rebadge of A2C/PPO entropy bonus.

(3) **Vs. GAE structure being decorative.** This is the user's third
question: is the GAE exponential averaging load-bearing, or is the
mechanism really just "REINFORCE with entropy difference per step?"
The ablation plan addresses this only partially. The primary
ablation (`c_t ≡ 1`) removes the entropy-flow weight entirely and
tests EFLO vs. trajectory-log-likelihood ascent; it does not isolate
the GAE-exponential-averaging *machinery* from the residual choice.
A cleaner discriminator on the GAE machinery would set `λ = 0`
(c_t = `δ^H_t`, one-step forward-entropy-difference) vs. `λ = 0.95`
and check whether discriminator (iii) (entropy preservation) and the
hypervolume-axis behavior differ between them. The hypothesis does
not require this comparison, but the cosine-alignment discriminator
(b) does fire for any non-uniform `c_t`, including `λ = 0`. So:
the GAE structure is not decisively isolated by the proposed
ablation, and the probe primarily tests "entropy-flow-weighted PG vs.
log-likelihood ascent" rather than "λ-interpolation matters." This is
a soft point: the residual *is* the load-bearing novelty by the
hypothesis's own framing (step 13: "exponential averaging is GAE's
machinery; the entropy-flow residual is the genuinely new
primitive"), and the residual is what the primary ablation isolates.
Acceptable for a first probe; a follow-up probe could add the
λ-sweep ablation if the primary signal is positive.

Search hits: "Entropy-Modulated Policy Gradients" (EMPG, arXiv
2509.09265, ICLR 2026 under review) modulates per-step PG by step
uncertainty *and* final task outcome — a reward-dependent
re-calibration that amplifies confident-correct and attenuates
uncertain steps. It uses entropy at one step as a *local modulator*,
not as a *temporal forward-flow estimator across steps*, and it
requires reward (outcome) to determine sign and magnitude. EFLO is
reward-free and uses a temporal sum of entropy *differences* with no
outcome term. Different mechanism, different objective. No rebadge.

Other searches ("discounted action entropy" policy gradient,
"entropy residual" lambda return GAE) returned no direct matches. I
did not find a published method that uses
`Σ_l (γλ)^l (H_{t+l+1} - H_{t+l})` as a credit weight. The mechanism
appears genuinely novel.

Dead-family check (per `prior_attempts.md`):

- Family A (bucket+vote): no buckets. Out.
- Family B (pairwise traj): single rollout. Out.
- Family C (within-trajectory geometric): the residual is on the
  *policy manifold* (action-distribution entropy), not on the
  observation/cumulant trace. Out.
- Family D (reward-free + reward-gated): EFLO is reward-free
  *throughout*, with no gate. Not Family D.
- Family E (avoid value vocabulary, keep value structure): `H(π_θ(·|s))`
  does not future-compress return. It is read in closed form from the
  policy network's output at `s`. Not a renamed value function.
- Family F (hand-engineered priors): entropy is canonical, not
  hand-designed.
- Family G (mechanism stack): one primitive (Â_t^H(λ)), one update.
- Family I (per-channel parameter-space aggregation): one scalar
  weight per step, no per-channel decomposition.

No dead-family match.

## Implementability and ablation check

Implementability: an Engineer can write `train.py` as a standard
REINFORCE-style loop with two non-standard pieces:

1. After rollout, compute `H_t` for each visited state by a forward
   pass of the policy net at `s_t` and a Shannon-entropy of the
   softmax. This is one extra forward pass per visited state (or the
   per-state entropy can be cached during rollout for free, since
   the policy is queried at `s_t` to sample `a_t`).
2. Run the canonical right-to-left GAE recursion on `δ^H_t = H_{t+1}
   - H_t` to get `A_t`, then `c_t = A_t.detach()`.
3. Loss is `-Σ_t c_t · log π_θ(a_t | s_t)` (negated for ascent under
   minimizer).

`train_ablate.py` flips step 4 to `c_t = 1` and skips steps 2-3 of
the GAE recursion. Mechanical.

The Engineer does **not** need to consume `info["vector"]` because
EFLO reads no reward — neither scalar nor vector. `harness.py`
unconditionally injects `info["vector"]` on vector envs (line 160 of
harness.py wraps step to add `info["vector"] = vec`), so the env
contract is honored on the env side; the algorithm simply ignores it.

**Vector-reward compliance.** This is the user's most important
question. The substrate rule in `CLAUDE.md` says: "For vector envs,
training must consume `info["vector"]`. Training only on scalar reward
in vector envs is scalarization and is disallowed." `prior_attempts.md`
disqualifies `wᵀr` for any fixed or learned `w`. EFLO does neither: it
trains on **no reward at all**. There is no `wᵀr` projection, no
single-channel selection, no fixed weight applied to a vector return.
The intent of the scalarization disqualifier is to prevent collapse of
vector reward to scalar; a genuinely reward-free update does not
collapse anything, because there is no reward signal in the update
path. This is structurally distinct from TRACE (run 23), which used
`info["vector"][0] = wᵀr with w=[1,0]` as the return signal. EFLO's
return signal is the empty set. Compliant.

A residual concern: the **secondary score-axis claim** in (d) — final
hypervolume score on DST-concave — is computed by `harness.py` from
env returns (the harness sees `info["vector"]` independently of the
algorithm), so the *evaluation* is on the vector reward, but the
*training* is reward-free. This is a clean separation and is exactly
what the substrate intends.

Ablation quality: the primary `c_t ≡ 1` ablation is load-bearing for
the entropy-flow-weight-vs-log-likelihood-ascent contrast (predicted
discriminator (iii): EFLO preserves H_t, ablation collapses H_t to 0).
This isolates the entropy *residual* as the novelty. It does not
isolate the GAE-exponential-averaging *machinery* (which would
require a `λ=0` arm). The optional `c_t ~ Uniform(-1,1)` sanity
ablation isolates "non-uniformity matters" from "entropy-flow
correlation matters" — a useful third arm. Adequate for a first
probe.

## Decision

`probe`. The probe has:

- a coherent principle (ascend the discounted forward-state-conditional
  policy-entropy functional);
- one typed primitive (Â_t^H(λ): R^T-valued GAE-style estimator on
  policy-entropy residuals);
- an implementable update rule against the existing harness contract,
  with no missing pieces;
- a load-bearing primary ablation (`c_t ≡ 1`) that cleanly tests
  whether the entropy-flow weight is causally responsible for entropy
  preservation;
- a falsifiable empirical claim with multiple discriminators (a)–(d)
  that fire from rollout 1 (mechanism presence) plus a score-axis
  outcome partition;
- a credible novelty boundary distinguishing EFLO from SAC (no
  Q/Bellman), entropy-bonus PG (multiplicative temporal-flow vs.
  additive single-step gradient), GAE (entropy residual vs. value-TD
  residual), state-MaxEnt (action entropy in closed form vs. visitation
  density), and TRACE (temporal entropy-flow vs. squared single-step
  log-prob gap);
- explicit proof debt (convergence to stationary points of J^H,
  stop-gradient bias bound, bias-variance theorem analogue, connection
  to Hazan visitation-entropy) listed honestly rather than claimed.

Vector-reward concern resolved: EFLO is genuinely reward-free, not
scalarizing. The TRACE rejection precedent does not apply.

The novelty boundary is the heaviest dimension here, and the analysis
holds: `Â_t^H(λ)` as a sole credit-assignment weight on the score-
function gradient is structurally distinct from every named method
checked.

Soft point (not a blocker): the primary ablation isolates the entropy
*residual* but not the GAE-exponential-averaging machinery. The
hypothesis explicitly frames the residual as the novelty, so this is
internally consistent. A second-iteration probe could add a `λ=0` arm
to test whether λ-interpolation is load-bearing if (a)–(c) fire.

Empirical signal worth the panel run: yes. The training-dynamics
discriminators (a)–(c) fire from rollout 1 regardless of reward
discovery, so even on a 120s budget the probe yields informative
signal about whether forward-entropy-flow ascent preserves vs.
collapses action entropy on a non-trivial substrate.
