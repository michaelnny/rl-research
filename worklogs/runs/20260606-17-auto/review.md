---
verdict: probe
reviewer_run: 20260606-17-auto
hypothesis_type: probe
---

## Summary

SNELL replaces REINFORCE's full-trajectory return-weighting with a
predictable, learned stopping-time truncation driven by a Snell-envelope
continuation value regressed to suffix-maxima of cumulative reward;
probe approved because the primitive is typed and load-bearing for the
ablation, the update is implementable, and the discriminating observable
makes the ablation contrast falsifiable.

## Schema check

`candidate.json` parses and matches the field constraints
(`update_family=direct_policy_update`, `memory=network`,
`claimed_stage=quick`, `uses_vector_reward=false`,
`nearest_disqualifier=policy_gradient`). Prose alignment:

- `principle` and `## Principle` agree (Snell-envelope-truncated
  REINFORCE on the prefix `t <= tau*` with locked-in `R_{tau*}` weight).
- `primitive_name` / `primitive_type` match `## Primitive`
  (state-indexed continuation value satisfying a max-inside-expectation
  fixed-point).
- `claimed_stage` (`quick`), `empirical_claim`, and `falsifier` align
  with `## Empirical claim`. The discriminating observables
  (mean(tau*)/T < 1 in >=30% of episodes; corr(tau*, R_{tau*}) > 0.3)
  are concretely stated in both.
- `ablation_plan` (uniform random threshold per rollout, same prefix
  gradient + locked-in-reward weighting) matches `## Ablation plan`
  faithfully.
- `nearest_disqualifier=policy_gradient` and `novelty_boundary` align
  with `## Novelty boundary` (a)-(i).

No schema/prose contradictions.

## Coherence check

Step-by-step on the derivation:

1. The Snell-envelope statement `U_t = max(R_t, E[U_{t+1}|F_t])` and
   the predictable optimal stopping time `tau* = min{t : U_t = R_t}`
   are standard (Snell 1952; Peskir-Shiryaev 2006). Correct.
2. The **operator-identity claim** in step 4 ("max-inside-expectation
   distinct from Bellman") is partially loose: the operator
   `T_Snell f(s) = E[max(R_{t+1}, R_{t+1} + gamma f(s'))]
   = E[R_{t+1}] + gamma E[max(0, f(s'))]`. When `f >= 0` (likely on
   CartPole where rewards are non-negative), this collapses to
   `E[R_{t+1}] + gamma E[f(s')]` — a Bellman expectation operator with
   an extra constant. The structural separator from `V` is therefore
   weaker than the prose suggests on the quick stage. This is a
   coherence wobble but the author lists contraction as proof debt
   (item 1) and the empirical claim does not depend on `C != V`; it
   depends on whether the predictable-stopping rule shapes returns
   differently from random truncation. Acceptable as proof debt.
3. The regression target `target_t = max(R_{t+1}, ..., R_T)` is a
   well-defined supervised target. On monotone-cumulative-reward envs
   (CartPole) this collapses to `R_T` (the final cumulative return),
   so on quick `C` will learn a final-return predictor. This is a
   non-trivial scalar function of state and is enough for the stopping
   rule to vary by state.
4. The stopping rule `R_t >= C(s_t)` is causal/predictable — only
   `F_t`-measurable inputs. Distinct from DUAL-IR (run 15) which used
   the anticipative arg-sup-time. Verified.
5. The truncated REINFORCE update `g = sum_{t<=tau*} R_{tau*} *
   grad log pi(a_t|s_t)` is well-defined and implementable.
6. The bias of this gradient is non-trivial (item 4 of proof debt) —
   the author flags it explicitly rather than papering over it.

Verdict on coherence: load-bearing steps follow with explicit proof
debt; the operator-distinction prose is overclaimed on monotone-reward
envs, but the empirical contrast does not rely on it.

## Novelty check

Searched on the principle, not the name:
- "predictable stopping time" + "policy gradient" + REINFORCE: no hit
  in standard PG literature.
- Becker-Cheridito-Jentzen 2019 (deep optimal stopping for American
  options): learns stopping decisions for *evaluation* of fixed payoff
  processes, not for policy improvement. Different problem and
  different mechanism.
- Truncated/horizon-limited PG (e.g., PPO truncation, n-step PG):
  fixed horizon, not state-conditional and learned. SNELL's tau* is
  state-conditional and learned.
- DUAL-IR (run 15, this loop): anticipative arg-sup-time, not
  predictable stopping. Mathematically distinct (causal vs.
  anticipative).
- Dead families A-H: SNELL is not bucketed-tensor (A), not pairwise-
  trajectory (B), not within-trajectory geometric statistic (C — `C`
  is a learned function, not a within-trajectory geometric quantity),
  not reward-free-then-gated (D), not cochain (H).
- Family E ("avoid value vocabulary"): `C` does **not** straightforwardly
  satisfy a Bellman equation, but on monotone-reward envs the
  distinction collapses (see coherence note above). This is the
  closest dead-family risk. The probe nonetheless has a load-bearing
  ablation (random threshold) that will tell us whether the
  state-conditional learned nature of `C` matters or whether the
  algorithm is decorative truncation.

The nearest disqualifier listed (`policy_gradient`) is honest. The
novelty claim — predictable-stopping truncation of the prefix with
locked-in path-statistic weighting — is a structural modification, not
a rename, and is testable against the random-threshold ablation.

## Implementability and ablation check

`train.py` needs:
- a policy network (standard MLP for CartPole/Acrobot),
- a continuation-value network `C_phi : S -> R` (same shape),
- episode rollout, online cumulative `R_t` computation,
- predictable-stopping computation `tau* = min{t : R_t >= C_phi(s_t)}`,
- truncated REINFORCE update on prefix with `R_{tau*}` weight,
- MSE regression of `C_phi(s_t)` toward suffix-max
  `target_t = max(R_{t+1..T})` over the rollout buffer.

All standard PyTorch. No missing pieces. Logging of mean(tau*)/T and
corr(tau*, R_{tau*}) over batched rollouts is straightforward.

`train_ablate.py` removes `C_phi`, draws
`theta_random ~ Uniform(0, max_t R_t)` per rollout, computes
`tau_random = min{t : R_t >= theta_random}`, and applies the same
truncated-REINFORCE update with `R_{tau_random}` weight. This holds
fixed: prefix truncation, locked-in-reward weighting. It varies: the
state-conditional learned nature of the threshold. Correctly
load-bearing for the primitive.

Vector-reward concern: claimed stage is `quick` (CartPole-v1,
Acrobot-v1), both scalar. `uses_vector_reward=false`. No
scalarization risk on this stage. The hypothesis does include a
forward-looking note about component-sum for vector envs but it does
not affect this probe.

## Decision

Approve `probe`. Despite the operator-distinction overclaim
(max-inside vs. max-outside collapses on monotone-reward envs), the
probe has:

- a typed primitive (`C : S -> R`) regressed to a concrete supervised
  target (suffix-max of cumulative reward),
- an implementable update rule (truncated REINFORCE with locked-in
  reward),
- a load-bearing ablation that tests whether the learned,
  state-conditional, predictable-stopping nature of the primitive
  matters vs. random truncation with identical bookkeeping,
- a discriminating observable (`corr(tau*, R_{tau*})`) with sharp
  thresholds,
- explicit proof debt on contraction, regression convergence,
  improvement theorem, and gradient bias.

The empirical signal that would justify investing in the missing
contraction/improvement proofs is exactly what the quick-stage probe
is designed to extract. Engineer should proceed.
