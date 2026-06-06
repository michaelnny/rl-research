# 20260606-14-auto -- NORMAL (Normal-cONe Active-set policy iteRation viA aLternating projection) [probe]

## Principle

The optimal policy at each state is the unique fixed point of the
**normal-cone projection** `Π_{N_Δ}` onto the simplex's polar of the
advantage vector — i.e., π*(·|s) is characterized as the projection of
any zero vector onto the closed convex set
`{p ∈ Δ : ⟨A(s,·), p⟩ ≥ ⟨A(s,·), q⟩ ∀ q ∈ Δ}`, computed by **Dykstra's
alternating projection** between the simplex constraint and the
maximal-advantage halfspace, with no Bellman backup on the value side.

## Primitive

The **per-state active-set indicator**

  `χ : S → 2^A`  (codomain = power set of actions)

mapping each state `s` to the set of actions `χ(s) ⊆ A` that are
**active** in the variational-inequality sense:

  `χ(s) := { a ∈ A : ⟨A(s, ·), e_a − π(·|s)⟩ = 0 }`

where `A(s, a)` is the advantage estimate at `(s, a)` and `e_a` is the
canonical basis vector. Equivalently, χ(s) is the support of any policy
π̃ in the variational-inequality solution set at s:
`{π̃ : ⟨A(s,·), π̃ − π̃'⟩ ≤ 0 ∀ π̃' ∈ Δ}`.

This is **one typed object** per state — a binary mask over actions —
distinct from a Q-function (real-valued) or a policy (real-valued
distribution). The active-set indicator is the **Lagrange-multiplier
support** of the per-state simplex VI.

## Derivation sketch

1. **VI form of optimality.** A policy π* is optimal iff for every
   state s the per-state variational inequality holds:
   `⟨−A^{π*}(s, ·), π'(·|s) − π*(·|s)⟩ ≥ 0 ∀ π' ∈ Δ`,
   where `A^π(s,a) = Q^π(s,a) − V^π(s)`. This is **Stampacchia VI**
   (Stampacchia 1964; Facchinei-Pang 2003 §1.1) on the policy simplex.

2. **Normal-cone characterization.** The VI is equivalent to
   `−A^{π*}(s,·) ∈ N_Δ(π*(·|s))`, where `N_Δ(p)` is the normal cone
   of the simplex Δ at p (Rockafellar-Wets 1998 §6). For the simplex,
   `N_Δ(p) = { v ∈ ℝ^A : v_a ≤ c ∀a, v_a = c if p_a > 0 }` for some
   c ∈ ℝ. So at optimum, all *supported* actions have equal advantage
   and *unsupported* actions have weakly worse advantage.

3. **Active-set primitive.** Given any A(s,·) (advantage estimator —
   not necessarily exact A^π), define the **active set**
   `χ(s) = argmax_a A(s,a)` (with ties broken into a maximal set).
   The unique policy supported on χ(s) with all probability on the
   tie-set is the **VI solution at s for that A**. Crucially, χ
   depends only on the *order* of A(s,·), not its magnitudes.

4. **Why projection, not backup.** Standard Q-learning updates A's
   *magnitudes* via Bellman backup. We instead update A's *order* —
   specifically, the normal-cone signature χ — via **Dykstra's
   alternating projection** (Dykstra 1983; Boyle-Dykstra 1986)
   between two convex sets in `ℝ^{S × A}`:
   - C₁ = {A : A(s,·) ≥ 0 for some action at every s} (each state has
     an active action — this is the simplex normal-cone constraint
     written on advantages: the max is non-negative).
   - C₂ = {A : A(s,a) = r(s,a) − r̄(s) + γ E[max_{a'} A(s', a')] −
     γ Ē[max A(s', ·)]} (the **advantage consistency cone**: A is the
     advantage of *some* policy whose value at s is `r̄(s) + γ Ē[max
     A(s', ·)]`).
   These two convex cones intersect at the optimal A*, and Dykstra's
   algorithm converges to that intersection from any initialization
   (Han-Lou 1990; Bauschke-Borwein 1996, Theorem 3.18).

5. **Self-bootstrapping at zero reward.** Unlike Bellman backup, which
   stalls when reward is identically zero (gradients vanish — see the
   LYRA cold-start failure in run 13), Dykstra projection on the
   active-set signature is **driven by the structural constraint
   that some action must be active per state**, irrespective of
   reward magnitudes. The signature χ updates whenever the *relative
   ordering* of A(s,·) changes; with zero reward but non-zero
   transitions, χ propagates from terminal states (where A = 0
   uniformly) outward via the consistency-cone projection (step 4
   constraint C₂). This is the fresh structural property: the
   primitive carries information through the order on A, even when
   magnitudes are zero.

6. **Tie-breaking by perturbation.** When A(s,·) has ties (e.g., at
   initialization, all entries are zero so χ(s) = A trivially), the
   active set is `A` itself and the policy is uniform. As the C₂
   projection carries reward information back from terminal/rewarded
   states, ties break and χ shrinks. We **never need ε-greedy
   exploration**: the uniform policy is the natural initialization.

7. **Update rule (online sample-based).** Instead of Dykstra in the
   batch setting, run a sample-based version:
   - Maintain advantage estimates `A(s,a)` (a network or table).
   - At each transition `(s, a, r, s')`, compute the C₂-residual:
     `ε₂ = r + γ max_{a'} A(s', a') − A(s,a) − [γ Ē_{s'}[max A(s',·)]]`
     where the bracketed term is the *baseline* (running mean of
     `γ max_{a'} A(s', a')` over all transitions, not just the
     sampled action).
   - Update `A(s,a) ← A(s,a) + α · ε₂`. (This is the C₂
     projection step.)
   - Project to C₁: clip `max_a A(s,·)` to be ≥ 0 by adding a state-
     wise constant if needed: `A(s,·) ← A(s,·) + max(0, −max_a A(s,·))`.
   - Recompute χ(s) = argmax_a A(s,·) (the *active set indicator*).
   - **Act**: π(a|s) = uniform on χ(s) (active-set policy).

   The active-set indicator χ is the load-bearing primitive;
   advantages serve only to compute χ.

8. **Proof debt.** We do not yet prove that the sample-based
   alternating-projection update converges to A* in the stochastic
   setting. The batch-deterministic version converges by Boyle-Dykstra
   1986; the stochastic extension requires a step-size schedule and
   variance bound on ε₂, analogous to but distinct from the standard
   Q-learning convergence proof (Watkins-Dayan 1992) because the
   target operator is not a contraction in sup-norm — it is a
   composition of two cone projections.

## Update rule

```
Inputs: env, discount γ, learning rate α, baseline rate β
Init:   advantage net A_θ(s, a) (or table); baseline scalar V̄ = 0
Init:   policy: uniform random (χ(s) = A for all s by symmetric init)

For each episode:
  Roll out s_0, a_0, r_0, s_1, ... using π(a|s) = Uniform(χ(s))
  where χ(s) = argmax_a A_θ(s, a) (with ties → multi-element set).

  For each transition (s, a, r, s'):
    # 1. C₂-projection residual: advantage consistency cone
    M_next = max_{a'} A_θ(s', a')                # sup of next advantage
    V̄ ← (1 − β) V̄ + β · γ · M_next                # running baseline
    target = r + γ · M_next − V̄                  # consistency target
    ε₂ = target − A_θ(s, a)

    # 2. C₂-projection update on advantages
    grad-step on θ to reduce (A_θ(s, a) − target)²  with step α
    # equivalent table update: A(s,a) ← A(s,a) + α · ε₂

    # 3. C₁-projection: enforce that max_a A(s,·) ≥ 0 at visited s
    m_s = max_a A_θ(s, ·)
    if m_s < 0:
      shift A_θ(s, ·) ← A_θ(s, ·) + (−m_s)        # state-wise constant

    # 4. Recompute χ(s) = argmax (returned to sampler at next visit)

# At eval: π(a|s) = Uniform(argmax_a A_θ(s, a))
```

The key algorithmic difference from Q-learning: the **policy is
uniform on the argmax set χ(s), not greedy on Q**, and the **C₁
shift is a per-state additive constant** that is invisible to
standard Bellman analysis because it preserves order on A(s,·) but
keeps the C₁ cone constraint active. The shift is what couples the
two projections in the Dykstra sense.

## Empirical claim

stage: quick
claim: On the quick stage (CartPole, dense reward), NORMAL with the
active-set indicator χ should learn at a rate comparable to or
exceeding the matched-architecture REINFORCE/Q-learning baseline,
while exhibiting **lower variance in the early-training reward curve**
because the uniform-on-argmax policy provides built-in exploration
that does not collapse (no ε-greedy schedule needed), and the C₁
projection step prevents pathological "all-actions-look-bad"
collapse where vanilla Q would assign near-zero values to all
actions and lose discrimination.

The quick stage is appropriate because: (i) reward is dense, so the
C₂ projection has strong driving signal from step 1; (ii) the action
space is small (|A|=2 for CartPole), so the active-set primitive χ
is meaningful and changes are visible; (iii) the cold-start problem
that killed LYRA (run 13) is avoided.

falsifier: If NORMAL fails to learn on CartPole (mean episodic
return at 120s budget < 50, vs. random ~20 and PPO ~200), or if it
learns at rate **identical** to a bare Q-learning ablation (showing
the active-set primitive and C₁ shift are decorative), the principle
is falsified.

## Ablation plan

Replace the **active-set indicator χ with a hard greedy policy**
(argmax tie-broken by index, no uniform-on-argmax) **and remove the
C₁ shift step**. Concretely in `train_ablate.py`:
1. Sampling: `a = argmax_a A_θ(s, a)` deterministically (with ε-greedy
   schedule matching standard Q-learning at ε=0.1, decaying), instead
   of `a ~ Uniform(χ(s))`.
2. Remove the C₁ shift: never add the state-wise constant.

This collapses the algorithm to **bare advantage-baselined Q-learning
with a running scalar baseline**, which is a standard known method.
If the ablation matches NORMAL's learning curve on CartPole, the
active-set + C₁ projection primitive is decorative; if NORMAL is
strictly better (especially at early-training variance), the
primitive is load-bearing.

A second, weaker ablation: replace the C₁ shift `A(s,·) ← A(s,·) +
max(0, −max A(s,·))` with a **random** state-wise shift drawn from
N(0, 1) at each visit. If random shift performs equally to the C₁
shift, the projection structure is irrelevant — only the magnitude-
disruption matters.

## Novelty boundary

Closest known methods:

(a) **Q-learning with greedy policy** (Watkins 1989): updates Q
    via Bellman backup, acts ε-greedy. NORMAL does NOT do Bellman
    backup on Q; the C₂ "consistency" update has the form
    `A(s,a) + α (r + γ max A(s',·) − V̄ − A(s,a))` which differs
    from Q-learning by the **subtraction of a running baseline V̄**
    and by **acting on advantages, not Q-values**. Critically,
    NORMAL acts uniformly on χ(s), not greedily; this is not
    ε-greedy because χ(s) is the set of *exact* argmax actions
    (often a singleton at convergence, often a multi-element set
    early), not a fixed exploration rate.

(b) **Advantage-Actor-Critic (Mnih et al. 2016)**: separate critic
    network for V, policy gradient on actor with advantage as
    weight. NORMAL has no actor network and no policy gradient;
    the policy is *constructed* from the active set χ, not
    *parameterized*.

(c) **Mirror descent / NPG** (Kakade 2002): multiplicative-weights
    update on policy logits. NORMAL has no log-prob update and no
    KL term; the update is a **projection on the advantage cone**,
    not a Bregman step.

(d) **Munchausen RL** (Vieillard et al. 2020): adds log-policy
    bonus to Bellman target. NORMAL has no entropy bonus, no
    KL-bonus, and the target subtracts a *baseline*, not a
    log-policy.

(e) **Dual averaging on the simplex** (Nesterov 2009; Xiao 2010):
    accumulates gradients and projects onto Δ. NORMAL projects
    onto a *different* set: the active-set cone C₁ is not a
    simplex, and the C₂ cone is not a function-space norm ball.

(f) **Dead family A (bucketed-tensor + partial-order vote)**:
    NORMAL is **not** in family A — there is no partial-order vote
    over channels; the active set χ is the standard argmax over
    advantages (a real-valued vector), and the only "vote" is the
    standard argmax. There is no second axis (channel/cumulant)
    to disagree on.

(g) **Dead family E (avoid value vocabulary, keep value
    structure)**: NORMAL DOES use advantage A as an intermediate
    quantity, but the **load-bearing primitive is the active-set
    indicator χ**, not A's magnitudes. The ablation that replaces
    χ with greedy-on-A reduces NORMAL to standard advantage-Q;
    if that ablation matches, family E classification is correct
    and we lose. We claim it will not match because the C₁ shift
    and the uniform-on-argmax sampling change the dynamics in a
    way standard Q-learning does not.

The structural difference from Q-learning is the **two-cone
Dykstra projection structure**: the C₁ constraint that
max_a A(s,·) ≥ 0 (some action is active at every state) is **not**
present in Bellman backup — Bellman allows max_a Q(s,·) to be
arbitrary. The C₁ shift is the load-bearing novelty.

## Proof debt

1. **Convergence theorem (open).** Show that the sample-based
   alternating-projection update on (A_θ, V̄) converges to the
   intersection point A* (where A*(s,·) is the advantage of the
   optimal policy) under standard Robbins-Monro step-size
   conditions on α and β. The deterministic batch case follows
   from Boyle-Dykstra 1986 because both C₁ (a halfspace
   intersection) and C₂ (an affine subspace once the policy is
   fixed) are closed convex; the stochastic case requires a
   Markov-chain version of the alternating-projection argument.

2. **Active-set fixed-point characterization.** Prove that at any
   fixed point of the iteration, χ(s) ⊆ argmax_a A^{π*}(s,a) where
   π* is the optimal policy. (This says the active-set indicator
   converges to the optimal action set, not just to a stationary
   set of the iteration.)

3. **Consistency-cone closure.** Prove that the consistency cone
   C₂, defined by `A(s,a) = r(s,a) − r̄(s) + γ E[max A(s', ·)] −
   γ Ē[max A(s', ·)]`, is closed and convex in ℝ^{S × A}. This is
   non-trivial because the max operator is non-linear; the proof
   strategy is to observe that fixing the active-set indicator χ
   linearizes the max, and the cone is a union of linear
   subspaces over all possible χ — but this union is generally
   not convex, so the argument requires more care.

4. **Comparison-of-rates theorem.** Compare the convergence rate
   of NORMAL to that of Q-learning under matched conditions; the
   conjecture is that NORMAL's two-cone projection achieves
   polynomial rate `O(1/T)` because Dykstra has linear convergence
   under regularity (Bauschke-Borwein 1996), versus Q-learning's
   contraction rate γ^T which is exponential but with a rate
   bounded by γ. The crossover point would depend on the spectral
   gap of the policy chain.

The load-bearing improvement claim (4) is what would justify the
algorithm if empirical signal appears.
