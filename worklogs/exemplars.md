# exemplars.md — what an algorithm looks like at the bar we want

**Read this before you propose anything.** The mission is to find the next
AlphaZero-class RL algorithm. The list below is *what that bar looks
like* — not a menu to imitate. If your proposal does not have a kernel
of comparable quality, do not write it down.

Every entry has the same shape:

- **Optimization principle** — the statement of what is being optimized.
- **Derivation** — how the update rule falls out of the principle.
- **Primitive** — the one mathematical object the algorithm computes and
  manipulates.
- **Theorem** — the convergence / improvement guarantee, with the
  condition under which it holds.

Note what is *not* on the list: a "predicted failure modes" section, a
"scaling story," a "side-information channel" enumeration, a
"nearest-prior audit." Those are corpus-management fields the project
loop uses for bookkeeping. They are not what makes an algorithm an
algorithm. The exemplars below have none of them and lose nothing.

---

## Q-learning (Watkins, 1989)

- **Principle.** Solve the Bellman optimality equation
  `Q*(s,a) = E[r + γ max_{a'} Q*(s', a')]` by stochastic fixed-point
  iteration.
- **Derivation.** The Bellman optimality operator `T*` is a γ-contraction
  in sup-norm. By Banach fixed point, repeatedly applying `T*` to any
  initial `Q` converges to `Q*`. Online sample-based application of `T*`
  with appropriate step sizes is the Q-learning update
  `Q(s,a) ← Q(s,a) + α (r + γ max_{a'} Q(s', a') − Q(s,a))`.
- **Primitive.** The action-value function `Q : S × A → ℝ`.
- **Theorem.** Under standard Robbins–Monro step sizes and infinite
  visitation, `Q → Q*` almost surely (Watkins & Dayan 1992).

## Policy gradient (Williams 1992; Sutton et al. 1999)

- **Principle.** Maximize expected return `J(θ) = E_τ~π_θ[R(τ)]` directly
  in policy parameter space.
- **Derivation.** The policy gradient theorem:
  `∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) · A^π(s_t, a_t)]`.
  Score-function identity + advantage decomposition; no model of the
  environment required.
- **Primitive.** The score function `∇_θ log π_θ(a|s)` weighted by an
  advantage estimator.
- **Theorem.** Stochastic gradient ascent on `J` converges to a local
  maximum under standard conditions on step sizes and variance.

## TRPO / PPO (Schulman et al. 2015, 2017)

- **Principle.** Maximize a *trust-region surrogate* of expected return:
  largest improvement permitted by a KL constraint between consecutive
  policies. PPO replaces the constraint with a clipped likelihood-ratio
  objective that enforces it implicitly.
- **Derivation.** Kakade–Langford performance-difference lemma gives an
  exact identity for `J(π') − J(π)` in terms of advantages under `π` and
  the state-visitation shift. Bound the visitation shift by a KL
  divergence and you get a monotonically improving lower-bound surrogate.
- **Primitive.** The clipped likelihood ratio `r_t(θ) · A_t` with
  `r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)` clipped to `[1-ε, 1+ε]`.
- **Theorem.** Under exact updates, monotone improvement of `J`
  (Kakade–Langford 2002, Schulman 2015 §3).

## Mirror descent / natural policy gradient (Kakade 2002; Beck–Teboulle 2003)

- **Principle.** Steepest ascent on `J` measured in a geometry that
  respects the policy manifold. KL is the canonical Bregman divergence
  on probability distributions; the natural gradient is its steepest-
  ascent direction.
- **Derivation.** From mirror descent: minimize a linearization of `−J`
  plus a Bregman divergence step penalty. With KL as the Bregman, the
  closed-form solution is multiplicative-weights-style update on the
  policy logits, scaled by the Fisher information matrix.
- **Primitive.** The natural gradient `F(θ)^{-1} ∇_θ J(θ)`.
- **Theorem.** Mirror descent converges at rate `O(1/√T)` for convex
  objectives; the policy mirror descent variant converges to optimal
  policy under tabular conditions (Agarwal et al. 2021).

## AlphaZero (Silver et al. 2017, 2018)

- **Principle.** Self-play policy iteration where MCTS is the policy
  improvement operator and the network is trained to distill the MCTS
  visit distribution and predict the game outcome.
- **Derivation.** MCTS with PUCT exploration produces a search-improved
  policy `π_MCTS` from any prior policy `π_θ`. Train `π_θ` to match
  `π_MCTS` (cross-entropy on visit counts) and `v_θ` to match the
  self-play outcome (MSE on z). At fixed points, search improvement is
  zero — the network has absorbed search.
- **Primitive.** The pair `(π_θ, v_θ)` of a single neural network whose
  outputs are the prior policy and value used to guide MCTS.
- **Theorem.** Empirical: superhuman in Go, chess, shogi from random
  initialization with no domain knowledge beyond rules.

## Soft actor-critic / max-entropy RL (Haarnoja et al. 2018; Ziebart 2010)

- **Principle.** Maximize expected return *plus* policy entropy:
  `J(π) = E[Σ_t r_t + α H(π(·|s_t))]`. Entropy is not a regularizer; it
  is part of the objective.
- **Derivation.** The soft Bellman equation
  `Q_soft(s,a) = r + γ E[V_soft(s')]`,
  `V_soft(s) = α log Σ_a exp(Q_soft(s,a)/α)`,
  has a unique fixed point. The optimal policy is
  `π*(a|s) ∝ exp(Q_soft(s,a)/α)` — the softmax of soft Q.
- **Primitive.** Soft Q-function, with the log-sum-exp value and
  Boltzmann policy as direct consequences.
- **Theorem.** Soft policy iteration converges to the soft-optimal
  policy in tabular settings (Haarnoja et al. 2018, Theorem 1).

## MCTS / UCT (Kocsis & Szepesvári 2006)

- **Principle.** Apply UCB1 regret bounds at every node of a search
  tree, treating each node's children as a multi-armed bandit. Build
  the tree by selectively expanding the most promising paths.
- **Derivation.** UCB1 has logarithmic regret in the bandit setting.
  Recursively applying it down the tree gives an algorithm that
  asymptotically explores like the optimal minimax search while
  respecting a finite simulation budget.
- **Primitive.** The visit-count and value statistics
  `(N(s,a), Q̂(s,a))` at each node, combined into the PUCT/UCB1
  selection score.
- **Theorem.** UCT's value estimate at the root converges to the
  minimax value as the number of simulations grows
  (Kocsis & Szepesvári 2006, Theorem 6).

## GAE — Generalized Advantage Estimation (Schulman et al. 2016)

- **Principle.** Trade off bias and variance in advantage estimation by
  exponentially-weighted averaging of n-step TD residuals.
- **Derivation.** Define `δ_t = r_t + γV(s_{t+1}) − V(s_t)`. The
  GAE(λ) advantage is `Â_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}`. λ=0
  gives the high-bias TD(0) advantage; λ=1 gives the unbiased
  Monte-Carlo advantage; intermediate λ smoothly interpolates.
- **Primitive.** The exponentially-weighted sum of TD residuals.
- **Theorem.** GAE is the unique estimator that is both an exponential
  average of n-step TD residuals and an unbiased estimator of the
  advantage in expectation (Schulman 2016, §3).

---

## What these have in common

- **The principle is one sentence.** Maximize J under a KL trust region.
  Solve a fixed-point equation. Apply UCB1 at every tree node. Distill
  search into the prior. Maximize entropy-augmented return. If you
  can't write the principle in one sentence, you don't have one.

- **The primitive is one object.** Q. The score function. The clipped
  ratio. A neural net that outputs (π, v). The visit-count statistic at
  a search-tree node. **Not** a per-(state-cluster, action, channel)
  tensor with a partial-order indicator on top of it. **Not** a stack
  of cross-attention between three named primitives.

- **There is a theorem.** Banach fixed point. Policy improvement under
  KL constraint. UCT regret bound. They might assume tabular MDPs or
  exact updates, but the theorem is there. Without one, you have a
  heuristic.

- **The math came first.** The implementation followed. None of them
  started from "what tensor can I bucket the experience into?" They
  started from a principle (fixed point, gradient, regret, search,
  entropy) and the algorithm came out.

## What makes a good RL algorithm new

A new entry on this list is one that does at least one of:

- Solves a problem the existing list cannot. (AlphaZero on Go vs. all
  prior search-based methods.)
- Replaces a load-bearing assumption. (PPO replacing TRPO's hard KL
  constraint with a clipped surrogate.)
- Establishes a *new* principle, not a new tensor. (Mirror descent vs.
  vanilla gradient descent.)

A new entry is **not** any of:

- A new bucketing scheme. (Bucket the state by hash, by cluster, by
  policy regime, by exit hash, by terminal hash — none of these are
  algorithms.)
- A new partial-order voting rule on a tensor. (Pareto, Kemeny, strict
  superset, sup-norm, dominance count — none of these are algorithms.)
- A new geometric quantity computed from observations. (Lévy area,
  hull volume, signature, JSD, TV — these are statistics, not
  algorithms.)
- A new offline supervised projection of cumulants. (See the entire
  attempt history.)

If your proposal is one of these last four shapes, it is a heuristic,
not the next AlphaZero. Do not write it down.
