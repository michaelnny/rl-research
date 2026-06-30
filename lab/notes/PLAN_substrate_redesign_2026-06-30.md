# Proposed substrate redesign — plan synopsis

This is the plan a reviewer has been asked to red-team. It is a
proposal, not a commitment. Nothing has been implemented.

## Trigger

The user pointed out that the current substrate is not honestly long-
horizon or action-complex:

- Horizons in the registry: 60–180 steps (`docs/SUBSTRATE_MAP.md`).
  Atari is 2,700–10,800; MuJoCo locomotion is 1,000.
- Action dims in the registry: 2, 5, 8. AlphaZero's action space is
  362 (Go).
- The lab's mission talks about finding algorithms in the class of
  GAE / MCTS / AlphaZero, which is the class of problem where horizon
  and structured actions are the defining hardness axes.

User's target: H=5k–10k, action dim "more complex" (clarified to mean
continuous 32–128, not hybrid/combinatorial).

## What the user decided

Across three follow-up questions:

1. **Horizon**: "Mixed — 1k baseline, 10k stretch." Smoke variants
   stay short for cheap iteration; canonical / Large jump to 10k.
2. **Action complexity**: "Higher continuous dim (32–128)" only.
   User declined discrete / hybrid / combinatorial after I clarified
   that those require new env classes, not just config bumps.
3. **Substrate scope**: "Redesign in place."
4. **Long-horizon structure**: "Drop maze, build a new family."
5. **Baseline rebuild**: "Full rebuild including CEM."
6. **Workflow**: "Plan first, implement after approval."

## What I would change

### Drop the maze family

`RecoverablePointMaze-{Small, v0, HD}-v0` — retired. Rationale:

- Single-goal-then-idle dynamics. Once the agent reaches the goal,
  the remaining horizon is structureless.
- Multi-waypoint patches don't earn the long-horizon label; they
  stitch N short-horizon problems together.
- The spatial-control flavor is a smaller fraction of the algorithm-
  class problem than the allocation/scheduling flavor that resource
  already does well.

### Redesign the resource family

`RecoverableResourceAllocation-{Small, v0, Large}-v0` keeps its
identity but the dynamics change:

- **Rolling demand schedule**: each of K projects has a periodic
  demand schedule across the horizon (peaks at deterministic times).
  Allocations only "count" when delivered into a demand window. The
  agent is decision-saturated for the full horizon.
- **Service-level vector reward** becomes a window-weighted sum of
  demand met over time, not an end-state read.
- **K bumped**: Small → 8 projects, v0 → 32, Large → 128.
- **Horizons**: Small → 500, v0 → 1000, Large → 10000.

### Add a new family: inventory routing / logistics

Tentative name: `RecoverableInventoryRouting-{Small, v0, Large}-v0`.

- A fleet (1–4 agents, depending on tier) services demand across an
  N-node graph (N=20, 50, 128 across tiers).
- At each step, the action is `Box([0,1]^N)` interpreted as the
  fraction of fleet capacity to release toward each node this step.
  Travel time on graph edges is deterministic.
- Demand at each node accumulates over time; missing a node's demand
  window costs `neg_lateness`.
- Terminal vector: `(success, fill_rate, neg_distance, neg_lateness, neg_overstock)`.
- Horizons: Small 500, v0 1000, Large 10000.

### Substrate boundary unchanged

- Still terminal-only reward (the window/service signals aggregate
  into terminal vector; the env still returns zero per step).
- Still deterministic given seed.
- Still no per-step shaping rewards in the env.
- Still NumPy / optional PyTorch only for baselines.

### Baselines

- `RandomPolicy`, `ZeroPolicy` unchanged (still apply to any env).
- `ResourceGreedyPolicy` extended to be demand-window-aware. A
  greedy-but-not-trivial heuristic is the right difficulty signal.
- New `InventoryNearestUrgentPolicy` for the routing family.
- `train_cem`, `train_reinforce` unchanged structurally; they will
  just take 10× longer at H=10k.

### Re-run

- Full `experiments/run_baselines.py` sweep on the new envs.
- New `docs/baseline_report.md`.
- Updated `DESIGN.md` (environment contract section).
- Updated `docs/SUBSTRATE_MAP.md`.
- Tests under `tests/` extended to cover the new dynamics
  (deterministic resets, terminal-only reward at H=10k, demand-window
  accounting, graph topology determinism).
- `tests/` should remain green at 25+ tests.

### Workflow

- Branch off master, not lab/auto.
- Plan reviewed (this document + Codex red-team) before any code is
  written.
- After implementation, the user reviews before merge.
- Loop on `lab/auto` is paused while master is moving.

## What this plan does NOT do

- Does not introduce discrete or hybrid action spaces. The user
  declined this. MCTS / AlphaZero-style algorithms are therefore
  structurally disadvantaged; the lab's mission statement claims
  to want algorithms in that class but the substrate would not
  admit their natural action space.
- Does not change the substrate's "no per-step reward shaping" rule.
- Does not introduce stochasticity in the environment (open question,
  leaning toward keeping fully deterministic).
- Does not introduce partial observability.

## Open design questions

- Should the demand schedule be visible in the observation, or
  inferable from history only? Visible = planning problem;
  inferable = real RL.
- For the routing family, should the graph be the same topology
  across resets (per tier) or seeded per reset?
- CEM at H=10k will be very slow. Is the result informative, or
  just expensive?
- "Recoverability" — does it still mean the same thing at H=10k as
  it did at H=120, where one bad step can be undone in 10 more?
  At H=10k, a bad step at t=100 may be effectively unrecoverable
  by t=9000 simply because the relevant demand window is gone.

These are exactly the kinds of things the red-team should hammer on.
