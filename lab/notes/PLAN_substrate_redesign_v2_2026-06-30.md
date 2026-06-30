# Substrate redesign — consolidated plan (v2, 2026-06-30)

This supersedes `lab/notes/PLAN_substrate_redesign_2026-06-30.md`. It
incorporates Codex's red-team
(`lab/notes/codex_redteam_substrate_redesign_2026-06-30.md`) and
counter-design (`lab/notes/codex_counterdesign_substrate_2026-06-30.md`),
and the user's three follow-up decisions:

1. Adopt Codex's counter-design as the new plan.
2. Implement all three tiers; defer Large (H=10k) baselines.
3. Update mission text (CLAUDE.md, docs/LAB.md, system prompts) to
   drop AlphaZero / MCTS in the same redesign.

## What is being built

Two families, three tiers each, plus updated mission text.

### Family A — `RecoverableKeyFuelMaze`

Continuous-control with genuine long-horizon coupling. Replaces
`RecoverablePointMaze-*-v0`.

- **Action**: `Box([-1, 1]^D)`. World samples a deterministic
  `A_world ∈ R^{2×D}` actuator matrix; physical force is
  `clip(A_world @ action, max_force)`. Actuator usage also incurs
  heat and energy drain through world-specific positive weights, so
  different action vectors that produce the same force differ in
  fuel/heat cost. This makes `D` behaviorally relevant, not just
  syntactic.
- **Observation**: position, velocity, fuel, heat, damage, repair
  debt, inventory of keys/seals, current gate-phase features for
  *known* gates, egocentric ray distances to walls/hazards, relative
  vectors to a small set of known landmarks, compact route manifest
  of remaining required seals, previous-action summary, `t/H`.
  Deliberately *not* a global occupancy grid. Small tier may include
  more global debug features.
- **Dynamics**: damped point mass with collisions (recoverable, not
  terminating). Fuel decreases with distance, actuator effort, heat,
  and collision damage. Fuel stations recharge a bounded amount then
  cool down or deplete. Repair pads convert time → reduced damage
  with diminishing returns. Keys collected by dwelling in key
  regions; doors open with required inventory OR during open phase
  of timed gate. Some corridors are one-way or high-friction (never
  unrecoverable). Final seals require visiting regions with the
  right inventory and remaining fuel/heat margin.
- **Long-horizon coupling** (six independent mechanisms):
  - Fuel is a global budget; recharge choices are local.
  - Key order changes which corridors are available later.
  - Timed gates couple early route speed to later access.
  - Heat/damage create hidden opportunity cost of aggressive
    early control.
  - Some fuel stations are behind doors → route + resource planning
    are coupled.
  - Final objective is a *set* of obligations, not a single endpoint.
- **Terminal vector** (9 components, all larger-is-better):
  `(success, seal_completion, key_coverage, fuel_margin, neg_damage,
  neg_lateness, neg_energy, neg_collision, route_efficiency)`.
- **Tiers**:
  - Small: H=500, D=16, ~24×24 units, 3–5 rooms, 2 key types, 2
    seals, 1 gate, 2 fuel stations, 1 repair pad.
  - v0: H=2000, D=32, ~48×48, 8–12 rooms, 4 key types, 5–7 seals,
    3–5 gates, 4–6 fuel, 2 repair pads.
  - Large: H=10000, D=64, ~96×96, 20–30 rooms, 6–8 key types,
    12–18 seals, 8–12 gates, 8–12 fuel, 4 repair pads.

### Family B — `RecoverableCapacityScheduling`

Allocation/scheduling with multiple non-trivial cross-time
couplings. Replaces `RecoverableResourceAllocation-*-v0`.

- **Action**: `Box([-1, 1]^D)`, deterministically decoded into:
  - per-project allocation logits (K projects)
  - per-mode allocation logits (M production modes)
  - per-mode maintenance intensity
  - per-mode setup-change intensity
  - per-product buffer/inventory release intensity
  - softmax/projection enforces capacity budgets while preserving
    smoothness.
- **Observation**: per-project (backlog, cumulative service, soft-
  deadline slack, priority, quality shortfall, next-window summary);
  per-mode (capacity, setup mixture, wear, heat, maintenance debt,
  recent utilization); per-product (inventory, age/perishability,
  reserved commitments); aggregate future-demand sketches at
  multiple horizons (next 16/64/256 steps — *not* full 10k-step
  table); previous-action aggregates; `t/H`.
- **Dynamics**: production for project k depends on compatible
  modes, setup alignment, wear, heat, inventory, quality state.
  Mode wear grows with utilization and aggressive setup changes.
  Maintenance reduces wear but consumes capacity. Heat decays slowly
  and lowers effective capacity if accumulated. Setup state has
  inertia — switching product families takes many steps. Inventory
  can be pre-built but is capacity-limited and may perish. Contracts
  have prerequisite chains/bundle requirements. Failing one project
  can lower related-project value but never ends the episode.
  Backlog can be partially recovered later with lateness penalty.
- **Long-horizon coupling** (six mechanisms again):
  - Current allocation → future mode wear and heat.
  - Skipping maintenance early raises future production cost.
  - Setup changes have multi-hundred-step consequences at v0/Large.
  - Inventory built too early may perish; too late misses
    deadlines.
  - Contract bundles make service order consequential.
  - Project priorities differ → a uniform fill-rate policy is
    dominated.
- **Terminal vector** (11 components, all larger-is-better):
  `(success, weighted_fill_rate, mandatory_fill_rate, neg_lateness,
  neg_shortfall_tail, neg_wear, neg_heat_violation, neg_setup_churn,
  neg_inventory_waste, neg_energy, resilience_margin)`.
- **Tiers**:
  - Small: H=500, D=32, K=16, M=4, P=4, 2 mandatory bundles,
    setup ~20–50 steps, maintenance visible in 100–200 steps.
  - v0: H=2000, D=64, K=48, M=8, P=8, 6–10 bundles,
    setup ~100–300 steps, maintenance visible in 300–800 steps.
  - Large: H=10000, D=128, K=128, M=16, P=16, 20–30 bundles,
    setup ~500–1500 steps, maintenance visible in 1000–4000 steps.

### Substrate-wide changes

- **Held-out evaluation is now a first-class substrate property.**
  Train seeds vs validation seeds vs held-out seeds, defined per
  tier. The env contract documents what varies with seed. Small
  includes a tiny fixed debug seed set for tests.
- **Deterministic seed → world** preserved exactly as before.
  Generalization pressure comes from held-out worlds, not from
  transition noise.
- **Terminal-only reward** preserved. The per-step structure
  aggregates into the terminal vector; no shaping in env.
- **Reward orientation** preserved: all components larger-is-better;
  costs appear as `neg_*`.
- **No baseline RL libraries**, NumPy + optional PyTorch only.

### Baseline portfolios

Each family ships with multiple cheap policies plus a decomposition
diagnostic. *No single heuristic is the difficulty signal.*

**Maze**:
- `ZeroPolicy`, `RandomPolicy` (sanity)
- `LocalWallFollowerPolicy` (geometry-only)
- `GreedyNearestLandmarkPolicy` (myopic baseline)
- `KeyDoorRulePolicy` (hand-coded FSA over visible requirements)
- `FuelAwareGreedyPolicy` (greedy + fuel reserve thresholds)
- `GraphOracleRoute + LowLevelPD` (privileged-info upper-bound
  diagnostic, not a learner baseline)
- `ShortHorizonCEM/MPC` over 50/100/250-step windows
  (decomposition diagnostic — if it solves Large, the long-horizon
  claim is false)

**Scheduling**:
- `ZeroPolicy`, `RandomPolicy`, `UniformAllocationPolicy`
- `EarliestDeadlinePolicy`
- `BacklogPriorityPolicy` (backlog × priority / compat-cost)
- `MaintenanceThresholdPolicy` (wear/heat-aware EDP)
- `SetupAwareGreedyPolicy`
- `InventoryBufferPolicy` (fixed safety-stock rule for forecast
  bursts)
- `ShortHorizonRolloutPolicy` over 25/100/500-step horizons
  (decomposition diagnostic)
- `RelaxedOracle` on Small only, NumPy + simple search (headroom
  estimator; not a learner baseline)

### Mission text update (same commit)

- `CLAUDE.md`: replace "Q-learning, PPO, AlphaZero, mirror descent,
  SAC, MCTS, GAE" with "PPO, SAC, CEM, mirror descent, GAE-style
  credit assignment, trajectory-level vector-reward methods." Add a
  one-line note that AlphaZero/MCTS-class algorithms need a
  discrete-action substrate the lab does not currently provide.
- `docs/LAB.md`: same change in the "What this lab is" section, plus
  a pointer to `lab/notes/codex_counterdesign_substrate_2026-06-30.md`
  for the rationale.
- `lab/prompts/claude_system.md`, `lab/prompts/codex_system.md`:
  same change wherever the algorithm class is named.
- No other doc edits in the same commit beyond what the substrate
  redesign forces (`docs/SUBSTRATE_MAP.md`, `DESIGN.md`,
  `docs/baseline_report.md`).

## Acceptance gates (the implementation contract)

The substrate is not merged until every one of these passes. Each
gate has a test or a probe that produces a falsifiable result.

1. **Determinism gate.** Same seed + same action sequence → same
   terminal vector exactly (mod float epsilon). Different seeds →
   measurably different worlds.
2. **Terminal-only gate.** `reward == 0` for every non-terminal
   step. Diagnostic traces may exist in `info` but learners don't
   read them.
3. **Feasibility gate.** Privileged diagnostic or conservative
   oracle solves a high fraction of generated Small/v0 worlds.
   Failure = generator bug, not difficulty.
4. **No-idle-tail gate.** For baseline + oracle trajectories,
   meaningful state changes / obligations occur throughout the
   episode. The final 25% of horizon must affect terminal-vector
   components on most worlds.
5. **Lookahead-depth gate.** Short receding-horizon policies
   underperform longer-horizon diagnostics. Maze: 50/100-step MPC
   does not match graph-level planning on v0/Large. Scheduling:
   100-step rollout does not match 500/1000-step rollout when
   maintenance/setup coupling is active.
6. **Myopic-gap gate.** Greedy nearest-target / earliest-deadline
   policies leave significant headroom in *multiple* terminal
   vector components, not just an arbitrary scalar weight.
7. **Recoverability gate.** Inject perturbations at early, middle,
   late times. Terminal vector degrades *gradedly* with policy
   strength. No single mistake = total collapse, no single mistake
   = zero effect.
8. **Action-complexity gate.** Top-1/top-2 action sparsification
   does *not* preserve most return in v0/Large. Appending no-op
   dims is forbidden; actuator/allocation dims must have
   measurable behavioral effect.
9. **Seed-generalization gate.** Policies tuned on train seeds are
   evaluated on held-out worlds. A large train→held-out gap is a
   reportable warning. A policy that hard-codes one world fails
   the gate.
10. **Reward-normalization gate.** Component magnitudes are
    comparable across Small/v0/Large after normalization. No cost
    grows linearly with H unless explicitly normalized.
11. **Baseline-portfolio gate.** Each family ships with ≥6 cheap
    policies plus ≥1 decomposition diagnostic. No one heuristic is
    the sole difficulty signal.
12. **Runtime gate.** Small episodes run quickly enough for smoke
    tests (~30s wall-clock per episode upper bound). Large episodes
    run within ~5 minutes per episode on a laptop CPU.

## Workflow and scope

- **Branch off master**, not lab/auto. The lab loop pauses while
  master is moving.
- **Tier rollout**: Small + v0 + Large all get implemented. Large's
  full baseline sweep (including CEM and the decomposition
  diagnostic across all seeds) is deferred to a separate follow-up
  session. Small + v0 baselines are run as part of this redesign.
- **What the baseline report covers in this round**: random,
  heuristic portfolio, and ShortHorizonRollout/MPC diagnostic on
  Small + v0 of each family. CEM on Small + v0 (Codex flagged CEM
  on Large as expensive and low-signal — agreed). Large is built
  but baselines are TODO.
- **Tests**: substrate tests under `tests/` extended to cover the
  new dynamics, including the determinism, terminal-only,
  feasibility, and seed-generalization gates. Expect ~50 tests
  total when done (current is 25).
- **Docs touched**: `DESIGN.md`, `docs/SUBSTRATE_MAP.md`,
  `docs/baseline_report.md`, `docs/AGENT_GUIDE.md` (env IDs +
  recommended first targets), plus the mission-text updates listed
  above.

## Things this plan deliberately does not do

- No discrete or hybrid action spaces. The lab's mission is
  narrowed to continuous-action algorithms.
- No environment stochasticity. Determinism is preserved;
  generalization pressure comes from held-out worlds.
- No partial observability beyond the lossy-future-summary
  observations described above.
- No routing/VRP family — `Box([0,1]^N)` over nodes is fluid flow,
  not routing. Codex's red-team argued and I agree.
- No vendored OR solver. The closest we get is a NumPy-only
  `RelaxedOracle` on Small instances for headroom estimation.
- No commits to master until the user reviews this plan and the
  acceptance gates pass on the implementation.

## Implementation order (when approved)

1. Update mission text in CLAUDE.md / docs/LAB.md / system prompts.
2. Build `RecoverableCapacityScheduling` family + tests (it's the
   harder of the two and informs the substrate generator
   conventions).
3. Build `RecoverableKeyFuelMaze` family + tests.
4. Build the baseline portfolios for both families.
5. Wire env IDs into `registered_envs()`; delete the old
   `RecoverablePointMaze-*` and `RecoverableResourceAllocation-*`
   registrations.
6. Run Small + v0 baseline sweep → new `docs/baseline_report.md`.
7. Update `docs/SUBSTRATE_MAP.md`, `DESIGN.md`,
   `docs/AGENT_GUIDE.md`.
8. Verify all 12 acceptance gates with a final test pass.
9. Single curated commit to master.

## What the user is approving when they approve this plan

- The two families, their action/observation/dynamics contracts,
  their tier ladders.
- The acceptance gates as the implementation contract.
- The mission narrowing (drop AlphaZero/MCTS).
- The deferral of Large baselines (Large envs *are* built; their
  full baselines come later).
- The expected scope: ~1500–2500 lines of substrate + baselines +
  tests + docs; ~10–30 minutes of baseline wall-clock for the
  Small+v0 sweep.
