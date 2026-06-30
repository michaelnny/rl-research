# Counter-design — substrate 2026-06-30

## Thesis

This substrate should expose a specific deficiency in continuous-action RL algorithms: their tendency to solve the locally obvious control problem while failing to preserve latent optionality for decisions whose consequences are only visible thousands of steps later. I would not optimize for horizon length or action dimension as labels. I would optimize for delayed coupling: energy, setup, inventory, wear, route order, and commitments that remain recoverable but make early choices measurably change the feasible Pareto frontier at the end of the episode. The narrowed mission is therefore: terminal-only, deterministic, continuous-control worlds where PPO/SAC/CEM/mirror-descent/GAE-style methods must learn policies that manage long-lived state, not just repeat a myopic controller inside a long rollout.

## Families

### Family A: RecoverableKeyFuelMaze

- Mission relevance:
  - This is the spatial continuous-control family the lab should keep.
  - It tests embodied long-horizon credit assignment rather than pure dispatch.
  - It discriminates between policies that can navigate locally and policies that can manage route order, energy, inventory, and timed access over a long episode.
  - It is especially useful for PPO/SAC/GAE-style methods because the observation is dense in physical state but the reward is terminal-only.
  - It is useful for CEM/mirror-descent methods because the action space is high-dimensional and redundant, so naive open-loop search should struggle on held-out worlds.

- Name:
  - `RecoverableKeyFuelMaze-Small-v0`
  - `RecoverableKeyFuelMaze-v0`
  - `RecoverableKeyFuelMaze-Large-v0`

- Core idea:
  - A point-mass robot moves in a continuous 2-D maze.
  - The robot must visit a set of semantic regions before the horizon ends.
  - Regions include fuel stations, key shrines, locks, final seals, repair pads, and timed gates.
  - The final score depends on the set and order of obligations completed, the residual energy, damage, route efficiency, and lateness.
  - No per-step reward is emitted.
  - The long-horizon challenge is not "go to waypoint 1, then waypoint 2".
  - The challenge is to choose a route order that preserves enough fuel, unlocks later doors, catches gate phases, and avoids accumulating damage that lowers terminal capacity.

- Action space:
  - `Box(low=-1.0, high=1.0, shape=(D,), dtype=float32)`.
  - The action is not directly `(ax, ay)`.
  - Each world samples a deterministic actuator matrix `A_world in R^{2 x D}`.
  - Force is `clip(A_world @ action, max_force)`.
  - Actuator use also generates heat and energy drain through world-specific positive weights.
  - This makes `D` syntactically and behaviorally relevant without changing the continuous-control nature.
  - Redundant actuators mean multiple actions can produce similar force but different heat/fuel costs.
  - Tiers:
    - Small: `D=16` for smoke speed.
    - v0: `D=32`.
    - Large: `D=64`.
  - I would not push maze to `D=128` initially; resource allocation can carry the highest-dimensional action stress.

- Observation:
  - A flat `Box` observation.
  - Contents:
    - normalized time remaining;
    - position and velocity;
    - current fuel, heat, damage, and repair debt;
    - binary/continuous inventory vector for keys and seals;
    - door/gate phase features for gates currently known to the agent;
    - egocentric ray distances to walls and hazards, e.g. 32/48/64 rays by tier;
    - relative vectors to a small public set of known landmarks;
    - a compact route manifest: remaining required seal types and key requirements;
    - previous action summary for actuator hysteresis.
  - The observation should not include a full global occupancy grid for v0/Large.
  - Small may include more global debugging features.
  - The policy gets enough information to act, but not a free full-world shortest-path table.
  - The generator and seed contract remain deterministic; the anti-memorization mechanism is held-out worlds, not stochastic transitions.

- Dynamics:
  - State includes position, velocity, fuel, heat, damage, inventory, door states, gate phases, and visited-region counters.
  - Movement is deterministic Euler integration with wall collision and friction.
  - Collisions add damage and may rebound the robot but do not terminate the episode.
  - Fuel decreases with distance, actuator effort, heat, and collision damage.
  - Fuel stations recharge a bounded amount and then enter a long cooldown or deplete permanently.
  - Repair pads convert time/fuel into reduced damage with diminishing returns.
  - Keys are collected by dwelling in a key region for enough continuous time.
  - Doors open only if the inventory requirement is met, or if a timed gate is in an open phase.
  - Some corridors are one-way or high-friction, but never create literal unrecoverable death.
  - Timed gates have deterministic periods and phases sampled by seed.
  - Gate misses cost waiting, detouring, or fuel; they should not hard-fail the episode.
  - Final seals require visiting regions with the right inventory and sufficient remaining fuel/heat margin.

- Long-horizon coupling:
  - Fuel is a global budget with local recharge choices.
  - Key order changes which corridors are available later.
  - Timed gates couple early route speed to later access.
  - Heat/damage create hidden opportunity cost for aggressive early control.
  - Repair and recharge decisions consume time that may move the agent out of gate phase.
  - Some fuel stations are behind doors, so route planning and resource planning are coupled.
  - The final objective is a set of obligations, not a single endpoint.
  - A policy that greedily moves toward the nearest unvisited target should be visibly suboptimal.
  - A policy that follows a precomputed route for one seed should fail to generalize to held-out layouts.
  - A 10k-step episode should contain changing strategic pressure, not 1k useful steps plus 9k idle.

- Terminal vector:
  - `success`: 1 if all required seals are completed and the robot ends in an extraction zone; else 0.
  - `seal_completion`: completed required seals divided by required seals.
  - `key_coverage`: collected useful keys divided by keys needed for an oracle route.
  - `fuel_margin`: final fuel normalized to `[0,1]`, with a small negative extension for fuel-starved crawling.
  - `neg_damage`: negative normalized damage.
  - `neg_lateness`: negative average normalized lateness of seal completions relative to soft deadlines.
  - `neg_energy`: negative actuator energy divided by a tier-specific oracle-normalized scale.
  - `neg_collision`: negative collision impulse/damage normalized by horizon.
  - `route_efficiency`: oracle lower-bound path length divided by actual traversed path length, clipped to `[0,1]`.
  - All components are terminal-only.
  - Default scalarization may exist, but reports must show vector components.

- Tiers:
  - Small:
    - horizon `H=500`;
    - action dim `D=16`;
    - map size about `24 x 24` continuous units;
    - 3-5 rooms;
    - 2 key types;
    - 2 seals;
    - 1 timed gate;
    - 2 fuel stations;
    - 1 repair pad;
    - target runtime well under 30 seconds per episode in pure Python, normally far below that.
  - v0:
    - horizon `H=2000`;
    - action dim `D=32`;
    - map size about `48 x 48`;
    - 8-12 rooms;
    - 4 key types;
    - 5-7 seals;
    - 3-5 timed gates;
    - 4-6 fuel stations;
    - 2 repair pads;
    - enough branching that route order matters.
  - Large:
    - horizon `H=10000`;
    - action dim `D=64`;
    - map size about `96 x 96`;
    - 20-30 rooms;
    - 6-8 key types;
    - 12-18 seals;
    - 8-12 timed gates;
    - 8-12 fuel stations;
    - 4 repair pads;
    - target runtime under about 5 minutes per episode on a laptop CPU, with vectorized ray casting if needed.

- World generator:
  - Seed samples room graph, room geometry, wall layout, landmark positions, key placement, seal requirements, fuel-station capacities, gate phases, and actuator matrix.
  - The generator must reject worlds that fail a cheap oracle-feasibility check.
  - The oracle check should be conservative and graph-level, not a full optimal controller.
  - Small can have a fixed public debug world in addition to seeded worlds.
  - v0 and Large should never mean one fixed public layout.

- Failure modes considered and rejected:
  - Rejected: simple multi-waypoint maze with visible next target.
    - It decomposes into repeated local navigation.
  - Rejected: single goal with huge horizon.
    - It creates idle-tail artifacts.
  - Rejected: irreversible death on early route mistakes.
    - It violates recoverability and turns learning into brittle avoidance.
  - Rejected: full global map plus full future gate schedule as a shortest-path planning benchmark.
    - It would be a deterministic planner test more than an RL substrate.
  - Rejected: hidden deterministic map with one training seed.
    - It would invite memorization.
  - Rejected: making `D=128` by appending no-op actuators.
    - It would fake action complexity.

- Baseline portfolio:
  - `ZeroPolicy`:
    - Confirms terminal success is not accidental.
  - `RandomPolicy`:
    - Measures exploration rarity and collision/fuel scaling.
  - `LocalWallFollowerPolicy`:
    - Tests whether geometry alone solves too much.
  - `GreedyNearestLandmarkPolicy`:
    - Moves toward the nearest useful visible region.
    - Should get partial completion but not solve v0/Large.
  - `KeyDoorRulePolicy`:
    - Hand-coded finite-state route rule using visible key/door requirements.
    - Useful as a non-learning lower bound.
  - `FuelAwareGreedyPolicy`:
    - Greedy target selection with fuel reserve thresholds.
    - Tests whether resource coupling is shallow.
  - `GraphOracleRoute + LowLevelPD` diagnostic:
    - Uses privileged graph information to estimate an upper bound.
    - Should not be a normal learner baseline.
    - It is important for validating that worlds are feasible.
  - `ShortHorizonCEM/MPC` diagnostic:
    - Plans over 50/100/250-step windows.
    - If it solves Large, the long-horizon claim fails.

### Family B: RecoverableCapacityScheduling

- Mission relevance:
  - This is the allocation/scheduling-flavored family the lab needs.
  - It keeps the spirit of the existing resource environment but removes the easy completion-then-idle structure.
  - It targets continuous high-dimensional allocation, long-lived capacity, delayed maintenance, and terminal vector trade-offs.
  - It is designed so a rolling demand window policy is not enough.
  - It should discriminate between myopic throughput maximization and policies that preserve future productive capacity.

- Name:
  - `RecoverableCapacityScheduling-Small-v0`
  - `RecoverableCapacityScheduling-v0`
  - `RecoverableCapacityScheduling-Large-v0`

- Core idea:
  - There are `K` contracts/projects and `M` production modes/resources.
  - Each project has a deterministic demand calendar, deadline profile, quality requirement, and compatibility with production modes.
  - Each mode has capacity, setup state, wear, heat, inventory buffers, and maintenance options.
  - Actions continuously allocate capacity among projects, setup changes, buffer production, and maintenance.
  - Demand is not just a rolling independent window because serving current demand changes future capacity, setup, inventory, and wear.
  - The terminal vector scores total service, lateness, contract priority, safety, inventory waste, energy, setup churn, and residual capacity.

- Action space:
  - `Box(low=-1.0, high=1.0, shape=(D,), dtype=float32)`.
  - Internally transformed deterministically into:
    - project allocation logits for `K` projects;
    - mode allocation logits for `M` resources;
    - maintenance intensity for each mode;
    - setup-change intensity for each mode;
    - buffer/inventory release intensity for each product class.
  - A soft projection enforces capacity budgets but preserves continuous differentiability-like behavior.
  - Actions can be dense: allocating a little to many projects is legal but incurs coordination/setup costs.
  - Tiers:
    - Small: `D=32`.
    - v0: `D=64`.
    - Large: `D=128`.

- Observation:
  - A flat `Box` observation.
  - Contents:
    - normalized time and phase features;
    - per-project backlog, cumulative service, soft-deadline slack, priority, quality shortfall, and next-window summary;
    - per-mode capacity, setup mixture, wear, heat, maintenance debt, and recent utilization;
    - per-product inventory, age/perishability, and reserved commitments;
    - aggregate future-demand sketches over multiple horizons, e.g. next 16/64/256 steps, not a full 10k-step table;
    - previous action aggregates for switching-cost dynamics.
  - The observation should be Markov or close to Markov with respect to the exposed summary.
  - If exact full calendars are hidden, that is acceptable only because train/eval worlds are split.
  - The benchmark should document exactly what future summaries are visible.

- Dynamics:
  - Each step converts action into production, maintenance, setup movement, and inventory release.
  - Production for project `k` depends on compatible modes, current setup alignment, wear, heat, inventory availability, and quality state.
  - Mode wear increases with utilization and aggressive setup changes.
  - Maintenance reduces wear but consumes capacity and may require setup downtime.
  - Heat decays slowly and lowers effective capacity if allowed to accumulate.
  - Setup state has inertia; changing from one product family to another takes many steps.
  - Inventory can be built ahead, but it is capacity-limited and may perish or become obsolete.
  - Some contracts have prerequisite chains or bundle requirements.
  - Failing one project can lower the value of related projects, but not end the episode.
  - Backlog can be partially recovered later with lateness penalties.
  - Demand calendars and project compatibility are sampled by seed and deterministic thereafter.

- Long-horizon coupling:
  - Current allocation changes future mode wear and heat.
  - Maintenance skipped early raises future production cost or lowers capacity.
  - Setup changes have multi-hundred-step consequences at v0/Large.
  - Inventory built too early may perish; inventory built too late misses deadlines.
  - Contract bundles make service order consequential.
  - Project priorities are not identical, so a uniform fill-rate policy can be dominated.
  - Future demand summaries are lossy; policies must learn robust allocation rules across worlds.
  - A greedy earliest-deadline policy should be good enough to be informative but should leave obvious terminal-vector headroom.
  - A short receding horizon should fail when maintenance/setup debt only becomes costly after its lookahead window.

- Terminal vector:
  - `success`: 1 if all mandatory contracts meet minimum service and safety thresholds; else 0.
  - `weighted_fill_rate`: priority-weighted served demand divided by priority-weighted total demand.
  - `mandatory_fill_rate`: fill rate on mandatory contracts only.
  - `neg_lateness`: negative normalized weighted lateness.
  - `neg_shortfall_tail`: negative penalty for demand still unserved at horizon.
  - `neg_wear`: negative normalized terminal wear and maintenance debt.
  - `neg_heat_violation`: negative normalized overheating exposure.
  - `neg_setup_churn`: negative normalized setup movement and switching cost.
  - `neg_inventory_waste`: negative obsolete/perished/unused inventory.
  - `neg_energy`: negative energy/capacity expenditure normalized by a feasible-oracle scale.
  - `resilience_margin`: normalized residual effective capacity after satisfying commitments.
  - Components should be magnitude-comparable across tiers after normalization.

- Tiers:
  - Small:
    - horizon `H=500`;
    - action dim `D=32`;
    - projects `K=16`;
    - modes/resources `M=4`;
    - product families `P=4`;
    - 2 mandatory contract bundles;
    - setup time constants around 20-50 steps;
    - maintenance effects visible within 100-200 steps.
  - v0:
    - horizon `H=2000`;
    - action dim `D=64`;
    - projects `K=48`;
    - modes/resources `M=8`;
    - product families `P=8`;
    - 6-10 contract bundles;
    - setup time constants around 100-300 steps;
    - maintenance effects visible after 300-800 steps.
  - Large:
    - horizon `H=10000`;
    - action dim `D=128`;
    - projects `K=128`;
    - modes/resources `M=16`;
    - product families `P=16`;
    - 20-30 contract bundles;
    - setup time constants around 500-1500 steps;
    - maintenance effects visible after 1000-4000 steps.

- World generator:
  - Seed samples project calendars, priorities, compatibility matrix, setup graph, capacity profiles, initial wear, maintenance efficiency, inventory perishability, and bundle/prerequisite graph.
  - The generator should include regimes:
    - smooth demand worlds;
    - bursty demand worlds;
    - maintenance-critical worlds;
    - setup-critical worlds;
    - inventory-critical worlds.
  - Evaluation should stratify across these regimes.
  - Worlds that are impossible under a simple relaxed feasibility check should be rejected.
  - Worlds that are solved by a one-step greedy check should also be rejected for v0/Large.

- Failure modes considered and rejected:
  - Rejected: independent rolling demand windows.
    - They decompose into repeated short-horizon dispatch.
  - Rejected: fixed demand schedule per env ID.
    - It invites hard-coded calendars.
  - Rejected: pure fluid allocation with no setup, wear, or inventory state.
    - It becomes a convex-ish rate control problem.
  - Rejected: one irreversible missed deadline causing terminal failure.
    - It violates recoverability.
  - Rejected: full future demand table in observation for all 10k steps.
    - It turns the task into deterministic offline planning.
  - Rejected: hidden single world with no held-out seeds.
    - It turns the task into memorization.

- Baseline portfolio:
  - `ZeroPolicy`:
    - Confirms no free service.
  - `RandomPolicy`:
    - Measures action-space difficulty and safety scaling.
  - `UniformAllocationPolicy`:
    - Detects whether contracts are too homogeneous.
  - `EarliestDeadlinePolicy`:
    - Myopic urgency baseline.
  - `BacklogPriorityPolicy`:
    - Allocates by backlog times priority divided by compatibility cost.
  - `MaintenanceThresholdPolicy`:
    - Adds simple wear/heat thresholds to backlog priority.
  - `SetupAwareGreedyPolicy`:
    - Penalizes allocations that require large setup moves.
  - `InventoryBufferPolicy`:
    - Builds inventory for forecast bursts with a fixed safety-stock rule.
  - `ShortHorizonRolloutPolicy`:
    - Greedy rollout over 25/100/500-step horizons depending on tier.
    - This is the key decomposition diagnostic.
  - `RelaxedOracle` diagnostic:
    - Solves a coarse aggregate planning relaxation on Small only, if implementable with NumPy and simple search.
    - It estimates headroom; it is not required as a training baseline.

## Evaluation distribution

- Determinism contract:
  - `reset(seed=s)` must deterministically generate the same world and initial state for that tier.
  - Given the same action sequence, the same terminal vector must result exactly up to floating point tolerance.
  - No transition stochasticity is introduced.
  - The seed is the world.
  - Determinism should be tested at the world-descriptor level and rollout level.

- What is fixed across resets:
  - Tier config.
  - Reward component definitions and normalization constants.
  - Action and observation schema.
  - Generator family and parameter ranges.
  - Baseline policy code.
  - Public train/eval seed lists once published.

- What varies with seed:
  - Maze:
    - room graph;
    - geometry;
    - obstacle placement;
    - key/seal/fuel/repair placement;
    - gate periods and phases;
    - actuator matrix and actuator costs.
  - Scheduling:
    - project calendar;
    - priorities;
    - compatibility matrix;
    - setup graph;
    - capacity profile;
    - initial wear/heat/inventory;
    - maintenance efficiency;
    - contract bundle graph.

- Train/eval split:
  - Small:
    - include a tiny fixed debug set, e.g. seeds `0..9`;
    - include held-out smoke seeds, e.g. `1000..1099`.
  - v0:
    - public train seeds, e.g. `0..999`;
    - public validation seeds, e.g. `1000..1199`;
    - held-out evaluation seeds, e.g. `10000..10199`.
  - Large:
    - public development seeds, e.g. `0..99`;
    - held-out evaluation seeds, e.g. `20000..20049`;
    - optional hidden-later challenge seeds if the lab ever needs a leaderboard-like evaluation.

- Held-out evaluation definition:
  - A result is not considered substrate-valid if it is reported only on train seeds.
  - Held-out evaluation means new generated worlds, not new initial noise in the same world.
  - The environment generator is public; the specific held-out seed list can be public for reproducibility.
  - For serious claims, also report a second seed block chosen after algorithm design freezes.
  - Algorithms may know the generator class but should not use privileged world descriptors unavailable through observation unless explicitly labeled as oracle/planner diagnostics.

- Reporting:
  - Report vector components, not only scalar return.
  - Report scalarization sensitivity under at least three weight settings:
    - success-heavy;
    - efficiency-heavy;
    - safety/resilience-heavy.
  - Report mean, standard deviation, median, and worst-quartile over worlds.
  - Report per-regime breakdown for scheduling.
  - Report maze breakdown by key/fuel/gate density.
  - Report compute budget and environment steps.

## Acceptance criteria

- Determinism gate:
  - Same seed plus same action sequence gives the same terminal vector.
  - Different seeds generate measurably different worlds.

- Terminal-only gate:
  - `reward == 0` for every nonterminal step.
  - The terminal reward vector contains all scoring information.
  - Diagnostic traces may exist in `info` for tests, but learner-facing baselines should not use shaping rewards.

- Feasibility gate:
  - A privileged diagnostic or conservative oracle can solve or nearly solve a high fraction of generated Small/v0 worlds.
  - If the oracle cannot find feasible solutions, failures are generator bugs, not benchmark difficulty.

- No idle-tail gate:
  - For baseline and oracle trajectories, meaningful state changes and obligations occur throughout the episode.
  - The final 25% of the horizon must affect terminal vector components on most worlds.

- Lookahead-depth gate:
  - Short receding-horizon policies should be worse than longer-horizon diagnostics.
  - For Maze, 50/100-step MPC should not match graph-level route planning on v0/Large.
  - For Scheduling, 100-step rollout should not match 500/1000-step rollout when maintenance/setup coupling is active.

- Myopic-gap gate:
  - Greedy nearest-target and earliest-deadline policies must leave significant vector headroom.
  - The headroom should appear in more than one component, not only in an arbitrary scalar weight.

- Recoverability gate:
  - Inject perturbations at early, middle, and late times.
  - Perturbations should cause graded terminal-vector degradation.
  - Single mistakes should not usually cause total collapse.
  - Single mistakes should not usually have zero effect.
  - Recovery curves should distinguish stronger policies from weaker policies.

- Action-complexity gate:
  - Top-1/top-2 action sparsification should not preserve most return in v0/Large when dense action use is claimed.
  - Appending no-op dimensions is forbidden.
  - Actuator/allocation dimensions must have measurable behavioral effect.

- Seed-generalization gate:
  - Policies tuned on train seeds must be evaluated on held-out worlds.
  - A large train-to-held-out gap is a warning sign and must be reported.
  - A policy that hard-codes one world should fail the gate.

- Reward-normalization gate:
  - Component magnitudes should be comparable across Small/v0/Large after normalization.
  - No cost component should grow linearly with horizon unless explicitly normalized.
  - Scalar reports should not be dominated by an accidental scale mismatch.

- Baseline-portfolio gate:
  - Each family must ship with several weak-to-moderate heuristics and at least one decomposition diagnostic.
  - Do not tune one heuristic until it succeeds and then treat it as the sole difficulty signal.

- Runtime gate:
  - Small episodes should run quickly enough for smoke tests.
  - Large episodes should plausibly run within about five minutes on a laptop CPU.
  - If ray casting or scheduling updates violate this, simplify geometry/state before adding dependencies.

- Vector-reporting gate:
  - Baseline reports must include component-wise results and Pareto-style comparisons.
  - A single scalar success table is insufficient for this substrate.

## What I am giving up

- I am giving up AlphaZero/MCTS-native discrete structure.
  - The mission is now continuous-action algorithms, so I am not trying to smuggle in discrete combinatorial actions.

- I am giving up a true vehicle-routing family.
  - Continuous relaxations of routing too easily become fluid dispatch while pretending to be combinatorial routing.
  - The scheduling family captures allocation pressure without making a misleading routing claim.

- I am giving up full stochastic robustness.
  - The settled contract is deterministic seed-to-world.
  - Generalization pressure comes from held-out generated worlds, not transition noise.

- I am giving up perfect Markov/full-information purity in every tier.
  - Some future summaries are intentionally compressed.
  - That is acceptable if the observation contract is documented and held-out worlds prevent single-world memorization.

- I am giving up universal benchmark breadth.
  - Two families will not cover all continuous-control RL.
  - They are enough for the lab's current goal: expose long-horizon coupling under terminal-only vector rewards.

- I am giving up very high-dimensional embodied control in Maze.
  - Maze uses 16/32/64 dimensions, not 128.
  - The scheduling family carries the 128-dimensional action stress more naturally.

- I am giving up cheap black-box CEM on Large as a central result.
  - CEM remains useful on Small/v0 and as short-horizon diagnostics.
  - Full Large CEM may be too expensive and too low-signal.

## Open questions for the proposer

1. For Maze, how much partial observability is acceptable?
   - I recommend local geometry plus compact landmark/manifest features, not a full global map.

2. For Scheduling, should future demand be exposed as lossy multi-scale summaries or as an exact finite calendar prefix?
   - I recommend lossy summaries for v0/Large and exact short calendars for Small debugging.

3. What scalarization weights should be treated as default, and which claims require vector/Pareto reporting instead of scalar return?
   - I recommend making scalar return secondary in baseline reports.

4. How strict should held-out evaluation be during development?
   - I recommend public validation seeds for iteration and a separate seed block used only after algorithm changes freeze.

5. Are oracle diagnostics allowed to use privileged world descriptors if they are clearly labeled and never compared as learner policies?
   - I recommend yes, because feasibility and headroom checks are otherwise too hard to interpret.
