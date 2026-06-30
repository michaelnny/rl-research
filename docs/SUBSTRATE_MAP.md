# Substrate map

One-page cheat sheet for the `rlh_bench` substrate after the v2
redesign (2026-06-30). The substrate is the **frozen** problem
definition — environments, registry, metrics, wrappers, and reference
baselines. Algorithms live **outside** the substrate. Do not edit
substrate files to rescue a candidate algorithm (see `CLAUDE.md`).

## Entry points

```python
from rlh_bench import (
    RewardSpec,
    CapacitySchedulingConfig, RecoverableCapacitySchedulingEnv,
    DEFAULT_SCHEDULING_REWARD_SPEC,
    KeyFuelMazeConfig, RecoverableKeyFuelMazeEnv,
    DEFAULT_KEYFUEL_REWARD_SPEC,
    make_env, registered_envs,
    rollout, evaluate_policy, pareto_non_dominated,
)
from rlh_bench.baselines import SCHEDULING_BASELINES, MAZE_BASELINES
from rlh_bench.seed_bands import seed_band_for
```

## Env contract (`rlh_bench.core`)

```python
obs, info = env.reset(seed=int | None)
obs, reward, terminated, truncated, info = env.step(action)
```

Every env exposes `observation_space`, `action_space`, `reward_dim`,
`reward_spec`, `diagnostics()`. Non-terminal reward is **zero**; the
only non-zero reward arrives on the terminal step at `t == horizon`.
`terminated` flips on; `truncated` is always `False`. Calling `step()`
after termination raises `RuntimeError` — `reset()` is mandatory.

`reward_mode` controls return shape:
- `"scalar"` (default): `reward` is `float = RewardSpec.scalarize(vec)`.
- `"vector"`: `reward` is the raw `np.float32` vector, shape `(reward_dim,)`.

Either way, the **vector is always in `info["reward_vector"]`** and the
component names are in `info["reward_names"]`. `info["is_success"]` is a
bool.

## Seed → world contract

`reset(seed=s)` deterministically generates the entire world from `s`
(graph topology, demand schedule, actuator matrix, key/seal placement,
etc.). Same seed → same world. Different seeds → measurably different
worlds.

Held-out evaluation is a first-class substrate property. Each tier
publishes train / validation / held-out / debug seed ranges via
`rlh_bench.seed_bands.seed_band_for(env_id)`. Algorithms that train
against the public train block should be reported against the held-out
block.

## Registry (`rlh_bench.envs.registration`)

| Env ID                                       | Family    | H      | Action dim | Notes                              |
| -------------------------------------------- | --------- | ------ | ---------- | ---------------------------------- |
| `RecoverableCapacityScheduling-Small-v0`     | Scheduling| 500    | 32         | K=16 M=4 P=4; smoke tier           |
| `RecoverableCapacityScheduling-v0`           | Scheduling| 2000   | 96         | K=48 M=8 P=8; canonical            |
| `RecoverableCapacityScheduling-Large-v0`     | Scheduling| 10000  | 224        | K=128 M=16 P=16; stretch           |
| `RecoverableKeyFuelMaze-Small-v0`            | Maze      | 500    | 16         | 24×24 world; 2 keys / 2 seals      |
| `RecoverableKeyFuelMaze-v0`                  | Maze      | 2000   | 32         | 48×48 world; 4 keys / 6 seals      |
| `RecoverableKeyFuelMaze-Large-v0`            | Maze      | 10000  | 64         | 96×96 world; 6 keys / 12 seals     |

`make_env(env_id, reward_mode="scalar"|"vector", ...)` forwards extra
kwargs to the env constructor. `registered_envs()` returns the sorted
tuple of IDs.

## Family A — `RecoverableCapacitySchedulingEnv`

Allocation / scheduling with multiple cross-time couplings (wear,
setup inertia, heat, inventory perishability, contract bundles).

- **Action `Box([-1, 1]^D)`** decomposed by the env into:
  - project allocation logits (K)
  - mode allocation logits (M)
  - per-mode maintenance intensity (M)
  - per-mode setup-change intensity (M)
  - per-product inventory release (P)
  - trailing dims beyond `K + 3M + P` are unused (allows the registry to advertise larger action spaces at higher tiers without changing semantics).
- **Obs**: per-project (cumulative service, backlog, deadline slack, priority); per-mode (utilization EMA, wear, heat, maintenance debt); setup mixture (M×P); per-product (inventory, age); multi-scale future demand summaries (16/64/256-step windows); previous-action aggregates; `t/H`.
- **Dynamics**: production per project = `mode_capacity × setup_alignment × compat × proj_alloc`. Capacity reduces with wear and heat. Setup retargeting costs same-step capacity. Inventory builds from unused capacity and can perish. Contract bundles require all-of-N projects above quality threshold.
- **Terminal vector** (`DEFAULT_SCHEDULING_REWARD_SPEC`):
  - 11 components: `(success, weighted_fill_rate, mandatory_fill_rate, neg_lateness, neg_shortfall_tail, neg_wear, neg_heat_violation, neg_setup_churn, neg_inventory_waste, neg_energy, resilience_margin)`.

## Family B — `RecoverableKeyFuelMazeEnv`

Continuous-control 2D point-mass with a structured task: visit a set
of key regions (dwell to collect), then visit seal regions (each
requires specific keys + an optional timed-gate open phase), then
reach an extraction zone. All before fuel runs out.

- **Action `Box([-1, 1]^D)`** mapped through a per-world deterministic actuator matrix `A ∈ R^{2 × D}` into 2-D force. Per-actuator energy/heat cost weights are also per-world. Higher D means more redundant actuators, NOT more force dimensions; choosing the right basis costs less fuel/heat.
- **Obs**: position(2), velocity(2), fuel(1), heat(1), damage(1), keys held(K_t), seal status(S), 3 nearest unfinished landmarks (each: dx, dy, kind one-hot[4], key-type one-hot[K_t]), gate phases(G), `t/H`, prev-action energy.
- **Dynamics**: damped point mass with soft boundary collisions. Fuel decreases with distance + actuator energy. Fuel stations recharge with cooldowns. Keys collected by dwelling in a key region for several steps. Seals require keys + optional gate-open phase. Final extraction zone for success.
- **Terminal vector** (`DEFAULT_KEYFUEL_REWARD_SPEC`):
  - 9 components: `(success, seal_completion, key_coverage, fuel_margin, neg_damage, neg_lateness, neg_energy, neg_collision, route_efficiency)`.

## Reward orientation

All vector components are **larger-is-better**. Costs / delays /
violations appear as negative quantities (`neg_*`). This makes the
vectors directly usable for Pareto analysis, hypervolume, and convex
scalarization. `pareto_non_dominated(points)` returns a boolean mask
under that maximization assumption.

## Metrics (`rlh_bench.metrics`)

- `rollout(env, policy, seed=None, max_steps=None) -> EpisodeResult` —
  one episode. `EpisodeResult` holds `scalar_return`, `reward_vector`,
  `length`, `terminated`, `truncated`, `info`.
- `evaluate_policy(env_factory, policy_factory, episodes=5, seed=0) -> EvaluationSummary` —
  rebuilds env and policy each episode, with seeds `seed + ep`. Policy
  factory may take `()` or `(env)`. Reports `mean_return`, `std_return`,
  `mean_reward_vector`, `success_rate`, `mean_length`, `episodes`.
- `first_success_episode(successes)` — 1-indexed first hit, or `None`.

## Wrappers

- `ScalarizeRewardWrapper(env, weights)` — collapse a vector-mode env back to a scalar with user-chosen weights. Vector still in `info["reward_vector"]`.
- `GymnasiumAdapter(env)` — wrap as `gymnasium.Env` for external code. Optional dep; converts the internal `Box`/`Discrete`/`MultiDiscrete` spaces.

## Reference baselines (`rlh_bench.baselines`)

Each new family ships with a *portfolio* of cheap heuristics plus a
decomposition diagnostic. No single heuristic is the difficulty
signal. The decomposition diagnostic checks whether the env truly
rewards long-horizon credit assignment — if it solves the env, the
long-horizon claim fails.

**Scheduling** (`baselines.scheduling.SCHEDULING_BASELINES`):
zero, uniform, capacity_push, backlog_priority, earliest_deadline,
maintenance_aware, setup_aware, short_horizon_rollout (decomposition
diagnostic). `capacity_push` is a high-cost stress diagnostic — it
succeeds by running all-out production with no maintenance / setup
retargeting, paying heavily in wear / inventory waste / energy /
resilience. Treat its success as "feasibility floor", not as a
target to merely match.

**Maze** (`baselines.maze.MAZE_BASELINES`):
zero, random_constant, greedy_landmark, fuel_aware_greedy,
efficient_actuator, short_horizon_lookahead (decomposition diagnostic).

**Maze oracle diagnostics** (`baselines.maze.MAZE_ORACLE_DIAGNOSTICS`):
`MazeOracleRoutePlannerPolicy`. Reads privileged env-internal state
(waypoint coordinates, gate phases). NOT comparable to the learner-
facing portfolio. Exists to verify feasibility — if the oracle
succeeds, the env admits a competent policy. Never cite "beat the
oracle" — the oracle's whole point is that learners shouldn't beat it
via observation-only policies.

**Legacy** (`baselines.random`, `baselines.cem`, `baselines.reinforce`):
`RandomPolicy`, `ZeroPolicy`, `train_cem`, `train_reinforce` — still
available; CEM and REINFORCE on the new envs use the standard
`LinearPolicy`.

These baselines are *not* contenders; their role is to confirm tasks
are feasible-but-non-trivial. New algorithm work should beat them, not
re-derive them.

## Substrate boundary — what an algorithm may and may not touch

May:
- Build new policies, learners, replay buffers, optimizers, exploration bonuses, hindsight relabelers, scalarization schedulers, etc., as *components* in `experiments/` (or a new dir of yours), importing the substrate.
- Read `info["reward_vector"]` / `info["reward_names"]` / `info["is_success"]` — those are the substrate's intended outputs.
- Read the **public model API**: properties exposed on the env
  object. For KeyFuelMaze this includes ``env.actuator_matrix``
  (the per-world deterministic mapping from D-dim action to 2-D
  force) and ``env.seed`` (current world seed for cache
  invalidation). These are treated as known-model information
  analogous to a robot knowing its own kinematic structure; they
  do not reveal task state (key positions, gate phases, etc.).
- Construct any wrapper of your own, including custom Gym adapters.

May not:
- Read underscore-prefixed env attributes (`env._key_positions`,
  `env._seal_gate_requirements`, `env._gate_phases`, etc.). Those
  are the privileged hooks used by oracle diagnostics, and a
  baseline that reads them is no longer a baseline.
- Edit anything under `src/rlh_bench/`, including registry, metrics, reward specs, and reward orientations, to make an algorithm work.
- Add per-step shaping rewards back into the env (the terminal-only property is load-bearing).
- Use a scalarization step inside the learner and call it "vector RL" (see `CLAUDE.md`).
- Pull in baseline RL libraries (stable-baselines3, RLlib, cleanrl, etc.). NumPy and the optional PyTorch dep are the bar.

## Legacy env classes

The pre-redesign families remain importable from
`rlh_bench.envs.continuous_maze.RecoverablePointMazeEnv` and
`rlh_bench.envs.resource_allocation.RecoverableResourceAllocationEnv`,
but are **not** in `registered_envs()`. They are kept for backward
compatibility with code that referenced them; new work targets the
two new families above.
