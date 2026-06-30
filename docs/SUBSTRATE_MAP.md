# Substrate map

One-page cheat sheet for the `rlh_bench` substrate. The substrate is the
**frozen** problem definition — environments, registry, metrics,
wrappers, and reference baselines. Algorithms live **outside** the
substrate. Do not edit substrate files to rescue a candidate algorithm
(see `CLAUDE.md`).

## Entry points

```python
from rlh_bench import (
    RewardSpec,
    RecoverablePointMazeEnv, RecoverableMazeConfig, Rectangle, DEFAULT_MAZE_REWARD_SPEC,
    RecoverableResourceAllocationEnv, ResourceAllocationConfig, DEFAULT_RESOURCE_REWARD_SPEC,
    make_env, registered_envs,
    rollout, evaluate_policy, pareto_non_dominated,
)
```

`from rlh_bench.wrappers import ScalarizeRewardWrapper, GymnasiumAdapter`
gives optional interop. `from rlh_bench.baselines import ...` exposes
the reference baselines.

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
bool. Vector-reward learning means consuming that vector pre-scalarization
(see `CLAUDE.md`); collapsing it to a weighted scalar before the learner
is scalarization, not vector learning.

## Registry (`rlh_bench.envs.registration`)

| Env ID                                    | Family   | Horizon | Action dim | Notes                                |
| ----------------------------------------- | -------- | ------- | ---------- | ------------------------------------ |
| `RecoverablePointMaze-Small-v0`           | Maze     | 120     | 2          | shorter; quick smoke tests           |
| `RecoverablePointMaze-v0`                 | Maze     | 160     | 2          | canonical maze                       |
| `RecoverablePointMaze-HD-v0`              | Maze     | 180     | 8          | redundant actuator pairs             |
| `RecoverableResourceAllocation-Small-v0`  | Resource | 60      | 4          | shorter; quick smoke tests           |
| `RecoverableResourceAllocation-v0`        | Resource | 100     | 5          | canonical allocator                  |
| `RecoverableResourceAllocation-Large-v0`  | Resource | 120     | 8          | larger structured action             |

`make_env(env_id, reward_mode="scalar"|"vector", ...)` forwards extra
kwargs to the env constructor. `registered_envs()` returns the sorted
tuple of IDs.

## Family A — `RecoverablePointMazeEnv`

- **Obs (7,)**: `[x, y, vx, vy, goal_x, goal_y, t / H]` (vels scaled to ~[-1, 1]).
- **Action `[-1, 1]^d`**, `d` even. Pairs of channels are averaged into a 2D acceleration with weights `1 / (1 + i)`, so extra dims are redundant actuators, not new degrees of freedom.
- **Dynamics**: damped point mass, soft collisions against `Rectangle` obstacles and the unit-square boundary. Collisions are recoverable: velocity bounces, position is clamped, the episode keeps going.
- **Terminal vector** (`DEFAULT_MAZE_REWARD_SPEC`):
  - names: `("success", "neg_final_distance", "neg_energy", "neg_collisions", "neg_path_length")`
  - weights: `(1.0, 0.30, 0.003, 0.03, 0.02)`
- **Difficulty knobs** (`RecoverableMazeConfig`): `horizon`, `action_dim`, `start`, `goal`, `goal_radius`, `obstacles`, `dt`, `acceleration_scale`, `damping`, `max_speed`, `agent_radius`.

## Family B — `RecoverableResourceAllocationEnv`

- **Obs (3K+1,)**: `[progress_ratio[K], readiness[K], last_allocation[K], t / H]`.
- **Action `[0, 1]^K`**, projected to the per-step `budget` (default 1.0).
- **Dynamics**: each project accumulates `efficiency_i * readiness_i * allocation_i`, capped at `demand_i * progress_cap_factor`. `readiness_i = min_readiness + (1 - min_readiness) * clip(ratio_{i-1}, 0, 1)` — downstream is never fully locked, so early bad allocations are wasteful but recoverable. Per-project allocation above `safe_allocation` accrues a quadratic safety violation.
- **Terminal vector** (`DEFAULT_RESOURCE_REWARD_SPEC`):
  - names: `("success", "service_level", "neg_cost", "neg_delay", "neg_safety_violation")`
  - weights: `(1.0, 0.65, 0.003, 0.10, 0.08)`
- **Difficulty knobs** (`ResourceAllocationConfig`): `horizon`, `num_projects`, `budget`, `demand`, `efficiency`, `cost`, `deadlines`, `min_readiness`, `safe_allocation`, `progress_cap_factor`.

## Reward orientation

All vector components are **larger-is-better**. Costs/delays/collisions
appear as negative quantities (`neg_*`). This makes the vectors directly
usable for Pareto analysis, hypervolume, and convex scalarization.
`pareto_non_dominated(points)` returns a boolean mask under that
maximization assumption.

## Metrics (`rlh_bench.metrics`)

- `rollout(env, policy, seed=None, max_steps=None) -> EpisodeResult` —
  one episode. `EpisodeResult` holds `scalar_return` (always a scalar,
  computed via `reward_spec.scalarize` when the env is in vector mode),
  `reward_vector`, `length`, `terminated`, `truncated`, `info`.
- `evaluate_policy(env_factory, policy_factory, episodes=5, seed=0) -> EvaluationSummary` —
  rebuilds env and policy each episode, with seeds `seed + ep`. Policy
  factory may take `()` or `(env)`. Reports `mean_return`, `std_return`,
  `mean_reward_vector`, `success_rate`, `mean_length`, `episodes`.
- `first_success_episode(successes)` — 1-indexed first hit, or `None`.

## Wrappers

- `ScalarizeRewardWrapper(env, weights)` — collapse a vector-mode env
  back to a scalar with user-chosen weights. Vector still in
  `info["reward_vector"]`.
- `GymnasiumAdapter(env)` — wrap as `gymnasium.Env` for external code.
  Optional dep; converts the internal `Box`/`Discrete`/`MultiDiscrete`
  spaces.

## Reference baselines (`rlh_bench.baselines`)

- `RandomPolicy(action_space, seed)`, `ZeroPolicy(action_space)`.
- `MazeWaypointPolicy(env)` — PD waypoint follower for the maze.
- `ResourceGreedyPolicy(env)` — greedy ready-then-deficient allocator.
- `make_heuristic_policy(env)` — returns the right one by type.
- `train_cem(env_factory, iterations, population, elite_frac, init_std, min_std, eval_episodes, seed) -> CEMResult` —
  CEM over a linear tanh policy (`LinearPolicy`, params shape
  `action_dim * (obs_dim + 1)`). NumPy only.
- `train_reinforce(env_factory, episodes, hidden_size, lr, gamma, entropy_coef, seed) -> ReinforceResult` —
  tiny Gaussian-MLP REINFORCE with a moving-average baseline. Requires
  the `[torch]` extra. CPU-only on macOS is fine.

These baselines are *not* contenders; their role is to confirm tasks
are feasible-but-non-trivial. New algorithm work should beat them, not
re-derive them.

## Substrate boundary — what an algorithm may and may not touch

May:
- Build new policies, learners, replay buffers, optimizers, exploration
  bonuses, hindsight relabelers, scalarization schedulers, etc., as
  *components* in `experiments/` (or a new dir of yours), importing
  the substrate.
- Read `info["reward_vector"]` / `info["reward_names"]` /
  `info["is_success"]` — those are the substrate's intended outputs.
- Construct any wrapper of your own, including custom Gym adapters.

May not:
- Edit anything under `src/rlh_bench/`, including registry, metrics,
  reward specs, and reward orientations, to make an algorithm work.
- Add per-step shaping rewards back into the env (the terminal-only
  property is load-bearing).
- Use a scalarization step inside the learner and call it "vector RL"
  (see `CLAUDE.md`).
- Pull in baseline RL libraries (stable-baselines3, RLlib, cleanrl,
  etc.). NumPy and the optional PyTorch dep are the bar.
