# RLH Bench Design Document

## 1. Research motivation

The package is designed for a specific problem class:

> Deterministic, recoverable, long-horizon decision problems with sparse or terminal-only environmental feedback, hard exploration, difficult credit assignment, non-trivial action spaces, and optional terminal vector-valued outcomes.

The immediate goal is not to propose a new RL algorithm. The goal is to create environments that are clean enough to support Phase 1 and Phase 2 research decisions.

## 2. Non-goals

The package intentionally avoids:

- real robot hardware;
- expensive robot simulation as the primary dependency;
- large vision models or transformer policies;
- dense reward shaping;
- stochasticity in the initial benchmark;
- tiny four-action-only gridworlds as the primary task family;
- intrinsic rewards or auxiliary training rewards.

## 3. Common environment contract

Each environment follows a Gymnasium-like API:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

The package is self-contained and does not require Gymnasium, but `GymnasiumAdapter` is provided for projects that want Gymnasium interop.

All environments expose:

```python
env.observation_space
env.action_space
env.reward_dim
env.reward_spec
env.diagnostics()
```

The scalar/vector reward behavior is controlled with `reward_mode`:

- `reward_mode="scalar"`: return a weighted scalar reward; always include `info["reward_vector"]`.
- `reward_mode="vector"`: return the full reward vector as the reward.

Non-terminal reward is always zero. Terminal feedback occurs at the fixed horizon.

## 4. Reward-vector convention

All reward-vector components are oriented so that larger is better.

Costs are therefore represented as negative quantities:

```text
neg_energy
neg_cost
neg_delay
neg_safety_violation
```

This makes vector returns compatible with Pareto analysis, hypervolume, and simple scalarization.

## 5. Environment family A: RecoverablePointMaze

### Purpose

A continuous-control task that isolates recoverable long-horizon exploration and terminal credit assignment without requiring heavy physics.

### State

The observation is:

```text
[x, y, vx, vy, goal_x, goal_y, t / H]
```

### Action

The action is continuous:

```text
a ∈ [-1, 1]^d
```

`d` is configurable and must be even. For `d > 2`, action pairs are redundant actuator channels combined into a 2D acceleration. This gives a simple action-dimensionality knob without changing the underlying task semantics.

### Dynamics

The agent is a point mass with damped velocity. Obstacles and boundaries cause soft collisions. A collision does not terminate the episode; it changes velocity, wastes time, and contributes to the terminal collision outcome.

### Terminal vector

```text
[success, neg_final_distance, neg_energy, neg_collisions, neg_path_length]
```

### Why recoverable?

A bad action can cause a collision or waste energy, but the agent remains in the episode and can still reach the goal before the horizon.

### Difficulty knobs

- horizon;
- action dimension;
- obstacle layout;
- goal radius;
- acceleration scale;
- damping;
- max speed;
- terminal scalarization weights.

## 6. Environment family B: RecoverableResourceAllocation

### Purpose

A structured large-action task where terminal outcomes naturally form a vector: success, service, cost, delay, and safety/constraint violation.

### State

For `K` projects, the observation is:

```text
[progress_ratio[K], readiness[K], last_allocation[K], t / H]
```

### Action

The action is a continuous allocation vector:

```text
a ∈ [0, 1]^K
```

If `sum(a)` exceeds the per-step budget, it is deterministically projected back to the budget.

### Dynamics

Each project accumulates progress from allocated resources. Downstream projects have soft dependencies on upstream progress. Allocating to a downstream project too early is inefficient but not fatal, because readiness is never exactly zero.

### Terminal vector

```text
[success, service_level, neg_cost, neg_delay, neg_safety_violation]
```

### Why recoverable?

Bad early allocations waste budget and hurt delay/cost, but later allocations can still partially or fully recover service-level and success outcomes.

### Difficulty knobs

- horizon;
- number of projects/action dimension;
- budget;
- demand;
- efficiency;
- dependency readiness;
- safe-allocation threshold;
- deadlines;
- terminal scalarization weights.

## 7. Baseline design

Baselines are intentionally lightweight:

- random policy: establishes exploration rarity;
- heuristic policies: establish feasibility and recoverability;
- CEM: CPU-friendly classic policy-search baseline;
- REINFORCE: optional small policy-gradient baseline.

The baselines are not expected to be state of the art. Their role is to validate that the environments are neither impossible nor trivial.

## 8. Recommended benchmark diagnostics

For Phase 2, report at least:

```text
random-policy success rate
heuristic success rate
episodes to first success
terminal reward vector
scalarized return
recovery after injected early mistakes
performance as horizon increases
performance as action dimension increases
Pareto non-dominated outcome set for vector rewards
```

The key diagnostic is recoverability: after an early mistake, the episode should not collapse into a useless trajectory. The terminal outcome should still contain meaningful information about how well the agent recovered.

## 9. Extension points

The package is deliberately small. Useful next extensions include:

- additional maze layouts;
- structured scheduling variants;
- hybrid discrete-continuous actions;
- stochastic variants after deterministic behavior is understood;
- multi-objective evaluation metrics such as hypervolume;
- benchmark report scripts for full Phase 2 diagnostics.
