# RLH Bench

**RLH Bench** is a small research package for studying **recoverable long-horizon sparse-feedback reinforcement learning**.

It contains two deterministic benchmark families designed for the Phase 1 / Phase 2 research goal:

1. **RecoverablePointMaze** — continuous-control navigation with soft collisions, long horizons, terminal-only feedback, and optional terminal vector rewards.
2. **RecoverableResourceAllocation** — structured continuous allocation with soft dependencies, large action dimension, long horizons, terminal-only feedback, and natural vector outcomes.

The environments are intentionally lightweight: they use NumPy only, run on CPU, and do not require robot simulators, large models, transformers, or GPUs.

---

## Target problem class

The package focuses on deterministic episodic tasks with these properties:

- long adjustable horizon;
- sparse or terminal-only environmental feedback;
- hard but recoverable exploration;
- difficult credit assignment;
- continuous or large structured action spaces;
- cheap deterministic simulation;
- optional terminal vector-valued environmental reward.

The vector rewards are **environment outcomes**, not intrinsic rewards and not auxiliary shaping. During an episode, reward is zero. At the terminal step, the environment reports a vector such as:

```text
[success, negative final distance, negative energy, negative collisions, negative path length]
```

or:

```text
[success, service level, negative cost, negative delay, negative safety violation]
```

---

## Installation

From the package directory:

```bash
pip install -e .
```

For optional PyTorch REINFORCE baseline:

```bash
pip install -e .[torch]
```

For optional Gymnasium interop:

```bash
pip install -e .[gymnasium]
```

The core environments and tests require only NumPy and pytest.

---

## Quick start

```python
from rlh_bench.envs import make_env
from rlh_bench.baselines import make_heuristic_policy
from rlh_bench.metrics import rollout

# Scalar reward mode: returned reward is weighted scalar, vector is still in info.
env = make_env("RecoverablePointMaze-v0")
policy = make_heuristic_policy(env)
result = rollout(env, policy, seed=0)

print(result.scalar_return)
print(result.reward_vector)
print(result.info["reward_names"])
```

Vector reward mode:

```python
env = make_env("RecoverableResourceAllocation-v0", reward_mode="vector")
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print(reward)                 # terminal vector reward
print(info["reward_vector"])  # same vector
```

---

## Available environment IDs

```python
from rlh_bench.envs import registered_envs
print(registered_envs())
```

Current IDs:

```text
RecoverablePointMaze-Small-v0
RecoverablePointMaze-v0
RecoverablePointMaze-HD-v0
RecoverableResourceAllocation-Small-v0
RecoverableResourceAllocation-v0
RecoverableResourceAllocation-Large-v0
```

---

## Baselines

The package includes lightweight baselines:

- `RandomPolicy`
- `ZeroPolicy`
- `MazeWaypointPolicy`
- `ResourceGreedyPolicy`
- `train_cem` — CPU-friendly Cross-Entropy Method policy search
- `train_reinforce` — optional PyTorch REINFORCE baseline

Example:

```bash
PYTHONPATH=src python examples/run_heuristics.py
PYTHONPATH=src python examples/train_cem.py
```

---

## Tests

Run from the package root:

```bash
PYTHONPATH=src pytest -q
```

The test suite checks:

- deterministic reset and rollout behavior;
- terminal-only reward behavior;
- vector reward mode;
- recoverability after bad actions;
- heuristic feasibility;
- registry construction;
- CEM and optional REINFORCE smoke tests;
- Pareto utility behavior.

---

## Research use

The package is meant to support the next research phase by giving you a controlled benchmark scaffold, not a final benchmark claim. Recommended next steps:

1. Run the included diagnostics across horizon, action dimension, and recoverability settings.
2. Decide which environment family best isolates the phenomenon you care about.
3. Add benchmark reports for random-policy success, first-success episodes, terminal vector trade-offs, and recovery after injected early mistakes.
4. Only then use these environments to compare new algorithms.

See `DESIGN.md` and `docs/PHASE_PLAN.md` for details.
