# Agent guide — plugging a candidate algorithm into the runner

This is the technical reference for the `implement` shape of a lab
session: when a researcher decides to write code, evaluate it, and
record the result, this is the protocol. For everything else (read,
play, propose, synthesize, tool-build, …), see
[`docs/LAB.md`](LAB.md) and the system prompt that loaded for the
session.

Substrate background: [`docs/SUBSTRATE_MAP.md`](SUBSTRATE_MAP.md).
Project rules: [`CLAUDE.md`](../CLAUDE.md).

## Probes vs. candidates

There are two shapes of "I wrote code this session":

- **probe** (`experiments/probes/<slug>.py`): a quick ablation, trace,
  or one-off script. Doesn't need to satisfy any protocol. Often the
  right shape for a `play` or `read` session that needs to *run*
  something to be honest. Mention the path in the journal entry.

- **candidate** (`experiments/algorithms/<slug>.py`): a candidate
  algorithm you actually want to evaluate against the baselines.
  Targets the `Algorithm` protocol below and gets a JSON record
  under `experiments/results/`.

Do not promote every probe into a full candidate. Most sessions that
write code should produce probes, not candidates.

## TL;DR for candidates

1. Substrate (`src/rlh_bench/`) is **frozen**. Don't edit it to make
   your algorithm work.
2. Build the candidate under `experiments/algorithms/<your_name>.py`.
3. It must satisfy the `Algorithm` protocol in
   `experiments/algorithms/runner.py` — `name` + `train(env_factory, *, seed) -> policy`.
4. Run it through `evaluate_algorithm(...)`. Drop the JSON record into
   `experiments/results/`.
5. Compare against `docs/baseline_report.md` (random / heuristic /
   CEM). "Beats baselines" means same-or-better success rate **and**
   a Pareto-competitive mean reward vector — not just higher scalar
   return.

## What counts as novelty

From `CLAUDE.md`:

> Find a novel RL algorithm of the same class as Q-learning, PPO,
> AlphaZero, mirror descent, SAC, MCTS, and GAE. Baseline modifications
> do not count as novelty.

Concretely:

- A new tweak on REINFORCE / CEM / random search **does not count**.
  Even if it wins, it's a baseline variant.
- A new credit-assignment scheme that gets terminal vector outcomes
  back to individual actions can count.
- A new exploration objective that uses recoverability structure can
  count.
- A new optimizer for trajectory-level scalarization-free objectives
  (preference-conditioned policies, hypervolume gradients, etc.) can
  count.
- An algorithm that consumes `info["reward_vector"]` natively, without
  flattening to a weighted scalar before the learner, is a strong
  candidate — vector reward is the load-bearing case (`CLAUDE.md`).

## Substrate boundary recap

May:
- Read `info["reward_vector"]`, `info["reward_names"]`,
  `info["is_success"]`.
- Construct any internal model, buffer, optimizer, wrapper.
- Use NumPy and (optionally) PyTorch.
- Use `make_env(env_id, reward_mode="vector")` and learn directly on
  the vector.

May not:
- Edit `src/rlh_bench/**` — environments, registry, metrics, reward
  specs, reward orientations.
- Add per-step shaping rewards back into the env (terminal-only is the
  load-bearing property).
- Collapse the reward vector to a scalar inside the learner and call
  the result "vector RL" — that is scalarization.
- Pull in baseline RL libraries: stable-baselines3, RLlib, cleanrl,
  Tianshou, etc.

## How to plug in

The protocol (`experiments/algorithms/runner.py`) is intentionally tiny:

```python
from experiments.algorithms.runner import Algorithm, evaluate_algorithm

class MyAlgorithm:
    name = "my-algorithm"

    def __init__(self, ...): ...

    def train(self, env_factory, *, seed):
        # env_factory() returns a fresh env in the requested reward_mode.
        # Spend rollouts however you like; return a callable `policy(obs) -> action`.
        ...
        return policy
```

Then evaluate:

```python
record = evaluate_algorithm(
    algorithm=MyAlgorithm(...),
    env_id="RecoverablePointMaze-v0",
    train_seed=0,
    eval_seeds=range(20),
    reward_mode="scalar",          # or "vector" for true vector-reward learners
    save_to="experiments/results/my_algorithm__pointmaze.json",
)
print(record.success_rate, record.mean_return, record.mean_reward_vector)
```

`evaluate_algorithm` trains once (seeded by `train_seed`) and rolls the
returned policy over each `eval_seed`. The JSON record has fields:
`success_rate`, `mean_return`, `std_return`, `mean_reward_vector`,
`reward_names`, `mean_length`, `first_success_episode`,
`pareto_non_dominated_count`, `train_seconds`, and `per_episode[]`.

A complete worked example is in
`experiments/algorithms/example_random_shooting.py` — not a real
contender, just the right shape.

## Evaluation protocol

For each candidate algorithm, report on every env ID returned by
`rlh_bench.registered_envs()` (currently 6), with:

- `eval_seeds=range(20)` minimum;
- both `reward_mode="scalar"` (for headline numbers) and
  `reward_mode="vector"` (for any vector-native learner);
- training seed varied across at least 3 seeds when claiming the result
  is not seed-luck.

Aim for a markdown report in `docs/` that puts your numbers next to
`docs/baseline_report.md` row by row. A new row is interesting when:

- success rate ≥ heuristic success rate **and** mean reward vector is
  not dominated by the heuristic, OR
- the algorithm solves an env that random + heuristic + CEM all leave
  at success = 0 (currently the resource-allocation Small and Large
  configurations).

## Recommended first targets

In order of cost:

1. `RecoverableResourceAllocation-Small-v0` — small action dim,
   short horizon, **heuristic and CEM both fail at success = 0**. A
   small win here would be the most informative signal.
2. `RecoverableResourceAllocation-Large-v0` — same shape, larger action
   space; useful for testing scaling.
3. `RecoverablePointMaze-HD-v0` — redundant actuator pairs; tests
   whether your method exploits the redundancy or wastes capacity.
4. `RecoverableResourceAllocation-v0` — canonical allocator. Heuristic
   succeeds; if you beat it on the vector you're learning useful
   structure beyond the greedy rule.

## File layout

```
experiments/
    run_baselines.py            # Phase 1 baseline sweep -> docs/baseline_report.md
    algorithms/
        runner.py               # Algorithm protocol + evaluate_algorithm
        example_random_shooting.py
        <your_algorithm>.py
    probes/
        <your_probe>.py         # one-off scripts that don't need a protocol
    results/
        baselines.json
        example_random_shooting.json
        <your_algorithm>__<env_id>.json
docs/
    SUBSTRATE_MAP.md            # what the substrate offers
    AGENT_GUIDE.md              # this file
    baseline_report.md          # honest baseline numbers
```

## Hot-path commands

Setup is in [`README.md`](../README.md). Once the venv exists:

```bash
# tests, baselines, example
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py
PYTHONPATH=src:. .venv/bin/python experiments/algorithms/example_random_shooting.py

# run a candidate
PYTHONPATH=src:. .venv/bin/python experiments/algorithms/<your_algorithm>.py
```

Use `PYTHONPATH=src:.` whenever you import `experiments.*` so both the
substrate package and the experiments package resolve.
