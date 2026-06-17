# Test Report

Date: 2026-06-17

Environment used for validation:

```text
Python package root: /mnt/data/rlh_bench
Core dependencies: NumPy, pytest
Optional dependency available during smoke test: PyTorch
Gymnasium was not required for the core test suite.
Editable install was also smoke-tested with `pip install -e . --no-deps --no-build-isolation`.
```

## Commands run

```bash
cd /mnt/data/rlh_bench
python -m pip install -e . --no-deps --no-build-isolation
pytest -q
PYTHONPATH=src python examples/run_heuristics.py
PYTHONPATH=src python examples/train_cem.py
PYTHONPATH=src python examples/train_reinforce.py
```

## Pytest result

```text
25 passed
```

## Coverage by behavior

The test suite validates:

- self-contained space classes;
- deterministic resets;
- terminal-only scalar reward behavior;
- terminal-only vector reward behavior;
- fixed-horizon termination;
- error on `step()` after termination;
- recoverability after repeated bad actions;
- high-dimensional continuous action configurations;
- environment registry construction;
- random rollout smoke tests;
- heuristic feasibility;
- CEM policy-search smoke test;
- optional PyTorch REINFORCE smoke test;
- Pareto non-dominated utility behavior.

## Heuristic smoke-test outcomes

The built-in heuristics reached success on the default canonical environments:

```text
RecoverablePointMaze-v0:
  success: 1.0
  final_distance: approximately 0.013
  collisions: 0

RecoverableResourceAllocation-v0:
  success: 1.0
  service_level: 1.0
  safety_violation: 0.0
```

These results are not meant to establish algorithmic performance. They verify that the tasks are feasible, recoverable, and not accidentally impossible.
