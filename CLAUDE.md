# CLAUDE.md

Project instructions for `rl-research`.

## Mission

Find a novel RL algorithm of the same class as Q-learning, PPO,
AlphaZero, mirror descent, SAC, MCTS, and GAE. Baseline modifications do
not count as novelty.

The substrate is `rlh_bench` (vendored under `src/rlh_bench/`):
deterministic, recoverable, long-horizon, sparse / terminal-only
feedback, optional terminal vector reward. See `README.md` and
`DESIGN.md` for the environment contract and `docs/PHASE_PLAN.md` for
the staged research plan.

## Substrate boundary

- The `rlh_bench` package, its environments, registry, metrics, and
  baselines are the fixed substrate. Do not edit them mid-attempt to
  rescue a candidate algorithm.
- Vector reward mode is the load-bearing case. Algorithms targeting
  vector rewards must consume `info["reward_vector"]` (or the vector
  reward returned in `reward_mode="vector"`); collapsing it to a scalar
  before the learner is scalarization, not vector-reward learning.
- Non-terminal reward is always zero. Algorithms must handle terminal-
  only feedback rather than rely on per-step shaping.
- No baseline RL libraries. NumPy and the optional PyTorch dep are
  fine; replay buffers, optimizers, and wrappers are components, not
  baselines.

## Hot-path commands

```bash
pip install -e .                       # core (numpy only)
pip install -e .[torch]                # + PyTorch REINFORCE baseline
pip install -e .[gymnasium]            # + Gymnasium adapter
pip install -e .[dev]                  # + pytest, ruff

PYTHONPATH=src pytest -q               # full test suite
PYTHONPATH=src python examples/run_heuristics.py
PYTHONPATH=src python examples/train_cem.py
PYTHONPATH=src python examples/train_reinforce.py   # requires torch
```

## Repository layout

- `src/rlh_bench/` — environments, registry, metrics, wrappers, baselines.
- `tests/` — substrate regression tests.
- `examples/` — runnable baselines.
- `docs/PHASE_PLAN.md` — Phase 1 (problem validation) → Phase 2
  (benchmark prototype) → Phase 3 (algorithm research).
- `DESIGN.md` — environment contracts and reward-vector convention.
