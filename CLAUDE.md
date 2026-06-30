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
- `experiments/` — non-substrate research scaffold:
  `run_baselines.py` for the Phase 1 sweep, `algorithms/runner.py`
  with the `Algorithm` protocol + `evaluate_algorithm`, and
  `algorithms/<name>.py` for candidates.
- `docs/PHASE_PLAN.md` — Phase 1 (problem validation) → Phase 2
  (benchmark prototype) → Phase 3 (algorithm research).
- `docs/SUBSTRATE_MAP.md` — one-page cheat-sheet of the substrate API.
- `docs/AGENT_GUIDE.md` — how a coding agent plugs in a new algorithm,
  what counts as novelty, and the evaluation protocol.
- `docs/baseline_report.md` — honest random / heuristic / CEM numbers
  across all six env IDs; the bar a candidate must beat.
- `DESIGN.md` — environment contracts and reward-vector convention.

## Local env

A project-local `.venv` (Python 3.12, `uv`) is installed with the
`[dev,torch,gymnasium]` extras. Run the suite with
`PYTHONPATH=src .venv/bin/python -m pytest -q` and candidate algorithms
with `PYTHONPATH=src:. .venv/bin/python experiments/algorithms/<x>.py`.

## The lab

This project runs an autonomous research lab against the substrate.
Read `docs/LAB.md` for the lab's spirit (journal-as-product, no
verdicts, bad ideas welcome) and `lab/README.md` for the operator's
manual. The loop is `lab/run_lab.sh`. Per iteration it invokes
`claude -p` (Opus, max effort) to write a journal entry under
`docs/journal/`, then `codex exec -p jelly` to append a `## Peer note`
section to the same entry, then commits with a descriptive message.
The loop auto-branches to `lab/auto`; `master` is for hand-curated
commits.
