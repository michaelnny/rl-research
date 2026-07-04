# CLAUDE.md

Project-level rules of engagement for the `rl-research` lab. Both
human contributors and AI agents read this file. The lab's *spirit*
is in [`docs/LAB.md`](docs/LAB.md); this file is the *rules*.

## Mission

Find a novel **continuous-action** RL algorithm of the same class as
PPO, SAC, CEM, mirror descent, GAE-style credit assignment, or
trajectory-level vector-reward methods. Baseline modifications do not
count as novelty.

The substrate is continuous-action only by design (see
`lab/notes/planning/PLAN_substrate_redesign_v2_2026-06-30.md` for the
rationale). AlphaZero / MCTS-class algorithms have a different action-
space substrate requirement (discrete / structured-search) that this
lab does not currently provide.

The substrate is `rlh_bench` (vendored under `src/rlh_bench/`):
deterministic, recoverable, long-horizon, sparse / terminal-only
feedback, optional terminal vector reward, continuous action spaces
throughout. See [`docs/SUBSTRATE_MAP.md`](docs/SUBSTRATE_MAP.md) for
the one-page API.

## Substrate boundary (the only hard rules)

These are about protecting the substrate, not about gating ideas:

1. **Do not edit anything under `src/rlh_bench/`** to make a candidate
   algorithm work. The substrate is frozen; algorithms are built
   around it, not into it.
2. **Do not introduce per-step reward shaping into the env.** Terminal-
   only feedback is load-bearing for the problem class.
3. **Do not use baseline RL libraries** — stable-baselines3, RLlib,
   cleanrl, Tianshou, etc. NumPy and optional PyTorch are the bar.
   Replay buffers, optimizers, and wrappers you build yourself are
   components, not baselines.
4. **Vector reward learning means consuming `info["reward_vector"]`.**
   If a learner takes the vector and immediately collapses it to a
   scalar, that is scalarization, not vector RL. Call it what it is.

If a candidate algorithm tempts you to break one of these to "make it
work", record what you wanted to break and why in the journal — that
is information about the algorithm. Then do not break it.

## How the lab operates

The repository runs an autonomous research loop. Two AI agents
(Claude and Codex/jelly) take turns each iteration: Claude writes a
journal entry to `lab/journal/`, Codex appends a `## Peer note` to
the same entry. The loop is [`lab/run_lab.sh`](lab/run_lab.sh); the
operator's manual is [`lab/README.md`](lab/README.md); the lab's
spirit is [`docs/LAB.md`](docs/LAB.md).

The loop auto-branches off `master` to `lab/auto` on first run.
`master` is for hand-curated commits — substrate changes, prompt
edits, README updates. `lab/auto` is for the loop's iteration
commits. Do not let the loop write to `master`.

## Repository layout

- `src/rlh_bench/` — the substrate (frozen). Environments, registry,
  metrics, wrappers, reference baselines.
- `tests/` — substrate regression tests. Should be 60+ passed.
- `examples/` — bare-bones substrate demos.
- `experiments/` — non-substrate research scaffold:
  - `run_baselines.py` — the baseline sweep over the registered
    envs.
  - `algorithms/runner.py` — the `Algorithm` protocol and
    `evaluate_algorithm`.
  - `algorithms/<name>.py` — candidate algorithms (when an agent
    decides to make one).
  - `probes/<name>.py` — one-off probes, ablations, traces.
  - `results/` — JSON records produced by the runner.
- `docs/` — orientation and lab artifacts:
  - [`LAB.md`](docs/LAB.md) — lab spirit.
  - [`SUBSTRATE_MAP.md`](docs/SUBSTRATE_MAP.md) — one-page substrate
    API, the single source of truth for env semantics.
  - [`AGENT_GUIDE.md`](docs/AGENT_GUIDE.md) — how to plug a candidate
    into the runner.
  - [`baseline_report.md`](docs/baseline_report.md) — honest baseline
    portfolio numbers on the registered envs.
- `lab/` — the loop itself:
  - [`run_lab.sh`](lab/run_lab.sh) — the loop.
  - [`journal/`](lab/journal/) — append-only research journal.
  - [`prompts/`](lab/prompts/) — production prompts for Claude
    sessions, Codex peer notes, and Codex steering memos.
  - `logs/` — ignored process logs and PID file.
  - `runs/` — ignored per-session stdout/stderr/prompt artifacts.
  - [`notes/`](lab/notes/) — lab-meta artifacts (substrate design,
    peer reviews, audit). Subfolders: `planning/`, `reviews/`,
    `briefs/`. See [`lab/notes/README.md`](lab/notes/README.md) for
    the index.
- [`README.md`](README.md) — top-level orientation and new-machine
  setup walkthrough.

## Local environment

A project-local `.venv` (Python 3.12, `uv`) is the canonical
environment. Bootstrap is in [`README.md`](README.md):

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e ".[dev,torch,gymnasium]"
```

Hot-path commands:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py
PYTHONPATH=src:. .venv/bin/python experiments/algorithms/<name>.py
```

Use `PYTHONPATH=src:.` whenever importing `experiments.*` so both the
substrate and the experiments package resolve.
