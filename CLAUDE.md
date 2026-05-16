# CLAUDE.md

Guidance for coding agents working in this repository.

## Mission

This repo is a **research substrate**: agents are expected to invent and evaluate new RL
algorithms on classic (non-LLM) RL problems. The repo intentionally ships only environments
and compute primitives. **Algorithms are the work product, not a dependency.**

## Hard rules

- **Do not add RL algorithm libraries** as dependencies. That includes (but is not limited to)
  Stable-Baselines3, CleanRL, Tianshou, RLlib, Acme, Coax, garage. If you need a baseline for
  comparison, implement it from primitives in this repo.
- **Pin every new dependency to an exact version** (`pkg==X.Y.Z`) in `pyproject.toml`. This
  project relies on full reproducibility.
- **Use `uv` for every package operation.** Never invoke `pip` directly. Add deps with
  `uv add <pkg>==<ver>`; add dev deps with `uv add --dev <pkg>==<ver>`.
- **Single-GPU assumption.** Hardware is one RTX 3090 Ti (24 GB). Do not introduce
  multi-GPU/distributed code paths unless explicitly requested.
- **TensorBoard, not wandb.** Logs go under `runs/` (gitignored).

## Tech stack (locked)

- Python 3.12, managed via `uv`
- PyTorch with CUDA 12.8 wheels (no JAX)
- Gymnasium 1.3 (with `classic-control`, `box2d`, `mujoco`, `atari`, `other`) + dm-control + ale-py
- Ruff (lint + format), pytest, pre-commit

## Layout conventions

```
src/rl_research/     # shared primitives ONLY (buffers, env wrappers, telemetry)
experiments/<name>/  # one directory per algorithm/experiment, self-contained
tests/               # pytest tests for primitives in src/
runs/                # TensorBoard logs / checkpoints (gitignored)
```

When inventing a new algorithm, create a new `experiments/<name>/` directory and keep the
training script self-contained — easy to read, easy to delete. Promote code into
`src/rl_research/` only when it is reused by ≥2 experiments.

## Workflow

1. `uv sync` — install pinned deps.
2. `uv run python experiments/<name>/train.py` — run an experiment.
3. `uv run tensorboard --logdir runs` — inspect telemetry.
4. `uv run ruff check && uv run ruff format && uv run pytest` before committing.
