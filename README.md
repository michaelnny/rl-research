# rl-research

A substrate for coding agents to research new RL algorithms on classic (non-LLM) RL problems.

## What this is (and isn't)

This repo provides only **environments** and **compute primitives** — no pre-built RL algorithms.
The goal is for coding agents to **invent algorithms from scratch** on top of PyTorch and the
standard env interfaces.

Installed by design:
- **PyTorch** (CUDA 12.8 wheels) + TensorBoard
- **Gymnasium** with `classic-control`, `box2d`, `mujoco`, `atari`, `other` extras
- **MuJoCo** + **dm-control** for continuous-control benchmarks
- **ALE-py** for Atari (ROMs are bundled)

Deliberately **not** installed: Stable-Baselines3, CleanRL, Tianshou, RLlib, or any other RL
algorithm library. If a baseline is needed, implement it from primitives.

## Setup

```bash
uv sync
```

Hardware: tested on a single NVIDIA RTX 3090 Ti (24 GB) with driver supporting CUDA 12.8.

## Layout

```
src/rl_research/   # shared primitives (env wrappers, replay buffers, telemetry helpers)
tests/             # pytest suite
runs/              # TensorBoard logs and checkpoints (gitignored)
```

## Quality

- `uv run ruff check` / `uv run ruff format`
- `uv run pytest`
- `uv run pre-commit install` once, then hooks run on every commit.

## Smoke test

```bash
uv run python tests/smoke_test.py
```
