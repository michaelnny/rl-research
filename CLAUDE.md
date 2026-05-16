# CLAUDE.md

Guidance for coding agents working in this repository.

## Mission

This repo is a **research substrate**: agents are expected to invent and evaluate
new RL algorithms on classic (non-LLM) RL problems. The repo intentionally ships
only environments and compute primitives. **Algorithms are the work product, not
a dependency.**

The full mission, hard rules, and disqualifiers live in
**[docs/charter.md](docs/charter.md)**. Read it before doing anything substantive.

## Source-of-truth docs

These are the anti-divergence specs. Every role and every iteration of the loop
reads them. Do NOT re-derive their content from this file or from prose
elsewhere — read the docs themselves.

- **[docs/charter.md](docs/charter.md)** — mission, hard rules, disqualifiers.
- **[docs/loop.md](docs/loop.md)** — the four roles and per-iteration state machine.
- **[docs/contract.md](docs/contract.md)** — the run artifact contract (what
  every run directory must contain, schema for `result.json`).
- **[docs/benchmarks.md](docs/benchmarks.md)** — three primary benchmarks and two
  sanity envs.
- **[docs/roles/](docs/roles/)** — per-role prompts (`researcher.md`,
  `reviewer.md`, `operator.md`, `curator.md`).

## Hard rules (summary — see charter for full list)

- **No imported RL algorithm libraries.** SB3 / CleanRL / Tianshou / RLlib /
  Acme / Coax / garage are forbidden. Single exception: the own-authored PPO
  baseline at `src/rl_research/baselines/ppo.py`.
- **Symbolic search.** Each iteration produces a self-contained `train.py`.
- **2-hour wallclock cap** per run in early-phase exploration. Hard kill at
  the OS level.
- **Promotion is curatorial, never numerical.**
- **Pin every dependency** to an exact version.
- **Use `uv` for all package operations.** Never invoke `pip` directly. Add
  deps with `uv add <pkg>==<ver>`.
- **Single-GPU assumption.** One RTX 3090 Ti (24 GB).
- **TensorBoard, not wandb.** Logs go under `runs/` (gitignored) and
  `lab/runs/<run_id>/tb/` (committed).
- **Default branch is `master`**, not `main`.

## Layout conventions

```
src/rl_research/
  contract.py          # run artifact contract enforcement (next_run_id, write_result, validate_result_json)
  baselines/
    ppo.py             # the single allowed RL algorithm (frozen yardstick)
docs/                  # source-of-truth specs (charter, loop, contract, benchmarks, roles/)
lab/                   # operational corpus (read+written by the loop)
  ledger.jsonl         # append-only one-line-per-run summary
  lessons.md           # Curator-distilled findings
  threads/             # per-direction thread state
  runs/<run_id>/       # per-run artifacts (hypothesis, train.py, result.json, tb/, ...)
  baselines/           # frozen baseline runs (random, PPO)
  templates/           # template hypothesis.md and friends
  result.schema.json   # machine-readable schema for result.json
tests/                 # pytest suite for src/rl_research primitives
```

`runs/` (top-level, gitignored) is for ad-hoc TensorBoard logs only. The
**research corpus** lives at `lab/runs/`.

## Workflow

- **Quality:** `uv run ruff check && uv run ruff format && uv run pytest`
- **Smoke test stack:** `uv run python tests/smoke_test.py`
- **Run an experiment** (manual): `uv run python lab/runs/<run_id>/train.py
  --env <env> --seed <s> --total-env-steps <n> --max-wallclock-s <s> --logdir
  lab/runs/<run_id>/tb/<seed>`
- **Inspect telemetry:** `uv run tensorboard --logdir lab/runs/<run_id>/tb`

## Working as a sub-agent

If you have been spawned in a specific role (Researcher / Reviewer / Operator /
Curator), your role prompt at `docs/roles/<your-role>.md` is your operating
instructions. It supersedes anything in this file *except* the hard rules in
`docs/charter.md`, which are non-negotiable.
