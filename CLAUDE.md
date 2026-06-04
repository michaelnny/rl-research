# CLAUDE.md

Guidance for AI coding agents working in this repository.

## What this repo is

A minimal substrate for an autonomous AI agent to discover novel RL
algorithms targeting **long-horizon sparse-reward** and **vector-reward**
problems. Modeled on the autoresearch design philosophy: a frozen harness,
one agent-editable file, one panel runner, and an instruction sheet — that
is the entire surface.

The agent's job: edit `train.py` to invent an RL algorithm; run the panel;
keep the commit if the panel score advanced; otherwise discard. Loop.

## Source of truth

- **[program.md](program.md)** — the agent's instruction sheet. Goal,
  setup, contract, experiment loop, keep/discard rule, NEVER-STOP clause.
  This is what the agent reads at the start of a session.
- **[prior_attempts.md](prior_attempts.md)** — 14 prior failed directions,
  cross-attempt failure modes, disqualifier-family list. The agent reads
  this between iterations to avoid rebadges.
- **[panel.md](panel.md)** — design rationale for the two-tier panel
  (5 smoke + 4 hard). Each env is mapped to the specific failure mode it
  detects and the prior attempts that died to it. Read once at session
  start so you understand what each panel score is actually measuring.

Read these three files before doing anything substantive in this repo.

## Layout

```
harness.py              # frozen — env factory, evaluate, panel, hypervolume, baseline loader
train.py                # agent-editable — the RL algorithm lives here
run_panel.py            # frozen — runs train.py vs each panel env, aggregates panel score
program.md              # the agent's instruction sheet
prior_attempts.md       # one-paragraph index of failed directions + disqualifier list
panel.md                # rationale for each panel env vs failure modes
worklogs/               # research memory — see worklogs/README.md
  README.md             # explains the template + index split
  TEMPLATE.md           # fixed per-attempt template (frontmatter + sections)
  attempts/             # database: one bounded file per attempt (NN-<slug>.md)
baselines.json          # frozen smoke-tier baselines (built by scripts/build_baselines.py)
baselines_hard.json     # frozen hard-tier baselines (published_sota + our_baseline)
baselines/              # baseline `train()` implementations (random, eps_greedy_q, count_bonus)
scripts/build_baselines.py
tests/test_harness.py   # smoke test for the harness pipeline
results.tsv             # one row per panel sweep — gitignored
runs/last/              # per-env stdout from the most recent sweep — gitignored
```

## Hard rules

- **No imported RL algorithm libraries.** SB3 / CleanRL / Tianshou / RLlib /
  Acme / Coax / garage are forbidden. The agent invents the algorithm; it
  does not assemble one.
- **Pin every dependency** to an exact version.
- **Use `uv` for all package operations.** Never invoke `pip` directly. Add
  deps with `uv add <pkg>==<ver>`.
- **Single-GPU assumption.** One RTX 3090 Ti (24 GB). JAX (Craftax) is forced
  to CPU via `JAX_PLATFORMS=cpu` to leave VRAM for torch.
- **MiniHack is linux-only** — pyproject pins it behind
  `sys_platform == 'linux'`. macOS dev environments skip it.
- **Default branch is `master`**, not `main`.
- **The agent does not modify** `harness.py`, `run_panel.py`,
  `baselines.json`, or `baselines_hard.json`. Tampering invalidates the
  panel score.

## Workflow

```bash
# Install
uv sync

# Smoke test the harness end-to-end (random policy on one env)
uv run pytest tests/test_harness.py -k smoke

# Quality gate
uv run ruff check && uv run ruff format && uv run pytest

# One smoke panel sweep (~5 min: 5 envs × 300s in parallel)
uv run run_panel.py > run.log 2>&1
grep "^panel_tier\|^panel_n_beat\|^panel_wallclock_s" run.log

# One hard panel sweep (~2 h: Option-B grouped — Craftax solo, then 3 in parallel)
uv run run_panel.py --hard > run_hard.log 2>&1

# Quick sanity (one env: deep-sea-treasure-concave-v0, ~5 min)
uv run run_panel.py --quick > run.log 2>&1
```

## Loop driver

The experiment loop runs as plain text instructions in
[program.md](program.md). The agent reads it at the start of a session,
agrees on a run tag, creates a branch `rl/<tag>`, and starts iterating.
There is no slash command, no orchestrator daemon, no per-role subagent.
The NEVER-STOP clause in program.md is binding: once the experiment loop
begins, the agent does not pause to ask whether to continue.
