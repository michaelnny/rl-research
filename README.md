# rl-research

Substrate for AI coding agents to autonomously discover new families of
reinforcement-learning algorithms on classic (non-LLM) RL problems.

> **Mission, in one line:** find a *third* family of RL algorithms — distinct
> from value-based and policy-based — through autonomous research by coding
> agents. The full charter lives at [docs/charter.md](docs/charter.md).

This is a **research substrate**, not an algorithm library. The repo ships
environments, evaluation primitives, and a frozen PPO yardstick. Algorithms are
the *work product*, not a dependency. SB3 / CleanRL / Tianshou / RLlib /
Acme / Coax / garage are forbidden.

---

## Read first

Three docs anchor everything else. Read them before changing anything substantive:

1. **[docs/charter.md](docs/charter.md)** — mission, hard rules, disqualifiers.
2. **[docs/loop.md](docs/loop.md)** — the four roles and per-iteration state machine.
3. **[docs/contract.md](docs/contract.md)** — the run artifact contract.

Then [docs/benchmarks.md](docs/benchmarks.md), [docs/sota.md](docs/sota.md), and
the per-role prompts in [docs/roles/](docs/roles/).

For day-to-day operations of the autonomous loop, see
**[docs/operations.md](docs/operations.md)**.

---

## Quickstart

```bash
# Install (uv handles the venv + pinned deps)
uv sync

# Verify the install end-to-end (CUDA, gymnasium, dm_control, TB)
make smoke

# Quality gate (ruff + tests)
make check

# Dashboard for the autonomous loop
make status
```

Common targets (`make help` for the full list):

| target           | what it does                                                       |
| ---------------- | ------------------------------------------------------------------ |
| `make smoke`     | verifies install: CUDA available, envs step, TB writes events     |
| `make test`      | full pytest suite (CPU-only; the GPU runs are in baselines/)      |
| `make check`     | the full quality gate: ruff check + format + pytest               |
| `make status`    | one-screen ops dashboard (corpus, GPU, disk, log)                 |
| `make preflight` | health check used between iterations of the autonomous loop      |
| `make stats`     | refresh `lab/CORPUS_STATS.md` from the ledger                     |
| `make loop`      | start the headless loop in a tmux session named `loop`            |
| `make stop`      | request halt (writes `lab/HALT_REQUESTED.md`)                     |

---

## Running the autonomous loop

The loop is **four-role** (Researcher → Reviewer → Engineer → Curator) and
runs headlessly via `claude -p '/iterate'` inside a tmux session.

```bash
# Start
make loop          # creates tmux session 'loop'
tmux attach -t loop

# Monitor (read-only, safe to run while loop is active)
make status
tail -f lab/iterations.log

# Stop (graceful — finishes current iteration, then exits)
make stop
```

The loop is designed to run unattended for **weeks**. See
[docs/operations.md](docs/operations.md) for: failure modes, recovery, log
rotation cadence, backup strategy, and what to inspect when something goes
wrong.

---

## Layout

```
src/rl_research/        # the framework primitives
  baselines/ppo.py      # the only allowed RL algo (frozen yardstick)
  contract.py           # run-artifact contract + ledger writer
  runtime.py            # CLI / seeding / wallclock / config.json / param checksum
  envs.py               # vector adapters: gym-classic / atari / minecart / dm_control
  evaluate.py           # algorithm-agnostic deterministic eval (takes a policy_fn)
  tb.py                 # TB scalar names + eval cadence (matches the contract)
  checkpoints.py        # atomic save/load with retention

docs/                   # source-of-truth specs
  charter.md            # mission + hard rules
  loop.md               # state machine
  contract.md           # what every run dir must contain
  benchmarks.md         # 3 primary + 2 sanity envs
  sota.md               # published SOTA references
  operations.md         # day-to-day ops manual
  roles/                # per-role prompts (also loaded into .claude/agents/)

lab/                    # the operational corpus (read+written by the loop)
  ledger.jsonl          # append-only one-line-per-run summary
  lessons.md            # Curator-distilled findings
  threads/              # per-direction thread state
  runs/<run_id>/        # per-run artifacts (hypothesis, train.py, result.json, tb/, ...)
  baselines/            # frozen baseline runs (random, PPO)
  templates/            # template hypothesis.md + algorithm-agnostic train.py skeleton
  CORPUS_STATS.md       # auto-generated corpus surface (refreshed each iteration)
  result.schema.json    # machine-readable schema for result.json

tests/                  # pytest suite for src/rl_research primitives
scripts/                # ops scripts (loop.sh, preflight.sh, status.sh, ...)
.claude/                # agent definitions and the /iterate slash command
```

---

## Constraints baked into the substrate

- **Default branch is `master`**, not `main`.
- **Use `uv` for all package operations.** Never `pip`. Add deps with
  `uv add <pkg>==<ver>`.
- **Pin every dependency** to an exact version.
- **Single-GPU assumption.** One RTX 3090 Ti, 24 GB.
- **TensorBoard, not wandb.** Logs under `runs/` (gitignored, ad-hoc) and
  `lab/runs/<run_id>/tb/` (committed corpus).
- **2-hour wallclock cap per run** in early-phase exploration. Hard kill at
  the OS level on overrun.
- **Promotion is curatorial, never numerical.**

The full set lives in [docs/charter.md](docs/charter.md).
