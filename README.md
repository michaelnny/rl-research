# rl-research

Substrate for an autonomous AI coding agent to discover novel RL
algorithms on **long-horizon sparse-reward** and **vector-reward**
problems.

> **Mission, in one line:** find a behavior-improvement primitive that
> handles long-horizon sparse-reward and vector-reward problems natively,
> replaces what value *does* (future compression, temporal composition,
> local improvement), and is structurally distinct from every prior
> failed attempt and every standard RL family. Full goal in
> [program.md](program.md).

The repo intentionally ships only environments, the eval protocol, the
panel runner, and an instruction sheet. **Algorithms are the work product,
not a dependency.** SB3 / CleanRL / Tianshou / RLlib / Acme / Coax /
garage are forbidden.

---

## Read first

Two files anchor everything:

1. **[program.md](program.md)** — the agent's instruction sheet. Goal,
   setup, experiment loop, keep/discard rule.
2. **[prior_attempts.md](prior_attempts.md)** — 14 prior failed directions
   plus the disqualifier-family list. Read this between iterations to
   avoid rebadges.

[CLAUDE.md](CLAUDE.md) is the entry point for AI coding agents.

The detailed research memory — one bounded file per attempt — lives
under [`worklogs/`](worklogs/README.md). The agent appends there during
the loop using [`worklogs/TEMPLATE.md`](worklogs/TEMPLATE.md).

---

## Quickstart

```bash
# Install (uv handles the venv + pinned deps)
uv sync

# Smoke test the harness pipeline (random policy on one env)
uv run pytest tests/test_harness.py -k smoke

# Quality gate
uv run ruff check && uv run ruff format && uv run pytest

# Build the smoke-tier baseline scores (~5 min/baseline × 5 envs × 3 baselines = ~75 min;
# pass --time-budget-s 60 for a quick pre-build)
uv run python scripts/build_baselines.py
```

### Linux: system prerequisites

The hard-tier env list pulls some sdist-only deps that build from source.
Install the C/C++ build chain once before the first `uv sync`:

```bash
sudo apt install -y build-essential cmake ninja-build bison flex
```

`bison` and `flex` are required by `nle` (NetHack Learning Environment),
the upstream of `minihack`. `cmake` and `ninja` drive its build.

GPU side: the lockfile pins `torch==2.11.0+cu128` from the pytorch cu128
index, which bundles its own CUDA runtime — no system-wide CUDA toolkit
is needed for execution. A modern NVIDIA driver (570+) is required.

---

## Running the loop

The agent reads [program.md](program.md), agrees on a run tag with the
human, creates a branch `rl/<tag>`, and iterates.

- **Smoke tier** (default, ~5 min/sweep): 5 envs in parallel. The agent
  iterates against this constantly — ~100 sweeps fit in an 8-hour night.
- **Hard tier** (`--hard`, ~2 h/sweep with Option-B grouped scheduling): 4
  envs. Reserved for promoted candidates and periodic checkpoints.

```bash
# One smoke sweep — typically ~5 min
uv run run_panel.py > run.log 2>&1
grep "^panel_tier\|^panel_n_beat\|^panel_wallclock_s" run.log

# One hard sweep — typically ~2 h
uv run run_panel.py --hard > run_hard.log 2>&1
```

The agent appends one row to `results.tsv` per sweep, then `git commit -am`
or `git reset --hard HEAD~1` based on whether the smoke tuple
`(n_beat_random, n_beat_strong)` advanced.

---

## Layout

```
harness.py              # frozen — env factory, evaluate, panel, hypervolume, baseline loader
train.py                # agent-editable — the RL algorithm lives here (default: random policy)
run_panel.py            # frozen — runs train.py × each panel env, aggregates panel score
program.md              # the agent's instruction sheet
prior_attempts.md       # one-paragraph index of failed directions + disqualifier list
panel.md                # rationale for each panel env vs failure modes
worklogs/               # research memory — see worklogs/README.md
  README.md             #   explains the template + index split
  TEMPLATE.md           #   fixed per-attempt template
  attempts/             #   database: one bounded file per attempt
baselines/              # `random.py`, `eps_greedy_q.py`, `count_bonus.py`
scripts/build_baselines.py
tests/test_harness.py
baselines.json          # frozen smoke-tier baselines (committed; built once)
baselines_hard.json     # frozen hard-tier baselines (published_sota + our_baseline)
results.tsv             # one row per sweep — gitignored
runs/last/              # per-env stdout from the most recent sweep — gitignored
```

---

## Panel

**Smoke tier** (5 envs, ~5 min/sweep parallel — what the agent iterates against):

| Env | Type | Channels |
| --- | --- | --- |
| `MiniGrid-DoorKey-8x8-v0` | scalar | 1 |
| `MiniGrid-KeyCorridorS3R3-v0` | scalar | 1 |
| `deep-sea-treasure-concave-v0` | vector | 2 |
| `minecart-v0` | vector | 3 |
| `mo-reacher-v4` | vector | 4 |

**Hard tier** (4 envs, ~2 h/sweep — Option-B grouped: Phase 1 Craftax solo,
Phase 2 the rest in parallel — the bar a serious candidate must push):

| Env | Type | Channels |
| --- | --- | --- |
| `Craftax-Symbolic-v1` | scalar | 1 |
| `MiniHack-Quest-Hard-v0` | scalar | 1 |
| `mo-halfcheetah-v4` | vector | 2 |
| `Humanoid-v5` | scalar | 1 |

Vector envs inject `info['vector']: np.ndarray` of shape `(k,)` on every
`step()`. Consuming the scalar `reward` instead of `info['vector']` on a
vector env is a scalarized-vector-reward rebadge — flagged as a disqualifier
(see [prior_attempts.md](prior_attempts.md)).

Score per env: scalar envs → mean episode return; vector envs → Pareto
hypervolume vs a fixed reference point.

Panel score: tuple `(n_beat_random, n_beat_strong)` over the active tier.

---

## Constraints baked into the substrate

- **Default branch is `master`**, not `main`.
- **`uv` only** — never `pip`. Pin every dep.
- **Single-GPU assumption.** One RTX 3090 Ti, 24 GB. JAX (Craftax) is forced
  to CPU via `JAX_PLATFORMS=cpu` so it doesn't fight torch for VRAM.
- **MiniHack is linux-only** — the wheel doesn't exist on PyPI for any
  platform and the sdist build needs cmake. The pyproject pins it behind
  `sys_platform == 'linux'`. macOS workstations skip MiniHack at install
  time and rely on the linux box for hard sweeps.
- **Per-env wallclock cap:** `harness.TIME_BUDGET_SMOKE = 300` (smoke);
  `harness.TIME_BUDGET_HARD = 3600` (hard).
- **Eval cap:** `N_EVAL_EPISODES = 20` deterministic episodes per env.
- **No RL algorithm libraries.** Inventing, not assembling.
