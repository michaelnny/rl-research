# rl-research

An autonomous reinforcement-learning research lab. Two AI agents
(Claude and Codex) take turns working against a fixed problem
substrate, keeping a research journal that — over many sessions — is
intended to make a novel RL algorithm likely.

The product is the journal at `lab/journal/`. If a novel algorithm
emerges, it will be a downstream consequence of the journal being
honest and varied.

## What the lab is trying to find

A novel **continuous-action** RL algorithm in the same class as PPO,
SAC, CEM, mirror descent, GAE-style credit assignment, or trajectory-
level vector-reward methods. Not a tweak on an existing algorithm; an
idea that wasn't there before. The mission, hard rules, and substrate
boundary are in [`CLAUDE.md`](CLAUDE.md). The lab's spirit — no
verdicts, journal-as-product, bad ideas welcome — is in
[`docs/LAB.md`](docs/LAB.md). The substrate is continuous-action only
by design; AlphaZero / MCTS-class algorithms need a different action
substrate that this lab does not currently provide.

## What "substrate" means here

The substrate is `rlh_bench` (vendored at `src/rlh_bench/`): two
deterministic, recoverable, long-horizon continuous-control
environments with terminal-only sparse vector rewards. Current
environments and reward semantics are summarized in
[`docs/SUBSTRATE_MAP.md`](docs/SUBSTRATE_MAP.md); baseline numbers
that any candidate algorithm has to beat are in
[`docs/baseline_report.md`](docs/baseline_report.md). The substrate
is frozen — the lab works around it, not on it.

## Repository layout

```
src/rlh_bench/                # the substrate (frozen)
tests/                        # substrate regression tests
examples/                     # bare-bones substrate demos
experiments/
  run_baselines.py            # baseline portfolio sweep
  algorithms/runner.py        # Algorithm protocol + evaluate_algorithm
  algorithms/<name>.py        # candidate algorithms (when implemented)
  probes/<name>.py            # one-off probes and ablations
  results/                    # JSON records emitted by the runner
docs/
  LAB.md                      # lab spirit (no verdicts, journal-as-product)
  SUBSTRATE_MAP.md            # one-page substrate API
  AGENT_GUIDE.md              # how to plug a candidate into the runner
  baseline_report.md          # honest baseline portfolio numbers
lab/
  README.md                   # operator's manual
  run_lab.sh                  # the dumb loop
  journal/                    # the lab's research journal — append-only
  prompts/                    # system + per-session prompts (production)
  notes/                      # lab-meta artifacts (planning, reviews, briefs)
  logs/                       # ignored process logs and pid file
  runs/                       # ignored per-session stdout/stderr/prompts
CLAUDE.md                     # project-level rules of engagement
```

## Setting it up on a new machine

Tested on macOS 14+ (Apple Silicon) with Python 3.12. Linux should
work but isn't covered by the smoke tests.

### Prerequisites

- [`uv`](https://github.com/astral-sh/uv) ≥ 0.11 — the Python project
  manager used here. Install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have
  it yet.
- Python 3.12 (any patch). `uv` can install one if you don't have it:
  `uv python install 3.12`.
- `git`.
- [Claude Code](https://docs.claude.com/en/docs/claude-code/) CLI
  v2.1+ — needed by the loop. Authenticated with whatever method your
  Anthropic account uses.
- [Codex CLI](https://developers.openai.com/codex/) v0.142+ — needed
  by the loop. A custom profile named `hai` is expected at
  `~/.codex/hai.config.toml` pointing at whichever provider you use.
  The provider's API key must be exported in the environment variable
  the profile names (e.g. `HAI_OPENAI_API_KEY`).

### Bootstrap

```bash
git clone <this-repo> rl-research
cd rl-research

# Create the project-local venv with editable install + extras
uv venv --python 3.12 .venv
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e ".[dev,torch,gymnasium]"

# Verify the substrate works
PYTHONPATH=src .venv/bin/python -m pytest -q

# Optional: re-run the baseline sweep on this machine
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py
```

### Verify your CLI overrides work

The lab relies on two CLI-level overrides that aren't well-documented
upstream. Confirm both before starting the loop.

Claude (replaces the ~37k-token default system prompt; auto-memory
also disabled by `--bare`):

```bash
claude -p --bare --system-prompt-file lab/prompts/claude_system.md \
  --no-session-persistence --model haiku \
  "Reply with exactly OK_CLAUDE." < /dev/null
```

Codex (replaces `base_instructions` for this run only):

```bash
codex -a never exec -p hai \
  -c "model_instructions_file=\"$(pwd)/lab/prompts/codex_system.md\"" \
  --sandbox read-only --skip-git-repo-check --ephemeral \
  'In one sentence, what kind of lab are you in?' < /dev/null
```

Codex should answer something like *"I'm in a small autonomous
reinforcement-learning research lab."* If it says it's a coding
agent, the override didn't load — check that the path in
`-c model_instructions_file=...` is absolute and the file exists.

### Launch the loop

```bash
# Refuses to start unless the worktree is clean and HAI_OPENAI_API_KEY
# is exported in this shell.
git status --short

# Forever, detached:
mkdir -p lab/logs
nohup bash lab/run_lab.sh > lab/logs/console.log 2>&1 &

# Or in tmux (easier to detach/reattach):
tmux new -s rl-lab 'bash lab/run_lab.sh'
```

The loop auto-branches off `master` to `lab/auto`, so the main branch
is never auto-committed to. It refuses to start from a dirty worktree,
because each successful autonomous session is committed with `git add -A`.
Regular iterations are Claude journal entries followed by
mandatory Codex peer notes. After several regular sessions, the loop
inserts a Codex-authored steering memo before the next Claude session
to break local-search drift.

### Watching it

```bash
tail -f lab/logs/run.log
tail -f lab/logs/console.log   # only needed for shell-level stderr/stdout
ls -lat lab/runs/ | head       # newest raw run artifacts first
ls -lat lab/journal/ | head        # newest entries first
git log --oneline lab/auto | head   # commits the loop has made
```

### Stopping it

```bash
kill $(cat lab/logs/run.pid)        # nohup
# or Ctrl-C inside the tmux session
```

The loop catches `SIGINT`/`SIGTERM`, stops the active agent process
tree if one is running, and removes `lab/logs/run.pid`.

## How sessions work

Per regular iteration (≈ 5-20 minutes wall-clock on Opus max-effort):

1. Loop reads the next session number from `lab/journal/`.
2. Loop invokes `claude -p --bare --system-prompt-file lab/prompts/claude_system.md ...` with a tiny user prompt that names the session number. Claude reads the recent journal, picks a session kind (read/play/propose/implement/synthesize/tool-build), does the work, and writes `lab/journal/sessionNNNN-<slug>.md` ending in an empty `## Peer note` section.
3. Loop invokes `codex -a never exec -p hai -c model_instructions_file=... ...` pointing at the new entry. Codex reads the entry (and any artifacts it depends on) and appends a peer note. If Codex exits nonzero, leaves the placeholder unchanged, or edits the wrong journal file, the loop stops instead of committing.
4. Loop commits with a descriptive non-verdictive message and starts the next iteration.

Every `STEERING_INTERVAL` regular Claude sessions (default: 5), the
next session number is used for a Codex-authored steering memo instead:
`lab/journal/sessionNNNN-codex-steering.md`. Claude's following
session is explicitly prompted to pick up one of the memo's leads.
Set `STEERING_INTERVAL=0` is not supported; use a larger positive
integer if you want steering less often.

Full operator's manual is in [`lab/README.md`](lab/README.md).

## When to intervene

Read `lab/README.md` → "What 'good' looks like" for the early-warning
signs (session-kind monoculture, peer notes turning into verdicts,
files under `src/rlh_bench/` changing). The runner now refuses to
commit substrate changes, but the loop is still dumb on purpose; the
human's job is to watch the journal and intervene by hand when the
disposition slips.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The substrate test suite checks deterministic resets, terminal-only
reward behavior, vector reward mode, recoverability after bad
actions, registry construction, baseline portfolio honesty
(public-model baselines vs oracle separation), held-out seed
contract, idle-tail measurement, acceptance gates, CEM / optional
REINFORCE smoke tests, and Pareto utility behavior. It should be
60+ passed on a working install.

## License

MIT — see [`LICENSE`](LICENSE).
