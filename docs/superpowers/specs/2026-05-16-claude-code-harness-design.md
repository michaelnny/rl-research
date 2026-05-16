---
title: Claude Code agent harness for rl-research
date: 2026-05-16
status: design — pending user review
---

# Claude Code agent harness for rl-research

## 1. Goal and design philosophy

**Goal.** Express the autonomous research loop in Claude Code primitives so
it runs *headless* (overnight / weekend) with minimal supervision, while
giving each role-specific subagent the maximum power Claude Code already
provides. The harness is committed to the repo: anyone with the repo can
reproduce the loop bit-for-bit.

**Design philosophy — substrate, not enforcement.** The harness creates
the conditions for the agents to do good research. It does not police
them, observe them, or anticipate their failure modes.

- We **do not reimplement** anything Claude Code already provides:
  retries, rate-limit backoff, auth, model routing, OTel emission, cost
  reporting via `--output-format json`, transcripts under
  `~/.claude/projects/`. These are taken as given.
- We **do not pre-design corpus artifacts**. If the Curator decides at
  iteration 30 that a coverage map or health snapshot would help, it
  writes one. The harness does not pre-create files like
  `landscape.md`, `health.md`, `cost.json`, `trace.jsonl`,
  `violations.jsonl`. These are emergent if useful, absent if not.
- We **do not impose per-role tool denylists or read budgets**. Every
  subagent gets the full Claude Code tool surface. The role prompt
  *orients* the agent toward source-of-truth docs; it does not police
  what the agent may touch.
- We **do not insert mid-iteration halts** based on cost, verdict
  trends, or any inspection logic. Compute policy lives on `train.py`
  itself (existing wallclock + retry-budget rules in
  `docs/charter.md`). Auto-halt is a single between-iteration check
  for `lab/HALT_REQUESTED.md` — which any agent (typically the
  Curator) can write when it judges the loop has stopped producing
  signal.

**Goal restated.** Four prompts + one slash command + one shell loop. The
research substrate (`docs/`, `lab/`, `src/rl_research/contract.py`) does
the rest.

## 2. Why subagents (sequential pipeline)

The official Claude Code docs draw a sharp distinction. From
`code.claude.com/docs/en/agent-teams` (the experimental
`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` mode):

> *"For sequential tasks where one role completes before the next begins,
>  a single session or subagents are more effective."*

Our loop is sequential by design — Researcher → Reviewer → Engineer →
Curator, with file-artifact handoffs (`hypothesis.md`, `review.md`,
`result.json`, ledger entry). There is no real-time peer debate inside
one iteration. Disagreement is expressed across iterations through the
corpus.

**Therefore: subagents.** Each role is a `.claude/agents/<name>.md`
file. The orchestrator (the `/iterate` slash command) spawns them in
order. Context isolation is per-subagent (each gets its own conversation
window); filesystem isolation is unnecessary because the artifacts are
how subagents communicate.

## 3. Role redesign — what changes in `docs/roles/`

The current `docs/roles/researcher.md` bundles two completely different
jobs into one role: Phase 1 (creative ideation) and Phase 2 (writing
self-contained PyTorch from primitives, debugging, hitting the
contract). The same trap as putting a research scientist on production
code duty.

**New role split — four roles, properly separated:**

| Role | Job | Model |
|------|-----|-------|
| Researcher | Phase 1 only. Writes `hypothesis.md`. Halts. Never touches code. Maximum generative breadth. | opus |
| Reviewer | Cheap text checkpoint. Reads `hypothesis.md`, writes `review.md` with verdict (`novel-direction` \| `known-rebadge` \| `needs-sharpening`). ~30s. | sonnet |
| Engineer | The heavy lifter. From approved hypothesis: writes `train.py` from primitives, runs Stage A, debugs failures with the 3-retry budget and `fix-N.md` notes, runs Stage B, writes `config.json` + `result.json`, appends ledger entry. Owns *everything* between approval and recorded result. | opus |
| Curator | Per-iteration meta-supervisor. Assigns `verdict_curator` to recent runs. May also (its judgment): prune `lessons.md`, archive stale threads, write a coverage map, decide on mass-run promotion, write `lab/HALT_REQUESTED.md`. | opus |

The current `docs/roles/operator.md`'s retry-with-`fix-N.md` logic was
already engineering work in disguise. Merging implementation +
execution + retry into one Engineer role matches the actual nature of
the work and removes a synthetic handoff.

**Source-of-truth doc edits required (implementation step 1):**

- `docs/roles/researcher.md` — strip Phase 2; the role ends at writing
  `hypothesis.md`. Keep all generative-discipline guidance (propose
  freely, no self-censorship, no numeric beat-baseline targets).
- `docs/roles/operator.md` → renamed and rewritten as
  `docs/roles/engineer.md`. Folds in the train.py-writing
  responsibility from the old researcher.md Phase 2.
- `docs/roles/reviewer.md` — minor edit to clarify it reviews
  `hypothesis.md` (no train.py review).
- `docs/roles/curator.md` — explicit grant of meta-supervisor
  authority: "you may create new corpus artifacts if you judge them
  useful, and you may write `lab/HALT_REQUESTED.md` if you judge the
  loop has stopped producing signal."
- `docs/loop.md` — update the four-role state machine diagram and
  per-role responsibilities to match.
- `docs/contract.md` — extend `status` enum with `abandoned-rebadge`
  and `abandoned-sharpening` (orchestrator writes minimal `result.json`
  with these statuses when revision cycles are exhausted).
- `lab/result.schema.json` — same status-enum extension.

## 4. Agent definitions (`.claude/agents/`)

Four files. Bodies are *short* — they orient the agent toward source of
truth, not restate the role doc inline. Tool surface is broad; Claude
Code's defaults are sufficient.

### 4.1 `researcher.md`

```yaml
---
name: researcher
description: Proposes a novel third-family RL hypothesis. Phase 1 only — writes hypothesis.md and run_id.txt, halts. Does not implement code.
model: opus
color: blue
---
```

Body covers, in ~15 lines:

- Read `docs/charter.md` and `docs/roles/researcher.md` first; they are
  source of truth.
- Also orient against `lab/lessons.md`, `lab/threads/*.md`, last 50
  lines of `lab/ledger.jsonl`.
- Allocate `run_id` via
  `uv run python -c "from rl_research.contract import next_run_id; print(next_run_id('<thread-slug>'))"`.
- Write `lab/runs/<run_id>/hypothesis.md` AND
  `lab/runs/<run_id>/run_id.txt` (single line: the run_id).
- Halt. Do not write `train.py`. The Engineer will.
- Propose freely — including ideas that look obvious or silly. The
  Reviewer is the gate; do not self-censor at this stage.

### 4.2 `reviewer.md`

```yaml
---
name: reviewer
description: Cheap text checkpoint on a hypothesis. Reads hypothesis.md, writes review.md with verdict. ~30s.
model: sonnet
color: yellow
---
```

Body covers:

- Read `docs/charter.md` §Disqualifiers and `docs/roles/reviewer.md`
  first.
- Read the hypothesis under review (path passed in invocation prompt).
- Write `lab/runs/<run_id>/review.md` with YAML frontmatter:
  `verdict: novel-direction | known-rebadge | needs-sharpening`,
  `reviewed_at: <iso8601>`.
- Body of review: 1–2 paragraphs. Cite specific lines for
  `known-rebadge`. Never propose your own hypothesis.
- A bad-but-novel idea is `novel-direction`. You are not a performance
  reviewer.

### 4.3 `engineer.md`

```yaml
---
name: engineer
description: Implements train.py from approved hypothesis, runs Stage A and Stage B per docs/roles/engineer.md, retries on Stage A failures with fix-N.md notes, records result.json and appends ledger entry. The heavy lifter.
model: opus
color: green
---
```

Body covers:

- Read `docs/charter.md`, `docs/roles/engineer.md`, `docs/contract.md`,
  `docs/benchmarks.md` first.
- Read the approved `hypothesis.md` and `review.md`.
- Write `lab/runs/<run_id>/train.py`. Self-contained. Allowed imports:
  `torch`, `numpy`, `gymnasium`, `dm_control`, `mo_gymnasium`, `ale_py`,
  `tensorboard`, anything in `src/rl_research/`. Forbidden imports:
  `stable_baselines3`, `cleanrl`, `tianshou`, `ray.rllib`, `acme`,
  `coax`, `garage`. Implement from primitives.
- Run Stage A (sanity gate, 5-min cap). On failure: up to 3 retries,
  each writing a `fix-N.md` explaining failure class, root-cause
  guess, what changed, why it should help. **Do not change** the
  algorithm core, learning-mechanism hyperparameters, or allowed
  imports during retries — if a retry would require that, it is a
  Stage A failure: stop and record.
- If Stage A passes, run Stage B (primary benchmark, 2-hour cap). No
  retries on Stage B. Failures are evidence: `killed-budget`,
  `killed-error`, `benchmark-failed`, `completed`.
- Write `config.json` and `result.json` per `docs/contract.md`.
- Validate result.json then append ledger entry:
  `uv run python -c "from rl_research.contract import validate_result_json, append_to_ledger; p='lab/runs/<run_id>/result.json'; validate_result_json(p); append_to_ledger(p)"`.
- Forbidden actions: editing `hypothesis.md`, tuning hyperparameters
  benchmark-specifically, suppressing failures, touching
  `lab/baselines/`.

### 4.4 `curator.md`

```yaml
---
name: curator
description: Per-iteration meta-supervisor. Assigns verdict_curator to recent runs. May also prune lessons, archive threads, write coverage maps, decide promotions, or write lab/HALT_REQUESTED.md if the loop has stopped producing signal.
model: opus
color: purple
---
```

Body covers:

- Read `docs/charter.md`, `docs/roles/curator.md`, `lab/ledger.jsonl`,
  `lab/lessons.md`, `lab/threads/*.md` first.
- For the run just completed: assign `verdict_curator` (`promising` |
  `dead-end` | `inconclusive`) — written via in-place edit of the
  matching ledger entry. Multi-criteria weighting: structural novelty,
  evidence quality, generality across pillars, implementation
  hardness, failure-mode informativeness. Never numerical thresholds.
- **You have meta-supervisor authority.** If you judge that:
  - `lessons.md` has grown noisy or contradictory → prune it (≤ 30
    active lessons; supersede stale entries explicitly).
  - Threads are sprawling or stale → archive them under
    `lab/threads/archive/`.
  - The corpus would benefit from a coverage map, health snapshot, or
    other artifact → write it. The harness does not pre-design these;
    you decide what's useful.
  - The loop has stopped producing signal (e.g., long stretches of
    `dead-end`/`inconclusive`, mode-collapse on one direction,
    Reviewer drift) → write `lab/HALT_REQUESTED.md` with your
    diagnosis. The wrapper will stop spawning new iterations.
- Mass-run promotion: recorded as a new run with `parent_run_id`; runs
  on primary benchmark + at least one additional pillar.
- Anti-patterns: don't promote on raw return, don't archive after one
  failure (≥ 3 negative runs), don't curate based on Researcher
  enthusiasm.

## 5. Slash command (`.claude/commands/iterate.md`)

One command. Pure orchestration glue. No flags, no scope creep.
Repetition is the wrapper's job (§6).

```yaml
---
description: One full pass of the rl-research loop (Researcher → Reviewer → Engineer → Curator).
argument-hint: "[thread-slug]"
allowed-tools: Agent, Read, Bash
model: sonnet
---
```

Body is a state-machine prompt to the orchestrator:

1. **Researcher (Phase 1).** Spawn `researcher` with: *"Propose a
   hypothesis. Read docs/charter.md and docs/roles/researcher.md first.
   Write lab/runs/&lt;run_id&gt;/hypothesis.md AND
   lab/runs/&lt;run_id&gt;/run_id.txt. Halt. Thread hint (may be empty):
   $ARGUMENTS"*.
2. **Recover run_id.** Read the most recently modified
   `lab/runs/*/run_id.txt`.
3. **Reviewer.** Spawn `reviewer` with: *"Review
   lab/runs/&lt;run_id&gt;/hypothesis.md per docs/roles/reviewer.md.
   Write lab/runs/&lt;run_id&gt;/review.md."*
4. **Branch on verdict** (parsed from `review.md` frontmatter):
   - `novel-direction` → step 5.
   - `needs-sharpening` (≤ 1 cycle) or `known-rebadge` (≤ 2 cycles) →
     re-spawn `researcher` with revision context → step 3.
   - Cycles exhausted → write a minimal `result.json` with
     `status="abandoned-rebadge"` or `"abandoned-sharpening"` via
     Bash + `rl_research.contract`, append ledger entry, jump to
     step 7.
5. **Engineer.** Spawn `engineer` with: *"The Reviewer approved
   lab/runs/&lt;run_id&gt;/hypothesis.md. Per docs/roles/engineer.md:
   write train.py, run Stage A then (if pass) Stage B, write
   config.json + result.json, validate and append ledger entry."*
6. **Curator.** Spawn `curator` with: *"The latest run is
   &lt;run_id&gt;. Per docs/roles/curator.md: at minimum assign
   verdict_curator for this run. Use your meta-supervisor judgment for
   anything else the corpus needs."*
7. **Output.** One-line summary: `run_id`, terminal status, verdict
   (if any), wallclock seconds.

The orchestrator does NOT write `hypothesis.md`, `train.py`, or
`result.json` (except the minimal abandoned-* result.json in step 4),
run `train.py` directly, or edit `lessons.md`/`ledger.jsonl`.

## 6. Headless wrapper (`scripts/loop.sh`)

The autonomous loop runs in tmux:

```bash
tmux new -s loop 'cd ~/projects/rl-research && scripts/loop.sh'
```

Detach with `^B d`. Kill with `tmux kill-session -t loop`.

The wrapper is **pure scaffolding**. Two knobs, three behaviors: daily
wallclock cap, halt on `lab/HALT_REQUESTED.md`, cooldown between
iterations. That's all.

```bash
#!/usr/bin/env bash
set -euo pipefail

LOOP_DAILY_HOURS=${LOOP_DAILY_HOURS:-16}
LOOP_COOLDOWN_S=${LOOP_COOLDOWN_S:-60}

start_ts=$(date +%s)
deadline=$((start_ts + LOOP_DAILY_HOURS * 3600))

while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -f lab/HALT_REQUESTED.md ]; then
    echo "$(date -Iseconds) HALT_REQUESTED.md present, stopping" \
      | tee -a lab/iterations.log
    break
  fi

  # If `claude -p` exits non-zero, that's already past Claude Code's
  # internal retry. Log and proceed — iterations are independent.
  claude -p '/iterate' --output-format json \
    | tee -a lab/iterations.log \
    || echo "$(date -Iseconds) /iterate exited non-zero, continuing" \
       | tee -a lab/iterations.log

  sleep "$LOOP_COOLDOWN_S"
done
```

`claude -p --output-format json` already prints `total_cost_usd` per
iteration; that lands in `lab/iterations.log` as part of the JSON
output line. No bespoke aggregation, no OTel plumbing, no cost.json
derivation. `/cost` is the live view; transcripts under
`~/.claude/projects/` are the deep audit trail.

## 7. File layout

What this design adds to the repo:

```
.claude/
  agents/
    researcher.md
    reviewer.md
    engineer.md
    curator.md
  commands/
    iterate.md
scripts/
  loop.sh                                          # +x
docs/
  charter.md                                       # unchanged
  loop.md                                          # state machine + role list updated
  contract.md                                      # +abandoned-* status enum entries
  benchmarks.md                                    # unchanged
  roles/
    researcher.md                                  # Phase 2 stripped
    reviewer.md                                    # minor clarification
    engineer.md                                    # NEW (replaces operator.md)
    curator.md                                     # +meta-supervisor authority
    operator.md                                    # DELETED (renamed to engineer.md)
lab/
  result.schema.json                               # +abandoned-* status entries
  iterations.log                                   # gitignored (wrapper output)
  HALT_REQUESTED.md                                # gitignored (Curator can write)
docs/superpowers/specs/
  2026-05-16-claude-code-harness-design.md         # this doc
```

`.gitignore` adds two lines: `lab/iterations.log` and
`lab/HALT_REQUESTED.md`. The existing `.claude/settings.local.json`
ignore stays. **No `.claude/settings.json` is added** — the user's
existing Claude Code configuration is sufficient and remains unchanged.

## 8. Out of scope (explicit)

- **Hooks** in `.claude/settings.json`. Agent prompts (with
  source-of-truth pointers) are sufficient.
- **Skills directory.** Anything an agent needs is either a charter
  rule (in their prompt) or a `uv run python -c "..."` invocation any
  agent can do via Bash.
- **Predefined corpus artifacts** — `landscape.md`, `health.md`,
  `cost.json`, `trace.jsonl`, `violations.jsonl`. The Curator may
  create such files if it judges them useful; the harness does not
  pre-design them.
- **Per-role tool denylists or read budgets.** Each subagent has the
  full Claude Code tool surface.
- **Wrapper-level adaptive halts** (consecutive-error counting,
  verdict-rate trends, daily USD caps). The only auto-halt is
  `HALT_REQUESTED.md`. Compute policy stays on `train.py`.
- **OpenTelemetry plumbing, derived observability, cost rollups.**
  `claude -p --output-format json` and `/cost` are sufficient.
- **Multi-Researcher parallelism.** Deferred per `docs/loop.md`
  §Concurrency.
- **Slash commands beyond `/iterate`.** `/curate`, `/replay`,
  `/inspect`, `/status` may be added when research surfaces a real
  need.

## 9. Implementation order

When the implementation plan is written, expect this order:

1. **Source-of-truth doc updates.**
   `docs/roles/researcher.md` (strip Phase 2),
   `docs/roles/operator.md` → `docs/roles/engineer.md` (rewrite for
   the merged role), `docs/roles/reviewer.md` (clarify),
   `docs/roles/curator.md` (meta-supervisor authority),
   `docs/loop.md` (state machine), `docs/contract.md` +
   `lab/result.schema.json` (`abandoned-*` status enum entries).
2. **Agent files.** Four `.claude/agents/*.md`. Self-test by manually
   spawning each agent against a stub `lab/runs/test/` dir.
3. **Slash command.** `.claude/commands/iterate.md`. Smoke-test
   `/iterate` against a tiny stub hypothesis (interactive, not
   headless).
4. **Headless wrapper.** `scripts/loop.sh`. Run for one iteration in
   tmux to verify.
5. **`.gitignore` update.** Two lines (`lab/iterations.log`,
   `lab/HALT_REQUESTED.md`).

The implementation plan (writing-plans skill) breaks each into
TDD-style steps with verification points.
