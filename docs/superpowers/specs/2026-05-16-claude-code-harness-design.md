---
title: Claude Code agent harness for rl-research
date: 2026-05-16
status: design — pending user review
---

# Claude Code agent harness for rl-research

## 1. Goal and non-goals

**Goal.** Express the four-role autonomous research loop defined in
[`docs/loop.md`](../../loop.md) entirely in Claude Code primitives, so the
loop runs *headless* (overnight / weekend) with minimal supervision and
maximum reproducibility. The harness is committed to the repo: anyone with
the repo can reproduce the loop bit-for-bit.

**Non-goals.**

- Not a Python orchestration layer. We do not import `claude_agent_sdk`,
  do not write a custom driver, do not re-implement what Claude Code
  already provides. The repository *is* the harness.
- Not blocking enforcement. Hooks are not added by this design. Agent
  prompts (strict, source-of-truth anchored) and skill invariants
  (validate-before-append, etc.) are the enforcement layer. Downstream
  failures and git-diff audit catch the rest.
- Not observability infrastructure. Per-run `cost.json` + `trace.jsonl`
  derived from transcripts at iteration end is sufficient; we do not
  stand up an OpenTelemetry backend.
- Not a multi-agent orchestrator UI. The four roles are sequential
  (file-handoff), not real-time peer debate. See §2.

## 2. Subagents, not agent teams — research finding

The official Claude Code docs draw a sharp distinction. From
`code.claude.com/docs/en/agent-teams` (the experimental
`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` mode):

> *"For sequential tasks where one role completes before the next begins,
>  a single session or subagents are more effective."*

The agent-teams mode is designed for parallel peer challenge (designer vs
coder, two reviewers disagreeing in real time) and explicitly carries
*"higher token cost than single sessions or subagents."* It also has
operational limits: no nested teams, no resume with in-process teammates,
one team per session.

Our loop is sequential by design. Researcher writes `hypothesis.md` →
Reviewer writes `review.md` → Operator writes `result.json` → Curator
updates `lessons.md` and the ledger. Handoffs are **file artifacts on
disk**, not real-time messaging. There is no peer debate inside one
iteration — disagreement is expressed across iterations through the
corpus.

**Therefore: subagents.** Each role is a `.claude/agents/<name>.md` file.
The orchestrator (a slash command) spawns them in order. Context isolation
is per-subagent (each gets its own conversation window); filesystem
isolation is unnecessary because the artifacts are how subagents
communicate.

## 3. Agent definitions (`.claude/agents/`)

Four agents, one per loop role. Frontmatter fields encode operational
posture (model, tools, memory). Bodies restate the hard constraints from
[`docs/charter.md`](../../charter.md) inline so they survive even if the
agent skips re-reading the doc.

### 3.1 Researcher

Frontmatter:

```yaml
---
name: researcher
description: Proposes a novel third-family RL hypothesis and, after Reviewer approval, implements its train.py. Invoke at iteration start, then again after Reviewer verdict is `novel-direction`.
model: opus
tools: Read, Grep, Glob, Write, Edit, Bash, Skill
memory: project
color: blue
---
```

Body covers (drawn verbatim from `docs/roles/researcher.md` plus the
generative-breadth directive):

- **Source of truth (read every invocation):** `docs/charter.md`,
  `docs/roles/researcher.md`, `docs/loop.md` §Researcher, `docs/contract.md`,
  `lab/lessons.md`, `lab/threads/*.md`, last 50 lines of `lab/ledger.jsonl`.
- **Two phases.** Phase 1 = write `lab/runs/<run_id>/hypothesis.md`; halt.
  Allocate `run_id` via
  `uv run python -c "from rl_research.contract import next_run_id; print(next_run_id('<thread-slug>'))"`.
  Phase 2 = write `lab/runs/<run_id>/train.py`; only after Reviewer wrote
  `verdict: novel-direction` in `review.md`.
- **Generative discipline — propose freely.** The Researcher's job is
  breadth and creativity, not pre-filtering. Propose any direction that
  is structurally interesting, including ideas that look obvious or naive.
  The Reviewer is the gate (~30s text check); self-censoring at the
  Researcher stage kills the loop's ability to surface non-obvious
  directions.
- **Awareness — what the Reviewer will reject as `known-rebadge`.** The
  four charter §Disqualifiers are listed *for awareness*, not as a filter:
  - `∇ log π(a|s) · A(s,a)` as the learning signal
  - `r + γ Q(s', a') - Q(s, a)` as the primary update target
  - the Bellman fixed-point as the optimization target
  - cross-entropy of policy vs an expert policy
- **Mechanical constraints (these break the run, not the idea).**
  `train.py` allowed imports: `torch`, `numpy`, `gymnasium`, `dm_control`,
  `mo_gymnasium`, `ale_py`, `tensorboard`, anything in `src/rl_research/`.
  Forbidden imports: `stable_baselines3`, `cleanrl`, `tianshou`,
  `ray.rllib`, `acme`, `coax`, `garage`. CLI must accept `--env --seed
  --total-env-steps --logdir --max-wallclock-s` and write `result.json`.
  Log `progress/param_checksum`. One hypothesis per iteration.
- **Performance is evidence, not objective.** Do not pin the hypothesis
  on a numeric beat-baseline target.

### 3.2 Reviewer

Frontmatter:

```yaml
---
name: reviewer
description: Reviews a Researcher hypothesis for structural novelty before any compute is spent. Cheapest checkpoint in the loop. ~30s text-only.
model: sonnet
tools: Read, Grep, Glob, Write, Skill
memory: project
color: yellow
---
```

`tools` deliberately excludes `Edit` and `Bash` — Reviewer writes only
`review.md`.

Body covers:

- **Source of truth:** `docs/charter.md` §Disqualifiers + §Anti-patterns,
  the hypothesis under review, prior thread runs if relevant.
- **Verdict labels** verbatim from `docs/roles/reviewer.md`:
  `novel-direction` | `known-rebadge` | `needs-sharpening`. Output
  schema includes YAML frontmatter with `verdict:` and `reviewed_at:`.
- **Bias to avoid.** Not a performance reviewer (a bad-but-novel idea is
  `novel-direction`). Not a stylistic reviewer.
- **Edge cases** for PPO citation, imitation hybrids, model-based,
  evolutionary methods — verbatim from the role doc.
- **Output discipline.** Terse. 1–2 paragraphs. Cite specific lines for
  `known-rebadge`. Never propose your own hypothesis.

### 3.3 Operator

Frontmatter:

```yaml
---
name: operator
description: Runs Stage A sanity gate then Stage B primary benchmark. Writes config.json + result.json + ledger entry. Only role with GPU access.
model: opus
tools: Read, Grep, Glob, Write, Edit, Bash, Skill
memory: project
color: green
---
```

Body covers:

- **Source of truth:** `docs/loop.md` §Operator, `docs/contract.md`,
  `docs/benchmarks.md`, the candidate's `hypothesis.md` and `train.py`.
- **Stage A — sanity gate.** Per `sanity_envs[0]`'s first seed, 50k env
  steps, 5-min wallclock cap. Pass criteria: exit 0, no NaN, parameter
  delta non-zero, `eval/return_mean` strictly above
  `lab/baselines/random.json` for that env.
- **Retry budget: 3 per env.** Each retry writes
  `lab/runs/<run_id>/fix-N.md` with failure class, root-cause guess, what
  changed, why it should help. **What you do NOT change in retries:**
  algorithm core update, learning-mechanism hyperparameters, allowed
  imports. If a retry would require changing the algorithm itself, that
  is a Stage A failure — log and stop.
- **Stage B — primary benchmark.** Per seed, 2-hour wallclock cap. **No
  retries on Stage B.** Failures are evidence: `killed-budget`,
  `killed-error`, `benchmark-failed`, `completed`.
- **Terminal step.** Write `result.json`, then invoke skill
  `validate-and-record-run` with the run_id. The skill enforces
  validate-before-append.
- **Forbidden actions.** Editing `hypothesis.md`. Tuning hyperparameters
  benchmark-specifically to make a candidate look better. Suppressing
  failures. Running outside the wallclock cap. Touching `lab/baselines/`.

### 3.4 Curator

Frontmatter:

```yaml
---
name: curator
description: Periodic (every ~10 runs) — assigns verdict_curator to uncurated runs, distills lessons.md, manages thread states, decides mass-run promotions. The only role allowed multi-criteria judgment.
model: opus
tools: Read, Grep, Glob, Write, Edit, Bash, Skill
memory: project
color: purple
---
```

Body covers:

- **Source of truth:** `docs/charter.md`, `lab/ledger.jsonl`,
  `lab/lessons.md`, `lab/threads/*.md`, all uncurated run dirs.
- **Verdict labels:** `promising` | `dead-end` | `inconclusive`. Written
  into matching ledger entries (in-place edit).
- **Multi-criteria weighting** verbatim from `docs/roles/curator.md`:
  structural novelty, evidence quality, generality across pillars,
  implementation hardness, failure-mode informativeness. **Never a
  numerical threshold. Never "X% better than PPO" framing.**
- **Mass-run promotion.** Recorded as a new run with `parent_run_id` in
  notes; runs on primary benchmark + at least one additional pillar.
- **`lessons.md` discipline.** Curated, not append-only. Replace
  superseded entries. Aim for ≤ 30 active lessons.
- **Anti-patterns** verbatim: don't promote on raw return, don't archive
  after one failure (≥ 3 negative runs), don't let `lessons.md` grow past
  100 entries, don't curate based on Researcher enthusiasm.
- **Curator may overrule the Reviewer.** The Reviewer is fast and
  shallow; the Curator is slow and deep. This is stated explicitly so the
  agent doesn't defer mistakenly.

## 4. Slash commands (`.claude/commands/`)

Two commands. Pure orchestration glue. No flags, no scope creep.
Repetition is a separate wrapper (§7); slash commands compose subagents
for one pass.

### 4.1 `/iterate`

```yaml
---
description: One full pass of the rl-research loop (Researcher → Reviewer → Operator, plus Curator if uncurated count ≥ 10).
argument-hint: "[thread-slug]"
allowed-tools: Agent, Read, Bash, Skill
model: sonnet
---
```

Body is a state-machine prompt to the orchestrator:

1. **Researcher Phase 1.** Spawn `researcher` with
   `"Propose a hypothesis (Phase 1 only). Halt after writing
   lab/runs/<run_id>/hypothesis.md. Thread hint (may be empty):
   $ARGUMENTS"`.
   The Researcher's return message must end with a line of the form
   `RUN_ID: <run_id>` so the orchestrator can parse it. As a fallback,
   the orchestrator reads the newest dir under `lab/runs/`.
2. **Reviewer.** Spawn `reviewer` with
   `"Review lab/runs/<run_id>/hypothesis.md. Write review.md."`.
   Read `review.md` frontmatter; parse `verdict`.
3. **Branch on verdict.**
   - `novel-direction` → step 4.
   - `needs-sharpening` (≤ 1 cycle) → re-spawn `researcher` with revision
     prompt → step 2.
   - `known-rebadge` (≤ 2 cycles) → re-spawn `researcher` with revision
     prompt → step 2.
   - Cycles exhausted → end iteration, append a ledger line with
     `status="abandoned-{rebadge,sharpening}"`.
4. **Researcher Phase 2.** Re-spawn `researcher` with `"Reviewer
   approved. Now write lab/runs/<run_id>/train.py per docs/contract.md.
   Halt."`.
5. **Operator.** Spawn `operator` with `"Execute lab/runs/<run_id>/
   train.py per docs/roles/operator.md. Stage A then (if pass) Stage B.
   Write config.json + result.json. Invoke skill validate-and-record-run."`.
6. **Curator check.** Bash:
   `uv run python -c "from rl_research.contract import count_uncurated; print(count_uncurated())"`.
   If ≥ 10, spawn `curator` with `"Run a curation pass per
   docs/roles/curator.md."`.
7. **Cost/trace derivation.** Invoke skill `derive-cost-trace` with
   `run_id`. Non-blocking — do not fail the iteration if the skill errors.
8. **Output.** One-line summary: `run_id`, terminal status, verdict (if
   any), wallclock seconds.

**Halt conditions** (any → end iteration immediately): revision budget
exhausted, Operator Stage A failure after 3 retries (ledger has the
entry), Operator Stage B terminal (ledger has the entry), any subagent
returns an error.

**What the orchestrator does NOT do.** Write `hypothesis.md`, `train.py`,
`result.json`, or any per-run file. Run `train.py` directly. Edit
`lessons.md` or `ledger.jsonl`.

### 4.2 `/curate`

```yaml
---
description: Run a curation pass on demand — assign verdict_curator, distill lessons.md, update threads, decide promotions.
allowed-tools: Agent
model: sonnet
---
```

Body: spawn `curator` with `"Run a curation pass per
docs/roles/curator.md."`. Output a one-line summary.

## 5. Skills (`.claude/skills/`)

Three skills. Each encodes a strict ordering or non-blocking constraint
that role prompts cannot enforce alone. Held back: one-liner Bash wrappers
(`allocate-run-id`, `append-to-ledger`, `scan-train-imports`) — those
inline cleanly in role prompts.

### 5.1 `check-novelty`

Used by the Reviewer. Pattern-scan only — produces evidence; the Reviewer
synthesizes the verdict. Isolates the regex-prone part from the judgment.

```yaml
---
name: check-novelty
description: Pattern-scan a hypothesis.md for the four charter §Disqualifiers. Produces structured findings the Reviewer synthesizes into a verdict. Necessary-not-sufficient evidence.
---
```

Body covers:

- **Inputs:** `$1` = path to `hypothesis.md`.
- **Procedure:** case-insensitive regex scan against four disqualifier
  classes pulled from `docs/charter.md`:
  - **PG family:** `∇ log π`, `nabla log pi`, `log[_ ]prob.*advantage`,
    function names `policy_gradient|actor_critic|ppo_step|grpo_step`.
  - **Q-family:** `r [+] γ Q`, `r [+] gamma [*] Q`, `td[_ ]error`,
    `Q\(s', a'\) - Q\(s, a\)`.
  - **Bellman:** `Bellman fixed-point`, `Bellman optimality`, `DP backup`.
  - **Imitation:** `cross-entropy.*expert`, `BC loss`,
    `behavior cloning vs`.
- **Output:** structured findings, one line per hit (`pattern`, `line`,
  `excerpt`), or `PASS: no disqualifier matches.`
- **What this skill does NOT do.** Write `review.md`. Predict
  performance. Flag stylistic issues. The Reviewer owns the verdict.
- **Failure modes.** Missing file → report and stop. Pattern miss does
  NOT mean the hypothesis is novel; the Reviewer still applies judgment
  for non-pattern rebadges (e.g., method using actor-critic under renamed
  variables).

### 5.2 `validate-and-record-run`

Used by the Operator at iteration end. Encodes the strict ordering from
`docs/contract.md`: `validate_result_json` MUST succeed before
`append_to_ledger`.

```yaml
---
name: validate-and-record-run
description: Validate a run's result.json against the schema, then append a ledger line. STRICT ORDERING — validation MUST succeed before append.
---
```

Body covers:

- **Inputs:** `$1` = run_id.
- **Procedure:**
  1. Verify `lab/runs/$1/result.json` exists; if not, stop with explicit
     error. Do NOT touch the ledger.
  2. `uv run python -c "from rl_research.contract import
     validate_result_json; validate_result_json('lab/runs/$1/
     result.json')"`. On exception: output `VALIDATION FAILED: <message>`
     and stop. Do NOT silently fix `result.json`. Do NOT append.
  3. `uv run python -c "from rl_research.contract import
     append_to_ledger; append_to_ledger('lab/runs/$1/result.json')"`.
  4. Confirm by reading the last line of `lab/ledger.jsonl`.
  5. Output: `Recorded $1: status=<status> verdict_curator=null`.
- **Failure modes.** Validation fail → Operator decides whether to repair
  `result.json` or accept the terminal state. Append fail (lock
  contention) → retry once, then surface.

### 5.3 `derive-cost-trace`

Non-blocking observability. Invoked once per iteration as the
orchestrator's last step.

```yaml
---
name: derive-cost-trace
description: NON-BLOCKING. Derive per-run cost.json and trace.jsonl from Claude Code session transcripts. Failures log to violations.jsonl but never fail the iteration.
disable-model-invocation: true
---
```

`disable-model-invocation: true` so only the orchestrator invokes it.

Body covers:

- **Inputs:** `$1` = run_id.
- **Procedure:**
  1. Locate the four subagent transcripts under
     `~/.claude/projects/<project-hash>/<session>.jsonl` whose tool calls
     touched `lab/runs/$1/`.
  2. Extract usage events (input/output/cache tokens, model_id, role) and
     tool_call events (tool, truncated input, timestamp).
  3. Write `lab/runs/$1/cost.json` and `lab/runs/$1/trace.jsonl` per the
     schemas in §6.
- **Hard constraints (per the observability directive):**
  - Total runtime ≤ 5 seconds. If exceeded, log violation and exit.
  - No network calls. No new Claude invocations. Read-only file ops.
  - On ANY exception: append one line to `lab/violations.jsonl` with
    `type="cost-trace-derivation"` + the error, exit 0. **Never propagate.**
  - The iteration is already complete when this skill runs.

## 6. Observability artifacts

No hooks. No OpenTelemetry. Three committed files plus one gitignored
log. Derived from transcripts that Claude Code already writes — zero new
infrastructure.

### `lab/runs/<run_id>/cost.json` (committed)

```json
{
  "run_id": "0007-energy-credit",
  "by_role": {
    "researcher": {"input": 12480, "output": 3120, "cache_read": 8200,
                   "cache_creation": 4200, "model": "claude-opus-4-7", "usd": 0.142},
    "reviewer":   {"...": "..."},
    "operator":   {"...": "..."},
    "curator":    {"...": "..."}
  },
  "total_usd": 0.84,
  "total_tokens": 91200
}
```

### `lab/runs/<run_id>/trace.jsonl` (committed)

One tool call per line, sorted by timestamp:

```json
{"ts":"2026-05-16T11:23:42Z","role":"operator","tool":"Bash","input_excerpt":"uv run python lab/runs/0007.../train.py --env CartPole-v1...","status":"ok","duration_ms":312000}
```

### `lab/violations.jsonl` (committed, append-only)

Skill-internal violations only. Schema:

```json
{"ts":"2026-05-16T11:30:14Z","type":"cost-trace-derivation","run_id":"0007-energy-credit","error":"transcript file not found"}
```

### `lab/iterations.log` (gitignored)

Orchestrator one-line summaries from the headless wrapper. Operational
noise, not corpus.

`/cost` UI and full transcripts under `~/.claude/projects/...` remain
available as the live source of truth — these committed artifacts are the
*reproducibility* slice.

## 7. Headless wrapper (`scripts/loop.sh`)

The autonomous loop runs in tmux:

```bash
tmux new -s loop 'cd ~/projects/rl-research && scripts/loop.sh'
```

Detach with `^B d`. Kill with `tmux kill-session -t loop`.

Three knobs as env vars (defaults inline). No flags. No internal state —
the ledger is the state.

```bash
#!/usr/bin/env bash
set -euo pipefail

LOOP_DAILY_HOURS=${LOOP_DAILY_HOURS:-16}
LOOP_COOLDOWN_S=${LOOP_COOLDOWN_S:-60}
LOOP_USD_CAP=${LOOP_USD_CAP:-50}   # daily soft cap, sourced from cost.json totals

start_ts=$(date +%s)
deadline=$((start_ts + LOOP_DAILY_HOURS * 3600))

while [ "$(date +%s)" -lt "$deadline" ]; do
  spent=$(uv run python -c "from rl_research.contract import usd_spent_today; print(usd_spent_today())")
  awk_cmp=$(awk -v a="$spent" -v b="$LOOP_USD_CAP" 'BEGIN{print (a>b)?1:0}')
  if [ "$awk_cmp" = "1" ]; then
    echo "$(date -Iseconds) USD cap reached ($spent ≥ $LOOP_USD_CAP), stopping" \
      | tee -a lab/iterations.log
    break
  fi

  claude -p '/iterate' --output-format json \
    | tee -a lab/iterations.log

  sleep "$LOOP_COOLDOWN_S"
done
```

`usd_spent_today()` is a small new helper in `src/rl_research/contract.py`
that sums today's `cost.json` totals.

## 8. File layout

What this design adds to the repo (everything committed except where
noted):

```
.claude/
  agents/
    researcher.md
    reviewer.md
    operator.md
    curator.md
  commands/
    iterate.md
    curate.md
  skills/
    check-novelty/SKILL.md
    validate-and-record-run/SKILL.md
    derive-cost-trace/SKILL.md
scripts/
  loop.sh                                          # +x
src/rl_research/
  contract.py                                      # +count_uncurated, +usd_spent_today
lab/
  violations.jsonl                                 # committed, append-only (skills only)
  iterations.log                                   # gitignored
  runs/<run_id>/
    cost.json                                      # per-run, committed
    trace.jsonl                                    # per-run, committed
docs/superpowers/specs/
  2026-05-16-claude-code-harness-design.md         # this doc
```

`.gitignore` adds one line: `lab/iterations.log`. The existing
`.claude/settings.local.json` ignore stays. **No `.claude/settings.json`
is added** — the user's existing Claude Code configuration is sufficient
and remains unchanged.

## 9. Out of scope (explicit)

- **Hooks** in `.claude/settings.json`. Agent prompts + skill invariants
  are the enforcement layer. Recurring oversteps that prompts don't catch
  may justify warn-only hooks later.
- **Custom base URL plumbing.** The user's existing Claude Code
  configuration handles this; the harness picks up
  `ANTHROPIC_BASE_URL` from the shell with no special handling.
- **Multi-Researcher parallelism.** Deferred until ≥ 1 working candidate
  exists per `docs/loop.md` §Concurrency.
- **OpenTelemetry / Prometheus.** Per-run committed artifacts are the
  reproducibility slice; live observability is `/cost` + transcripts.
- **Slash commands beyond `/iterate` and `/curate`.** `/iterate-thread`,
  `/replay`, `/inspect`, `/status` may be added when research surfaces a
  real need.

## 10. Implementation order

When the implementation plan is written, expect this order:

1. **`src/rl_research/contract.py` helpers.** Add `count_uncurated()` and
   `usd_spent_today()`. Tests.
2. **Agent files.** Four `.claude/agents/*.md`. Self-test by manually
   spawning each agent against a stub `lab/runs/test/` dir.
3. **Skill files.** Three `.claude/skills/*/SKILL.md`. Each has a small
   smoke test invocation.
4. **Slash commands.** Two `.claude/commands/*.md`. Smoke-test
   `/iterate` against a tiny stub hypothesis.
5. **Headless wrapper.** `scripts/loop.sh`. Run for one iteration in
   tmux to verify; check `lab/iterations.log` and per-run artifacts.
6. **`.gitignore` update.** One line.

The implementation plan (writing-plans skill) breaks each into TDD-style
steps with verification points.
