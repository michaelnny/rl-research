---
description: Autonomous research loop. Repeatedly runs Researcher → Reviewer → Engineer → Curator iterations until a halt condition trips. Designed for unattended weeks-long runs with NO user prompts between iterations.
argument-hint: "[--max-iters N] [--max-hours H]"
allowed-tools: Agent, Read, Write, Edit, Bash
model: sonnet
---

You are running the autonomous research loop of the rl-research project.
Your job is to keep firing iterations until a halt condition trips, with
NO human in the loop between iterations.

Each iteration: Researcher (Phase 1) → Reviewer → branch on verdict →
Researcher (Phase 2) → Engineer → Curator. All handoffs are file-based;
all subagents live under `.claude/agents/`.

## Arguments

Parse `$ARGUMENTS` for these optional flags (whitespace-separated):

- `--max-iters N` — stop after N completed iterations (default: unbounded).
- `--max-hours H` — stop after H wallclock hours from when this command
  started (default: unbounded).

If neither is given, the loop runs unbounded and stops only on the
file-based halt conditions below.

## State to track across iterations

`consecutive_errors` is **derived from `worklogs/ledger.jsonl` at each
pre-flight** — NOT held in your own context. Read the trailing N lines
of the ledger; count how many of the trailing entries have
`status` in {`killed-error`, `forbidden-import`} BEFORE a non-error
entry breaks the streak. This survives Claude context compaction and
session restarts; the in-context counter would not.

`iters_done` and `started_at` are tracked in your context for the
optional `--max-iters` and `--max-hours` guards. They reset when the
session restarts — that is acceptable, because those guards are
user-supplied conveniences, not safety mechanisms. The actual safety
mechanism (`consecutive_errors`) is durable.

## Pre-flight (before EACH iteration)

In this exact order. Any failure → log reason and exit cleanly.

1. **Halt-requested file.** If `worklogs/HALT_REQUESTED.md` exists, read
   it, print its contents, exit.
2. **Max iters / hours.** If `iters_done >= max_iters` (when set), exit.
   If wallclock since `started_at` >= `max_hours` (when set), exit.
3. **Disk cap.** `du -s worklogs/runs 2>/dev/null | awk '{print $1}'` —
   if > 5_000_000 (KB, ~5 GB), write a `HALT_REQUESTED.md` saying
   "disk cap exceeded" and exit.
4. **Substrate dirty.**
   `git status --porcelain harness.py run_panel.py baselines.json
    train.py prior_attempts.md worklogs/TEMPLATE.md`
   — if non-empty, write a `HALT_REQUESTED.md` with the diff and exit.
   The substrate must be clean before any iteration runs.

   **`train.py` is in this list specifically to detect contamination
   from a crashed prior iteration.** If a previous Engineer crashed
   between substituting the candidate and restoring the substrate,
   `train.py` at the repo root contains the candidate, not the
   substrate floor. Halting here prevents the next Engineer's `cp
   train.py train.py.bak` from baking that contamination in
   permanently.
5. **Consecutive errors.** Derive `consecutive_errors` from
   `worklogs/ledger.jsonl` (read trailing lines, count consecutive
   error statuses). If `consecutive_errors >= 3`, write a
   `HALT_REQUESTED.md` saying "three consecutive errored iterations —
   likely systemic regression, see worklogs/runs/*/result.json" and
   exit. (Systemic-regression circuit breaker.)

## One iteration — the state machine

### Step 0 — allocate run_id

```bash
uv run python scripts/next_run_id.py auto
```

Capture the printed `run_id` (format `<YYYYMMDD>-<NN>-<slug>`). Then
`mkdir -p worklogs/runs/<run_id>`.

### Step 1 — Researcher Phase 1

Spawn the `researcher` subagent (`subagent_type: researcher`) with this
exact prompt:

> Phase 1 only. Run ID is `<run_id>`. Read `prior_attempts.md` first,
> then propose a candidate by writing
> `worklogs/runs/<run_id>/hypothesis.md`. Halt after writing the file —
> do not write `train.py` yet.

After the subagent returns, verify `worklogs/runs/<run_id>/hypothesis.md`
exists. If not, write a stub `result.json` with
`status: killed-error, error: "researcher phase 1 produced no hypothesis"`
and skip to Step 5 (Curator).

### Step 2 — Reviewer

Spawn the `reviewer` subagent with this exact prompt:

> Review `worklogs/runs/<run_id>/hypothesis.md`. Write
> `worklogs/runs/<run_id>/review.md` with frontmatter
> `verdict: novel-direction | known-rebadge | needs-sharpening` and 1–2
> paragraphs of reasoning.

After return, parse the `verdict:` line out of `review.md`'s frontmatter.

### Step 3 — Branch on verdict

- **`novel-direction`** → continue to Step 4.

- **`needs-sharpening`** → respawn `researcher` ONCE with prompt:
  > Phase 1 revision. Reviewer wrote `worklogs/runs/<run_id>/review.md`.
  > Address the missing slots and rewrite hypothesis.md.

  Then respawn `reviewer` for a re-review. If the new verdict is still
  not `novel-direction`, write a stub `result.json` with
  `status: abandoned-sharpening` and skip to Step 5.

- **`known-rebadge`** → write a stub `result.json`:
  ```json
  {
    "run_id": "<run_id>",
    "stage": null,
    "envs": [],
    "scores": {},
    "beat_random": 0,
    "beat_strong": 0,
    "wallclock_s": 0.0,
    "n_retries": 0,
    "status": "abandoned-rebadge",
    "commit": "<git rev-parse HEAD>"
  }
  ```
  Then skip to Step 5.

### Step 4 — Researcher Phase 2 + Engineer

Researcher Phase 2 — spawn `researcher` with this exact prompt:

> Phase 2. Run ID is `<run_id>`. Reviewer approved with verdict
> `novel-direction`. Read `worklogs/runs/<run_id>/hypothesis.md` and
> `worklogs/runs/<run_id>/review.md`, then write
> `worklogs/runs/<run_id>/train.py` matching the substrate contract.

Verify `worklogs/runs/<run_id>/train.py` exists. If not, write a stub
`result.json` with `status: killed-error, error: "researcher phase 2
produced no train.py"` and skip to Step 5.

Engineer — spawn `engineer` with this exact prompt:

> Run `worklogs/runs/<run_id>/train.py` through the panel. Pick the stage
> from the hypothesis primary axis. Write
> `worklogs/runs/<run_id>/result.json` and ensure the repo-root `train.py`
> is restored from `train.py.bak` before you exit.

After return, verify `worklogs/runs/<run_id>/result.json` exists AND
that the repo-root `train.py` is unchanged
(`git diff --quiet train.py`). If `train.py` is dirty, restore it via
`git checkout -- train.py` and note the slip in your post-iteration log.

### Step 5 — Curator

Spawn the `curator` subagent with this exact prompt:

> Synthesize iteration `<run_id>`. Read hypothesis.md, review.md,
> result.json, panel.txt (if present), then write `curator.md`,
> append to `worklogs/ledger.jsonl`, and update the corpus per the
> verdict-conditional outputs in your agent definition.

After return, verify:
- `worklogs/runs/<run_id>/curator.md` exists.
- `worklogs/ledger.jsonl` has at least one line whose `run_id` matches.
- Substrate files are still clean (`git status --porcelain harness.py
  run_panel.py baselines.json` is empty).

**Recovery on missing ledger line.** If `curator.md` exists but the
ledger has no matching `run_id`, the Curator crashed between writing
its verdict file and appending to the ledger. Respawn `curator` ONCE
with this exact prompt:

> Recovery for `<run_id>`: your `curator.md` exists but no matching
> ledger line was appended. Re-read `curator.md` and append the
> corresponding line to `worklogs/ledger.jsonl`. Do not rewrite
> `curator.md` or modify `prior_attempts.md` / `worklogs/{attempts,
> candidates}/`.

If after the recovery attempt the ledger still has no matching line,
write `worklogs/HALT_REQUESTED.md` saying "Curator failed to append
ledger line for `<run_id>` after recovery attempt" and exit. Do NOT
proceed to the next iteration with a missing ledger entry — the
post-iteration step reads the ledger and would silently miscount
errors.

## Post-iteration update

1. `iters_done += 1`.
2. Read the last line of `worklogs/ledger.jsonl`. Parse `status`.
3. Print one summary line to stdout:
   `[research] iter=<iters_done> run_id=<...> verdict=<...>
    beat_strong=<N> status=<...>`.

(`consecutive_errors` is NOT tracked here — it is derived from the
ledger at each pre-flight; see "State to track across iterations".)

Then loop back to pre-flight.

## Cool-down between iterations

A short pause is useful for two reasons: (a) lets file-system writes
flush, (b) gives a watching human a chance to write
`HALT_REQUESTED.md`. Sleep 10s between iterations:

```bash
sleep 10
```

## Stop / resume

When `/research` exits cleanly (any halt reason), the corpus is in a
consistent state — every `worklogs/runs/<run_id>/` either has a
`curator.md` or is missing one because the iteration was interrupted
mid-run. The user can:

- Inspect `worklogs/ledger.jsonl` to see what happened.
- Inspect `worklogs/HALT_REQUESTED.md` for the halt reason.
- Run `/curate` to clean up any uncurated runs.
- Delete `worklogs/HALT_REQUESTED.md` and re-invoke `/research` to
  resume.

## Forbidden actions

- Editing `harness.py`, `run_panel.py`, `baselines.json`,
  `prior_attempts.md`, `worklogs/TEMPLATE.md`, `worklogs/attempts/*`.
- Skipping pre-flight checks "just to fit one more iteration in."
- Modifying `worklogs/HALT_REQUESTED.md` to suppress a halt — if a halt
  fired, the loop must exit. The user removes the file when they want
  to resume.
- Asking the user anything between iterations. The whole point of
  `/research` is unattended operation. If you find yourself wanting
  to ask the user a question, instead write
  `worklogs/HALT_REQUESTED.md` with the question and exit.

## Final summary on exit

Print to stdout:
- Halt reason
- `iters_done`
- Wallclock duration
- Last 5 ledger entries (run_id + verdict + status)
