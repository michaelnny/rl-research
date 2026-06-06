---
description: Autonomous research loop. Researcher → Reviewer → (Engineer →) Curator. Designed for unattended weeks-long runs. Empty-handed Researcher turns and Reviewer rejections are normal; the loop only halts on proven candidates, the disk cap, the consecutive-error circuit breaker, or an explicit halt request.
argument-hint: "[--max-iters N] [--max-hours H]"
allowed-tools: Agent, Read, Write, Edit, Bash
---

You are running the autonomous research loop of the rl-research project.
Each iteration: Researcher (proposes math or returns empty-hand) →
Reviewer (adversarial referee, default reject) → branch on verdict →
Engineer (only if Reviewer passed) → Curator (synthesizes the result).

**Empty-handed Researcher turns and Reviewer rejections are the
expected dominant outcome.** The loop is calibrated for a low-frequency
search of high-quality candidates, not a high-throughput search of
heuristics. A healthy week of unattended operation might produce 100+
empty-hand notes, ~10 reviewer-rejected proposals, ~1 reviewer-passed
proposal, and ideally one promotion that halts for user review.

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
of the ledger; count how many of the trailing entries have `status` in
{`killed-error`, `forbidden-import`} BEFORE a non-error entry breaks
the streak. This survives Claude context compaction and session
restarts.

`iters_done` and `started_at` are tracked in your context for the
optional `--max-iters` and `--max-hours` guards. They reset when the
session restarts — that is acceptable, because those guards are
user-supplied conveniences, not safety mechanisms.

## Pre-flight (before EACH iteration)

In this exact order. Any failure → log reason and exit cleanly.

1. **Halt-requested file.** If `worklogs/HALT_REQUESTED.md` exists,
   read it, print its contents, exit. The Curator writes this file on
   `proven-on-substrate` to halt the loop for user review; do not
   suppress it.
2. **Max iters / hours.** If `iters_done >= max_iters` (when set), exit.
   If wallclock since `started_at` >= `max_hours` (when set), exit.
3. **Disk cap.** `du -s worklogs/runs 2>/dev/null | awk '{print $1}'`
   — if > 5_000_000 (KB, ~5 GB), write a `HALT_REQUESTED.md` saying
   "disk cap exceeded" and exit.
4. **Substrate dirty.**
   `git status --porcelain harness.py run_panel.py baselines.json
    train.py worklogs/TEMPLATE.md`
   — if non-empty, write a `HALT_REQUESTED.md` with the diff and exit.
   The substrate must be clean before any iteration runs.

   The check intentionally lists ONLY sealed-immutable files plus
   `train.py` for contamination detection. `prior_attempts.md`,
   `worklogs/attempts/*`, `worklogs/runs/*`, `worklogs/promotions/*`,
   `worklogs/exemplars.md`, and `worklogs/ledger.jsonl` are corpus
   files owned by the Curator subagent and may be modified between
   iterations — they MUST NOT block the next iteration. Step 6
   (auto-commit) keeps the working tree clean across iterations.

5. **Consecutive errors.** Derive `consecutive_errors` from
   `worklogs/ledger.jsonl` (trailing lines, count consecutive error
   statuses). If `consecutive_errors >= 3`, write a
   `HALT_REQUESTED.md` saying "three consecutive errored iterations —
   likely systemic regression, see worklogs/runs/*/result.json" and
   exit. Empty-hand and reviewer-rejected verdicts do **not** count
   as errors for this circuit breaker.

## One iteration — the state machine

### Step 0 — allocate run_id

```bash
uv run python scripts/next_run_id.py auto
```

Capture the printed `run_id`. Then `mkdir -p worklogs/runs/<run_id>`.

### Step 1 — Researcher

Spawn the `researcher` subagent with this exact prompt:

> Run ID is `<run_id>`. Read `worklogs/exemplars.md` and
> `prior_attempts.md`. Either propose a candidate that meets the
> four-slot contract (principle, derivation, primitive, theorem) by
> writing `worklogs/runs/<run_id>/hypothesis.md`, or write the
> empty-hand note if nothing reaches the bar. Halt after writing the
> file. You do not write or read any code.

After the subagent returns, verify
`worklogs/runs/<run_id>/hypothesis.md` exists. If not, write a stub
`result.json` with `status: killed-error,
error: "researcher produced no hypothesis"` and skip to Step 5
(Curator).

Then read the hypothesis file's first line to detect the empty-hand
note (header of the form `# <run_id> — empty-handed`). If empty-hand:
write a stub `result.json`:

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
  "status": "no-proposal",
  "commit": "<git rev-parse HEAD>"
}
```

Skip to Step 5 (Curator) — do not spawn the Reviewer or Engineer.

### Step 2 — Reviewer

Spawn the `reviewer` subagent with this exact prompt:

> Review `worklogs/runs/<run_id>/hypothesis.md`. Check the math step
> by step. Search for the principle and update rule to confirm the
> proposal is not a renamed published method. Write
> `worklogs/runs/<run_id>/review.md` with frontmatter
> `verdict: pass | revise | reject` and reasoning.

After return, parse the `verdict:` line out of `review.md`'s
frontmatter.

### Step 3 — Branch on verdict

- **`pass`** → continue to Step 4.

- **`revise`** → respawn `researcher` ONCE with prompt:

  > Revise the hypothesis. Reviewer wrote
  > `worklogs/runs/<run_id>/review.md`. Address the specific fixes
  > listed in the Decision section and rewrite hypothesis.md. Do not
  > propose a different algorithm; only fix the issues the Reviewer
  > named. If the fixes would require abandoning the principle,
  > replace the file with the empty-hand note instead.

  Then respawn `reviewer` for a re-review. Whatever the second
  Reviewer's verdict is, accept it: `pass` → Step 4, anything else →
  write a stub `result.json` with `status: abandoned-revise` and skip
  to Step 5.

- **`reject`** → write a stub `result.json`:
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
    "status": "reviewer-rejected",
    "commit": "<git rev-parse HEAD>"
  }
  ```
  Then skip to Step 5.

### Step 4 — Engineer

Spawn `engineer` with this exact prompt:

> Run ID is `<run_id>`. Reviewer passed the proposal. Read
> `worklogs/runs/<run_id>/hypothesis.md` and
> `worklogs/runs/<run_id>/review.md`, then author
> `worklogs/runs/<run_id>/train.py` realizing the update rule
> faithfully. Pick the panel stage that exercises the principle.
> Run the panel, write `worklogs/runs/<run_id>/result.json`, and
> ensure the repo-root `train.py` is restored from `train.py.bak`
> before you exit.

After return, verify both `worklogs/runs/<run_id>/train.py` and
`worklogs/runs/<run_id>/result.json` exist AND that the repo-root
`train.py` is unchanged (`git diff --quiet train.py`). If `train.py`
is dirty, restore it via `git checkout -- train.py` and note the slip.
If `worklogs/runs/<run_id>/train.py` does not exist (Engineer crashed
before authoring), the Engineer should have written `result.json` with
`status: killed-error`; if even `result.json` is missing, write a stub
with `status: killed-error,
error: "engineer produced neither train.py nor result.json"` and
continue to Step 5.

### Step 5 — Curator

Spawn the `curator` subagent with this exact prompt:

> Synthesize iteration `<run_id>`. Read hypothesis.md, review.md (if
> present), result.json, panel.txt (if present), then write
> `curator.md`, append to `worklogs/ledger.jsonl`, and update the
> corpus per the verdict-conditional outputs in your agent definition.
> If the run is `proven-on-substrate`, also write
> `worklogs/HALT_REQUESTED.md` so the loop halts for user review.

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
> `curator.md` or modify `prior_attempts.md` / `worklogs/attempts/` /
> `worklogs/promotions/`.

If after the recovery attempt the ledger still has no matching line,
write `worklogs/HALT_REQUESTED.md` saying "Curator failed to append
ledger line for `<run_id>` after recovery attempt" and exit.

### Step 6 — auto-commit the iteration

Once Curator verification has passed:

```bash
git add worklogs/ prior_attempts.md
if ! git diff --cached --quiet; then
  git commit -m "iter <run_id>: <verdict>" -m "status: <status>" \
    || { echo "auto-commit failed for <run_id>" \
         > worklogs/HALT_REQUESTED.md; exit 0; }
fi
```

Use targeted paths (`worklogs/ prior_attempts.md`), not `git add -A`.
Empty-hand and reviewer-rejected iterations may produce no diff
beyond a ledger line — that is fine; the commit just captures the
ledger update.

## Post-iteration update

1. `iters_done += 1`.
2. Read the last line of `worklogs/ledger.jsonl`. Parse `verdict` and
   `status`.
3. Print one summary line to stdout:
   `[research] iter=<iters_done> run_id=<...> verdict=<...>
    status=<...>`.

Then loop back to pre-flight.

## Cool-down between iterations

Sleep 10s between iterations:

```bash
sleep 10
```

Empty-hand iterations will return very quickly, so the cool-down also
serves to keep the loop from spinning at full speed when the
Researcher has nothing to propose. If you find yourself in a long
streak of empty-hand verdicts (≥ 20), increase the cool-down to 60s
to reduce wasted cycles — but do **not** halt for emptiness. Empty is
the correct behavior; the loop survives weeks of empty turns.

## Stop / resume

When `/research` exits cleanly (any halt reason), the corpus is in a
consistent state. The user can:

- Inspect `worklogs/ledger.jsonl` and `worklogs/HALT_REQUESTED.md`.
- For a `proven-on-substrate` halt, inspect
  `worklogs/promotions/<run_id>.md`.
- Run `/curate` to clean up any uncurated runs.
- Delete `worklogs/HALT_REQUESTED.md` and re-invoke `/research` to
  resume.

## Forbidden actions

These bind the orchestrator (this `/research` loop), not the subagents.

- The orchestrator never edits `harness.py`, `run_panel.py`,
  `baselines.json`, `prior_attempts.md`, `worklogs/exemplars.md`,
  `worklogs/TEMPLATE.md`, `worklogs/attempts/*`, or any subagent's
  run directory contents. Subagents own their files.
- Skipping pre-flight checks "just to fit one more iteration in."
- Modifying `worklogs/HALT_REQUESTED.md` to suppress a halt.
- Auto-committing with `git add -A`.
- Asking the user anything between iterations.
- Counting `empty-hand` or `reviewer-rejected` as errors. They are
  the loop's expected dominant outcome.

## Final summary on exit

Print to stdout:
- Halt reason
- `iters_done`
- Wallclock duration
- Last 5 ledger entries (run_id + verdict + status)
