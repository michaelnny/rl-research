---
description: Autonomous empirical-probe research loop. Researcher -> schema validation -> Reviewer triage -> Engineer ladder/ablation run -> Curator. Halts on promotions, disk cap, repeated infrastructure errors, repeated no-panel iterations, or explicit halt request.
argument-hint: "[--max-iters N] [--max-hours H]"
allowed-tools: Agent, Read, Write, Edit, Bash
---

You are running the autonomous research loop of the rl-research project.
The loop is now probe-first: mathematical novelty and coherence are
screened before compute, but a full theorem is not required before the
Engineer runs the fixed panel. The panel must become part of the search
pressure.

Each iteration is:

Researcher writes a probe plus `candidate.json`, a negative closure, or
an empty-hand note -> the orchestrator validates the candidate schema ->
Reviewer performs novelty/coherence triage -> Engineer runs every
Reviewer-approved probe through the panel ladder and ablation -> Curator
converts the result into corpus signal.

The Reviewer is still adversarial about rebadges, dead families, and
incoherent updates. The Reviewer is no longer a theorem gate. Empirical
results are allowed to arrive before a convergence proof, matching the
historical path by which the exemplars in `worklogs/exemplars.md` became
worth theorizing about.

## Arguments

Parse `$ARGUMENTS` for optional flags:

- `--max-iters N` - stop after N completed iterations.
- `--max-hours H` - stop after H wallclock hours from command start.

If neither is given, run until a file-based halt condition triggers.

## State

`consecutive_errors` and `consecutive_no_panel` are derived from
`worklogs/ledger.jsonl` at each pre-flight. Do not keep them only in
context; context compaction and session restarts must not erase safety
state.

For the no-panel counter, count only ledger entries whose JSON object has
`"mode":"probe-v1"`. Legacy entries without that field do not count.

## Pre-flight Before Each Iteration

Stop cleanly on the first triggered condition.

1. **Halt request.** If `worklogs/HALT_REQUESTED.md` exists, read it,
   print its contents, and exit.
2. **Max iters / hours.** If the user-supplied guard is reached, exit.
3. **Disk cap.** Run `du -s worklogs/runs 2>/dev/null | awk '{print $1}'`.
   If it is greater than 5,000,000 KB, write `worklogs/HALT_REQUESTED.md`
   with "disk cap exceeded" and exit.
4. **Substrate dirty.** Run
   `git status --porcelain harness.py run_panel.py baselines.json train.py worklogs/TEMPLATE.md`.
   If non-empty, write `worklogs/HALT_REQUESTED.md` with the diff and
   exit. Corpus files are allowed to change between iterations; the
   fixed substrate is not.
5. **Consecutive infrastructure errors.** Read trailing ledger entries
   and count consecutive statuses in `{killed-error, killed-budget,
   forbidden-import}`. If the count is 3 or more, write a halt file:
   "three consecutive errored iterations - likely systemic regression".
6. **No-panel streak.** Among trailing `mode=probe-v1` entries, count
   consecutive entries with `stage` equal to null before a non-null
   stage breaks the streak. If the count reaches 8, write a halt file:
   "eight probe-v1 iterations produced no panel run - triage or
   Researcher has become too restrictive". This replaces the old
   empty-hand-streak breaker.

## One Iteration

### Step 0 - Allocate run_id

```bash
uv run python scripts/next_run_id.py auto
```

Capture the printed `run_id`. Create `worklogs/runs/<run_id>`.

### Step 1 - Researcher

Spawn `researcher` with this exact prompt:

> Run ID is `<run_id>`. Read `worklogs/exemplars.md`,
> `prior_attempts.md`, recent `worklogs/ledger.jsonl` entries, and
> recent hypothesis/curator summaries per your agent definition.
> Produce `worklogs/runs/<run_id>/hypothesis.md`. For a runnable
> `[probe]`, also produce `worklogs/runs/<run_id>/candidate.json` using
> the schema in your agent definition. Prefer a runnable `[probe]`:
> one-sentence principle, typed primitive, derivation sketch, update
> rule, empirical claim, ablation plan, novelty boundary, and explicit
> proof debt. A full theorem is not required. Write a
> `[negative-closure]` only when you can close a prior direction with a
> checkable negative result. Write `empty-handed` only when no coherent,
> non-rebadged probe can be made after reading recent failures. Do not
> read or write code. Halt after writing.

After return, verify `hypothesis.md` exists. If it does not, write a
minimal `result.json` with `status: killed-error` and skip to Curator.

Detect type from the first line:

- `# <run_id> - empty-handed` or `# <run_id> -- empty-handed`:
  write a stub `result.json` with `stage: null`, `status: no-proposal`,
  `mode: probe-v1`, then skip to Curator.
- Header containing `[negative-closure]`: continue to Reviewer.
- Header containing `[probe]`: verify `candidate.json` exists and run:

  ```bash
  uv run python scripts/validate_candidate.py worklogs/runs/<run_id>/candidate.json
  ```

  If validation fails, respawn Researcher once with:

  > `candidate.json` is missing or invalid for `<run_id>`. Run
  > `uv run python scripts/validate_candidate.py worklogs/runs/<run_id>/candidate.json`,
  > fix only the schema/mismatch issues, and leave the core principle
  > unchanged. If the principle cannot be represented honestly in the
  > schema, replace the hypothesis with an empty-hand note.

  Re-read the hypothesis header. If it became empty-handed, write the
  `no-proposal` stub and skip to Curator. If it remains a probe, re-run
  validation. If validation still fails, write a stub `result.json` with
  `status: candidate-invalid`, `stage: null`, `mode: probe-v1`, then
  skip to Curator.
- Any other header: continue to Reviewer, but the Reviewer should treat
  missing `[probe]` as a format issue and normally return `revise` or
  `reject`.

### Step 2 - Reviewer Triage

Spawn `reviewer` with this exact prompt:

> Review `worklogs/runs/<run_id>/hypothesis.md` and, for probes,
> `worklogs/runs/<run_id>/candidate.json`. This is a probe-first loop.
> Your job is to block rebadges, dead-family shapes, incoherent
> derivations, non-typed primitives, non-implementable update rules,
> missing/weak ablations, and vector scalarization. Do not require a
> convergence theorem before compute; explicit proof debt is allowed.
> For `[negative-closure]`, verify the negative result. Write
> `worklogs/runs/<run_id>/review.md`
> with frontmatter `verdict: probe | revise | reject | negative-closure`
> and `hypothesis_type: probe | negative-closure | empty-hand`.

Parse `verdict:` from `review.md`.

### Step 3 - Branch

- **`probe`** -> Engineer.
- **`negative-closure`** -> write stub `result.json` with `stage: null`,
  `status: negative-closure`, `mode: probe-v1`, then Curator.
- **`revise`** -> respawn `researcher` once:

  > Revise `worklogs/runs/<run_id>/hypothesis.md` using the Decision
  > section in `review.md`. Keep the same core principle; fix only the
  > issues named. If the principle cannot survive, replace the file with
  > an empty-hand note. Do not write code.

  Then respawn `reviewer` once. Accept the second verdict. `probe` goes
  to Engineer, `negative-closure` gets the stub above, anything else gets
  a stub `result.json` with `status: reviewer-rejected` and `mode:
  probe-v1`, then Curator.
- **`reject`** -> write stub `result.json` with `stage: null`,
  `status: reviewer-rejected`, `mode: probe-v1`, then Curator.

Each stub result must include:

```json
{
  "run_id": "<run_id>",
  "mode": "probe-v1",
  "stage": null,
  "envs": [],
  "scores": {},
  "beat_random": 0,
  "beat_strong": 0,
  "wallclock_s": 0.0,
  "n_retries": 0,
  "status": "<status>",
  "commit": "<git rev-parse HEAD>"
}
```

### Step 4 - Engineer

Spawn `engineer` with this exact prompt:

> Run ID is `<run_id>`. Reviewer approved this hypothesis for empirical
> probe. Read `CLAUDE.md`, `README.md`, `harness.py`, `run_panel.py`,
> repo-root `train.py`, `worklogs/runs/<run_id>/hypothesis.md`,
> `worklogs/runs/<run_id>/review.md`, and `candidate.json`. Author
> `worklogs/runs/<run_id>/train.py` realizing the probe's update rule
> faithfully, plus `worklogs/runs/<run_id>/train_ablate.py` implementing
> the ablation plan. Run the smoke -> claim -> ablation -> conditional
> confirmation ladder with
> `uv run python scripts/run_probe_ladder.py worklogs/runs/<run_id>`,
> verify `panel-*.txt` files and `result.json`, and leave repo-root
> `train.py` unchanged.

After return, verify:

- `worklogs/runs/<run_id>/train.py` exists.
- `worklogs/runs/<run_id>/train_ablate.py` exists.
- `worklogs/runs/<run_id>/result.json` exists.
- `git diff --quiet train.py` succeeds.

If the Engineer left root `train.py` dirty, write `worklogs/HALT_REQUESTED.md`
with "Engineer left train.py dirty for `<run_id>`" and exit rather than
silently continuing on a contaminated substrate.

If `result.json` is missing, write a minimal killed-error result and
continue to Curator.

### Step 5 - Curator

Spawn `curator` with this exact prompt:

> Synthesize iteration `<run_id>`. Read hypothesis.md, candidate.json (if
> present), review.md (if present), result.json, panel-*.txt or panel.txt
> (if present), and fix notes (if present). Write `curator.md`, append one JSON line to
> `worklogs/ledger.jsonl`, and update corpus files per your agent
> definition. If the verdict is `proven-on-substrate`, write
> `worklogs/HALT_REQUESTED.md` so the loop halts for user review.

Verify:

- `worklogs/runs/<run_id>/curator.md` exists.
- `worklogs/ledger.jsonl` has a line for `run_id`.
- `git status --porcelain harness.py run_panel.py baselines.json train.py`
  is empty.

If `curator.md` exists but the ledger line is missing, respawn Curator
once with:

> Recovery for `<run_id>`: `curator.md` exists but no matching ledger
> line was appended. Re-read `curator.md` and `result.json`, append the
> corresponding ledger line, and do not modify other corpus files.

If recovery fails, write a halt file and exit.

### Step 6 - Auto-commit Iteration Corpus

```bash
git add worklogs/ prior_attempts.md
if ! git diff --cached --quiet; then
  git commit -m "iter <run_id>: <verdict>" -m "status: <status>" || { echo "auto-commit failed for <run_id>" > worklogs/HALT_REQUESTED.md; exit 0; }
fi
```

Use targeted paths. Do not `git add -A`.

### Post-iteration

Increment `iters_done`, read the last ledger line, and print:

```text
[research] iter=<iters_done> run_id=<run_id> verdict=<verdict> status=<status> stage=<stage>
```

Sleep 10 seconds, then loop to pre-flight.

## Forbidden Actions

- Skipping pre-flight checks.
- Modifying `harness.py`, `run_panel.py`, `baselines.json`, or
  `worklogs/TEMPLATE.md` during an iteration.
- Suppressing `worklogs/HALT_REQUESTED.md`.
- Treating `reviewer-rejected`, `negative-closure`, `candidate-invalid`,
  or `no-proposal` as infrastructure errors. They count toward the
  no-panel breaker, not the error breaker.
- Asking the user between iterations.
- Auto-committing with `git add -A`.

## Final Summary On Exit

Print:

- halt reason
- `iters_done`
- wallclock duration
- last 5 ledger entries (`run_id`, `verdict`, `status`, `stage`)
