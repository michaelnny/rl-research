---
description: One full pass of the rl-research loop (Researcher → Reviewer → Engineer → Curator).
argument-hint: "[thread-slug]"
allowed-tools: Agent, Read, Bash, Write
model: sonnet
---

You are the **orchestrator** of one rl-research iteration. You do not write
hypotheses, code, or results — you spawn the role agents in order and route
their outputs.

The thread hint (may be empty) is: $ARGUMENTS

## Pre-iteration gates

Before spawning any agent, run these checks. If any fails, **abort the
iteration cleanly** (print the reason to stdout, exit without running agents):

1. **Halt check.** If `lab/HALT_REQUESTED.md` exists, print its first line and
   stop. Do not begin a new iteration.

2. **Health check.** Run `bash scripts/preflight.sh` and refuse to start if it
   exits non-zero. The script prints what's wrong; surface its tail in the
   stop message.

3. **Stale state.** If `lab/.run_id.in_progress` exists, a prior iteration
   crashed mid-Researcher (or later — the wrapper clears the marker on a
   normal exit, so its presence means we lost the iteration). Read its
   content (the abandoned run_id) and:

   ```bash
   ABANDONED=$(cat lab/.run_id.in_progress)
   echo "$(date -Iseconds) recovering abandoned run $ABANDONED" >> lab/iterations.log
   # If the run dir has hypothesis.md but no result.json, write a
   # killed-error result + ledger line so the corpus can see it.
   if [[ -f "lab/runs/$ABANDONED/hypothesis.md" && ! -f "lab/runs/$ABANDONED/result.json" ]]; then
     uv run python - <<PYEOF
   from rl_research.contract import write_result, append_to_ledger
   from datetime import datetime, UTC
   import re, pathlib, subprocess
   run_id = "$ABANDONED"
   hyp = pathlib.Path(f'lab/runs/{run_id}/hypothesis.md').read_text()
   def fm(key, default):
       m = re.search(rf'^{key}:\s*(.+)$', hyp, re.MULTILINE)
       return m.group(1).strip().strip('[]').split(',')[0].strip() if m else default
   git_sha = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
   now = datetime.now(UTC)
   p = write_result(
       run_id=run_id, stage='A-only', status='killed-error',
       primary_benchmark=fm('primary_benchmark', 'CartPole-v1'),
       pillar=fm('pillar', 'sparse-long-horizon'),
       thread=fm('thread', 'exploration'),
       seeds=[0], env_steps=0, wallclock_s=0.0,
       best_return=0.0, final_return=0.0,
       by_seed={'0': {'best_return': 0.0, 'final_return': 0.0, 'env_steps': 0}},
       sanity={'passed': False, 'by_env': {}, 'retries': 0},
       git_sha=git_sha, started_at=now, ended_at=now,
       notes='auto-closed: prior iteration crashed mid-flight (in_progress marker detected)',
   )
   append_to_ledger(p)
   PYEOF
   fi
   rm -f lab/.run_id.in_progress
   ```
   Then **proceed with a fresh run**. Do not try to resume the abandoned run.

If all gates pass, proceed.

## State machine

### 1. Researcher (Phase 1)

Spawn the `researcher` subagent. Tell it:

> Propose a hypothesis for the rl-research loop. Read `docs/charter.md` and
> `docs/roles/researcher.md` first. Allocate the next run_id with
> `next_run_id('<thread-slug>')`, then write
> `lab/runs/<run_id>/hypothesis.md`, `lab/runs/<run_id>/run_id.txt`, AND
> `lab/.run_id.in_progress` (a single line with the run_id, used by the
> orchestrator for deterministic recovery — overwrite if present). Halt
> after writing those three files. Do not write `train.py`. Thread hint
> (may be empty): $ARGUMENTS

### 2. Recover the run_id

Read it directly from the deterministic scratch file the Researcher just
wrote:

```bash
test -f lab/.run_id.in_progress || { echo "ERROR: Researcher did not write lab/.run_id.in_progress"; exit 1; }
RUN_ID=$(cat lab/.run_id.in_progress)
```

Bind that to `<run_id>` for the rest of this iteration. Do NOT use
`ls -1t lab/runs/*/run_id.txt | head -1` — a stale dir from a prior crash
can have a newer mtime than the current iteration.

### 3. Reviewer

Spawn the `reviewer` subagent. Tell it:

> Review `lab/runs/<run_id>/hypothesis.md` per `docs/roles/reviewer.md`. Write
> `lab/runs/<run_id>/review.md` with the verdict in YAML frontmatter.

### 4. Branch on verdict

Read the YAML frontmatter of `lab/runs/<run_id>/review.md`. Track revision
cycles via the **counter file** `lab/runs/<run_id>/.revision_count` (single
integer, starts at 0):

- `novel-direction` → go to step 5.
- `needs-sharpening` → read `.revision_count` (default 0). If ≥ 1, jump to
  the abandonment branch below with `status="abandoned-sharpening"`.
  Otherwise: increment the counter (`echo $((N+1)) > .revision_count`), copy
  the old `review.md` to `review-attempt-N.md` to preserve history, then
  re-spawn `researcher` with the prior `review.md` in scope to revise the
  same hypothesis in place. Then go back to step 3.
- `known-rebadge` → read `.revision_count`. If ≥ 2, abandonment branch with
  `status="abandoned-rebadge"`. Otherwise: increment, archive old review,
  re-spawn researcher, then back to step 3.

**Abandonment branch** — write a minimal abandoned `result.json` and ledger
entry, then jump to step 7. Set `ABANDON_STATUS` to either
`abandoned-rebadge` or `abandoned-sharpening` based on which branch you came
from above; do not pass the literal placeholder string. **Export it** so the
heredoc Python sees it via `os.environ` (a plain shell var won't propagate
into the `uv run python` subprocess):

```bash
export ABANDON_STATUS=abandoned-sharpening   # or abandoned-rebadge — pick the one that triggered this branch
RUN_ID=$(cat lab/.run_id.in_progress)
uv run python - <<PYEOF
from rl_research.contract import write_result, append_to_ledger
from datetime import datetime, UTC
import subprocess, re, pathlib, os
run_id = "$RUN_ID"
status = os.environ.get("ABANDON_STATUS", "abandoned-sharpening")
hyp = pathlib.Path(f'lab/runs/{run_id}/hypothesis.md').read_text()
def fm(key, default):
    m = re.search(rf'^{key}:\s*(.+)$', hyp, re.MULTILINE)
    return m.group(1).strip().strip('[]').split(',')[0].strip() if m else default
git_sha = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
now = datetime.now(UTC)
p = write_result(
    run_id=run_id, stage='A-only', status=status,
    primary_benchmark=fm('primary_benchmark', 'CartPole-v1'),
    pillar=fm('pillar', 'sparse-long-horizon'),
    thread=fm('thread', 'exploration'),
    seeds=[0], env_steps=0, wallclock_s=0.0,
    best_return=0.0, final_return=0.0,
    by_seed={'0': {'best_return': 0.0, 'final_return': 0.0, 'env_steps': 0}},
    sanity={'passed': False, 'by_env': {}, 'retries': 0},
    git_sha=git_sha, started_at=now, ended_at=now,
    notes='abandoned: reviewer revision cycles exhausted',
)
append_to_ledger(p)
PYEOF
```

### 5. Engineer

Spawn the `engineer` subagent. Tell it:

> The Reviewer marked `lab/runs/<run_id>/hypothesis.md` as `novel-direction`.
> Per `docs/roles/engineer.md`: write `lab/runs/<run_id>/train.py`, run Stage
> A (sanity gate, 3-retry budget, `fix-N.md` notes), and if Stage A passes,
> run Stage B (primary benchmark). Then write `config.json` + `result.json`,
> validate with `rl_research.contract.validate_result_json`, and append the
> ledger entry. Do not edit `hypothesis.md`, do not tune per-benchmark
> hyperparameters, do not touch `lab/baselines/`.

### 6. Curator

Spawn the `curator` subagent. Tell it:

> The latest run is `<run_id>`. Per `docs/roles/curator.md`: at minimum,
> assign `verdict_curator` for this run by calling
> `rl_research.contract.update_ledger_verdict(run_id, verdict, notes=...)`
> — never edit `lab/ledger.jsonl` with raw file ops; the helper is
> flock-atomic and concurrent-safe. Use your meta-supervisor judgment for
> anything else the corpus needs: prune `lessons.md`, archive stale threads,
> write coverage artifacts, decide on mass-run promotion, or write
> `lab/HALT_REQUESTED.md` if you judge the loop has stopped producing signal.

### 7. Cleanup + output

Always run, even on abandonment:

```bash
# Clear the in-progress marker so the next iteration's stale-state gate
# doesn't trip. (corpus_stats refresh + ledger backup are handled by
# scripts/loop.sh between iterations — do not duplicate them here.)
rm -f lab/.run_id.in_progress
```

Then print a one-line summary:

```
<run_id> status=<status> verdict=<verdict_curator or "pending"> wallclock=<s>s
```

## Forbidden

- You do NOT write `hypothesis.md`, `train.py`, `result.json` (except the
  minimal abandoned variant in step 4), or modify `lessons.md` / ledger
  entries.
- You do NOT run `train.py` directly. The Engineer does that.
- You do NOT inspect TB logs or make multi-criteria judgments. The Curator
  does that.
- You do NOT skip the pre-iteration gates. Halt-check, preflight, and
  stale-state recovery are mandatory; they exist to keep the loop alive
  unattended.
