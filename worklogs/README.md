# worklogs

Archived research memory + live iteration corpus.

## Layout

- `attempts/` — detailed records for historical failed or weak research
  directions. Sealed history. Do not delete or rewrite these files during
  harness work; the Curator may only **append** new entries.
- `candidates/` — alive-but-not-yet-tested directions, plus parked
  candidates the Curator wants the next iteration's Researcher to read
  first.
- `runs/` — live iteration scratch space. One subdirectory per iteration,
  named `<YYYYMMDD>-<NN>-<slug>`. Each holds:
    - `hypothesis.md` — Researcher Phase 1 artifact.
    - `review.md` — Reviewer verdict.
    - `train.py` — Researcher Phase 2 artifact (the candidate algorithm).
    - `panel.txt` — raw `run_panel.py` stdout.
    - `result.json` — Engineer-written structured summary.
    - `curator.md` — Curator's per-run verdict + lesson.
    - `fix-N.md` — Engineer retry notes (only if any retries happened).
- `ledger.jsonl` — append-only one-line-per-iteration index. Curator-written.
- `TEMPLATE.md` — schema for `attempts/` entries. Used when the Curator
  promotes a `failed-structural` run.
- `../prior_attempts.md` — compact negative-space index. Researcher and
  Reviewer read this every iteration; Curator may **append** a numbered
  entry on `failed-structural` verdicts.

## Source of truth

- The active harness (`harness.py`, `run_panel.py`, `train.py`) does not
  read this directory at runtime — it only matters to the agent loop.
- `attempts/` is the canonical record of what has been tried and failed.
- `candidates/` is the canonical record of what is alive but not yet
  conclusive.
- `runs/` is ephemeral in spirit but committed for reproducibility; old
  run directories may be garbage-collected by the user once the Curator
  has promoted or parked them.
