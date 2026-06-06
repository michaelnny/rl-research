# worklogs

Archived research memory plus live schema-backed probe-first iteration corpus.

## Layout

- `attempts/` - sealed detailed records for historical failed or rebadged
  mechanism families. Do not delete or rewrite old entries; Curator may
  append a new entry only for a family-level structural failure.
- `runs/` - one directory per iteration, named `<YYYYMMDD>-<NN>-<slug>`.
  Common files:
  - `hypothesis.md` - Researcher artifact: `[probe]`,
    `[negative-closure]`, or empty-hand note.
  - `candidate.json` - machine-readable probe schema, required for
    `[probe]` runs and validated by `scripts/validate_candidate.py`.
  - `review.md` - Reviewer triage verdict.
  - `train.py` - Engineer implementation for Reviewer-approved probes.
  - `train_ablate.py` - Engineer ablation of the claimed primitive.
  - `panel-*.txt` - raw `run_panel.py` stdout for smoke, claim,
    ablation, and confirmation ladder rungs.
  - `result.json` - structured run result, with `mode: probe-v1` for the
    current loop design.
  - `curator.md` - Curator verdict and lesson.
  - `fix-N.md` / `impl-blocker.md` - Engineer retry or blocker notes.
- `ledger.jsonl` - append-only one-line-per-iteration index. Curator
  written. New probe-first rows include `mode: probe-v1`.
- `TEMPLATE.md` - schema guidance for sealed `attempts/` entries.
- `../prior_attempts.md` - compact negative-space index. Researcher and
  Reviewer read this every iteration; Curator may update it sparingly.

## Source of Truth

- `harness.py`, `run_panel.py`, and repo-root `train.py` do not read this
  directory at runtime. Worklogs are for the agent loop and research
  memory.
- `runs/` is the canonical trail of probes, candidate schemas, panel
  evidence, ablations, and curator lessons.
- `attempts/` plus `prior_attempts.md` are the canonical negative-space
  record.
- There is no active candidates parking lot. Positive but incomplete runs
  are recorded as `empirical-signal` and rediscovered through recent
  curator summaries and the ledger.
