# Codex pre-merge review — outcome 2026-06-30

The proposer asked Codex for a final pre-merge review. Codex ran for
the full 10-minute window (timed out), but rather than write a
separate review document, it applied targeted fixes directly across
13 files as it found issues. The fixes are recorded below; this
file is the post-hoc summary.

## What Codex caught and fixed

### Stale env IDs in code + docs

- `docs/AGENT_GUIDE.md` example code block: `RecoverableKeyFuelMaze-v0`
  (deleted) → `RecoverableKeyFuelMaze-Small-v0`.
- `experiments/algorithms/runner.py` docstring example: same env-ID
  fix, plus `train_seeds=range(3)` (wrong param name) →
  `train_seed=0`.
- `experiments/probes/recoverability.py` hardcoded `[..., -v0, ...]`
  target list → now iterates `registered_envs()`. Probe will auto-
  pick up any tier re-registered later.

### Stale prose in operator-facing docs

- `docs/LAB.md` Pointers section: "baseline_report.md ... Small + v0
  tiers ... Large baselines are deferred" → "currently registered
  Small-tier envs. v0/Large were removed pending validation."
- `docs/SUBSTRATE_MAP.md` Scheduling action-layout note: "(32 / 80
  / 192)" (referenced the deleted tiers' action_dims) → "(Small:
  32); future v0/Large re-registrations should preserve the same
  no-trailing-control rule."
- `experiments/run_baselines.py` module docstring, `_select_envs`
  docstring, `--include-large` help text, and the report prose:
  all updated to reflect the strict registry. The `--include-large`
  flag still exists for compatibility but its help text now
  explains it only has effect once Large tiers are re-registered.

### Misleading inline comments

- `src/rlh_bench/envs/capacity_scheduling.py` had a comment
  claiming "Modes start at heat 0 but with a small randomized
  initial wear so worlds differ from the start." The actual code
  initializes `_wear` to all zeros; the seeded world variation
  comes from demand, compatibility, setup graph, bundles, and
  setup mixture (which are sampled, not from wear). Comment
  corrected.
- `src/rlh_bench/envs/keyfuel_maze.py` had observation-layout
  comments using "inventory" (it's "keys held") and "kind_one_hot
  (3+n_key_types)" (it's `(4)` plus a `(K_t)` key-index one-hot).
  Comments aligned with the actual `obs_dim` formula.

### Number drift in audit documents

- `lab/notes/acceptance_gates_audit_2026-06-30.md` gate 4 row
  claimed "Tail-zero probe + 3 tests" but only 2 tests exist
  (the cross-family parametrization counts as one). Fixed to "2
  tests."
- Gate 11 phrasing: "9 honest learner-facing policies plus 1
  decomposition diagnostic" → "9 learner-facing policies,
  including 1 decomposition diagnostic" (the diagnostic is one of
  the 9, not separate from them).
- `lab/notes/strict_registry_outcome_2026-06-30.md` claimed
  "Test count: 94 → 77" (the immediate pre-cleanup number). The
  full cleanup brought it to 64; doc now records 64.

### Version mismatch

- `pyproject.toml` said `version = "0.1.0"`. `src/rlh_bench/__init__.py`
  said `__version__ = "0.2.0"`. Now both say `0.2.0`.

### Regenerated artifact

- `docs/baseline_report.md` and `experiments/results/baselines.json`
  re-run on the strict registry. Success rates / reward vectors are
  identical to the pre-review numbers (substrate dynamics
  unchanged); `sec/ep` numbers shifted slightly due to wall-clock
  variance, which is expected.

## What Codex did NOT flag

Through 10 minutes of review across env code, baselines, tests,
docs, the lab loop, and the substrate map, Codex did NOT identify:

- Any bug in env dynamics.
- Any test that was passing for the wrong reason.
- Any acceptance gate claim unsupported by code.
- Any dishonest oracle leak in the baseline portfolio.
- Any contradiction between docs and tests / code.

The substrate's substance held up under the review. Codex's pass
was entirely cleanup-and-precision work — fixing comments, stale
references, and version metadata. That is the kind of pass you
expect on a mature artifact about to merge, not on a buggy one.

## Tests

`PYTHONPATH=src .venv/bin/python -m pytest -q` → 64 passed after
all edits.

## Verdict

Merge-ready. The remaining work (re-registering v0/Large after
real validation, the memory planner that would let Maze-v0 pass
gate 3, the varied-depth probe for gate 5) is **future research
work**, not pre-merge cleanup.
