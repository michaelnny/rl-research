# One-shot brief — final pre-merge review

The substrate redesign branch (`lab/substrate-redesign`) is approaching
merge readiness. Several Codex passes have happened:

1. Initial substrate design (red-team, counter-design, v2 plan)
2. CapacityScheduling Small calibration
3. KeyFuelMaze family + baseline portfolios
4. v0 calibration + adversarial self-review (oracle separation)
5. Reward normalization audit (gate 10)
6. Held-out seed protocol (gate 9)
7. (proposer-written) idle-tail probe + tests (gate 4)
8. (proposer-written) recoverability v0 test (gate 7 extension)

Now: final review of the merged state. Look for anything that
shouldn't merge to master.

Read this brief as your full identity for this single invocation.

## What to review

Look at the current state of the `lab/substrate-redesign` branch.
Recent commits (newest first):
  - d47c2e9 gate 4 idle-tail measurement
  - 8dcc1c6 gate 9 held-out seed protocol
  - 95fd721 gate 10 reward normalization
  - ee8fa15 v0 calibration + adversarial review
  - f301b08 register envs + baseline portfolios + sweep
  - 29bb85b RecoverableKeyFuelMaze
  - 9d15c80 RecoverableCapacityScheduling
  - 1eb36b1 world_gen + seed_bands
  - f377344 mission narrow to continuous-action
  - 6a401c4 planning artifacts

Files of interest to spot-check:
  - `src/rlh_bench/envs/capacity_scheduling.py` (~860 lines, primary env)
  - `src/rlh_bench/envs/keyfuel_maze.py` (~670 lines)
  - `src/rlh_bench/baselines/scheduling.py` (~360 lines)
  - `src/rlh_bench/baselines/maze.py` (~360 lines)
  - `experiments/run_baselines.py` (sweep)
  - `lab/notes/acceptance_gates_audit_2026-06-30.md` (gate status)
  - `docs/baseline_report.md` (current numbers)
  - `docs/SUBSTRATE_MAP.md` (one-page API doc)
  - `tests/` (70 tests)

## Concrete checks

1. **Does the substrate actually deliver what `CLAUDE.md` claims?**
   - Mission text says "deterministic, recoverable, long-horizon,
     terminal-only sparse feedback, optional terminal vector reward,
     continuous action spaces throughout."
   - Do all 6 registered envs satisfy this on inspection? Any
     hidden non-determinism, per-step reward shaping, or
     discrete actions sneaking in?

2. **Are the baselines honest?**
   - Already separated oracle from learner-facing portfolio (your
     earlier review). Quick scan: do any other policies in
     `SCHEDULING_BASELINES` or `MAZE_BASELINES` reach for env
     internals or do training that shouldn't happen at inference
     time?

3. **Does the test suite over-test or under-test?**
   - 70 tests is a lot. Are any redundant or testing the wrong
     thing? Are there obvious holes (e.g. an acceptance gate
     claimed ✓ in the audit that has no test backing it)?

4. **Documentation honesty.**
   - `docs/baseline_report.md` numbers were generated from a
     specific commit's state. Are they stale relative to the
     code? Quick spot check.
   - `docs/SUBSTRATE_MAP.md` lists capacity_push, oracle, etc.
     correctly?

5. **Gate audit honesty.**
   - `lab/notes/acceptance_gates_audit_2026-06-30.md` claims
     several ✓ for v0 / Large. Are those backed by tests / probes,
     or just structural claims?

6. **Any remaining bugs you flagged but didn't fix.**
   - You earlier flagged "_gate_phases sampled twice" — the
     proposer checked and called it a false alarm. Was the
     proposer right?
   - You earlier flagged that some maze baselines used `_seed` via
     getattr — that was cleaned up; confirm no remaining
     underscore-only accesses in the honest portfolio.

## Deliverable

Write your final review to
`lab/notes/codex_final_review_2026-06-30.md`. Default 200-400 lines.

For each section, one of:
  - "Ready to merge: <reason>"
  - "Should fix before merge: <concrete issue + suggested fix>"
  - "Acceptable but flag for follow-up: <concern + how to track>"

Apply fixes directly only when clearly right; defer judgment
calls to the proposer.

## Rules

- Read freely. Modify only if you find a clear bug. Tests must
  still pass: `PYTHONPATH=src .venv/bin/python -m pytest -q`.
- Do not commit; the proposer will commit after reading your
  review.

## How a session ends

When the final review is written.
