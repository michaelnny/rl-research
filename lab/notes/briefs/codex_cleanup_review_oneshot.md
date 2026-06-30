# One-shot brief — final cleanup review

Several Codex review passes have happened already; the last one
(`lab/notes/codex_final_review_2026-06-30.md`) identified 3 hard
blockers and several "acceptable but flag" items. The blockers were
fixed (commit 0471553). The "acceptable but flag" items were
addressed in subsequent cleanup commits:

  - 0471553 Fix 3 blockers (action_dim no padding, route_efficiency
    weighted by progress, stale docs/prompts)
  - f29d591 Wire setup_graph into setup churn cost (was dead code)
  - 183aff3 Strengthen recoverability tests + document public model
    API + cross-family invariants

The branch is now at 94 passing tests, up from 70 at your last review.

Run a *final cleanup review* — look for anything that remains. The
proposer is about to declare merge-ready, so this is the last gate.

Read this brief as your full identity for this single invocation.

## Things to look at

1. **setup_graph wiring**: I rewired setup churn in
   `src/rlh_bench/envs/capacity_scheduling.py` to use the
   per-world `_setup_graph` (transition cost graph between product
   families). Look at the new code (around lines 558-590). Does
   the mass-routing math make sense? Are there edge cases I missed
   (e.g. when no mass moves, division by zero, asymmetric flow
   cases)?

2. **Recoverability tests**: `tests/test_recoverability_gate.py`
   has two new tests. Are the assertions strict enough to be a
   real gate, or are they trivially satisfied?

3. **Public model API documentation**: I documented
   `env.actuator_matrix` and `env.seed` as part of the "public
   model API" that baselines may read, distinguishing them from
   the underscore-prefixed task state. Is this a coherent
   distinction, or am I papering over the same gate-honesty
   issue you flagged before?

4. **Cross-family invariants**: 20 parametrized tests in
   `tests/test_cross_family_invariants.py`. They run across all
   six envs. Are any of them slow enough to be a regression
   concern? Are any of them trivially passing (testing the wrong
   thing)?

5. **Gates audit honesty**: `lab/notes/acceptance_gates_audit_2026-06-30.md`
   now claims more ✓ than before. For each ✓, can you actually
   find the test/probe backing it? Spot check at least 3 gates.

6. **Anything else you find**. The branch is large now; you may
   spot issues outside the items above. Be specific.

## What to do

Write your review to
`lab/notes/codex_cleanup_review_2026-06-30.md`. For each section,
one of:
  - "Merge-ready: <reason>"
  - "Should fix before merge: <concrete issue + suggested fix>"
  - "Acceptable but flag for follow-up: <concern + how to track>"

Apply fixes directly only when clearly right. Defer judgment calls
to the proposer.

## Rules

- Read freely. Modify only if you find a clear bug. Tests must
  still pass: `PYTHONPATH=src .venv/bin/python -m pytest -q`
- Do not commit.

## How a session ends

When the review file is written.
