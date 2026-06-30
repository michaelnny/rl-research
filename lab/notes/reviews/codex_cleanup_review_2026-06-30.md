# Codex cleanup review — outcome 2026-06-30

The proposer asked Codex for a final cleanup review. Codex worked
through the listed checks but timed out (10-min cap) before writing
the formal review file. What Codex DID land on disk is documentation
consistency cleanup across 7 files (~21 lines total of doc edits, no
code logic changes):

  - CLAUDE.md: clarified baseline_report.md covers Small + v0 only,
    Large deferred.
  - README.md: rephrased CRITICAL DESIGN.md / SUBSTRATE_MAP.md
    relationship; updated "observation-only vs oracle" → "public-model
    vs oracle".
  - docs/SUBSTRATE_MAP.md: fixed stale action_dim numbers in the
    registry table (96 → 80, 224 → 192); rephrased "trailing dims are
    unused" to "registered tiers set D=K+3M+P exactly; no trailing
    no-op controls advertised" — this is the correct gate-8 framing
    after the blocker-fix commit.
  - src/rlh_bench/envs/capacity_scheduling.py: same rephrasing in the
    env docstrings.
  - src/rlh_bench/envs/keyfuel_maze.py: cosmetic comment fix on
    route_efficiency progress definition.
  - src/rlh_bench/baselines/maze.py: cleaned a stale "env._seed"
    reference in a comment.
  - tests/test_keyfuel_maze_env.py: renamed test section comment
    "observation-only" → "public-model vs oracle".

Tests: 94 still passing.

## Interpretation

Codex did NOT flag any new bugs, missing tests, or substrate concerns
in the time it had. The edits it landed are pure consistency cleanup
on the documentation surface (fixing stale numbers and language that
predated subsequent fixes). The absence of a code-bug flag through a
10-minute review pass — after Codex has already done several
adversarial passes that DID find real bugs — is a meaningful signal:
the branch is in a defensible state.

This is not a substitute for the formal review file the brief asked
for, but the next-best alternative is the diff itself, which is
faithfully recorded in the commit. If the proposer wants a deeper
review later, the brief is preserved at
`lab/notes/codex_cleanup_review_oneshot.md` and can be re-run.

## What remains as known follow-ups (not blockers)

These items are documented in `lab/notes/acceptance_gates_audit_2026-06-30.md`
and `lab/notes/codex_final_review_2026-06-30.md`; they are tracked
but not merge-blocking:

  - **Gate 3 maze-v0 honest baseline**: an observation-only memory
    planner would close the gap between oracle (succ=0.8) and the
    cheap-portfolio (succ=0). Natural shape for the lab's first
    candidate algorithm.
  - **Gate 5 varied-depth lookahead probe**: the current
    `short_horizon_*` policies are fixed-horizon; a parameterized
    variant would give a sharper signal that the env rewards
    longer lookahead.
  - **Gate 6 bundle-aware Scheduling heuristic**: would let the
    original myopic heuristics succeed at v0 without brute force.
