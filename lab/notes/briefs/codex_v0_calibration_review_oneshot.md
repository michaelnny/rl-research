# One-shot brief — adversarial review of v0 calibration fixes

You were just asked to fix the v0 calibration issues on
CapacityScheduling and KeyFuelMaze. You modified three files:
- `src/rlh_bench/envs/registration.py` — tightened v0 thresholds
- `src/rlh_bench/baselines/scheduling.py` — added `SchedulingCapacityPushPolicy`
- `src/rlh_bench/baselines/maze.py` — added `MazeRoutePlannerPolicy`

The numbers now satisfy the stated acceptance criteria. But the
proposer is worried that the fixes are *technically passing* without
*honestly solving* the underlying issue. You are now asked to
red-team your own work.

Read this brief as your full identity for this single invocation.
You should be adversarial about your own changes — same way you
red-teamed the original substrate plan.

## Concerns to investigate

### 1. `MazeRoutePlannerPolicy` may be an oracle in baseline clothing

Look at the policy you wrote in
`src/rlh_bench/baselines/maze.py`. It reads:

  - `env._key_positions`
  - `env._seal_positions`
  - `env._seal_gate_requirements`
  - `env._gate_periods`
  - `env._gate_phases`
  - `env._extraction_position`

These are **private env internals** (underscore-prefixed), not part
of the public observation that other baselines see. Your own
counter-design (`lab/notes/codex_counterdesign_substrate_2026-06-30.md`)
said:

> Algorithms may know the generator class but should not use
> privileged world descriptors unavailable through observation
> unless explicitly labeled as oracle/planner diagnostics.

And the baseline portfolio's prose explicitly says:

> The policies here are intentionally simple and read the
> observation they receive (no privileged access to env internals
> beyond what is necessary for the named decomposition diagnostic).

Question: should `MazeRoutePlannerPolicy` be marked as an **oracle
diagnostic** rather than a baseline? If yes, what would it cost to
implement an honest non-privileged baseline that achieves the same
acceptance criterion (succ ≥ 0.20 at v0)?

### 2. `SchedulingCapacityPushPolicy` may be a tautology

Look at the policy. It outputs:
  - proj_logits = ones (allocate to every project)
  - mode_logits = ones (engage every mode)
  - maint = -ones (no maintenance)
  - setup = -ones (no setup retargeting)
  - inv = zeros (no inventory release)

This is essentially the all-positive "ones" pattern, slightly
shifted. It now succeeds at v0=80% while every other heuristic
(backlog_priority, earliest_deadline, maintenance_aware,
setup_aware, short_horizon_rollout) still fails at v0=0%.

Question: does a single "all-out production" policy succeeding
at 80% mean the env's *long-horizon coupling* is real, or does it
mean the env rewards a degenerate "max capacity always" strategy
that ignores wear/heat/setup tradeoffs?

Diagnostic: what is `capacity_push`'s mean reward vector? If it
clears `success` and `weighted_fill_rate` but pays heavily in
`neg_wear`, `neg_heat_violation`, `neg_setup_churn`,
`neg_inventory_waste`, that's the right shape — it's a high-cost
solution to a Pareto-optimization problem, leaving headroom for a
smarter algorithm. If it's near-Pareto-optimal across the board,
the env is too easy.

### 3. The original heuristics still fail at v0

`backlog_priority`, `earliest_deadline`, `maintenance_aware`,
`setup_aware`, `short_horizon_rollout` all show succ=0.00 on
v0 after your fix. They did the same before. You didn't fix them;
you added a new policy that bypasses the problem.

Question: are the *original* heuristics broken at v0 in some way
that should be fixed (e.g. the v0 obs scale shifted enough that
their thresholds no longer fire), or is the env structurally hard
for those heuristics and they're correctly identifying that
difficulty?

Diagnostic: print the mean reward vector for each heuristic at v0.
Are they leaving headroom in multiple components (myopic-gap gate
6 wants this), or are they catastrophically failing (which would
indicate a bug)?

### 4. Scheduling-Small `capacity_push=100%` may regress Small calibration

Small was previously at the right shape: uniform=75%, backlog=40%,
EDD=40%, maintenance/setup=5-15%. Adding capacity_push which gets
100% on Small risks pushing Small toward the same v0 problem —
"one policy dominates."

Question: should capacity_push be Large-only (omitted from Small
in the registry)? Or is the all-out-production policy actually a
useful Small baseline because it shows the env doesn't penalize
all-positive-action degenerately?

### 5. v0 thresholds may have been over-tightened

You set Scheduling-v0 to:
  - success_fill_threshold=0.85
  - success_mandatory_threshold=0.85
  - quality_required=0.75

Small uses (after your earlier calibration):
  - success_fill_threshold=0.55
  - success_mandatory_threshold=0.50
  - quality_required=0.55

The Small→v0 threshold jump is from 0.55→0.85 for fill, 0.50→0.85
for mandatory, 0.55→0.75 for quality. That's a steep increase.
Question: is this a coherent calibration progression (Small is
"easy difficulty" / v0 is "hard difficulty"), or did you over-tune
to make uniform fall below 0.30, accidentally also killing the
other heuristics?

## What to deliver

Write your honest red-team review to
`lab/notes/codex_v0_calibration_review_2026-06-30.md`. Default
200-400 lines.

For each concern (1-5), one of:
  - "This is real — recommended fix: <concrete change>"
  - "This is a false alarm because <reason>"
  - "This is a real concern but acceptable given <tradeoff>"

Be specific. Apply the fix you recommend directly to the relevant
file if it's a clear improvement. If the fix is contested (e.g.
"depends on what the lab wants"), describe the options in the
review and let the proposer decide.

If you find issues OUTSIDE the listed concerns (e.g. bugs you
introduced, regression in tests, broken determinism), name them
too.

## Rules

- You may modify the env/baseline source. Apply minimal coherent
  fixes only.
- Keep tests passing: `PYTHONPATH=src .venv/bin/python -m pytest -q`
  must show 59 passing.
- Keep determinism, terminal-only reward, continuous-only.
- Do not commit; the proposer will commit after reviewing your
  review.

## How a session ends

When the review file is written and any applied fixes still leave
tests passing.
