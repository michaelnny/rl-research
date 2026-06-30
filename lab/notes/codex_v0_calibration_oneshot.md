# One-shot brief — v0-tier calibration pass (substrate redesign)

You previously diagnosed and fixed CapacityScheduling Small in
`lab/notes/codex_capacity_scheduling_calibration_2026-06-30.md`.
That fix landed (commit 9d15c80). The Small tier is correctly
calibrated. We now have v0 numbers from a fresh baseline sweep on
both families, and two issues need a focused calibration pass.

Read this brief as your full identity for this single invocation.

## What ran

`PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20`

Wrote `experiments/results/baselines.json` and
`docs/baseline_report.md`. Numbers below.

## Issue 1: CapacityScheduling-v0 is solved by uniform policy

```
CapacityScheduling-v0  (H=2000, K=48, M=8, P=8)
  zero                  succ=0.00 return=-0.660
  random                succ=0.00 return=-0.363
  uniform               succ=1.00 return=+1.571        <-- FAIL
  backlog_priority      succ=0.00 return=-0.191
  earliest_deadline     succ=0.05 return=-0.012
  maintenance_aware     succ=0.00 return=-0.133
  setup_aware           succ=0.00 return=-0.327
  short_horizon_rollout succ=0.00 return=-0.068
```

A uniform allocation policy (`SchedulingUniformPolicy` with intensity
0.5) is solving v0 100% of the time. This violates **acceptance
gate 6 (myopic-gap)** — a uniform policy is the simplest possible
"myopic" allocation and it dominates v0.

Hypothesis: at v0 the horizon is 2000 steps with K=48 projects, so
the total normalized demand is 48 and a uniform allocation at
intensity 0.5 fills each project's demand schedule across enough of
the horizon to clear the bundle-satisfaction threshold (0.55).

What I want you to check:
  - Are the success thresholds calibrated relative to H=2000? At
    Small H=500 the threshold worked, but at v0 the policy has 4x
    more time to accumulate service.
  - Does the priority distribution matter? With priorities uniform
    in [0.5, 1.5], weighted_fill_rate equals plain fill_rate up to
    re-weighting. So a uniform policy gets the same weighted fill
    as a non-uniform one if all projects get the same service.
  - Is the bundle structure (n_bundles=8, size 2-4 out of K=48)
    too sparse? With K=48 and bundles covering only ~12-32 unique
    projects, most projects are *not* in any bundle, so a uniform
    policy that misses a few projects can still satisfy all bundles
    if those misses happen to be outside bundles.

## Issue 2: KeyFuelMaze-v0 has no policy succeeding

```
KeyFuelMaze-v0  (H=2000, D=32, world 48x48, K_t=4, S=6, G=3)
  zero                     succ=0.00 return=+0.184
  random                   succ=0.00 return=+0.064
  random_constant          succ=0.00 return=+0.023
  greedy_landmark          succ=0.00 return=+0.146
  fuel_aware_greedy        succ=0.00 return=+0.146
  efficient_actuator       succ=0.00 return=+0.225
  short_horizon_lookahead  succ=0.00 return=+0.152
```

Zero policy returns +0.184 because the lateness/route components
default to mildly positive when nothing happens. But no policy
solves the env — `succ=0.00` across the board. This violates
**acceptance gate 3 (feasibility)** — the env is supposed to be
solvable by a competent algorithm.

What I want you to check:
  - Is the env actually feasible at v0? With 6 seals each requiring
    1-2 specific keys, plus 3 timed gates, plus extraction, plus 200
    units of starting fuel — can a hand-coded optimal-ish policy
    even reach success in 2000 steps?
  - Is the greedy_landmark policy actually steering correctly? It
    reads `landmark_0` from the observation but the observation
    layout might have a sign / order bug.
  - Should the v0 tier have a less ambitious objective (e.g. n_seals=4
    instead of 6, or fewer gates)?

## What to deliver

Apply targeted fixes to:
- `src/rlh_bench/envs/capacity_scheduling.py` (for issue 1)
- `src/rlh_bench/envs/keyfuel_maze.py` (for issue 2)

Optionally also tune:
- `src/rlh_bench/envs/registration.py` (if a tier needs different
  config knobs, e.g. fewer seals or different bundle structure)
- `src/rlh_bench/baselines/scheduling.py` or
  `src/rlh_bench/baselines/maze.py` (if the issue is a buggy
  baseline rather than the env)

Then re-run the sweep:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20
```

And report the new numbers in
`lab/notes/codex_v0_calibration_2026-06-30.md`.

## Acceptance criteria for this pass

  - CapacityScheduling-v0 with uniform policy: succ ≤ 0.30 and
    leaves headroom in *multiple* terminal-vector components.
  - CapacityScheduling-v0 with at least one of (backlog_priority,
    earliest_deadline, short_horizon_rollout): succ ≥ 0.40.
  - KeyFuelMaze-v0 with at least one policy (preferably
    short_horizon_lookahead or fuel_aware_greedy): succ ≥ 0.20.
  - All zero/random sanity policies: succ ≤ 0.05 on both v0 envs.
  - Small tiers still pass their existing calibration shape
    (don't break what's working).

## Rules

- You may modify env source and registry config but NOT the action
  layout or terminal reward vector shape.
- Keep determinism. Keep terminal-only reward. Keep continuous-only.
- Run `PYTHONPATH=src .venv/bin/python -m pytest -q` before
  finishing — must remain 59 passing.

## How a session ends

When you have:
  - Updated env source (and optionally registry/baselines)
  - A re-run baseline sweep showing the v0 issues are fixed
  - A calibration note at
    `lab/notes/codex_v0_calibration_2026-06-30.md` with before /
    after numbers
  - Tests still passing

Do not commit.
