# One-shot brief — v0-tier calibration pass (second attempt)

You previously fixed the CapacityScheduling Small calibration
(commit 9d15c80). A first v0-calibration attempt timed out before
producing fixes. This brief is **tightened to be one-shot
actionable** — do not re-diagnose what we already understand; apply
the smallest changes that satisfy the acceptance criteria and write
the calibration note.

Read this brief as your full identity for this single invocation.

## What is broken

Run `PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20`. v0 results:

```
CapacityScheduling-v0  (H=2000, K=48, M=8, P=8, action_dim=96)
  zero            succ=0.00  return=-0.660
  random          succ=0.00  return=-0.363
  uniform         succ=1.00  return=+1.571   <-- FAIL gate 6
  backlog_priority succ=0.00 return=-0.191
  earliest_deadline succ=0.05 return=-0.012
  short_horizon_rollout succ=0.00 return=-0.068

KeyFuelMaze-v0  (H=2000, D=32, world 48x48, K_t=4, S=6, G=3)
  every policy:   succ=0.00       <-- FAIL gate 3
```

## Hypotheses

You diagnosed these in your earlier note (calibration of Small).
Reasoning extended to v0:

**Scheduling-v0 uniform-dominance** — the success thresholds
(`success_fill_threshold=0.55`, `success_mandatory_threshold=0.50`,
`quality_required=0.55`) were tuned for Small (H=500). At H=2000
with K=48 projects and the same per-project demand normalization (=1.0 across the horizon), uniform allocation has 4x more time per project. Add bundles: with `n_bundles=8` of size 2-4 over K=48 projects, the bundle structure may also be too forgiving.

**Maze-v0 0% success** — the v0 maze has 6 seals each requiring 1-2
specific keys (K_t=4), plus 3 timed gates, plus extraction, all
within H=2000 in a 48x48 world. With max_speed=4.0 and dt=0.1, the
agent moves at most 0.4 units/step. The world diameter is ~67
units. A naive tour visits 4 keys + 6 seals + extraction = 11
waypoints; with avg 20 units between them, total ~220 units of
travel at 0.4/step = 550 steps for just travel, plus key dwell
(4 steps × 4 keys = 16 steps), plus seal completion (instant if
keys + gate-open). The math says it's borderline-feasible but
random / naive greedy will exceed fuel before reaching enough
seals. Also: the **landmark observation might have a wrong sign**
or **uses world-units when the policy expects normalized units**.

## What to do (in order)

### Step 1: Verify the maze landmark observation sign

In `src/rlh_bench/envs/keyfuel_maze.py`, look at
`_nearest_landmarks`. The `rel = position - pos` direction is
returned without normalization (in world units). Then in
`src/rlh_bench/baselines/maze.py` `MazeGreedyLandmarkPolicy`
multiplies `direction = landmark[:2]` by the actuator matrix
transpose.

Check: does the policy actually move toward landmarks? Add a
print or trace inside one of the test policies. If not, fix the
sign or normalization.

### Step 2: Make Scheduling-v0 harder

Smallest possible change: tighten the success thresholds **only
for the v0 registry config**, not the dataclass default. In
`src/rlh_bench/envs/registration.py`, the `_scheduling_default`
function constructs `CapacitySchedulingConfig()` — add explicit
thresholds:

```python
def _scheduling_default(**kwargs: Any) -> RecoverableCapacitySchedulingEnv:
    return RecoverableCapacitySchedulingEnv(
        config=CapacitySchedulingConfig(
            # v0 needs tighter thresholds than Small because the longer
            # horizon means uniform allocation easily clears 0.55.
            success_fill_threshold=0.85,
            success_mandatory_threshold=0.70,
            quality_required=0.75,
        ),
        **kwargs,
    )
```

(Adjust numbers as needed. Don't break Small — Small uses the
dataclass defaults still.)

### Step 3: Make Maze-v0 less brutal *or* add a stronger baseline

Two paths, your call which to apply:

**Path A: scale the world down at v0.** Reduce `world_size` from
48 to 36, keep n_seals=6, etc. Less travel, more room for
multi-seal completion in the horizon.

**Path B: improve the greedy/lookahead baselines.** The current
`MazeShortHorizonLookaheadPolicy` is a cheap proxy. A stronger
baseline (PD-controller-like, multi-actuator scaling by distance)
might already reach 20% success without any env change. If so,
that's *better* — it means the env was always feasible and we just
needed a more capable baseline.

Try Path B first (no env change needed). Only fall back to Path A
if no baseline can reach success≥0.20.

### Step 4: Re-run and verify

```bash
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20
```

Acceptance criteria (all must hold):
  - CapacityScheduling-v0 uniform: succ ≤ 0.30
  - CapacityScheduling-v0 at least one heuristic: succ ≥ 0.40
  - KeyFuelMaze-v0 at least one heuristic: succ ≥ 0.20
  - CapacityScheduling-Small: spread unchanged (no regression)
  - KeyFuelMaze-Small: spread unchanged (no regression)
  - All zero / random: succ ≤ 0.05 on both v0 envs
  - `PYTHONPATH=src .venv/bin/python -m pytest -q` shows 59 passing

### Step 5: Write the calibration note

Write to `lab/notes/codex_v0_calibration_2026-06-30.md`:

```markdown
# v0 calibration — 2026-06-30

## What was wrong
<one paragraph per env>

## What I changed
<concise diff summary; reference files + lines>

## Before / after baseline numbers
<table per env>

## Caveats / next pass
<anything you saw that the next session should look at>
```

## Rules

- Modify env source / registry config / baselines as needed.
- Do NOT change the action layout or terminal reward vector shape.
- Keep determinism, terminal-only reward, continuous-only.
- Run pytest; must still pass 59 tests.
- Do not commit; the proposer will commit after a review pass.

## How a session ends

When:
  - Acceptance criteria above all pass on a fresh sweep.
  - Calibration note is written.
  - Tests still pass.

Don't overthink. Apply minimal coherent changes; refer to the
existing Small calibration patterns where possible.
