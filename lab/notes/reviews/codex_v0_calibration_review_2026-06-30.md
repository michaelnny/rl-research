# Red-team review of v0 calibration fixes

Date: 2026-06-30
Reviewer: Codex, adversarial pass over my own v0 calibration patch
Scope: `RecoverableCapacityScheduling-v0` and `RecoverableKeyFuelMaze-v0`

## Executive summary

The v0 numbers technically passed the stated acceptance criteria, but one of
those passes was not honest.

The `MazeRoutePlannerPolicy` I added was an oracle. It ignored its observation
and read generated-world internals for waypoint positions, gate requirements,
gate periods/phases, and extraction. That violates the baseline portfolio's own
contract. I applied a minimal source fix: it is now named
`MazeOracleRoutePlannerPolicy`, has public name `oracle_route_planner`, is moved
out of `MAZE_BASELINES`, and is listed under `MAZE_ORACLE_DIAGNOSTICS`.

That means the honest learner-facing KeyFuelMaze-v0 portfolio no longer has a
baseline with success >= 0.20. The old acceptance criterion should be marked as
unmet until either (a) a non-privileged observation-only baseline is implemented,
or (b) the env/observation is adjusted so a non-oracle planner can succeed.

The scheduling fix is more nuanced. `capacity_push` is a degenerate high-capacity
stress policy, but it is not an oracle. Its reward vector shows the right broad
shape: it buys success/fill by maxing wear, burning energy, destroying resilience,
and accumulating huge inventory waste. It is not Pareto-clean. However, the fact
that it gets 80% on v0 and 100% on Small is a warning that the scalarized report
can over-reward brute production. The env remains useful only if we interpret
vector components and Pareto tradeoffs, not scalar return alone.

I did not change Scheduling thresholds in this pass. They may be over-tuned, but
that choice is contested: relaxing them lets uniform solve v0 too often, while
keeping them kills all original myopic heuristics. This should be a calibration
decision, not a silent red-team edit.

## Commands / diagnostics run

I reran the requested v0 Scheduling diagnostic with 20 seeds:

```text
PYTHONPATH=src .venv/bin/python - <<'PY'
from experiments.run_baselines import _summarize
from rlh_bench.baselines.scheduling import SCHEDULING_BASELINES
from rlh_bench.baselines.random import RandomPolicy, ZeroPolicy
...
PY
```

Observed `RecoverableCapacityScheduling-v0` reward vectors:

```text
zero                     succ=0.00 ret=-0.676 vec=[0.0, 0.0, 0.0, -13.3264, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
random                   succ=0.00 ret=-0.378 vec=[0.0, 0.3496, 0.0062, -8.3489, -0.4036, -1.0, -0.6309, -0.0566, 0.0, -0.3325, 0.0]
uniform                  succ=0.30 ret=0.817 vec=[0.30, 0.9190, 0.7125, -1.2622, -0.0165, -0.5231, 0.0, -0.0034, -0.0365, -0.1458, 0.4769]
capacity_push            succ=0.80 ret=1.208 vec=[0.80, 0.9712, 0.9000, -0.3530, -0.0097, -1.0, 0.0, 0.0, -6.1570, -0.7500, 0.0]
backlog_priority         succ=0.00 ret=-0.213 vec=[0.0, 0.4739, 0.0062, -6.7758, -0.2924, -1.0, -0.0112, -0.0160, 0.0, -0.1477, 0.0]
earliest_deadline        succ=0.00 ret=-0.112 vec=[0.0, 0.5442, 0.0250, -5.5220, -0.2306, -0.9986, -0.0034, -0.0128, 0.0, -0.1984, 0.0014]
maintenance_aware        succ=0.00 ret=-0.166 vec=[0.0, 0.5068, 0.0062, -6.8619, -0.2591, -0.5997, 0.0, -0.0129, 0.0, -0.1722, 0.3843]
setup_aware              succ=0.00 ret=-0.342 vec=[0.0, 0.3657, 0.0062, -8.2395, -0.3942, -1.0, -0.1026, -0.0166, 0.0, -0.1743, 0.0]
short_horizon_rollout    succ=0.00 ret=-0.111 vec=[0.0, 0.5565, 0.0125, -5.6101, -0.2198, -0.9642, -0.0005, -0.0146, 0.0, -0.0970, 0.0170]
```

I also reran `RecoverableKeyFuelMaze-v0` after moving the route planner out of
the learner-facing baseline list:

```text
zero                     succ=0.00 ret=0.184 vec=[0.0, 0.0, 0.0, 1.0250, 0.0, -0.3750, 0.0, 0.0, 1.0]
random                   succ=0.00 ret=0.064 vec=[0.0, 0.0, 0.0500, 0.1935, -1.0, -0.3750, -0.3663, 0.0, 1.0]
random_constant          succ=0.00 ret=0.023 vec=[0.0, 0.0, 0.0500, 0.0120, -1.0, -0.3750, -0.4281, -0.7470, 1.0]
greedy_landmark          succ=0.00 ret=0.146 vec=[0.0, 0.0333, 0.1750, 0.8808, -1.0, -0.3603, -0.1374, 0.0, 0.8713]
fuel_aware_greedy        succ=0.00 ret=0.146 vec=[0.0, 0.0333, 0.1750, 0.8808, -1.0, -0.3603, -0.1374, 0.0, 0.8713]
efficient_actuator       succ=0.00 ret=0.225 vec=[0.0, 0.0167, 0.1250, 1.2470, 0.0, -0.3698, -0.0084, 0.0, 1.0]
short_horizon_lookahead  succ=0.00 ret=0.152 vec=[0.0, 0.0500, 0.1750, 0.8291, -1.0, -0.3530, -0.1383, 0.0, 0.9275]
oracle_route_planner     succ=0.80 ret=1.421 vec=[0.80, 0.9667, 1.0, 1.2522, 0.0, -0.0911, -0.0070, 0.0, 0.6189]
```

I ran a Scheduling threshold sensitivity probe by recomputing success on the same
v0 rollouts under three alternative terminal criteria:

```text
Recomputed success rates on v0 rollouts under alternative terminal thresholds:
(fill, mandatory, project_quality)

uniform
  mean weighted_fill 0.919 mean mandatory by q {0.55: 0.887, 0.65: 0.781, 0.75: 0.713}
  Small thresholds: 1.00
  mid thresholds  : 0.65
  current v0      : 0.30

capacity_push
  mean weighted_fill 0.971 mean mandatory by q {0.55: 0.956, 0.65: 0.95, 0.75: 0.9}
  Small thresholds: 0.95
  mid thresholds  : 0.95
  current v0      : 0.80

backlog_priority
  mean weighted_fill 0.474 mean mandatory by q {0.55: 0.037, 0.65: 0.006, 0.75: 0.006}
  Small thresholds: 0.00
  mid thresholds  : 0.00
  current v0      : 0.00

earliest_deadline
  mean weighted_fill 0.544 mean mandatory by q {0.55: 0.156, 0.65: 0.075, 0.75: 0.025}
  Small thresholds: 0.05
  mid thresholds  : 0.00
  current v0      : 0.00

maintenance_aware
  mean weighted_fill 0.507 mean mandatory by q {0.55: 0.075, 0.65: 0.006, 0.75: 0.006}
  Small thresholds: 0.00
  mid thresholds  : 0.00
  current v0      : 0.00

setup_aware
  mean weighted_fill 0.366 mean mandatory by q {0.55: 0.006, 0.65: 0.006, 0.75: 0.006}
  Small thresholds: 0.00
  mid thresholds  : 0.00
  current v0      : 0.00

short_horizon_rollout
  mean weighted_fill 0.557 mean mandatory by q {0.55: 0.119, 0.65: 0.031, 0.75: 0.013}
  Small thresholds: 0.00
  mid thresholds  : 0.00
  current v0      : 0.00
```

## Concern 1: `MazeRoutePlannerPolicy` may be an oracle in baseline clothing

Verdict: **This is real — recommended fix: mark it as an oracle diagnostic and
remove it from the learner-facing maze baseline portfolio.**

The policy was worse than merely "strong". Its `__call__` did `del obs` and then
used private generated-world state:

- `_key_positions`
- `_seal_positions`
- `_seal_gate_requirements`
- `_gate_periods`
- `_gate_phases`
- `_extraction_position`
- `_vel`

That is exactly the privileged descriptor access that the counter-design said
must not be counted as a baseline unless explicitly labeled oracle/planner
diagnostic. The observation exposes only current state, three nearest unfinished
landmarks, coarse gate-open features, and status vectors. It does not expose all
waypoint coordinates, seal requirements, gate-to-seal bindings, or extraction
coordinates unless extraction happens to be among the nearest landmarks.

Applied source fix:

- Renamed the implementation to `MazeOracleRoutePlannerPolicy`.
- Changed policy name from `route_planner` to `oracle_route_planner`.
- Removed it from `MAZE_BASELINES`.
- Added `MAZE_ORACLE_DIAGNOSTICS = [MazeOracleRoutePlannerPolicy]`.
- Exported `MAZE_ORACLE_DIAGNOSTICS` from `rlh_bench.baselines`.
- Left a backward-compatible alias `MazeRoutePlannerPolicy =
  MazeOracleRoutePlannerPolicy`, but the public policy name and registry status
  now disclose oracle status.

Cost of an honest non-privileged baseline with `succ >= 0.20` on KeyFuelMaze-v0:

- Non-trivial. Probably 200-500 LOC, not a one-line tuning patch.
- It needs memory. The public observation only gives the nearest three
  unfinished landmarks; an honest planner must build a discovered landmark map
  over time from observations.
- It needs an observation-only controller. The PD inverse can still use the
  public `actuator_matrix` property if that is considered part of the public API,
  but velocity should be read from observation rather than `_vel`.
- It needs seal retry logic. Since seal key requirements and gate bindings are
  hidden, the policy can only infer failure by entering a seal radius and seeing
  no status change, then revisit later after more keys or a different gate phase.
- It needs fuel routing. The current `fuel_aware_greedy` is effectively identical
  to greedy on v0 because fuel stations only matter once fuel is low and nearest
  task pursuit has already caused heat/damage/route issues.
- It probably needs a persistent finite-state route plan: collect all observed
  keys, sweep seals, revisit blocked seals, then extract when the extraction
  landmark has been observed.

Alternative if the lab wants a planner baseline instead of a memory baseline:
expose a public task descriptor or expand the observation to include all
landmark coordinates plus requirements. That would make the environment more of
a known-model planning/control task and less of a partial-information task. I do
not recommend silently letting baselines read underscores.

Immediate consequence: the previous KeyFuelMaze-v0 acceptance claim is invalid.
The oracle reaches 80%, but honest learner-facing baselines remain at 0% success
on the 20-seed panel.

## Concern 2: `SchedulingCapacityPushPolicy` may be a tautology

Verdict: **This is a real concern but acceptable given the vector tradeoff; do
not treat scalar success as proof of long-horizon coupling.**

`capacity_push` is indeed close to an all-positive production pattern:

- project logits = ones
- mode logits = ones
- maintenance = -ones (no maintenance)
- setup = -ones (no retargeting)
- inventory release = default zeros

It is not clever. It ignores deadlines, compatibility structure except through
the env's own production projection, setup retargeting, inventory, wear, and heat
management. If it were near-optimal across vector components, I would call the
env too easy.

The v0 reward vector says it is *not* near-Pareto-optimal:

```text
capacity_push vec=[0.80, 0.9712, 0.9000, -0.3530, -0.0097,
                   -1.0, 0.0, 0.0, -6.1570, -0.7500, 0.0]
```

Interpretation:

- It clears success/fill/mandatory fill.
- It drives final wear to the maximum penalty (`neg_wear=-1.0`).
- It has zero resilience margin.
- It pays very high normalized energy (`neg_energy=-0.75`).
- It pays enormous inventory waste (`neg_inventory_waste=-6.157`).
- It is better than uniform on lateness and shortfall because brute supply works,
  but worse than many baselines on wear, energy, inventory, and resilience.

So the honest statement is: `capacity_push` is a high-cost feasibility policy,
not evidence that the scheduling task is solved. A novel algorithm should be
required to match its success/fill while materially improving at least wear,
inventory waste, energy, and resilience.

Recommended follow-up, not applied here:

- In reports, label `capacity_push` as a "stress/feasibility diagnostic" rather
  than a normal heuristic baseline.
- Add a Pareto or component-dominance table; otherwise scalar return overweights
  success and underweights giant inventory waste.
- Consider increasing scalar weight on `neg_inventory_waste` only if scalar
  return is intended to be meaningful. For vector-RL benchmarking, leave scalar
  secondary.

## Concern 3: The original scheduling heuristics still fail at v0

Verdict: **This is a real concern but mostly acceptable; they are structurally
myopic, not obviously broken by observation scale.**

The original heuristics are not all returning nonsense. They do produce moderate
weighted fill:

- backlog_priority: weighted fill 0.474
- earliest_deadline: weighted fill 0.544
- maintenance_aware: weighted fill 0.507
- setup_aware: weighted fill 0.366
- short_horizon_rollout: weighted fill 0.557

Their catastrophic failure is in mandatory bundle completion:

- backlog_priority: mandatory 0.006 at quality 0.75
- earliest_deadline: mandatory 0.025 at quality 0.75
- maintenance_aware: mandatory 0.006 at quality 0.75
- setup_aware: mandatory 0.006 at quality 0.75
- short_horizon_rollout: mandatory 0.013 at quality 0.75

The threshold sensitivity probe shows that simply restoring Small thresholds on
v0 would not rescue them. Under `(fill=0.55, mandatory=0.50, quality=0.55)`, only
`earliest_deadline` gets 5% success; the rest remain 0%.

That points to structural myopia:

- These policies chase current backlog/urgency.
- They do not explicitly ensure every member of random contract bundles reaches
  the per-project quality threshold.
- They do not preserve mode health/resilience except for the maintenance-aware
  variant, which then loses too much fill.
- They do not plan setup/family alignment over long windows.

I do not see evidence that their thresholds "no longer fire" because observation
scale shifted. They fire enough to produce 36-56% weighted fill. They just spread
service in a way that misses bundle all-members constraints.

Recommended follow-up, not applied here:

- Add a genuinely bundle-aware observation-only heuristic, e.g.
  `bundle_completion_policy`, if the acceptance criterion requires a non-brute
  Scheduling baseline to reach success.
- Make the short-horizon diagnostic actually simulate/copy the env or at least
  reason about bundles; the current proxy is too close to backlog priority.
- Keep the current failing myopic baselines because their component vectors are
  informative. Do not delete them just because they fail success.

## Concern 4: Scheduling-Small `capacity_push=100%` may regress Small calibration

Verdict: **This is a real concern but acceptable if `capacity_push` is treated as
a high-cost feasibility/stress diagnostic; do not hide it from Small by default.**

Small before the patch had a nice shape:

- uniform: 75% success
- backlog/EDD: 40% success
- maintenance/setup: 5-15% success

After adding `capacity_push`, Small has a 100% success row:

```text
capacity_push Small vec=[1.000, 0.968, 0.975, -0.118, -0.001,
                         -1.000, 0.000, 0.000, -2.610, -0.875, 0.000]
```

This does create a visual "one policy dominates success" problem. But it also
reveals a real property of the environment: on Small, brute all-out production
can meet obligations before its costs matter enough to block success.

I do not recommend omitting it from Small just to preserve a prettier baseline
report. Hiding the row would make the calibration look harder than it is. The
right mitigation is labeling and vector/Pareto reporting:

- Keep it visible on Small.
- Mark it as a stress diagnostic in prose.
- Require candidate algorithms to improve the cost components, not merely match
  success.

If the lab wants Small to remain a gentle-but-nondegenerate calibration tier,
then env-level changes are better than hiding the baseline:

- raise inventory-waste consequences,
- make no-maintenance wear reduce late-horizon fill more sharply,
- increase Small mandatory threshold slightly,
- or lower Small capacity slack.

I did not apply those changes because they would require a new full Small/v0
calibration sweep.

## Concern 5: v0 thresholds may have been over-tightened

Verdict: **This is a real concern but contested; I do not recommend a silent
threshold edit in this red-team pass.**

Current Scheduling-v0 thresholds:

- `success_fill_threshold=0.85`
- `success_mandatory_threshold=0.85`
- `quality_required=0.75`

Small thresholds:

- `success_fill_threshold=0.55`
- `success_mandatory_threshold=0.50`
- `quality_required=0.55`

That jump is steep. The threshold sensitivity probe confirms that the current
v0 setting is almost exactly tuned to put uniform at the acceptance boundary:

- uniform at Small thresholds on v0: 100% success
- uniform at mid thresholds `(0.75, 0.70, 0.65)`: 65% success
- uniform at current v0 thresholds: 30% success

So yes, the v0 thresholds were tightened primarily to stop uniform from solving.
That is not automatically wrong: a longer horizon gives uniform enough time to
serve most projects, so v0 needs stricter terminal criteria than Small. But the
current values also make the original myopic heuristics all fail success.

Important nuance: the original heuristics still fail even under Small thresholds
on v0. The strict thresholds are not the only reason they fail; bundle coverage
is the issue. Relaxing thresholds enough to make them succeed would likely make
uniform dominate again.

Recommended options for proposer decision:

1. Keep current thresholds and explicitly state that v0 is a hard quality/bundle
   tier. Then add a non-degenerate bundle-aware baseline later.
2. Use mid thresholds around `(fill=0.75, mandatory=0.70, quality=0.65)`. This
   makes uniform too strong at 65%, so it weakens the acceptance story unless
   another env knob penalizes uniform.
3. Change dynamics instead of thresholds: increase long-horizon cost of brute
   production and uniform service, while preserving enough capacity for a
   bundle-aware policy.

I recommend option 1 for now, plus better reporting language. I did not change
thresholds in `registration.py`.

## Additional issues outside the five listed concerns

1. **Baseline report is now stale after the oracle fix.**
   `docs/baseline_report.md` currently contains the old `route_planner` row as a
   normal baseline. After the source change, rerunning `experiments/run_baselines.py`
   will omit that row from learner-facing maze baselines. I did not regenerate the
   report in this review because the proposer should first decide whether to add
   an honest maze baseline or accept that KeyFuelMaze-v0 currently lacks one.

2. **`docs/SUBSTRATE_MAP.md` appears stale for Scheduling.**
   Its Scheduling baseline list omits `capacity_push`. This predates this review
   and should be updated when the baseline taxonomy is settled.

3. **`RecoverableKeyFuelMazeEnv.reset()` samples `_gate_phases` twice.**
   The first assignment is immediately overwritten. This is deterministic but
   consumes extra RNG state and silently shifts later sampled world components.
   I did not change it here because removing it would rebaseline all maze seed
   worlds. It should be fixed only with an explicit baseline regeneration.

4. **Some maze baselines still use private `_seed` for actuator-matrix caching.**
   This is not an oracle world descriptor in the same way waypoint positions are,
   and some policies recache every call because `RecoverableKeyFuelMazeEnv` lacks
   a public `seed` property. It is worth cleaning by adding a public `seed`
   property or dropping the cache. I did not treat this as acceptance-blocking.

## Final recommendation

- Treat the previous KeyFuelMaze-v0 acceptance as failed: the 80% success came
  from an oracle diagnostic, not a baseline.
- Keep the oracle planner, but only as `MAZE_ORACLE_DIAGNOSTICS`.
- Do not hide `capacity_push`; label it as a brute-force stress diagnostic and
  judge candidates on vector/Pareto improvements.
- Do not silently relax Scheduling-v0 thresholds. They are suspiciously tuned,
  but relaxing them makes uniform too strong unless dynamics change too.
- Next concrete work item: implement either an honest observation-only maze
  memory planner or expose a public planning descriptor and explicitly classify
  the task as known-model planning.
