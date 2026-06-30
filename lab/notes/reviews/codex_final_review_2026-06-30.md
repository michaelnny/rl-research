# Codex final pre-merge review — substrate redesign

Date: 2026-06-30
Branch: `lab/substrate-redesign`
Reviewed working tree state as-is.
I found no hidden stochasticity, no discrete action spaces in the six registered
envs, and no per-step nonzero rewards in the env implementations.
I did find three things I would not merge silently:
1. `RecoverableCapacityScheduling-v0` and `-Large-v0` advertise trailing action
   dimensions that have no control effect; they only add terminal energy cost.
2. `RecoverableKeyFuelMaze` gives a perfect `route_efficiency` component to
   incomplete/idle trajectories, including the zero policy.
3. The gate audit and several operator-facing docs/prompts are stale or
   internally contradictory relative to the current code.
The test suite passes:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q
......................................................................   [100%]
70 passed in 35.5s
```

I only wrote this review file. I did not modify envs, baselines, or docs.
At session start the tree already had uncommitted changes in:
- `lab/notes/acceptance_gates_audit_2026-06-30.md`
- `tests/test_capacity_scheduling_env.py`
- untracked `lab/notes/codex_final_review_oneshot.md`
I treated those as part of the state to review and did not overwrite them.

---

## 1. Registered substrate / CLAUDE mission compliance

Should fix before merge: CapacityScheduling v0/Large violate the action-complexity
spirit by carrying trailing dimensions with no positive control semantics.
The six registered env IDs are exactly the intended two-family × three-tier set:
- `RecoverableCapacityScheduling-Small-v0`
- `RecoverableCapacityScheduling-v0`
- `RecoverableCapacityScheduling-Large-v0`
- `RecoverableKeyFuelMaze-Small-v0`
- `RecoverableKeyFuelMaze-v0`
- `RecoverableKeyFuelMaze-Large-v0`
All six expose `Box([-1, 1]^D)` action spaces.
All six terminate only at the fixed horizon and return `truncated=False` in the
normal path.
I did not find any registered env with a `Discrete` or `MultiDiscrete` action
space.
The legacy `Discrete`/`MultiDiscrete` classes still exist in `spaces.py`, but
nothing in `registered_envs()` uses them.
The new envs seed their world generators from `np.random.default_rng(seed)` in
`reset()`.
Same seed + same action sequence is tested for both families and looks correct
on inspection.
Non-terminal reward is zero in both scalar and vector modes.
The per-step diagnostics in `info` expose state summaries, but the env reward
itself remains terminal-only.
The main mission gap is CapacityScheduling action dimensionality.
Current required semantic layout is:

```text
K project logits + M mode logits + M maintenance + M setup + P inventory
= K + 3M + P
```

For the registered tiers:
- Small: `K=16, M=4, P=4`, required `32`, registered `D=32` — OK.
- v0: `K=48, M=8, P=8`, required `80`, registered `D=96` — 16 trailing dims.
- Large: `K=128, M=16, P=16`, required `192`, registered `D=224` — 32 trailing dims.
`capacity_scheduling.py` documents these trailing dims as ignored by the env
semantics.
They do affect `neg_energy` because `_total_energy` sums over the whole raw
action vector, but they do not affect production, maintenance, setup, inventory,
wear, heat, lateness, success, or resilience except through that penalty.
That makes them not useful continuous controls; they are at best harmful dummy
channels.
The v2 plan's gate 8 says appending no-op dims is forbidden and actuator /
allocation dims must have measurable behavioral effect.
Suggested fix before merge:
- easiest: set Scheduling v0 `action_dim=80` and Large `action_dim=192`, update
  docs and regenerate affected baseline numbers; or
- better but more work: wire the extra dims into real behavior, e.g. secondary
  per-mode overdrive / quality / maintenance-allocation channels whose use has
  both upside and cost.
A narrow test should be added after the fix: changing any action dimension over
a fixed trajectory should be able to change at least one non-energy terminal
component on at least one seed.

---

## 2. Terminal vector semantics

Should fix before merge: KeyFuelMaze `route_efficiency` currently rewards
incomplete inaction.
`RecoverableKeyFuelMazeEnv._terminal_reward_vector()` computes:

```python
actual = max(self._total_distance, 1.0)
route_efficiency = min(self._oracle_route_length / actual, 1.0)
```

If the agent does not move, `actual == 1.0`, the oracle route length is much
larger than 1, and `route_efficiency` becomes `1.0`.
This is visible in the current baseline report:
- KeyFuelMaze Small zero policy: `route_efficiency = 1.000`
- KeyFuelMaze v0 zero policy: `route_efficiency = 1.000`
The zero policy therefore gets a maximum score on the route-quality component
while completing no keys, no seals, and no extraction.
That does not break success, but it does make one terminal-vector component
misleading and gives scalar reward for not attempting the route.
Suggested fix before merge:
- define route efficiency only for completed routes, e.g. `0.0` unless
  `self._extraction_reached`; or
- multiply by task progress, e.g. `route_efficiency *= seal_completion` and
  optionally require extraction for full credit; or
- replace it with a progress-normalized travel efficiency that cannot be maxed
  by zero distance.
After changing this, regenerate `docs/baseline_report.md` and add a regression
test that the zero policy's route-efficiency component is not maximal.

---

## 3. Determinism and hidden nondeterminism

Ready to merge: no hidden nondeterminism found in the registered envs.
Both new envs create all stochastic world state during `reset(seed=...)` using
the env-owned NumPy generator.
`step()` does not sample randomness.
`reset(seed=None)` maps to seed 0 in both new families, which is deterministic.
That is acceptable for this deterministic substrate, though it is not Gym's
usual random-on-None behavior.
The earlier `_gate_phases sampled twice` concern appears fixed / false for the
current state.
Current `RecoverableKeyFuelMazeEnv.reset()` samples `_gate_phases` once:

```python
self._gate_phases = self._rng.integers(0, self._gate_periods).astype(np.int64)
```

I found no second assignment.

---

## 4. Recoverability and long-horizon behavior

Acceptable but flag for follow-up: recoverability is structurally present, but
the current tests are weaker than the audit summary implies.
CapacityScheduling has recoverable state dynamics: bad actions do not terminate,
backlog can still be served later, and wear/heat/maintenance create graded
terminal effects.
KeyFuelMaze has nonterminal fuel exhaustion and boundary collisions; a mistake
continues the episode instead of ending it.
However, the tests do not yet establish a strong graded recoverability curve.
`test_recoverability_is_graded` says it checks early/mid/late degradation, but
it does not assert monotonicity.
Its no-collapse assertion is `min(early, mid, late) >= 0.0`, which is almost
tautological for a clipped fill-rate component.
The new v0 test is useful as a smoke test, but its second assertion allows the
burst to be slightly better than baseline (`mean_burst < mean_baseline + 0.05`).
That is fine as a loose guard against catastrophic collapse, but it does not
justify a strong gate-7 ✓ on its own.
Suggested follow-up:
- publish a small recoverability probe that injects bursts at several fractions
  of the horizon and reports multiple vector components;
- assert finite nontrivial degradation, not just non-collapse; and
- include KeyFuelMaze recoverability beyond fuel-exhaustion-not-terminal, e.g.
  heat/damage/collision perturbations with partial route recovery.

---

## 5. Baseline honesty

Ready to merge: the oracle/learner-facing separation is materially improved.
`MAZE_BASELINES` does not include `MazeOracleRoutePlannerPolicy`.
`MAZE_ORACLE_DIAGNOSTICS` contains the oracle route planner separately.
`tests/test_keyfuel_maze_env.py::test_oracle_route_planner_not_in_baseline_portfolio`
checks the separation and disjointness.
The previous `_seed` cache concern is cleaned up in the honest maze portfolio.
The learner-facing maze policies use `env.seed` and `env.actuator_matrix`, not
`env._seed`.
A grep shows private `env._...` accesses in `baselines/maze.py` are confined to
`MazeOracleRoutePlannerPolicy`.
That oracle still intentionally reads `_key_positions`, `_seal_positions`,
`_gate_periods`, `_gate_phases`, `_seal_gate_requirements`, `_extraction_position`,
and `_vel`.
That is correctly documented as privileged and outside `MAZE_BASELINES`.
Acceptable but flag for follow-up: the honest maze baselines are not purely
observation-only in the strict sense.
`MazeGreedyLandmarkPolicy`, `MazeFuelAwareGreedyPolicy`,
`MazeEfficientActuatorPolicy`, and `MazeShortHorizonRolloutPolicy` read the
public `env.actuator_matrix` property to map landmark directions into actuator
channels.
That is not a private underscore field, and it may be intended public model
information.
But `experiments/run_baselines.py` describes the learner-facing portfolio as
"observation-only, no privileged env-internal access".
If `actuator_matrix` is part of the public model API, the docs should say so.
If the baseline portfolio is meant to be observation-only, those policies should
not read it, or the matrix/cost basis should be included in the observation.
I would not block merge on this if the proposer explicitly classifies these as
"public-model controller" baselines rather than oracle baselines.

---

## 6. Baseline report freshness

Ready to merge: quick spot checks match the current code for the checked rows.
I re-ran `_summarize(...)` for the zero policy on:
- `RecoverableCapacityScheduling-Small-v0`
- `RecoverableKeyFuelMaze-Small-v0`
The resulting success rate, mean return, mean length, and rounded mean reward
vectors matched `experiments/results/baselines.json` exactly.
`docs/baseline_report.md` is therefore not stale relative to the current env code
for those spot-checked rows.
Acceptable but flag for follow-up: the report is not a held-out report.

The stored JSON has `held_out: null`; the report was generated on train-style seeds only.

That is acceptable as a baseline snapshot, but it should not be cited as gate-9
evidence.

If gate 9 is claimed ✓, add a held-out table by running:

```text
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --use-held-out
```

or explicitly state that the report remains train-seed-only.

---

## 7. `docs/SUBSTRATE_MAP.md`

Ready to merge: it correctly lists the six registered env IDs, `capacity_push`,
and the separated maze oracle diagnostics.

It also correctly documents terminal-only rewards, reward-vector orientation,
and the held-out seed-band API.

Acceptable but flag for follow-up: it transparently documents the Scheduling
trailing unused dimensions, but that transparency exposes the gate-8 issue noted
above.

If the extra Scheduling dims are removed or made semantic, update this page and
regenerate the baseline report.

Acceptable but flag for follow-up: `make_setup_graph()` / `_setup_graph` is
sampled for CapacityScheduling but never used by the env.

`world_gen.py` says setup graphs provide deterministic edge weights for setup
change costs.

Current Scheduling setup churn is plain L1 mixture movement times a scalar; the
sampled graph does not affect dynamics or rewards.

This is not a merge blocker by itself, but either use `_setup_graph` in churn /
setup-pressure costs or remove the claim and the unused sampled tensor.

---

## 8. Gate audit honesty

Should fix before merge: `lab/notes/acceptance_gates_audit_2026-06-30.md` is
internally inconsistent and overclaims several ✓ statuses.

The top/body still says the status is as of commit `3526fbc` and contains old
pre-calibration claims.

Examples:

- Gate 4 body says no formal idle-tail measurement was done and status is
  partial; the summary says ✓ with tail-zero probe + tests.
- Gate 9 body says infrastructure ready but runtime use pending; the summary
  says ✓ for all tiers.
- Gate 10 body says normalization is partial and needs a pass; the summary says
  ✓ after audit.
- Headline text still says Small passes 11/12 with no formal idle-tail test,
  contradicting the updated summary.

The uncommitted edit improves the summary row for gate 5/7, but the body still
needs to be reconciled.

Suggested fix: rewrite the audit as a final-state document, not a historical
appendix, or split it into "original audit" and "post-fix addendum" sections.

Gate statuses I would use after this review:

- Gate 1 determinism: ✓.
- Gate 2 terminal-only: ✓.
- Gate 3 feasibility: ✓/⚠, because maze feasibility relies on oracle diagnostics,
  not honest baselines.
- Gate 4 no-idle-tail: ⚠/✓ for tested Small/v0 paths; Large is probe-only unless
  the probe output is recorded.
- Gate 5 lookahead-depth: ⚠, because varied-depth probes are deferred.
- Gate 6 myopic-gap: ⚠, because `capacity_push` is a stress diagnostic and maze-v0
  honest baselines remain at 0% success.
- Gate 7 recoverability: ⚠, because tests are smoke tests, not graded curves.
- Gate 8 action-complexity: ✗ until Scheduling v0/Large trailing dims are fixed.
- Gate 9 seed-generalization: ⚠ unless held-out baseline/report evidence is added.
- Gate 10 reward-normalization: ⚠ unless KeyFuel and all tiers are explicitly
  covered, not just Capacity Small/Large.
- Gate 11 baseline portfolio: ✓, with oracle separation caveat.
- Gate 12 runtime: ✓ if measured numbers are retained.

---

## 9. Test suite coverage

Ready to merge: the suite passes and is not obviously bloated for a substrate
redesign of this size.

The 70 collected tests include legacy env compatibility tests, which is fine if
legacy classes remain importable.

Should fix before merge: add narrow regression tests for the two concrete env
bugs above.

Recommended tests:

- Scheduling v0/Large should have no trailing non-control dims, or every action
  dim should affect a non-energy terminal component in some controlled rollout.
- KeyFuel zero policy should not receive maximal `route_efficiency` when no
  extraction is reached.

Acceptable but flag for follow-up: several gate claims have only partial test
backing.

Coverage holes:

- Terminal-only is tested deeply on Small configs, not parametrized across all
  six registered env IDs.
- Idle-tail pytest coverage is Scheduling Small/v0 and Maze Small only.
- Reward-normalization pytest coverage is CapacityScheduling Small vs Large only;
  KeyFuelMaze is not covered.
- Held-out seed CLI/report generation is not tested, only seed-band ranges and
  representative world differences.
- Recoverability tests do not yet assert a graded curve.

These are not all merge blockers, but they should be tracked honestly in the
gate audit.

---

## 10. Other documentation surfaces

Should fix before merge: production-facing docs/prompts still contain stale
pre-redesign references.

Examples found by grep:

- `CLAUDE.md` says the test suite "Should be 25 passed"; current suite is 70.
- `README.md` says tests should be 25 passed.
- `docs/LAB.md` says `baseline_report.md` has numbers across all six env IDs;
  the current report covers Small + v0 only, with Large deferred.
- `docs/AGENT_GUIDE.md` example uses `RecoverablePointMaze-v0`, which is no
  longer registered.
- `experiments/algorithms/runner.py` docstring also uses `RecoverablePointMaze-v0`.
- `lab/prompts/claude_system.md` still lists the old `RecoverablePointMaze-*` and
  `RecoverableResourceAllocation-*` registry as the substrate highlights.

These are likely to misdirect the autonomous loop immediately after merge.

Suggested fix: update these docs/prompts in the same pre-merge cleanup as the
gate audit.

---

## 11. Final recommendation

Should fix before merge: do not merge the current state without addressing the
Scheduling trailing action dims, KeyFuel route-efficiency semantics, and stale
operator-facing documentation/gate audit.

After those are fixed, the remaining issues look acceptable as follow-up work:

- clarify whether public `actuator_matrix` access is allowed for learner-facing
  maze baselines;
- strengthen recoverability / idle-tail / held-out probes;
- decide whether to use or remove the unused setup graph; and
- regenerate baseline numbers when env semantics change.

The core substrate is close: deterministic, terminal-only, continuous-action
registered envs are in place, oracle separation is much cleaner, and tests pass.

I would call this merge-ready after the three blockers above are resolved and the
acceptance audit is made consistent with the actual evidence.
