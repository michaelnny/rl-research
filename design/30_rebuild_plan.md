# Rebuild plan

Status: initial automated-research release complete

Progress on 2026-07-13: Milestone 1's replacement kernel is implemented and
tested. The original Milestone 2 implementation was rejected because it made
tabular cue lookup the central learner. Neural FactorLab v1 and the batched
neural candidate protocol replace it; no v0 result carries over. The exact
`factorlab-small-v1` anchor passed all ten preregistered qualification gates on
2026-07-13 and is the only tier admitted to automated campaigns. Milestone 4's
deterministic controller, isolated neural candidate evaluator, provider-crossed
replication loop, fail-closed preflight, and process supervisor are implemented.
A seven-day campaign is a post-launch operational study requiring explicit
operator spend authorization, not a prerequisite hidden inside repository
finalization. No applied family is implemented or qualified. The legacy package, environments, tests,
experiments, outputs, reports, prompts, journals, examples, and shell loop have
been removed; Git history is the archive.

## Repository decision

The existing benchmark and lab loop will not be refactored into the replacement.
New code starts in new namespaces with new interfaces and tests. Git history is
the archive; there is no reason to preserve old architecture inside the final
tree.

The old code has been removed. No old environment, metric, prompt, runner, test,
or result is grandfathered, and no replacement namespace imports legacy code.

## Target layout

```text
src/
  rlx_bench/          # new task specs, generators, evaluator API, metrics
  rlx_agents/         # reference algorithms and candidate protocol
  rlx_lab/            # graph store, scheduler, workers, providers, executor
tests/
  bench/
  agents/
  lab/
campaigns/            # versioned campaign definitions, no mutable state
design/               # architecture and decision records
runtime/              # ignored DB, worktrees, logs, artifacts, caches
```

`rlh_bench`, the current `experiments` runner, and the shell `lab` loop are
deleted after their replacements pass the corresponding end-to-end tests.

## Milestone 1: scientific kernel

Deliverables:

- immutable research-graph schema and SQLite store;
- atomic queue/lease/recovery behavior;
- content-addressed artifact store;
- local executor with time/resource limits;
- Codex, Claude, and deterministic fake-provider adapters;
- isolated worktree manager;
- typed role schemas; and
- an end-to-end fake campaign that survives worker termination and restart.

Exit condition: a fake hypothesis progresses through implementation, execution,
analysis, and replication without shared mutable state or manual file discovery.

## Milestone 2: FactorLab

Deliverables:

- native structured action specification;
- procedural causal generator and privileged ground truth;
- terminal/sparse vector objective protocols;
- exact solver for tiny instances;
- intervention-based causal audit;
- training/evaluation budget enforcement; and
- a baseline fingerprint over horizon, causal lag, and action structure.
- continuous procedural observations and a nonlinear suite-shared kernel that
  require neural representation generalization;
- compact MLP/recurrent neural reference learners with parameter and GPU-hour
  accounting; and

Exit condition: changing one controlled factor produces the predicted baseline
sensitivity while non-target factors remain bounded.

## Milestone 3: applied families

Implement SlateMarket, GraphOps, and AssemblyLab one at a time. Each remains not
admitted until it has a feasibility bound, learner evidence, headroom,
generalization results, runtime measurements, and an independent audit.

Exit condition: at least two families are qualified and a known mechanism found
in FactorLab transfers under the same fixed-budget protocol.

## Milestone 4: autonomous campaign

- seed the graph with a sourced algorithm-family map;
- run parallel theory, falsification, and implementation branches;
- apply successive-halving budgets;
- cross providers for replication;
- generate daily human briefs and weekly contradiction/prior-art audits; and
- record harness incidents as a dataset for scheduler improvement.

Initial-release exit condition: the full primary/replica evidence DAG passes a
bounded end-to-end test; termination/restart recovery, protected-path checks,
candidate process isolation, and budget exhaustion are tested; and `doctor`
passes against the committed qualified tier and authenticated providers.

Operational validation target: seven uninterrupted days with no benchmark
mutation, no lost run provenance, bounded failure retries, and at least one
independently replicated algorithmic result (positive or negative).

## Milestone 5: remove legacy — completed 2026-07-13

Delete the old packages, tests, docs, prompts, baselines, results, and shell
loop. Rewrite the root README and project metadata around the new system. Run a
final audit that no new module imports or relies on legacy code.

## Post-release expansion

The running system is restricted to `factorlab-small-v1`. Future work may add
memory-lag tiers, other action/objective protocols, additional compact neural
references, and applied families only through new frozen qualification studies.
Live model output remains untrusted input throughout.
