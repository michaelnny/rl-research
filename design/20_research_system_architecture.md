# Autonomous research system architecture

Status: automated harness implemented; long-horizon benchmark admission pending

The implementation now covers the graph, artifacts, queue/leases/recovery,
provider adapters, worktree isolation, structured run parsing, campaign
lifecycle, hard attempt/wall budgets, portfolio branch allocation, incidents,
cross-provider replication DAGs, and the process-isolated candidate evaluator.
The release includes fail-closed admission, a campaign-scoped process
supervisor, repeated-trial evaluation, restart recovery, and deterministic
budgets. A seven-day production campaign is operational evidence collected
after an operator authorizes provider and accelerator spend; it is not replaced
by a software claim. Provider billing reconciliation, automatic promotion, and
applied-family transfer are later extensions and are not implied by this release.

## Core model

The harness is a search system over a typed, immutable research graph. Language
models propose graph transformations; deterministic code owns state,
permissions, execution, scoring, and recovery.

The system does not simulate a lab meeting and does not rely on agents being
consistent across sessions. Every model call is treated as an unreliable,
stateless worker that may time out, hallucinate, edit the wrong file, or produce
an invalid conclusion.

## Research graph

Primary node types:

- `Observation`: a sourced fact from code, literature, or a run;
- `Hypothesis`: mechanism, quantitative predictions, scope, and falsifier;
- `PriorArtClaim`: search query, source, relationship, and uncertainty;
- `ProbePlan`: smallest experiment that distinguishes named explanations;
- `Implementation`: immutable Git revision and build manifest;
- `Run`: exact command, environment, budget, seed, logs, and raw artifacts;
- `Analysis`: code-backed transformation from runs to findings;
- `Finding`: claim linked to the runs and analysis that support or contradict it;
- `Replication`: independent rerun from a clean revision;
- `Synthesis`: a view of live hypotheses, contradictions, and next information
  needs; and
- `Incident`: infrastructure or scientific-integrity failure.

Edges are typed: `motivates`, `tests`, `implements`, `produces`, `supports`,
`contradicts`, `replicates`, `supersedes`, and `cites`. Nodes are append-only;
incorrect findings are contradicted or superseded, not edited into correctness.

## Search loop

The atomic research loop is:

1. choose an open uncertainty, not merely an idea;
2. generate multiple candidate hypotheses or operators;
3. design the cheapest discriminating probe;
4. implement each viable probe in an isolated worktree;
5. execute fixed validation and measurement code;
6. attach results without letting the author reinterpret missing data;
7. allocate replication or ablation budget based on expected information gain;
8. update the research graph; and
9. periodically search prior work and synthesize across branches.

Ideation cannot recurse indefinitely. A branch receives additional budget only
after adding new empirical evidence, a sourced prior-art result, or a formal
derivation. Prose that only restates a parent does not count as progress.

## Search policy

The first scheduler will use an explicit portfolio rather than one LLM score:

- 35% exploit: follow branches with replicated capability improvements;
- 25% falsify: attack assumptions and run ablations;
- 20% explore: mechanisms distant from active branches;
- 10% transfer: test a mechanism on a second family; and
- 10% synthesize/prior-art: merge duplicates and detect rediscovery.

Within each pool, asynchronous successive halving allocates small probe budgets
before confirmation runs. A novelty descriptor based on mechanism and measured
fingerprint prevents five prompt variants of one idea from consuming the
exploration pool. These percentages are configuration, recorded per campaign,
and must be evaluated rather than treated as universal truths.

## Agent roles

Roles are permission and output contracts, not personalities.

- `mapper`: read-only code/literature observation;
- `theorist`: hypothesis and derivation, no code mutation;
- `probe_designer`: ProbePlan only;
- `implementer`: one worktree, no benchmark or evaluator writes;
- `skeptic`: read-only confound and falsification analysis;
- `analyst`: analysis code plus Finding candidates;
- `replicator`: clean implementation/run context, no author transcript;
- `prior_art`: sourced search only; and
- `synthesizer`: graph view, no authority to promote or merge.

Codex is initially preferred for implementation and repository analysis. Claude
is useful as an independent critic and alternate implementation path. The
scheduler deliberately crosses providers on replication and review, but does
not assume that “more agents” means independent evidence.

## Execution and isolation

- One durable SQLite database in WAL mode stores graph metadata, queues,
  leases, budgets, and events.
- Immutable large artifacts live in a content-addressed filesystem store.
- Every code-writing job receives a dedicated Git worktree and branch. Validated
  output is committed on the evidence branch; the worktree is removed after its
  evaluation while the commit remains inspectable.
- Candidate worktrees cannot write benchmark, evaluator, reference-result, or
  held-out-manifest paths.
- Experiments run through a local executor with CPU, memory, wall-time, and
  process-count limits. GPU use requires an explicit parameter, accelerator-time,
  and device-memory budget grant. Compact neural GPU training is the normal RL
  workload, not an exceptional algorithm class.
- Held-out evaluation runs in a separate checkout without agent tools and does
  not reveal world identifiers to training code.
- Provider output, stdout, stderr, exit code, duration, and usage are retained.
- Candidate measurements retain a neural model manifest, framework/device,
  trainable parameter count, binary checkpoint digest, and evaluator-measured
  training/accelerator-time upper bound.
- One high-entropy suite key is stored per campaign with owner-only permissions.
  Model-provider processes are OS-sandboxed from the runtime tree and strip all
  `RLX_` variables. The trusted outer evaluator receives only the path to its
  owner-only key file; its candidate child receives neither the path nor key.
  Both provider and candidate sandboxes deny process-information access.

No agent commits to or merges the integration branch. A deterministic
integration service may commit a verified worktree. Merge requires configured
tests and a graph link to the ProbePlan it implements.

## Reliability

Jobs transition atomically through `queued`, `leased`, `running`, and terminal
states. Workers heartbeat leases. A watchdog kills the entire process group on
timeout. Expired jobs are retried only for infrastructure-class failures and
only up to a limit. Scientific negative results are completed findings, not
retryable errors.

Supervisor shutdown sends SIGTERM to each worker. The worker explicitly
terminates every active provider, evaluator, and Git executor process group
before returning, and interrupted jobs are requeued as infrastructure failures;
model or candidate grandchildren cannot survive as orphaned processes.

The daemon is restart-safe and idempotent. It has no dependence on a model
creating a particular filename. Structured outputs are validated against role
schemas; code jobs additionally require an inspectable diff and configured
verification results.

After repeated failures the scheduler opens an Incident and shifts budget to
other branches. It never burns provider calls indefinitely on a broken job.

## Evidence and promotion

An implementation is eligible for the active comparison set only when:

1. the hypothesis predicted a result before the run;
2. the evaluator produced a complete fixed-budget record;
3. the claimed mechanism has an ablation or matched counterfactual;
4. an independent worker replicated it from a clean revision;
5. the effect survives multiple training seeds and worlds;
6. it transfers to a second diagnostic configuration or applied family; and
7. prior-art status is sourced or explicitly unresolved.

“Eligible” is not “novel,” “correct,” or “publishable.” It only means the result
is reproducible enough to justify more research budget.

## Human control points

The system may run ordinary probes continuously, but the following remain
explicit human actions in the first production version:

- changing benchmark or evaluator semantics;
- increasing campaign compute/provider budgets;
- exposing or publishing held-out results;
- merging a candidate into the canonical algorithm set; and
- making a public novelty claim or submission.

These boundaries can be revisited after the system itself has a failure dataset.

## Operator surface

The command-line surface supports campaign creation and lifecycle control,
queue/campaign status, manual structured-job enqueueing, lease recovery,
`once`/`drain`/`daemon` worker modes, one-shot or daemon controllers, supervised
serving, and disposable human research briefs. Candidate promotion and branch
integration intentionally remain human-controlled actions rather than hidden
automation.

`doctor` verifies the committed snapshot, campaign key, schemas, provider CLI
compatibility, and macOS sandbox. `serve` refuses to start on failure, then
supervises a controller and a configurable number of campaign-scoped worker
processes with group termination on shutdown.

The brief is a view. Deleting it loses no scientific state.
