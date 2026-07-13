# RLX Research

RLX Research is a clean-sheet project for discovering reinforcement-learning
mechanisms for long causal horizons, large or structured action spaces, sparse
feedback, and vector-valued objectives—without requiring giant simulators or
large neural networks.

The clean-sheet vertical slice is executable end to end. FactorLab is still
under calibration, so campaign results are research evidence—not yet benchmark
claims—and the first unattended deployment must still complete its soak gate.

## Architecture

- [Foundations](design/00_foundations.md) defines the scientific target and the
  assumptions explicitly rejected from the earlier attempt.
- [Benchmark architecture](design/10_benchmark_architecture.md) defines the
  capability matrix, FactorLab diagnostics, applied families, metrics, and
  qualification studies.
- [Research-system architecture](design/20_research_system_architecture.md)
  defines the immutable evidence graph, search loop, isolation, and promotion
  requirements.
- [Rebuild plan](design/30_rebuild_plan.md) defines delivery order and deletion
  of the legacy implementation.
- [Candidate protocol](design/40_candidate_protocol.md) defines the executable
  learner/evaluator trust boundary.

## Clean-sheet boundary

All replacement work lives in `rlx_bench`, `rlx_agents`, and `rlx_lab`. These
packages do not import the previous `rlh_bench` implementation. Legacy code is
neither a specification nor a baseline result.

## Implemented benchmark kernel

`src/rlx_bench/` now provides the first FactorLab vertical slice:

- matched flat, embedded-catalog, factored-discrete, continuous, and
  conditional-hybrid action renderings over one canonical decision space;
- procedural long causal lags, terminal or sparse vector feedback, controlled
  objective conflict, memory lag, and additive/pairwise/threshold/prerequisite
  effects;
- separate learner and evaluator-only causal-inspector APIs;
- exact tiny-instance weighted and Pareto solvers plus intervention audits;
- evaluator-enforced transition, episode, wall, policy, and preference-query
  budgets;
- preference-panel, Pareto, hypervolume, and constraint metrics with fixed
  semantic normalization; and
- campaign-wide HMAC-derived public/held-out suites, hidden learnable cue
  transforms, and cryptographic commitments backed by 256-bit evaluator keys.

`src/rlx_agents/` contains real tabular and factorized policy-gradient
calibration learners, explicitly labeled privileged mechanism probes, and a
process-isolated candidate evaluator. Each configured training trial starts
fresh, and every held-out world starts a new process from that trial's
checkpoint without exposing world objects, transforms, or seeds.

FactorLab remains **under calibration**. The qualification state machine cannot
promote smoke observations; all ten studies require immutable evidence, and
statistics plus independent audit are not complete.

## Implemented research system

`src/rlx_lab/` provides:

- a transactional SQLite research graph and durable job queue;
- atomic leases, heartbeats, bounded retries, dependency blocking, and recovery;
- content-addressed immutable artifacts;
- shell-free process execution with time, output, CPU, and memory limits;
- isolated Git worktrees with protected-path enforcement;
- committed evidence branches plus automatic worktree cleanup after evaluation;
- structured Codex, Claude, and deterministic fake-provider adapters;
- agent and evaluator workers;
- a deterministic controller that expands hypotheses into falsification,
  matched probes, isolated primary and cross-provider replication
  implementations, hidden-world runs, findings, and periodic synthesis;
- weighted exploit/falsify/explore/transfer/synthesis branch allocation;
- atomic campaign wall/provider/run budgets, controller leases, incidents, and
  pause/resume/stop controls; and
- high-entropy per-campaign evaluator keys that model providers cannot read;
- fail-closed production preflight and a process-supervised multi-worker
  `serve` operation; and
- queue/status/recovery/brief/controller/worker daemon operations.

The end-to-end tests run a fake campaign through hypothesis, isolated
implementation, measurement, independent reimplementation, replication, and an
evidence-backed finding.

## Calibration preview

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_agents.cli smoke \
  --output runtime/factorlab-smoke.json
```

This produces a provisional report whose `qualified` field is necessarily
false.

## Campaign operation

The canonical branch must be clean and committed because every implementation
worktree is derived from `HEAD`.

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime init \
  --name first-campaign \
  --question "Which credit-assignment mechanisms remain stable as causal lag grows?"

PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime doctor --campaign CAMPAIGN_ID --live-providers

PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime serve --campaign CAMPAIGN_ID --workers 3
```

`init` creates an owner-only 256-bit suite key under `runtime/secrets/`. `serve`
will not start if the repository snapshot, key, schemas, provider commands, or
OS sandbox fail preflight. `serve` also performs authenticated structured calls
through every provider configured by the campaign. It then supervises
campaign-scoped worker processes and
stops them as a group. Live Codex and Claude campaigns use the operator's
existing CLI authentication and remain bounded by the campaign policy.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q
PYTHONPATH=src:. .venv/bin/python -m ruff check \
  src/rlx_bench src/rlx_agents src/rlx_lab tests/bench tests/agents tests/lab
uv lock --check
```

## Legacy removal

The previous `rlh_bench` package, environments, baselines, tests, experiment
outputs, reports, examples, prompts, journals, and shell lab loop have been
deleted. Git history is the only archive. No replacement module imports or
depends on that implementation.
