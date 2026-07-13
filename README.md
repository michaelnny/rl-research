# RLX Research

RLX Research is a neural reinforcement-learning benchmark and autonomous R&D
system for long causal horizons, enormous structured action spaces, sparse or
terminal feedback, and genuinely vector-valued objectives.

“Compute-light” here means compact neural policies that train within hours on
one consumer GPU—including Apple Silicon through PyTorch MPS. It does not mean
tabular learning, linear cue maps, or avoiding neural networks. Learner runs are
bounded by parameters, transitions, wall time, accelerator time, and device
memory; LLM-scale learner training is outside the target envelope.

## Current architecture

- [Foundations](design/00_foundations.md) defines the scientific target,
  compact-neural compute envelope, and neural-task admissibility rules.
- [Benchmark architecture](design/10_benchmark_architecture.md) defines Neural
  FactorLab, capability sweeps, metrics, and evidence-gated qualification.
- [Research-system architecture](design/20_research_system_architecture.md)
  defines the immutable evidence graph, deterministic search policy, worker
  isolation, and promotion requirements.
- [Candidate protocol](design/40_candidate_protocol.md) defines batched neural
  interaction, model manifests, and bounded binary checkpoints.

The rejected tabular-centered implementation and the later 64-step experiment
are not research benchmarks. Both remain only in Git history.

## Long-horizon Neural FactorLab v1

`src/rlx_bench/` provides a procedural neural RL diagnostic family with:

- continuous signals, contexts, and dynamic state—no reusable finite cue
  dictionary;
- an evaluator-owned nonlinear neural task/dynamics kernel shared across
  public, tune, held-out, and audit worlds;
- delayed causal effects over 5,000, 10,000, and 20,000 steps with terminal-only or sparse
  vector reward;
- factored discrete spaces reaching at least `10^12` joint choices, plus native
  flat, catalog, continuous, and conditional-hybrid renderings;
- preference-conditioned, policy-coverage, and constrained objective semantics;
- evaluator-only tiny exact solvers, intervention audits, and a separate
  numerical implementation for independent qualification checks; and
- HMAC-derived world bands and commitments that do not expose held-out IDs,
  seeds, or task-kernel parameters.

`src/rlx_agents/` provides compact neural controls rather than research
conclusions:

- residual MLP, residual GRU, and bounded causal-attention branching
  actor-critics;
- CPU, CUDA, and MPS execution with a deterministic explicit/auto device rule;
- model manifests and hard parameter caps; and
- candidate protocol v2 with batched rollouts, fresh held-out processes, and
  SHA-256-verified binary checkpoints.

The active frozen protocol is
`campaigns/factorlab_long_v1/qualification_protocol.json`. Its anchor has a
5,000-step terminal-only vector return and `10^12` joint actions; qualification
also requires a measured 20,000-step scaling contrast. Until all ten gates pass
and a reviewed report is committed, preflight rejects automated research.

## Autonomous research loop

`src/rlx_lab/` implements:

- an append-only typed research graph and transactional SQLite job queue;
- deterministic exploit/falsify/explore/transfer/synthesis portfolio scheduling;
- typed hypotheses with quantitative predictions and falsifiers;
- isolated Git worktrees for every primary and independent implementation;
- Codex and Claude provider adapters with cross-provider replication;
- evaluator-enforced neural model and experiment budgets;
- sandboxed candidate execution with protected benchmark/evaluator paths;
- durable runs, findings, contradictions, replications, and incidents; and
- restart-safe controller/worker/daemon operation with bounded retries.

Model prose is never scientific evidence. Promotion requires immutable runs,
an ablation or matched counterfactual, independent implementation/replication,
multiple seeds and worlds, and a sourced prior-art status.

## Qualification

Create an owner-only 32-byte suite key under ignored runtime storage, then run
the frozen protocol:

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_agents.cli \
  --protocol campaigns/factorlab_long_v1/qualification_protocol.json \
  --key-file runtime/secrets/factorlab-long-5k-v1.key \
  --output-dir runtime/qualification/factorlab-long-5k-v1
```

This repository-finalization task does not run that command. A separate
qualification session follows [the execution handoff](design/50_execution_handoff.md).
The command exits successfully only if all ten scientific gates pass. Raw
held-out evidence remains under `runtime/`; only reviewed aggregate evidence
and its content digest may be promoted into the versioned campaign definition.

## Campaign operation (separate session, after admission)

The canonical branch must be clean and committed because implementation
worktrees are derived from `HEAD`.

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime init \
  --name first-neural-campaign \
  --question "Which neural credit mechanisms remain stable as causal lag grows?"

PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime doctor --campaign CAMPAIGN_ID --live-providers

PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime serve --campaign CAMPAIGN_ID --workers 3
```

`doctor` fails closed on an uncommitted snapshot, missing evaluator key,
unqualified tier, invalid schemas, provider incompatibility, or sandbox failure.
Starting continuous provider spend remains an explicit operator action.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q
PYTHONPATH=src:. .venv/bin/python -m ruff check \
  src/rlx_bench src/rlx_agents src/rlx_lab tests/bench tests/agents tests/lab
uv lock --check
```

Runtime databases, secrets, provider transcripts, raw held-out results, binary
checkpoints, and generated worktrees are ignored and must never be committed.
