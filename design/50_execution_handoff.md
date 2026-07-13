# Execution handoff

Status: design and implementation handoff; no live research authorized here

Repository construction, benchmark qualification, and autonomous algorithm
research are three different activities. This document is the boundary between
them. The repository-building session stops after verification. A separate
qualification session collects benchmark evidence. A later research session
may start provider workers only after the benchmark is admitted.

## State machine

1. **Implemented, not admitted.** `campaigns/factorlab_long_v1/definition.json`
   has `status=qualification_pending` and an empty `admitted_tiers` list. This
   is the current repository state.
2. **Qualified, awaiting review.** A complete execution of the frozen protocol
   writes raw evidence and a report under ignored `runtime/` storage. An exit
   code of zero is necessary but does not itself change repository state.
3. **Reviewed and admitted.** A reviewer checks the raw study, commits only the
   aggregate report, and adds the exact admitted scope and immutable digests to
   the campaign definition.
4. **Research enabled.** `doctor` verifies the committed admission, evaluator
   key, provider compatibility, process sandbox, and clean revision. Only then
   may an operator start `serve`.

The `rlx-lab` `controller`, `worker`, and `serve` entry points all refuse
execution while the tier is not admitted. Unit tests and development probes are
not admission evidence.

## Qualification-session contract

Use one clean checkout at the exact handoff commit. Do not edit the protocol,
benchmark, reference learner, thresholds, or seed list after inspecting any
held-out result.

First verify the checkout:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q
PYTHONPATH=src:. .venv/bin/python -m ruff check \
  src/rlx_bench src/rlx_agents src/rlx_lab tests/bench tests/agents tests/lab
uv lock --check
git status --short
```

Create or reuse one owner-only 32-byte suite key. The key and all raw evidence
remain under ignored `runtime/` storage:

```bash
install -d -m 700 runtime/secrets
(umask 077 && openssl rand 32 > runtime/secrets/factorlab-long-5k-v1.key)
```

Run exactly the frozen study:

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_agents.cli \
  --protocol campaigns/factorlab_long_v1/qualification_protocol.json \
  --key-file runtime/secrets/factorlab-long-5k-v1.key \
  --output-dir runtime/qualification/factorlab-long-5k-v1
```

Device selection is intentionally one rule: `auto` chooses CUDA, then MPS,
then CPU. `--device cpu|cuda|mps` is only an explicit replication override.

The session must preserve the process exit code, stderr progress stream,
`evidence.json`, and `qualification-report.json`. Interruption, any failed seed,
a nonzero exit, `qualified=false`, a missing gate, or a digest mismatch leaves
the tier unadmitted. Do not combine partial attempts.

## Admission-review contract

Admission is a repository change made only after reviewing the completed raw
study. The reviewer must confirm:

- all ten required checks are `verified` and cite the same evidence and
  protocol digests;
- the task is the 5,000-step, terminal-only, two-objective, 12-by-10 factored
  anchor and the scaling record reaches 20,000 steps;
- all five preregistered training seeds completed inside the per-seed reference
  bound;
- learnability, confidence interval, headroom, generalization, specificity,
  causal audit, throughput, storage, and independent-audit thresholds passed;
- no held-out identifier or evaluator-only metadata was exposed; and
- the report was produced from the final committed implementation and frozen
  protocol.

Copy only the reviewed aggregate report to
`campaigns/factorlab_long_v1/qualification/factorlab-long-5k-v1-report.json`.
Do not commit `evidence.json`, keys, logs, checkpoints, provider transcripts, or
worktrees. Update the definition to `status=qualified`, add
`factorlab-long-5k-v1` to `admitted_tiers`, and add one
`qualification_reports.factorlab-long-5k-v1` entry containing:

- the versioned report and frozen protocol paths;
- the report, canonical protocol, and canonical evidence digests;
- nonempty `reviewed_on` and `reviewed_by` fields; and
- an `admitted_scope` exactly matching the campaign policy, including
  `wall_seconds_total=14400`.

The admission is invalid if any field is broadened beyond the executed study.
After committing the report and definition, rerun the full verification suite.

## Research-session contract

The research operator creates a new campaign only from the clean admitted
revision:

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime init \
  --name first-neural-campaign \
  --question "Which neural credit mechanisms remain stable as causal lag grows?"
```

Then run the production preflight. `--live-providers` performs authenticated
Codex and Claude compatibility calls and therefore requires explicit operator
authorization:

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime doctor \
  --campaign CAMPAIGN_ID --live-providers
```

Only a fully ready report permits the operator to start continuous work:

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_lab.cli \
  --repo . --runtime runtime serve \
  --campaign CAMPAIGN_ID --workers 3
```

`serve` is the production entry point because it preflights, supervises the
controller and workers, recovers expired leases, and terminates all child
process groups on shutdown. Provider and accelerator spend begins only with
this explicit command. A separate session owns monitoring, pause/stop choices,
scientific review, and any later budget increase.

## What this handoff does not claim

It does not claim that the pending benchmark is already qualified, that a live
campaign has been operated for seven days, that a new algorithm has been found,
or that the unimplemented memory/action/objective/applied-family tiers are
available. Those claims require their own immutable executions and reviews.
