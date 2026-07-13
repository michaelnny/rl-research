# RLX repository rules

Read the documents under `design/` before making architectural changes.
They are the current specification.

## Boundaries

- New work belongs in `rlx_*` namespaces. The deleted `rlh_bench`, old
  experiment runner, and shell lab architecture may not be reintroduced or
  copied from Git history without a written architecture decision.
- Candidate implementations may not alter `src/rlx_bench/`, evaluator code,
  task manifests, reference results, or held-out world definitions.
- Executable candidates must follow `design/40_candidate_protocol.md` and live
  only in the branch-specific path granted by their job.
- A benchmark is not qualified by tests alone. Qualification requires the study
  in `design/10_benchmark_architecture.md`.
- A model response, journal entry, or review score is not scientific evidence.
  Link claims to immutable runs, analyses, ablations, and replications.
- Do not expose held-out world identifiers or evaluator-only metadata to model
  workers or candidate training code.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q
PYTHONPATH=src:. .venv/bin/python -m ruff check \
  src/rlx_bench src/rlx_agents src/rlx_lab tests/bench tests/agents tests/lab
uv lock --check
```

Use isolated worktrees for code-writing agent jobs. Never commit runtime state,
provider transcripts, secrets, generated worktrees, or unreviewed held-out
results.
