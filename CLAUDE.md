# CLAUDE.md

Substrate rules and hot-path commands for `rl-research`. The research
mission and the candidate-shape requirements live in the subagent prompts
(`.claude/agents/researcher.md`, `.claude/agents/reviewer.md`); this file
is the operational spec that surrounds them.

The full original framing is in `RESEARCH_IDEA.md`. The substrate panel
we run on (5 envs at 120 s/env each) is a deliberately speed-optimized
*subset* of the benchmark ladder in `RESEARCH_IDEA.md` — diagnostic gates
for the autonomous loop, not the target settings. Read `RESEARCH_IDEA.md`
when you need the long-form goal/context; the subagent prompts already
contain the operative version.

## Read first

1. `README.md` — hot-path commands, active envs.
2. `prior_attempts.md` — failed-direction index and disqualifier-family
   list. The Researcher and Reviewer read this every iteration.
3. `worklogs/candidates/*.md` — alive-but-not-yet-conclusive candidates.

`worklogs/attempts/<NN>-<slug>.md` per-attempt detail files are archival
evidence written by the Curator. Open them only when the compact
`prior_attempts.md` index is genuinely insufficient for the question at
hand; the index is written to be self-sufficient.

## Substrate boundary

The repo is intentionally small. Do not add orchestration frameworks, new
docs, or benchmark tiers unless the user explicitly asks.

- Edit `train.py` (or `worklogs/runs/<run_id>/train.py`) for candidate
  algorithms. Only the Engineer subagent touches code.
- Keep `harness.py`, `run_panel.py`, and `baselines.json` fixed during an
  algorithm attempt. Tampering invalidates the panel score.
- For vector envs, training MUST consume `info["vector"]`. Optimizing the
  scalar `reward` on a vector env is a scalarization rebadge by
  definition — the vector envs are designed to detect this.
- Run the smallest relevant stage first:

```bash
uv run run_panel.py --stage sparse --time-budget-s 120
uv run run_panel.py --stage vector --time-budget-s 120
uv run run_panel.py --stage core   --time-budget-s 120  # sparse + vector
uv run run_panel.py --stage all    --time-budget-s 120  # adds Craftax
```

## Subagent roles (canonical scope)

- **Researcher** — proposes ideas. Reads `prior_attempts.md` +
  `worklogs/candidates/*.md` and the mission spine in its own prompt.
  Writes `worklogs/runs/<run_id>/hypothesis.md`. Never reads or writes
  code.
- **Reviewer** — cheap structural-novelty gate. Reads
  `worklogs/runs/<run_id>/hypothesis.md` + `prior_attempts.md`. Writes
  `worklogs/runs/<run_id>/review.md` with one of three verdicts.
- **Engineer** — authors `worklogs/runs/<run_id>/train.py` from the
  hypothesis, runs it through the panel, captures
  `worklogs/runs/<run_id>/result.json`. Restores the repo-root
  `train.py` before exiting.
- **Curator** — synthesizes hypothesis + review + result into a per-run
  verdict. Writes `worklogs/runs/<run_id>/curator.md`, appends to
  `worklogs/ledger.jsonl`, and updates the corpus
  (`prior_attempts.md` for `failed-structural`,
  `worklogs/candidates/<slug>.md` for `alive-*` /
  `failed-implementation`).

The orchestration is `.claude/commands/research.md`.
