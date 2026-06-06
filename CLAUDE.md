# CLAUDE.md

Substrate rules and hot-path commands for `rl-research`. The research
mission and quality bar live in `worklogs/exemplars.md` and the subagent
prompts (`.claude/agents/{researcher,reviewer,engineer,curator}.md`);
this file is the operational spec that surrounds them.

The original framing is in `RESEARCH_IDEA.md`.

## Mission, in one sentence

Find the next AlphaZero-class RL algorithm. The bar is calibrated against
`worklogs/exemplars.md`. Most loop iterations correctly produce no
proposal; that is the design.

## Read first

1. `worklogs/exemplars.md` — what the bar looks like (Q-learning, PPO,
   AlphaZero, mirror descent, SAC, MCTS, GAE).
2. `prior_attempts.md` — dead mechanism *families* (A–G) and the
   standard disqualifier list. Family-level only; the appendix maps
   individual sealed attempts to families if the Reviewer needs to
   disambiguate a borderline rebadge claim.
3. `README.md` — hot-path commands and active envs.

`worklogs/attempts/<NN>-<slug>.md` per-attempt detail files are sealed
archival evidence. Open one only when a specific structural-distinction
question requires the math from a prior attempt; the family list is
designed to be self-sufficient otherwise.

`worklogs/_archive/candidates/` holds the parking-lot files from the
prior loop design. Preserved for traceability; not active corpus.

## Substrate boundary

The repo is intentionally small. Do not add orchestration frameworks,
new docs, or benchmark tiers unless the user explicitly asks.

- The Engineer subagent edits `train.py` (or
  `worklogs/runs/<run_id>/train.py`) for candidate algorithms. No other
  role touches code.
- Keep `harness.py`, `run_panel.py`, and `baselines.json` fixed during
  an algorithm attempt. Tampering invalidates the panel score.
- For vector envs, training MUST consume `info["vector"]`. Optimizing
  the scalar `reward` on a vector env is a scalarization rebadge by
  definition.
- Run the smallest relevant stage first:

```bash
uv run run_panel.py --stage sparse --time-budget-s 120
uv run run_panel.py --stage vector --time-budget-s 120
uv run run_panel.py --stage core   --time-budget-s 120  # sparse + vector
uv run run_panel.py --stage all    --time-budget-s 120  # adds Craftax
```

## Subagent roles (canonical scope)

- **Researcher** — proposes ideas under the four-slot contract
  (principle, derivation, primitive, theorem) or returns the empty-hand
  note. Reads `worklogs/exemplars.md` + `prior_attempts.md` and may
  use web search for mathematical machinery (not for RL paper
  imitation). Writes `worklogs/runs/<run_id>/hypothesis.md`. Never
  reads or writes code.
- **Reviewer** — adversarial referee. Default verdict is `reject`.
  Verifies the derivation step-by-step and searches the web to confirm
  the proposal is not a renamed published method. Reads
  `worklogs/runs/<run_id>/hypothesis.md` + `prior_attempts.md` (family
  level) + `worklogs/exemplars.md`. Writes
  `worklogs/runs/<run_id>/review.md` with verdict
  `pass | revise | reject`.
- **Engineer** — runs only when Reviewer's verdict is `pass`. Authors
  `worklogs/runs/<run_id>/train.py` from the hypothesis, runs it
  through the panel, captures `worklogs/runs/<run_id>/result.json`.
  Restores the repo-root `train.py` before exiting.
- **Curator** — synthesizes hypothesis + review + result into a per-run
  verdict (`proven-on-substrate`, `structural-failure`,
  `implementation-failure`, `null-result`, `empty-hand`,
  `reviewer-rejected`). Writes `worklogs/runs/<run_id>/curator.md`,
  appends to `worklogs/ledger.jsonl`, and updates the corpus. On
  `proven-on-substrate`, also writes `worklogs/HALT_REQUESTED.md` to
  halt the loop for user review.

The orchestration is `.claude/commands/research.md`.

## Loop economics

The loop expects a high empty-hand and reviewer-rejected rate. A
healthy week of unattended operation:

- 100+ empty-hand notes (Researcher correctly couldn't reach the bar)
- ~10 reviewer-rejected proposals (Reviewer caught flaws)
- ~1 reviewer-passed proposal that got run through the panel
- ideally 1 promotion that halts for user review

Anything denser than that means the bar slipped. Empty-hand turns and
rejections are the *correct* dominant outcome; the loop is calibrated
for low-frequency search of high-quality candidates, not
high-throughput search of heuristics.

## Promotion halts the loop

When the Curator records `proven-on-substrate`, it writes
`worklogs/HALT_REQUESTED.md`. The loop halts. The user reviews
`worklogs/promotions/<run_id>.md` and decides what happens next. The
loop does not auto-continue past a real promotion — the next step
after a real candidate is human attention, not another iteration.
