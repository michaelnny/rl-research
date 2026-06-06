# CLAUDE.md

Substrate rules and hot-path commands for `rl-research`. The research
mission and quality bar live in `worklogs/exemplars.md` and the subagent
prompts (`.claude/agents/{researcher,reviewer,engineer,curator}.md`);
this file is the operational spec that surrounds them.

The original framing is in `RESEARCH_IDEA.md`.

## Mission, in one sentence

Find the next AlphaZero-class RL algorithm. The bar is calibrated against
`worklogs/exemplars.md`. Most loop iterations produce a seed, a rejected
proposal, or an empty-hand note rather than a Reviewer-passed full
proposal — that is the design. A 15-iteration empty-hand streak halts
the loop as a design problem; see `.claude/commands/research.md`.

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

- **Researcher** — produces one of: a full proposal under the
  four-slot contract (principle, derivation, primitive, theorem),
  a **seed** (slots 1–3 filled, slot 4 replaced by an explicit
  open question — carried forward in the corpus for future
  closure), or an empty-hand note. Reads `worklogs/exemplars.md`,
  `prior_attempts.md`, and recent hypothesis headers + open seeds;
  may use web search for mathematical machinery (not for RL paper
  imitation). Writes `worklogs/runs/<run_id>/hypothesis.md`. Never
  reads or writes code.
- **Reviewer** — adversarial referee. Default verdict is `reject`.
  Verifies the derivation step-by-step and searches the web to
  confirm the proposal is not a renamed published method. Reads
  `worklogs/runs/<run_id>/hypothesis.md` + `prior_attempts.md`
  (family level) + `worklogs/exemplars.md`. Writes
  `worklogs/runs/<run_id>/review.md` with verdict
  `pass | pass-as-seed | revise | reject` (`pass-as-seed` applies
  only to seeds; the seed verdict still requires slots 1–3 at
  exemplar quality and a checkable open question).
- **Engineer** — runs only when Reviewer's verdict is `pass` on a
  full proposal. Authors `worklogs/runs/<run_id>/train.py` from the
  hypothesis, runs it through the panel, captures
  `worklogs/runs/<run_id>/result.json`. Restores the repo-root
  `train.py` before exiting. Does **not** run on seeds.
- **Curator** — synthesizes hypothesis + review + result into a
  per-run verdict (`proven-on-substrate`, `structural-failure`,
  `implementation-failure`, `null-result`, `seeded`, `empty-hand`,
  `reviewer-rejected`). Writes `worklogs/runs/<run_id>/curator.md`,
  appends to `worklogs/ledger.jsonl`, and updates the corpus. On
  `proven-on-substrate`, also writes `worklogs/HALT_REQUESTED.md`
  to halt the loop for user review.

The orchestration is `.claude/commands/research.md`.

## Loop economics

The loop's expected steady-state output mix per Researcher turn is
roughly:

- ~20% full proposals (most rejected by Reviewer)
- ~50% seeds (some closed by future iterations, most retiring stale)
- ~30% empty-hand notes (after honest seed-closure attempts and
  fresh-region attempts both failed to produce slots 1–3)

A healthy month of unattended operation might produce dozens of
seeds (most stale, some closed), reviewer-rejected proposals,
~1–3 reviewer-passed full proposals that get run through the panel,
and ideally one promotion that halts for user review.

Anything that looks denser than this — many full proposals passing
Reviewer per week — likely means the bar slipped. Anything that
looks far sparser — long pure-empty-hand streaks — means the seed
mechanism is failing and the loop converged on the empty-hand basin
(the pre-flight halts after 15 consecutive empty-hands as a design
breaker).

## Promotion halts the loop

When the Curator records `proven-on-substrate`, it writes
`worklogs/HALT_REQUESTED.md`. The loop halts. The user reviews
`worklogs/promotions/<run_id>.md` and decides what happens next. The
loop does not auto-continue past a real promotion — the next step
after a real candidate is human attention, not another iteration.
