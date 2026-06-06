# CLAUDE.md

Substrate rules and hot-path commands for `rl-research`. The research
mission and quality bar live in `worklogs/exemplars.md`; the operational
agent loop is in `.claude/commands/research.md` and
`.claude/agents/*.md`.

The project now uses a schema-backed probe-first loop. Novelty,
coherence, and ablation quality are screened before compute, but
theorem-level review no longer blocks the first empirical panel run. This fixes the failure mode documented in
`PROBLEM.md`: the previous loop made empirical evidence downstream of a
bar that the historical exemplars generally did not clear before being
tried.

## Mission, in one sentence

Find a novel RL algorithm of the same class as Q-learning, PPO,
AlphaZero, mirror descent, SAC, MCTS, and GAE, using the fixed panel as
an empirical filter without accepting baseline modifications as novelty.

## Read first

1. `worklogs/exemplars.md` - calibration set, not a menu.
2. `prior_attempts.md` - family-level negative space and disqualifiers.
3. `README.md` - active environments and hot-path commands.
4. `PROBLEM.md` - why the loop moved from theorem-gated to probe-first.

`worklogs/attempts/<NN>-<slug>.md` files are sealed archival evidence.
Open one only to resolve a specific structural distinction. The family
list in `prior_attempts.md` should normally be enough.

## Substrate boundary

- Keep `harness.py`, `run_panel.py`, and `baselines.json` fixed during an
  algorithm attempt.
- `run_panel.py --train-path <path>` can run a candidate or ablation file
  directly. The Engineer should leave repo-root `train.py` unchanged.
- For vector envs, training must consume `info["vector"]`. Training only
  on scalar reward in vector envs is scalarization and is disallowed.
- No baseline RL libraries are allowed. Neural nets, optimizers, replay
  buffers, and environment wrappers are allowed as components.

Run the smallest relevant stage first:

```bash
uv run run_panel.py --stage quick  --time-budget-s 120
uv run run_panel.py --stage sparse --time-budget-s 120
uv run run_panel.py --stage vector --time-budget-s 120
uv run run_panel.py --stage core   --time-budget-s 120
uv run run_panel.py --stage all    --time-budget-s 120
```

## Subagent roles

- **Researcher** - writes one runnable `[probe]`, one
  `[negative-closure]`, or an empty-hand note. A probe has a principle,
  typed primitive, derivation sketch, update rule, empirical claim,
  ablation plan, novelty boundary, proof debt, and a machine-readable
  `candidate.json`. The Researcher reads corpus summaries but never reads
  or writes code.
- **Reviewer** - triages schema validity, novelty, coherence,
  implementability, and ablation quality. The verdict set is
  `probe | revise | reject | negative-closure`. The Reviewer rejects
  rebadges and dead families, but does not require a convergence theorem
  before compute.
- **Engineer** - runs on Reviewer verdict `probe`. Authors
  `worklogs/runs/<run_id>/train.py` and `train_ablate.py`, runs
  `scripts/run_probe_ladder.py` for the smoke -> claim -> ablation ->
  conditional confirmation ladder, writes
  `panel-*.txt` and `result.json`, and leaves repo-root `train.py`
  unchanged.
- **Curator** - converts hypothesis + review + result into corpus signal,
  appends `worklogs/ledger.jsonl`, updates negative space only when a
  family-level lesson is warranted, and halts on `proven-on-substrate`.

## Loop economics

A healthy unattended run should produce frequent panel executions.
Reviewer rejections still happen, but a long streak of `stage: null`
entries in `mode=probe-v1` is a design failure because empirical signal
has again been pushed out of the loop. `/research` halts after eight
probe-v1 iterations without a panel run.

Expected output mix after the redesign:

- many `null-result` runs that faithfully tested coherent probes;
- many `ablation-failure` runs that show the claimed primitive was not
  load-bearing;
- some `empirical-signal` runs worth sharpening or retesting;
- occasional `reviewer-rejected` or `negative-closure` entries that keep
  the novelty boundary clean;
- rare `proven-on-substrate` promotions that halt for user review.

## Promotion halts the loop

When Curator records `proven-on-substrate`, it writes
`worklogs/HALT_REQUESTED.md` and `worklogs/promotions/<run_id>.md`. The
loop stops. The next step after a real substrate win is human review, not
another automatic iteration.
