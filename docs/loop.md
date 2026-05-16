# Loop specification

This document specifies how a single iteration of the autonomous research loop
proceeds, the four roles involved, and the state machine that governs transitions.

## Roles

```
+------------+        +-----------+        +------------+        +----------+
| Researcher |  --->  | Reviewer  |  --->  |  Operator  |  --->  | Curator  |
+------------+        +-----------+        +------------+        +----------+
   per-iter            per-iter             per-iter             periodic
```

Role prompts live at `docs/roles/<role>.md` and are loaded into the corresponding
Claude Code agent.

### Researcher

- **Reads:** `docs/charter.md`, `lab/lessons.md`, `lab/ledger.jsonl` (last 50 lines),
  `lab/threads/*.md`.
- **Writes (Phase 1):** `lab/runs/<run_id>/hypothesis.md` following the template in
  `docs/contract.md`. Then halts.
- **Writes (Phase 2, only after Reviewer approval):** `lab/runs/<run_id>/train.py`.
- **Constraint:** prompt is broad. No numeric targets. Goal is *new families*, not
  improvements. See `docs/charter.md`.

### Reviewer

- **Reads:** `lab/runs/<run_id>/hypothesis.md`, `docs/charter.md` (Disqualifiers).
- **Writes:** `lab/runs/<run_id>/review.md` with a verdict in
  `{novel-direction, known-rebadge, needs-sharpening}` plus 1-2 paragraphs of
  justification.
- **Cost:** ~30 seconds, text only.
- **Gate:**
  - `novel-direction` → Researcher proceeds to Phase 2.
  - `known-rebadge` → Researcher revises (max 2 cycles), then if still rebadge,
    abandon iteration and start a new one in a different thread.
  - `needs-sharpening` → Researcher revises (max 1 cycle).

### Operator

The Operator runs in two stages. Stage B runs only if Stage A passes.

#### Stage A — sanity gate

- Run `train.py --env <env> --seed <s> --total-env-steps <small> --max-wallclock-s 300 --logdir <path>`
  for each `(env, seed)` in the cross-product of `sanity_envs` (default
  `[CartPole-v1, Pendulum-v1]`) × first seed only.
- **Pass criteria** (per env):
  - Process exits 0.
  - No NaN in any logged scalar.
  - Parameter delta from start to end is non-zero (model actually trained).
  - `eval/return_mean` at end is *strictly above* the random-policy baseline for
    that env.
- **Retry budget:** up to 3 debug-and-rerun attempts per env. Each retry produces
  `lab/runs/<run_id>/fix-N.md` describing what was changed and why.
- If sanity fails after retries: write `result.json` with `stage="A-only"`,
  `status="sanity-failed"`, append to ledger, stop.

The sanity gate is a **bug filter, not a performance filter**. Do not select for
"high return on CartPole." Select for "the implementation works."

#### Stage B — primary benchmark

- Runs only if Stage A passed.
- Run `train.py --env <primary_benchmark> --seed <s> --total-env-steps <budget>
  --max-wallclock-s 7200 --logdir <path>` for each seed in `seeds`.
- **No retries on Stage B failures.** Log the failure as evidence and move on.
- On budget exceeded: SIGTERM, then SIGKILL after 30s grace, write `result.json`
  with `status="killed-budget"`. Partial logs are kept.
- On exception: write `result.json` with `status="killed-error"` and the
  traceback in `notes`.

### Curator

- **Cadence:** every 10 completed runs OR on user trigger.
- **Reads:** `lab/ledger.jsonl`, all run dirs since last curation, current
  `lab/lessons.md` and `lab/threads/*.md`.
- **Writes:**
  - `lab/lessons.md` — distilled findings, *curated* (not append-only). Replace
    superseded entries; do not let it grow unbounded.
  - `lab/threads/<thread>.md` — thread state in `{active, paused, archived}` plus
    a 1-paragraph status summary.
  - `verdict_curator` field in matching ledger entries — one of
    `{promising, dead-end, inconclusive}`.
  - Promotion decisions: a candidate may be promoted to a "mass run" with
    extended budget on all three primary benchmarks. Recorded as a new run with
    `parent_run_id` set in `notes`.
- **Constraint:** curatorial multi-criteria judgment. Never a fixed numerical gate.
  Weigh: structural novelty, evidence quality, generality across pillars,
  implementation reproducibility, failure-mode informativeness.

## State machine

```
[start]
   |
   v
proposed ----[Reviewer: needs-sharpening]----> revising ---+
   |                                                       |
   |---[Reviewer: novel-direction]----> approved           |
   |                                       |               |
   |---[Reviewer: known-rebadge × 3]-> abandoned (logged)  |
   |                                                       |
   +<------------------------------------------------------+
                                           |
                                           v
                                     code-ready
                                           |
                                  [Operator: Stage A]
                                           |
                          +----------------+----------------+
                          |                                 |
                         pass                              fail (after retries)
                          |                                 |
                  [Operator: Stage B]                  sanity-failed (terminal)
                          |
              +-----------+-----------+--------------+
              |           |           |              |
         completed   killed-budget  killed-error   benchmark-failed
              |           |           |              |
              +----[ ledger append ]--+--------------+
                              |
                              v
                          [iter end]
```

Every terminal state writes a `result.json` and a ledger line. **There are no silent
failures.**

## Retry budgets and rate limits

- Sanity gate retries: 3 per env per run.
- Reviewer revision cycles: 2 (rebadge) or 1 (sharpening).
- Sequential sanity-fail in the same thread: 3 → Curator must inspect.
- Iteration-level wallclock soft cap: 4h (covers Stage A + 3 retries + Stage B 2h).
- Daily total: configurable, default 16h (≈ 4 iterations / day).

## Failure handling principles

- Bug evidence is evidence. A `killed-error` outcome with a clean traceback is more
  valuable than a vague "didn't learn."
- A thread with 3 sequential sanity failures is a signal of *implementation
  hardness* of that family, not just bad luck — Curator must inspect before more
  proposals in the same thread.
- The Operator must never silently retry Stage B. Stage B failures are evidence,
  not noise.

## Concurrency

The MVP is sequential: one iteration at a time. Parallelism is reserved for:

- Per-seed evaluation in Stage B (run multiple seeds in parallel within the
  single 2h cap, GPU memory permitting).
- Curator-driven mass-run promotion (multiple benchmarks for the same `train.py`).

True multi-Researcher parallelism is deferred until we have ≥1 working candidate
to compare against.
