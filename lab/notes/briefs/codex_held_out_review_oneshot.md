# One-shot brief — review held-out seed protocol (gate 9)

The proposer added a `--use-held-out` flag to
`experiments/run_baselines.py` to satisfy acceptance gate 9
(seed-generalization). Review the implementation for bugs,
oversights, or honesty issues.

Read this brief as your full identity for this single invocation.

## What was changed

  - `experiments/run_baselines.py`:
    - Imports `rlh_bench.seed_bands.seed_band_for`.
    - `_summarize` now takes an explicit `seeds: list[int]`
      instead of `(episodes, seed)`; runs `rollout(env, policy, seed=s)`
      for each `s`.
    - `main()` accepts `--use-held-out`. When set, evaluates each
      baseline twice: once on `bands.train[:episodes]`, once on
      `bands.held_out[:episodes]`. Reports the gap.
    - `_render_markdown` adds train-vs-held-out columns when
      held-out data is present.

  - `tests/test_seed_bands.py` (new):
    - All registered envs have a `SeedBands` lookup.
    - Train and held-out ranges are disjoint.
    - Train and held-out seeds produce different worlds (compat
      matrix differs).
    - Smaller tiers don't have *more* train seeds than larger
      tiers.

  - `src/rlh_bench/seed_bands.py` was already in place (not
    changed in this pass).

## What you should check

1. **Disjointness gate**: are the train / validation / held-out /
   debug seed ranges in `DEFAULT_*_BANDS` actually disjoint?
   The test asserts train vs held-out disjointness. What about
   debug vs train (debug=range(0,10) is a subset of train=range(0,100)
   in DEFAULT_SMALL_BANDS — is that intentional)?

2. **World-difference gate**: the test only checks that train[0]
   gives a different world than held_out[0]. Does it actually check
   that *any two seeds anywhere in the bands* give different worlds?
   Could a generator collision make two world configurations identical
   despite different seeds?

3. **`--use-held-out` honesty**: when the flag is set, the policy
   is built once per env via `policy_factory(env)`. For stateful
   policies that *learn* from train seeds (e.g. a future
   `CemPolicy`), the policy must be trained on train seeds and
   then evaluated on held-out. Is the current code structure
   compatible with that, or does it silently re-train on held-out?

4. **Report integrity**: the markdown report uses the held-out
   format when `records["held_out"]` is non-empty. Are there
   edge cases (e.g. oracle diagnostics not in held-out section)?

5. **Use of `seed_band_for`**: `seed_band_for(env_id)` decides
   bands by string substring (`"Small" in env_id`). Brittle to
   renames. Is this a real concern at the current scale?

6. **Any silent off-by-one or shape bug** in the new `_summarize`
   or `main` rewrite.

## Deliverable

Write your review to
`lab/notes/codex_held_out_review_2026-06-30.md`. For each concern,
verdict: real / acceptable / false alarm. Apply minimal fixes only
where clearly right.

## Rules

- You may modify `experiments/run_baselines.py`,
  `src/rlh_bench/seed_bands.py`, and `tests/test_seed_bands.py`.
- Do NOT modify env source unless you find an actual bug.
- Tests must still pass after any changes:
  `PYTHONPATH=src .venv/bin/python -m pytest -q`
- Do not commit; the proposer will commit after review.

## How a session ends

When the review file is written.
