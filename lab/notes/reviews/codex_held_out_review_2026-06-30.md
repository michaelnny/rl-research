# Held-out seed protocol review (gate 9)

Reviewed `experiments/run_baselines.py`, `src/rlh_bench/seed_bands.py`, and
`tests/test_seed_bands.py` for the new `--use-held-out` path.

## Verdicts by concern

1. **Disjointness gate — real.**
   `train`, `validation`, and `held_out` were disjoint, but `debug` overlapped
   `train` in all default bands. Since debug worlds are public, overlap is not a
   held-out leak, but it makes the published bands ambiguous. Fixed by moving
   debug bands to separate high seed ranges and strengthening the test to check
   all pairwise band intersections.

2. **World-difference gate — real test weakness, not a found env bug.**
   The previous test only compared `train[0]` vs `held_out[0]` for scheduling.
   It did not prove every seed in every band maps to a unique world. Exhaustive
   collision proof is not practical in unit tests, especially for large tiers,
   but the generators use high-entropy continuous state so exact full-world
   collisions are unlikely. Strengthened the test to fingerprint representative
   train/validation/held-out/debug seeds for every registered env. Do not
   overclaim this as a mathematical uniqueness proof.

3. **`--use-held-out` honesty for trainable policies — real future hazard.**
   Current registered baseline factories are no-training heuristics, so the
   current report is not corrupted. However, `run_baselines.py` is still not a
   generic train/evaluate harness: `_summarize` rebuilds `policy_factory(env)`
   inside each rollout and calls it again for held-out. A future CEM-like policy
   placed in this portfolio could train again on held-out or otherwise bypass a
   train-on-train-seeds/evaluate-on-held-out protocol. Added an explicit
   docstring warning; trainable algorithms should use/refactor
   `experiments/algorithms/runner.py` instead.

4. **Report integrity — acceptable.**
   The markdown switches to train-vs-held-out columns per env when held-out rows
   exist. Oracle diagnostics remain in their separate non-comparable section and
   are evaluated on the train seeds only; that is acceptable because they are not
   learner-facing baselines. If oracle held-out feasibility is desired later, it
   should be a separate diagnostic table, not mixed into the baseline gap.

5. **`seed_band_for` substring matching — acceptable but brittle.**
   At the current registry scale this was unlikely to bite, but substring tier
   detection could misclassify future family names containing `Small`/`Large`.
   Hardened it to use the documented suffixes (`-Small-v0`, `-Large-v0`).

6. **Off-by-one / shape issues — real.**
   `--episodes 0` could crash via empty seed lists, and `--use-held-out` with
   `--episodes` larger than a band could silently compare different train and
   held-out episode counts. Fixed by rejecting non-positive episode counts,
   rejecting held-out runs whose requested episode count exceeds either band,
   and making `_summarize` reject empty seed lists.

## Files changed

- `src/rlh_bench/seed_bands.py`
  - Moved debug bands out of train ranges.
  - Changed tier selection from substring checks to exact suffix checks.
- `tests/test_seed_bands.py`
  - Now checks all four bands are pairwise disjoint.
  - Added representative per-env world-fingerprint smoke test.
- `experiments/run_baselines.py`
  - Added empty/invalid episode guards.
  - Prevents train/held-out episode-count mismatch when using seed bands.
  - Documents that baseline factories must be no-training per-rollout policies.

## Test result

`PYTHONPATH=src .venv/bin/python -m pytest -q` passed: 66 tests.
