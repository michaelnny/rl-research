# Fix 1: harness import bootstrap

The first `uv run python scripts/run_probe_ladder.py` produced
`final_score:` nan in 1.0s wallclock. Inspection of
`runs/last/deep-sea-treasure-concave-v0.log` shows:

```
ModuleNotFoundError: No module named 'harness'
```

Cause: when `run_panel.py` invokes the candidate via
`subprocess.run([sys.executable, str(train_path), ...], cwd=ROOT)`,
Python prepends the *script's* directory to `sys.path`, which is
`worklogs/runs/20260606-15-auto/`, not the project root. The repo-root
`harness.py` is therefore not importable.

Mechanical fix: insert the repo root into `sys.path` at the top of both
`train.py` and `train_ablate.py` before `import harness`. This is an
import-path mechanical typo class fix (allowed retry class), not an
algorithmic change.
