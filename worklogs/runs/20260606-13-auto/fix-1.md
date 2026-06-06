# fix-1: import harness from arbitrary cwd

When `run_panel.py` invokes `python worklogs/runs/<run_id>/train.py` as a
subprocess, Python automatically prepends the script's directory
(`worklogs/runs/<run_id>/`), not the repo root, to `sys.path`. So the
default `import harness` line failed with `ModuleNotFoundError`.

Fix: prepend the repo root (computed as the grandparent of the run
directory, i.e. four `parent`s from this file) to `sys.path` before
importing harness. Same change in both `train.py` and `train_ablate.py`.

This is a mechanical import-path retry, not an algorithmic change.
