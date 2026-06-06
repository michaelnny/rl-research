# fix-1: import-path mechanical retry

The subprocess launched by `run_panel.py --train-path
worklogs/runs/<run_id>/train.py` runs Python with the script's directory
on `sys.path`, not the repo root, so `import harness` raised
`ModuleNotFoundError`. Class: import/name typo (mechanical).

Fix: in both `train.py` and `train_ablate.py`, prepend the repo root
(parents[3] of the file path) to `sys.path` before importing harness. No
algorithmic change.
