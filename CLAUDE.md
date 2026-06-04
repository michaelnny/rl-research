# CLAUDE.md

This repo is intentionally small. Do not add orchestration frameworks, new docs,
or benchmark tiers unless the human explicitly asks.

## Read First

1. `README.md` for commands and active envs.
2. `prior_attempts.md` for failed families and disqualifiers.

Detailed attempt files under `worklogs/attempts/` are archival evidence. Keep
them intact. Open them only when the compact prior index is not enough.

## Algorithm Work

- Edit `train.py` for candidate algorithms.
- Keep `harness.py`, `run_panel.py`, and `baselines.json` fixed during an
  algorithm attempt.
- For vector envs, use `info["vector"]` during training.
- Run the smallest relevant stage first:

```bash
uv run run_panel.py --stage sparse --time-budget-s 120
uv run run_panel.py --stage vector --time-budget-s 120
uv run run_panel.py --stage all --time-budget-s 120
```

## Research Gate

Before coding, state:

```text
primitive:
improvement_operator:
side_information:
nearest_prior_or_disqualifier:
falsifier:
```

Kill the idea if it is a renamed value/Bellman/actor-critic/scalarization/
count/novelty/Go-Explore/options/HER/successor-feature family.
