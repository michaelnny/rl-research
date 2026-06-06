# rl-research

Lean substrate for fast RL algorithm probes on two axes:

- long-horizon sparse reward: `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-KeyCorridorS3R3-v0`
- native vector reward: `deep-sea-treasure-concave-v0`, `resource-gathering-v0`, `Craftax-Symbolic-v1`

The quality bar for proposed algorithms lives in `worklogs/exemplars.md`
(Q-learning, PPO, AlphaZero, mirror descent, SAC, MCTS, GAE — calibration,
not a menu). The negative space lives in `prior_attempts.md` as
family-level dead-mechanism descriptions, with sealed per-attempt detail
preserved in `worklogs/attempts/`.

## Hot Path

```bash
uv sync
uv run pytest
uv run python scripts/build_baselines.py
uv run run_panel.py --stage sparse --time-budget-s 120
uv run run_panel.py --stage vector --time-budget-s 120
uv run run_panel.py --stage all --time-budget-s 120
```

Edit only `train.py` for an algorithm attempt. It must expose:

```python
train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn
```

For vector envs, training must consume `info["vector"]`; using scalar reward as
the training signal is a scalarization rebadge.

## Files

```text
harness.py                  env factory, eval, hypervolume, baselines
train.py                    agent-editable algorithm entry point
run_panel.py                parallel stage runner
scripts/build_baselines.py  random-floor baseline builder
baselines.json              compact baseline scores for the four active envs
worklogs/exemplars.md       quality bar (calibration set, not menu)
prior_attempts.md           dead-mechanism families (negative space)
worklogs/attempts/          sealed per-attempt detail; do not delete
worklogs/_archive/          archived prior-loop artifacts (candidates/)
```

## Stages

| Stage | Envs |
| --- | --- |
| `quick` | DST-concave |
| `sparse` | DoorKey, KeyCorridor |
| `vector` | DST-concave, Resource-Gathering |
| `craft` | Craftax-Symbolic |
| `all` | all five active envs |

Keep/kill rule: a candidate must improve its claimed axis and explain why the
lift is not a known disqualifier from `prior_attempts.md`. Do not promote from a
single lucky env; confirm with multiple seeds before spending larger compute.
