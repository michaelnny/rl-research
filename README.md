# rl-research

Lean substrate for fast RL algorithm probes across three axes:

- long-horizon sparse reward: `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-KeyCorridorS3R3-v0`
- native vector reward: `deep-sea-treasure-concave-v0`, `resource-gathering-v0`
- open-ended symbolic control: `Craftax-Symbolic-v1`

The quality bar for proposed algorithms lives in `worklogs/exemplars.md`
(Q-learning, PPO, AlphaZero, mirror descent, SAC, MCTS, GAE - calibration,
not a menu). The negative space lives in `prior_attempts.md` as
family-level dead-mechanism descriptions, with sealed per-attempt detail
preserved in `worklogs/attempts/`.

The autonomous loop is schema-backed and probe-first: Researcher writes a
`candidate.json`, Reviewer blocks rebadges and incoherent updates, then
Engineer runs coherent novel probes plus ablations on the fixed panel
before theorem-level work is required. See `PROBLEM.md` and `CLAUDE.md`
for the rationale.

## Hot Path

```bash
uv sync
uv run pytest
uv run python scripts/build_baselines.py
uv run python scripts/validate_candidate.py worklogs/runs/<run_id>/candidate.json
uv run python scripts/run_probe_ladder.py worklogs/runs/<run_id>
uv run run_panel.py --stage sparse --time-budget-s 120
uv run run_panel.py --train-path worklogs/runs/<run_id>/train.py --stage vector --time-budget-s 120
uv run run_panel.py --stage vector --time-budget-s 120
uv run run_panel.py --stage all --time-budget-s 120
```

Manual single-file attempts can still edit repo-root `train.py`.
Automated probes should write `worklogs/runs/<run_id>/train.py` and
`train_ablate.py`, then run them with `run_panel.py --train-path`. Each
entry point must expose:

```python
train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn
```

For vector envs, training must consume `info["vector"]`; using scalar
reward as the training signal is a scalarization rebadge.

## Files

```text
harness.py                  env factory, eval, hypervolume, baselines
train.py                    agent-editable algorithm entry point
run_panel.py                parallel stage runner
scripts/build_baselines.py  random-floor baseline builder
scripts/validate_candidate.py schema check for candidate.json
scripts/run_probe_ladder.py smoke/claim/ablation/confirmation runner
baselines.json              compact baseline scores for the active envs
worklogs/exemplars.md       quality bar (calibration set, not menu)
prior_attempts.md           dead-mechanism families (negative space)
worklogs/runs/              probe, review, panel, result, curator trail
worklogs/attempts/          sealed per-attempt detail; do not delete
worklogs/_archive/          archived prior-loop artifacts
```

## Stages

| Stage | Envs |
| --- | --- |
| `quick` | DST-concave |
| `sparse` | DoorKey, KeyCorridor |
| `vector` | DST-concave, Resource-Gathering |
| `craft` | Craftax-Symbolic |
| `core` | sparse + vector |
| `all` | all five active envs |

Keep/kill rule: a candidate must improve its claimed axis and explain why
the lift is not a known disqualifier from `prior_attempts.md`. Do not
promote from a single lucky env or from a candidate that fails its own
ablation; confirm with multiple random seeds before spending larger
compute.
