"""Shared primitives for agent-authored RL algorithms.

Public API (importable as ``rl_research.<name>``):

  contract       run-artifact contract enforcement (next_run_id, write_result,
                 validate_result_json, append_to_ledger, update_ledger_verdict)
  runtime        WallclockBudget, RunningMeanStd, parse_train_cli,
                 seed_everything, param_checksum, write_config_json
  envs           vector adapters + per-env factories (gym-classic, gym-atari,
                 mo-minecart, dm-control), make_vec dispatcher
  evaluate       deterministic single-env evaluation across all four families
  tb             RunLogger + EvalCadence (contract scalar names + cadence)
  checkpoints    save_checkpoint / load_latest (atomic, with retention)
  baselines.ppo  the single allowed RL algorithm (frozen yardstick)

A candidate's ``train.py`` may import freely from this package. See
``docs/contract.md`` §train.py contract for the allowed-imports list.
"""

from . import checkpoints, contract, envs, evaluate, runtime, tb

__all__ = ["checkpoints", "contract", "envs", "evaluate", "runtime", "tb"]
