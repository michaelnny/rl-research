"""Unit tests for ``rl_research.tb``.

Verifies the contract scalar names and the EvalCadence schedule. Reads back
TB event files to assert the actual tags written, since those names are the
contract surface that other roles depend on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_research.tb import EvalCadence, RunLogger


def _tb_tags(logdir: Path) -> set[str]:
    """Read all tags written to a TB event file under ``logdir``."""
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = list(logdir.glob("events.out.tfevents.*"))
    assert files, f"no event file written under {logdir}"
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    return set(ea.Tags()["scalars"])


# -- RunLogger contract surface --------------------------------------------


def test_run_logger_emits_required_scalar_names(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log_eval(0, mean=10.0, std=1.0)
    logger.log_train(0, loss=0.5)
    logger.log_progress(0, env_steps=0, wallclock_s=0.0, param_checksum="00ff00ff")
    logger.close()
    tags = _tb_tags(tmp_path)
    assert {
        "eval/return_mean",
        "eval/return_std",
        "train/loss",
        "progress/env_steps",
        "progress/wallclock_s",
        "progress/param_checksum",
    }.issubset(tags)


def test_run_logger_per_channel_morl(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log_eval(0, mean=0.0, std=0.0, per_channel=np.array([1.0, 2.0, 3.0]))
    logger.close()
    tags = _tb_tags(tmp_path)
    assert {
        "eval/return_per_channel/0",
        "eval/return_per_channel/1",
        "eval/return_per_channel/2",
    }.issubset(tags)


def test_run_logger_no_per_channel_for_scalar_envs(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log_eval(0, mean=0.0, std=0.0, per_channel=None)
    logger.close()
    tags = _tb_tags(tmp_path)
    assert not any(t.startswith("eval/return_per_channel") for t in tags)


def test_run_logger_creates_logdir(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    logger = RunLogger(nested)
    assert nested.exists()
    logger.close()


def test_run_logger_param_checksum_is_optional(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log_progress(0, env_steps=10, wallclock_s=1.0)
    logger.close()
    tags = _tb_tags(tmp_path)
    assert "progress/env_steps" in tags
    assert "progress/param_checksum" not in tags


def test_run_logger_scalar_escape_hatch(tmp_path):
    logger = RunLogger(tmp_path)
    logger.scalar("custom/my_metric", 1.23, 0)
    logger.close()
    tags = _tb_tags(tmp_path)
    assert "custom/my_metric" in tags


# -- EvalCadence ------------------------------------------------------------


def test_eval_cadence_fires_at_least_n_evals():
    total = 100_000
    n_evals = 20
    cadence = EvalCadence(total_env_steps=total, n_evals=n_evals)
    fired = sum(1 for s in range(0, total + 1, 100) if cadence.maybe_eval(s))
    assert fired >= n_evals


def test_eval_cadence_interval():
    cadence = EvalCadence(total_env_steps=1000, n_evals=10)
    assert cadence.interval == 100


def test_eval_cadence_force():
    cadence = EvalCadence(total_env_steps=1000, n_evals=10)
    cadence.force(0)
    assert cadence.last_eval_step == 0
    # The next eval shouldn't fire until interval (=100) has passed.
    assert not cadence.maybe_eval(50)
    assert cadence.maybe_eval(100)


def test_eval_cadence_does_not_double_fire():
    cadence = EvalCadence(total_env_steps=1000, n_evals=10)
    assert cadence.maybe_eval(100)
    assert not cadence.maybe_eval(101)
    assert not cadence.maybe_eval(199)
    assert cadence.maybe_eval(200)


def test_eval_cadence_handles_short_runs():
    # If n_evals > total_env_steps, interval clamps to >=1.
    cadence = EvalCadence(total_env_steps=5, n_evals=20)
    assert cadence.interval >= 1
