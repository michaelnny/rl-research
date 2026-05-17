"""Unit tests for ``rl_research.runtime``.

Covers the algorithm-agnostic primitives that every ``train.py`` relies on:
``WallclockBudget``, ``RunningMeanStd``, ``param_checksum``, ``seed_everything``,
``parse_train_cli``, ``write_config_json``. These are fast, CPU-only, no RL deps.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch
from torch import nn

from rl_research.runtime import (
    RunningMeanStd,
    WallclockBudget,
    param_checksum,
    parse_train_cli,
    seed_everything,
    write_config_json,
)

# -- WallclockBudget --------------------------------------------------------


def test_wallclock_budget_not_expired_immediately():
    b = WallclockBudget(seconds=10.0)
    assert not b.expired()
    assert b.elapsed_s() < 0.1
    assert b.remaining_s() <= 10.0


def test_wallclock_budget_expires_after_zero_budget():
    b = WallclockBudget(seconds=0.0)
    assert b.expired()
    assert b.remaining_s() == 0.0


def test_wallclock_budget_grace():
    b = WallclockBudget(seconds=1.0)
    # Already expired w/ a huge grace.
    assert b.expired(grace_s=2.0)


def test_wallclock_budget_reset():
    b = WallclockBudget(seconds=10.0)
    time.sleep(0.05)
    elapsed_before = b.elapsed_s()
    assert elapsed_before > 0
    b.reset()
    assert b.elapsed_s() < elapsed_before


# -- RunningMeanStd ---------------------------------------------------------


def test_rms_matches_numpy_on_one_pass():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.5, scale=2.0, size=(10_000, 4))
    rms = RunningMeanStd(shape=(4,))
    rms.update(x)
    np.testing.assert_allclose(rms.mean, x.mean(axis=0), atol=1e-2)
    # var is biased (np.var ddof=0); rms.var matches biased estimate up to
    # the eps-init prior bias (count=1e-4 → negligible at N=1e4).
    np.testing.assert_allclose(rms.var, x.var(axis=0), atol=1e-2)


def test_rms_matches_after_chunked_updates():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=-1.0, scale=0.5, size=(2_000, 3))
    rms_one = RunningMeanStd(shape=(3,))
    rms_one.update(x)
    rms_chunked = RunningMeanStd(shape=(3,))
    for chunk in np.array_split(x, 13):
        rms_chunked.update(chunk)
    np.testing.assert_allclose(rms_chunked.mean, rms_one.mean, atol=1e-9)
    np.testing.assert_allclose(rms_chunked.var, rms_one.var, atol=1e-9)


def test_rms_normalize_clip():
    rms = RunningMeanStd(shape=(2,))
    rms.update(np.zeros((10, 2)))
    rms.update(np.ones((10, 2)))
    big = np.array([[1e6, -1e6]], dtype=np.float64)
    out = rms.normalize(big, clip=10.0)
    assert out.dtype == np.float32
    assert (out <= 10.0).all() and (out >= -10.0).all()


def test_rms_normalize_no_clip_returns_raw():
    rms = RunningMeanStd(shape=())
    rms.update(np.ones(100))
    val = np.array([10.0])
    out = rms.normalize(val, clip=None)
    assert out.shape == (1,)
    # No clip: the standardized value can be arbitrary.
    assert np.isfinite(out).all()


# -- param_checksum ---------------------------------------------------------


def test_param_checksum_stable_for_same_weights():
    torch.manual_seed(0)
    net1 = nn.Linear(8, 4)
    state = {k: v.clone() for k, v in net1.state_dict().items()}
    net2 = nn.Linear(8, 4)
    net2.load_state_dict(state)
    assert param_checksum(net1) == param_checksum(net2)


def test_param_checksum_changes_after_step():
    torch.manual_seed(0)
    net = nn.Linear(8, 4)
    before = param_checksum(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x = torch.randn(16, 8)
    y = torch.randn(16, 4)
    opt.zero_grad()
    ((net(x) - y) ** 2).mean().backward()
    opt.step()
    assert param_checksum(net) != before


def test_param_checksum_length_is_16_hex():
    net = nn.Linear(2, 2)
    s = param_checksum(net)
    assert len(s) == 16
    int(s, 16)  # parses as hex


# -- seed_everything --------------------------------------------------------


def test_seed_everything_reproducible_across_libs():
    # seed_everything seeds the legacy np.random global RNG, so that's what we
    # check here. Candidates that prefer np.random.Generator can build it from
    # any int they like; this test verifies the legacy path.
    seed_everything(42)
    a_np = np.random.rand(8)  # noqa: NPY002
    a_torch = torch.rand(8)
    seed_everything(42)
    b_np = np.random.rand(8)  # noqa: NPY002
    b_torch = torch.rand(8)
    np.testing.assert_array_equal(a_np, b_np)
    torch.testing.assert_close(a_torch, b_torch)


# -- parse_train_cli --------------------------------------------------------


def test_parse_train_cli_required_flags():
    args = parse_train_cli(
        [
            "--env",
            "CartPole-v1",
            "--seed",
            "0",
            "--total-env-steps",
            "1000",
            "--max-wallclock-s",
            "60",
            "--logdir",
            "/tmp/x",
        ]
    )
    assert args.env == "CartPole-v1"
    assert args.seed == 0
    assert args.total_env_steps == 1000
    assert args.max_wallclock_s == 60
    assert args.logdir == "/tmp/x"


def test_parse_train_cli_extra_flags():
    args = parse_train_cli(
        [
            "--env",
            "Pendulum-v1",
            "--seed",
            "1",
            "--total-env-steps",
            "100",
            "--max-wallclock-s",
            "10",
            "--logdir",
            "/tmp/x",
            "--my-coef",
            "0.7",
        ],
        extra=[("--my-coef", {"type": float, "default": 0.0})],
    )
    assert args.my_coef == pytest.approx(0.7)


def test_parse_train_cli_missing_required_exits():
    with pytest.raises(SystemExit):
        parse_train_cli(["--env", "CartPole-v1"])


# -- write_config_json ------------------------------------------------------


def test_write_config_json_round_trips(tmp_path):
    args = parse_train_cli(
        [
            "--env",
            "CartPole-v1",
            "--seed",
            "7",
            "--total-env-steps",
            "12345",
            "--max-wallclock-s",
            "60",
            "--logdir",
            str(tmp_path),
        ]
    )
    out = write_config_json(tmp_path, args)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["cli_args"]["env"] == "CartPole-v1"
    assert data["cli_args"]["seed"] == 7
    assert data["cli_args"]["total_env_steps"] == 12345
    assert "git_sha" in data and len(data["git_sha"]) >= 7
    assert "torch_version" in data
    assert data["started_at"].endswith("Z")


def test_write_config_json_accepts_dict(tmp_path):
    out = write_config_json(tmp_path, {"env": "X", "seed": 0})
    data = json.loads(out.read_text())
    assert data["cli_args"]["env"] == "X"
