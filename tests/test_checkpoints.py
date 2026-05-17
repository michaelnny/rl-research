"""Unit tests for ``rl_research.checkpoints``.

Verifies atomic write, latest-symlink update, retention pruning, and
round-trip restore. CPU-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_research.checkpoints import list_checkpoints, load_latest, save_checkpoint


def _state(net: nn.Module, step: int) -> dict:
    return {"model": net.state_dict(), "step": step}


def test_save_creates_file_and_symlink(tmp_path):
    net = nn.Linear(4, 2)
    save_checkpoint(tmp_path, _state(net, 100), step=100)
    cdir = tmp_path / "checkpoints"
    assert (cdir / "step-00000100.pt").exists()
    latest = cdir / "latest.pt"
    assert latest.is_symlink()
    assert latest.resolve().name == "step-00000100.pt"


def test_load_latest_round_trip(tmp_path):
    net = nn.Linear(4, 2)
    save_checkpoint(tmp_path, _state(net, 5), step=5)
    loaded = load_latest(tmp_path)
    assert loaded is not None
    assert loaded["step"] == 5
    # Tensors round-trip exactly.
    for k, v in net.state_dict().items():
        torch.testing.assert_close(loaded["model"][k], v)


def test_load_latest_returns_none_when_no_checkpoint(tmp_path):
    assert load_latest(tmp_path) is None


def test_save_updates_latest_to_newest(tmp_path):
    net = nn.Linear(2, 2)
    save_checkpoint(tmp_path, _state(net, 10), step=10)
    save_checkpoint(tmp_path, _state(net, 20), step=20)
    save_checkpoint(tmp_path, _state(net, 30), step=30)
    latest = tmp_path / "checkpoints" / "latest.pt"
    assert latest.resolve().name == "step-00000030.pt"
    loaded = load_latest(tmp_path)
    assert loaded["step"] == 30


def test_keep_last_n_prunes_older(tmp_path):
    net = nn.Linear(2, 2)
    for s in (1, 2, 3, 4, 5):
        save_checkpoint(tmp_path, _state(net, s), step=s, keep_last_n=2)
    files = list_checkpoints(tmp_path)
    # Only the last 2 step files survive; latest.pt is a separate symlink.
    assert [int(p.stem.split("-", 1)[1]) for p in files] == [4, 5]


def test_keep_last_n_none_disables_pruning(tmp_path):
    net = nn.Linear(2, 2)
    for s in (1, 2, 3):
        save_checkpoint(tmp_path, _state(net, s), step=s, keep_last_n=None)
    files = list_checkpoints(tmp_path)
    assert len(files) == 3


def test_list_checkpoints_returns_ascending(tmp_path):
    net = nn.Linear(2, 2)
    for s in (3, 1, 2):
        save_checkpoint(tmp_path, _state(net, s), step=s, keep_last_n=None)
    files = list_checkpoints(tmp_path)
    assert [int(p.stem.split("-", 1)[1]) for p in files] == [1, 2, 3]


def test_list_checkpoints_empty_when_no_dir(tmp_path):
    assert list_checkpoints(tmp_path) == []


def test_load_latest_falls_back_when_symlink_missing(tmp_path):
    net = nn.Linear(2, 2)
    save_checkpoint(tmp_path, _state(net, 7), step=7)
    cdir = tmp_path / "checkpoints"
    (cdir / "latest.pt").unlink()
    loaded = load_latest(tmp_path)
    assert loaded is not None and loaded["step"] == 7
