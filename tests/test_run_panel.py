from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import run_panel  # noqa: E402


def test_select_envs_override() -> None:
    args = argparse.Namespace(envs="deep-sea-treasure-concave-v0", stage="all")

    assert run_panel.select_envs(args) == ["deep-sea-treasure-concave-v0"]


def test_resolve_train_path_accepts_relative_repo_path() -> None:
    assert run_panel.resolve_train_path(Path("train.py")) == (ROOT / "train.py").resolve()


def test_resolve_train_path_rejects_missing_file() -> None:
    with pytest.raises(SystemExit):
        run_panel.resolve_train_path(Path("missing-train.py"))
