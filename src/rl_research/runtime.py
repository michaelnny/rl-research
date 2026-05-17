"""Reusable runtime primitives shared by the PPO baseline and every candidate
``train.py``. These exist so candidates do not re-derive boilerplate (CLI
parsing, seeding, wallclock budget, observation/reward statistics, parameter
checksums, ``config.json`` writing). The contract that ``train.py`` must follow
lives in ``docs/contract.md``; this module is the canonical implementation of
the parts of that contract that are algorithm-agnostic.

Public API:

    parse_train_cli(argv=None)           -> argparse.Namespace
    seed_everything(seed)                -> None
    WallclockBudget(seconds)             -> elapsed/expired/remaining helpers
    RunningMeanStd(shape=(), eps=1e-4)   -> Chan/Welford parallel running stats
    param_checksum(network)              -> 16-hex sha1 of all parameters
    write_config_json(logdir, args, ...) -> Path

Reference for ``RunningMeanStd``: gymnasium/wrappers/utils.py L31-67
(Chan, Golub, LeVeque 1979). Used by the canonical PPO obs/reward
normalization pipeline ([37D §6-9]); see ``baselines/ppo.py`` for the wiring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random as _stdlib_random
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UV_LOCK_PATH = REPO_ROOT / "uv.lock"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_train_cli(
    argv: Sequence[str] | None = None,
    *,
    description: str = "rl-research train.py",
    extra: list[tuple[str, dict[str, Any]]] | None = None,
) -> argparse.Namespace:
    """Parse the canonical ``train.py`` CLI.

    The five required flags below are mandated by ``docs/contract.md`` §train.py
    contract and must be present on every candidate ``train.py``. Algorithm
    extensions can add their own flags via ``extra=[("--my-flag", {"type": int,
    "default": 1}), ...]``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--env", required=True, help="gymnasium env id (e.g. CartPole-v1)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--max-wallclock-s", type=int, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    if extra:
        for flag, kwargs in extra:
            parser.add_argument(flag, **kwargs)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed every RNG that affects training: torch (CPU + all CUDA), numpy,
    Python ``random``, and the ``PYTHONHASHSEED`` env var. Idempotent. Does
    not enable cuDNN deterministic mode (the perf hit is large and the
    contract does not require bit-identical reruns)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    _stdlib_random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — torch + numpy code below uses both legacy and Generator APIs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Wallclock budget
# ---------------------------------------------------------------------------


class WallclockBudget:
    """Enforce ``--max-wallclock-s`` per ``docs/contract.md``.

    ``train.py`` should construct this once at start, then check ``expired()``
    at every eval boundary and exit cleanly (writing ``result.json``) if it
    returns True. Using ``time.monotonic`` (not ``time.time``) so wall jumps
    don't bork the budget.
    """

    def __init__(self, seconds: float):
        self._max = float(seconds)
        self._start = time.monotonic()

    def reset(self) -> None:
        self._start = time.monotonic()

    @property
    def started_at_monotonic(self) -> float:
        return self._start

    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    def remaining_s(self) -> float:
        return max(0.0, self._max - self.elapsed_s())

    def expired(self, grace_s: float = 0.0) -> bool:
        """True if the budget is exhausted (within ``grace_s`` of the cap)."""
        return self.elapsed_s() >= (self._max - grace_s)


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """Chan/Welford parallel running mean / variance.

    Port of ``gymnasium/wrappers/utils.py:RunningMeanStd`` (L31-67), citing
    Chan, Golub, LeVeque 1979. Initial ``count = eps`` and ``var = ones`` so
    that early-batch updates are well-conditioned. Identical defaults to
    gymnasium's wrapper; the PPO baseline relies on these choices to match
    [37D §6-9].
    """

    __slots__ = ("count", "mean", "var")

    def __init__(self, shape: tuple[int, ...] = (), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, batch: np.ndarray) -> None:
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray, clip: float | None = 10.0) -> np.ndarray:
        """Standardize ``x`` against the current running stats (no update).
        Used both for live normalization and for normalizing held-out eval
        observations against frozen stats."""
        normed = (x - self.mean) / np.sqrt(self.var + 1e-8)
        if clip is not None:
            normed = np.clip(normed, -clip, clip)
        return normed.astype(np.float32)


# ---------------------------------------------------------------------------
# Param checksum
# ---------------------------------------------------------------------------


def param_checksum(network: nn.Module) -> str:
    """16-hex sha1 of every parameter tensor's bytes. Stable across runs given
    identical weights. Used by the Stage A ``progress/param_checksum`` gate to
    verify that parameters actually moved during training."""
    h = hashlib.sha1()
    for p in network.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# config.json
# ---------------------------------------------------------------------------


@dataclass
class _ConfigPayload:
    cli_args: dict[str, Any]
    python_version: str
    torch_version: str
    torch_cuda_available: bool
    torch_cuda_device: str | None
    git_sha: str
    uv_lock_sha256: str
    started_at: str
    platform: str

    def asdict(self) -> dict[str, Any]:
        return {
            "cli_args": self.cli_args,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_available": self.torch_cuda_available,
            "torch_cuda_device": self.torch_cuda_device,
            "git_sha": self.git_sha,
            "uv_lock_sha256": self.uv_lock_sha256,
            "started_at": self.started_at,
            "platform": self.platform,
        }


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "0" * 40


def _uv_lock_sha256() -> str:
    if not UV_LOCK_PATH.exists():
        return ""
    h = hashlib.sha256()
    h.update(UV_LOCK_PATH.read_bytes())
    return h.hexdigest()


def write_config_json(
    logdir: str | Path,
    args: argparse.Namespace | dict[str, Any],
    *,
    started_at: datetime | None = None,
) -> Path:
    """Write ``<logdir>/config.json`` per ``docs/contract.md``.

    Captures the exact CLI invocation, environment SHAs and Python/Torch
    versions. The Engineer is responsible for calling this; ``train.py``
    typically does so right after parsing CLI flags.
    """
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    cli_args = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    payload = _ConfigPayload(
        cli_args=cli_args,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        torch_cuda_available=torch.cuda.is_available(),
        torch_cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        git_sha=_git_sha(),
        uv_lock_sha256=_uv_lock_sha256(),
        started_at=(started_at or datetime.now(UTC))
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        platform=platform.platform(),
    )
    out = logdir / "config.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload.asdict(), indent=2, sort_keys=True))
    tmp.replace(out)
    return out


__all__ = [
    "RunningMeanStd",
    "WallclockBudget",
    "param_checksum",
    "parse_train_cli",
    "seed_everything",
    "write_config_json",
]
