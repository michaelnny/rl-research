"""Atomic checkpoint save/restore.

Every ``train.py`` should checkpoint periodically so a crashed or budget-killed
run is not entirely lost. This module is intentionally generic: the caller
hands in a ``state`` dict (typically ``{"model": net.state_dict(), "optim":
opt.state_dict(), "rng": ..., "step": int}``) and ``load_latest`` returns the
same dict. We do not opine on what goes in the dict.

Layout under a logdir::

    <logdir>/checkpoints/step-NNNNNNNN.pt    # the actual files
    <logdir>/checkpoints/latest.pt            # symlink to the most recent

Atomicity: write to ``.pt.tmp`` then ``os.replace`` to the final name; the
``latest`` symlink is updated via ``os.replace`` on a temporary symlink so a
concurrent reader never sees a half-written symlink.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import torch


def _checkpoints_dir(logdir: str | Path) -> Path:
    p = Path(logdir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    logdir: str | Path,
    state: dict[str, Any],
    step: int,
    *,
    keep_last_n: int | None = 3,
) -> Path:
    """Atomically save ``state`` as ``<logdir>/checkpoints/step-NNNNNNNN.pt``
    and update the ``latest.pt`` symlink. Optionally prune older checkpoints
    beyond the last ``keep_last_n`` (set ``None`` to disable pruning)."""
    cdir = _checkpoints_dir(logdir)
    fname = cdir / f"step-{step:08d}.pt"
    tmp = fname.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, fname)

    latest = cdir / "latest.pt"
    latest_tmp = cdir / "latest.pt.tmp"
    if latest_tmp.exists() or latest_tmp.is_symlink():
        latest_tmp.unlink()
    os.symlink(fname.name, latest_tmp)
    os.replace(latest_tmp, latest)

    if keep_last_n is not None and keep_last_n > 0:
        existing = sorted(
            cdir.glob("step-*.pt"),
            key=lambda p: int(p.stem.split("-", 1)[1]),
        )
        for old in existing[:-keep_last_n]:
            with contextlib.suppress(OSError):
                old.unlink()
    return fname


def load_latest(logdir: str | Path, *, map_location: str | None = None) -> dict[str, Any] | None:
    """Return the most recent checkpoint dict under ``<logdir>/checkpoints``,
    or ``None`` if no checkpoint exists. ``map_location`` is forwarded to
    ``torch.load`` so a checkpoint saved on GPU can be reloaded on CPU.

    ``weights_only=True`` is set: checkpoints are produced by ``save_checkpoint``
    in this same repo and only contain tensors / dicts, so the unpickle restriction
    is safe. It also silences torch's FutureWarning and blocks an attacker who
    drops a malicious .pt into a run dir.
    """
    cdir = Path(logdir) / "checkpoints"
    latest = cdir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        return torch.load(latest, map_location=map_location, weights_only=True)
    candidates = sorted(
        cdir.glob("step-*.pt"),
        key=lambda p: int(p.stem.split("-", 1)[1]),
    )
    if not candidates:
        return None
    return torch.load(candidates[-1], map_location=map_location, weights_only=True)


def list_checkpoints(logdir: str | Path) -> list[Path]:
    """Return all step-numbered checkpoints in ascending order."""
    cdir = Path(logdir) / "checkpoints"
    if not cdir.exists():
        return []
    return sorted(
        cdir.glob("step-*.pt"),
        key=lambda p: int(p.stem.split("-", 1)[1]),
    )


__all__ = ["list_checkpoints", "load_latest", "save_checkpoint"]
