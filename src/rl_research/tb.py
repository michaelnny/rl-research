"""TensorBoard scalar logging that matches the run-artifact contract.

``docs/contract.md`` §"Required TensorBoard scalars" mandates these tags on
every ``train.py``:

  eval/return_mean
  eval/return_std
  eval/return_per_channel/<i>           # MORL only
  train/loss
  progress/env_steps
  progress/wallclock_s

The Engineer also writes ``progress/param_checksum`` so the Stage A "params
actually moved" check has telemetry to scan. ``RunLogger`` provides convenience
methods for each, plus an ``EvalCadence`` helper that schedules the contract's
"≥20 evals per run" requirement.

This is a thin wrapper, not an abstraction layer. Candidates that need
non-standard scalars can either (a) call ``logger.scalar(tag, value, step)``
directly, or (b) reach through ``logger.writer`` to use the underlying
``SummaryWriter`` API verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    """Wraps a ``SummaryWriter`` with helpers for the contract scalars.

    Owns the underlying writer; call ``close()`` at exit. Eval cadence is
    tracked separately via ``EvalCadence``.
    """

    def __init__(self, logdir: str | Path):
        from torch.utils.tensorboard import SummaryWriter

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(log_dir=str(self.logdir))

    # Generic escape hatch
    def scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    # Contract-shaped helpers
    def log_eval(
        self,
        step: int,
        mean: float,
        std: float,
        per_channel: np.ndarray | None = None,
    ) -> None:
        self.writer.add_scalar("eval/return_mean", float(mean), step)
        self.writer.add_scalar("eval/return_std", float(std), step)
        if per_channel is not None:
            for i, v in enumerate(per_channel):
                self.writer.add_scalar(f"eval/return_per_channel/{i}", float(v), step)

    def log_train(self, step: int, loss: float) -> None:
        self.writer.add_scalar("train/loss", float(loss), step)

    def log_progress(
        self,
        step: int,
        *,
        env_steps: int,
        wallclock_s: float,
        param_checksum: str | None = None,
    ) -> None:
        self.writer.add_scalar("progress/env_steps", int(env_steps), step)
        self.writer.add_scalar("progress/wallclock_s", float(wallclock_s), step)
        if param_checksum is not None:
            self.writer.add_scalar(
                "progress/param_checksum", int(param_checksum, 16) % (2**31), step
            )

    def close(self) -> None:
        self.writer.close()


class EvalCadence:
    """Schedule eval boundaries so the contract's "≥20 evals per run" holds.

    Use::

        cadence = EvalCadence(total_env_steps=cfg.total_env_steps, n_evals=20)
        cadence.maybe_eval(env_steps)            # bool: should we eval now?

    The first eval at ``step=0`` is the responsibility of the caller (random-init
    baseline); after that, ``maybe_eval`` returns True roughly every
    ``total_env_steps / n_evals`` env steps. The final eval at the very end of
    training is also the caller's responsibility (so the result.json can pin
    the *final* return regardless of cadence rounding).
    """

    def __init__(self, *, total_env_steps: int, n_evals: int = 20):
        self._interval = max(1, total_env_steps // n_evals)
        self._last_eval = -1

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def last_eval_step(self) -> int:
        return self._last_eval

    def maybe_eval(self, env_steps: int) -> bool:
        if env_steps - self._last_eval >= self._interval:
            self._last_eval = env_steps
            return True
        return False

    def force(self, env_steps: int) -> None:
        """Mark ``env_steps`` as the last eval (used for the initial step-0 eval)."""
        self._last_eval = env_steps


__all__ = ["EvalCadence", "RunLogger"]
