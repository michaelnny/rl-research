"""Command line entry point for reference calibration runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from .calibration import SmokeCalibrationSettings, run_smoke_calibration


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlx-calibrate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run provisional FactorLab calibration")
    smoke.add_argument("--learner-episodes", type=int, default=200)
    smoke.add_argument("--headroom-episodes", type=int, default=20)
    smoke.add_argument("--master-seed", type=int, default=20260713)
    smoke.add_argument("--output", type=Path, help="write JSON atomically instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        report = run_smoke_calibration(
            SmokeCalibrationSettings(
                learner_episodes=args.learner_episodes,
                headroom_episodes=args.headroom_episodes,
                master_seed=args.master_seed,
            )
        )
        content = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.output:
            _write_atomic(args.output, content)
        else:
            print(content, end="")
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
