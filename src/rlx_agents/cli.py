"""Command-line entry point for preregistered neural qualification studies."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Sequence

from .qualification_study import load_protocol, run_qualification_study


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


def _read_key(path: Path) -> bytes:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise SystemExit("suite key must be a regular owner-only file")
    content = path.read_bytes()
    if len(content) != 32:
        raise SystemExit("suite key must contain exactly 32 bytes")
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlx-qualify")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol, protocol_sha = load_protocol(args.protocol)
    result = run_qualification_study(
        protocol,
        protocol_sha256=protocol_sha,
        master_key=_read_key(args.key_file),
        device_override=args.device,
    )
    _write_atomic(
        args.output_dir / "evidence.json",
        json.dumps(result.evidence_bundle, indent=2, sort_keys=True) + "\n",
    )
    _write_atomic(
        args.output_dir / "qualification-report.json",
        json.dumps(result.report.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "qualified": result.report.qualified,
                "report_id": result.report.report_id,
                "evidence_sha256": result.evidence_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if result.report.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
