"""Allocate the next run_id for an iteration of the research loop.

Format: <YYYYMMDD>-<NN>-<slug>, where NN is a per-day counter starting at 01.
Single-writer (one Claude session at a time), so no locking is needed.

Usage:
    uv run python scripts/next_run_id.py [slug]

The slug defaults to "auto" if omitted; the Researcher may rename the run
directory later, but the orchestrator only needs a unique id for now.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
RUNS_DIR = ROOT / "worklogs" / "runs"
SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(raw: str) -> str:
    s = SLUG_RE.sub("-", raw.strip().lower()).strip("-")
    return s or "auto"


def next_run_id(slug: str) -> str:
    today = _dt.datetime.now().strftime("%Y%m%d")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith(today)]
    nums = []
    for name in existing:
        parts = name.split("-", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    nn = max(nums, default=0) + 1
    return f"{today}-{nn:02d}-{slugify(slug)}"


def main() -> None:
    slug = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print(next_run_id(slug))


if __name__ == "__main__":
    main()
