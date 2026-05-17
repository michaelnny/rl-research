"""Render lab/CORPUS_STATS.md from the ledger.

This file gives the Researcher and Curator a stable corpus surface that does
NOT degrade as the loop runs for weeks — unlike "last 50 lines of ledger",
which becomes "yesterday's runs" once you've done >50 in a few days.

Reads:
    lab/ledger.jsonl
    lab/threads/*.md (frontmatter only)

Writes:
    lab/CORPUS_STATS.md

Invoked by:
    scripts/loop.sh (between iterations)
    .claude/commands/iterate.md (post-iteration cleanup)

Idempotent and safe to re-run on a partial / stale ledger. Never mutates the
ledger; only reads.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB = REPO_ROOT / "lab"
LEDGER = LAB / "ledger.jsonl"
THREADS_DIR = LAB / "threads"
OUT = LAB / "CORPUS_STATS.md"

MODE_COLLAPSE_WINDOW = 20
MODE_COLLAPSE_THRESHOLD = 0.6


def _read_ledger() -> list[dict]:
    """Stream-read the ledger line by line. Skips malformed lines (preflight
    surfaces them separately) and never loads the full file as a single string.

    For a multi-week corpus the ledger may grow to hundreds of MB; reading it
    via ``read_text().splitlines()`` would briefly hold the whole thing in
    memory plus a parsed list. The streaming form keeps peak memory bounded by
    the parsed-row list alone.
    """
    if not LEDGER.exists():
        return []
    rows: list[dict] = []
    with LEDGER.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_thread_status() -> dict[str, str]:
    """Return {thread_slug: status} for all threads with frontmatter."""
    out: dict[str, str] = {}
    for p in sorted(THREADS_DIR.glob("*.md")):
        if p.name == "README.md":
            continue
        text = p.read_text()
        m = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            continue
        fm = m.group(1)
        status_m = re.search(r"^status:\s*(\S+)", fm, re.MULTILINE)
        if status_m:
            out[p.stem] = status_m.group(1).strip()
    return out


def _fmt_section(title: str, rows: list[tuple[str, int | float | str]]) -> str:
    if not rows:
        return f"## {title}\n\n_(empty)_\n"
    lines = [f"## {title}", ""]
    width = max(len(str(k)) for k, _ in rows)
    for k, v in rows:
        lines.append(f"- `{k:<{width}}` — {v}")
    return "\n".join(lines) + "\n"


def render() -> str:
    rows = _read_ledger()
    thread_status = _read_thread_status()
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if not rows:
        return f"# Corpus stats\n\n_Auto-generated {now}._\n\nLedger is empty — no runs yet.\n"

    n = len(rows)
    by_status = Counter(r.get("status", "?") for r in rows)
    by_pillar = Counter(r.get("pillar", "?") for r in rows)
    by_thread = Counter(r.get("thread", "?") for r in rows)
    by_verdict = Counter((r.get("verdict_curator") or "uncurated") for r in rows)

    # Mode-collapse signal: in the last N runs, what fraction are the same thread?
    recent = rows[-MODE_COLLAPSE_WINDOW:]
    recent_threads = Counter(r.get("thread", "?") for r in recent)
    if recent_threads:
        top_thread, top_count = recent_threads.most_common(1)[0]
        top_frac = top_count / len(recent)
        mode_collapse_alert = top_frac >= MODE_COLLAPSE_THRESHOLD
    else:
        top_thread = "—"
        top_count = 0
        top_frac = 0.0
        mode_collapse_alert = False

    # Per-thread last-run summary
    last_per_thread: dict[str, dict] = {}
    for r in rows:
        last_per_thread[r.get("thread", "?")] = r

    out: list[str] = []
    out.append(
        f"# Corpus stats\n\n_Auto-generated {now} from `lab/ledger.jsonl` ({n} runs total)._\n"
    )

    if mode_collapse_alert:
        out.append(
            f"> **MODE-COLLAPSE WARNING** — `{top_thread}` accounts for "
            f"{top_count}/{len(recent)} ({top_frac:.0%}) of the last "
            f"{MODE_COLLAPSE_WINDOW} runs. Researcher should propose in a "
            f"different thread next iteration. Curator should consider "
            f"writing `lab/HALT_REQUESTED.md` if this persists.\n"
        )

    out.append(
        _fmt_section(
            "Status histogram",
            [(k, v) for k, v in by_status.most_common()],
        )
    )
    out.append(
        _fmt_section(
            "Pillar histogram",
            [(k, v) for k, v in by_pillar.most_common()],
        )
    )
    out.append(
        _fmt_section(
            "Verdict (Curator)",
            [(k, v) for k, v in by_verdict.most_common()],
        )
    )

    # Thread table: for each thread we know about, last status + last run_id.
    thread_rows: list[tuple[str, str]] = []
    for thread, count in by_thread.most_common():
        last = last_per_thread.get(thread, {})
        status = thread_status.get(thread, "—")
        verdict = last.get("verdict_curator") or "—"
        thread_rows.append(
            (
                thread,
                f"runs={count}, last={last.get('run_id', '—')}, "
                f"last_status={last.get('status', '—')}, "
                f"verdict={verdict}, thread_status={status}",
            )
        )
    out.append(_fmt_section("Threads", thread_rows))

    # Recent failures grouped by class
    failures = defaultdict(list)
    for r in rows[-50:]:
        st = r.get("status", "")
        if st in {"sanity-failed", "killed-error", "benchmark-failed", "killed-budget"}:
            failures[st].append(r.get("run_id", "?"))
    if failures:
        out.append("## Recent failures (last 50 runs)\n")
        for cls, ids in failures.items():
            out.append(f"- **{cls}** ({len(ids)}): {', '.join(ids[-10:])}")
        out.append("")

    # Recent runs (last 10) for quick scan
    out.append("## Last 10 runs\n")
    for r in rows[-10:]:
        verdict = r.get("verdict_curator") or "pending"
        out.append(
            f"- `{r.get('run_id', '?')}` "
            f"thread=`{r.get('thread', '?')}` "
            f"status=`{r.get('status', '?')}` "
            f"verdict=`{verdict}` "
            f"wallclock={r.get('wallclock_s', 0):.0f}s"
        )

    return "\n".join(out) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    tmp = OUT.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(OUT)
    print(f"wrote {OUT} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
