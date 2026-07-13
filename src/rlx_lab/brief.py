"""Human-readable projection of graph state; never a source of truth."""

from __future__ import annotations

from rlx_lab.models import JobStatus
from rlx_lab.store import ResearchStore


def render_brief(store: ResearchStore, *, campaign_id: str | None = None) -> str:
    jobs = store.list_jobs(campaign_id=campaign_id)
    nodes = store.list_nodes()
    lines = ["# Research state brief", "", "Generated from the immutable research graph.", ""]
    lines.extend(("## Queue", ""))
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status.value] = counts.get(job.status.value, 0) + 1
    if counts:
        for status in JobStatus:
            if status.value in counts:
                lines.append(f"- {status.value}: {counts[status.value]}")
    else:
        lines.append("- no jobs")

    lines.extend(("", "## Evidence nodes", ""))
    if not nodes:
        lines.append("- no nodes")
    for node in nodes:
        summary = _summary(node.payload)
        lines.append(f"- `{node.id}` — **{node.kind}** — {summary}")

    failures = [job for job in jobs if job.status in {JobStatus.FAILED, JobStatus.BLOCKED}]
    lines.extend(("", "## Incidents and blocked work", ""))
    if not failures:
        lines.append("- none")
    for job in failures:
        lines.append(f"- `{job.id}`: {job.error_class or 'unknown'} — {job.error_detail or ''}")
    lines.append("")
    return "\n".join(lines)


def _summary(payload) -> str:
    if not isinstance(payload, dict):
        return "recorded"
    role_output = payload.get("role_output")
    if isinstance(role_output, dict):
        for key in ("title", "mechanism", "summary", "reason", "statement"):
            if isinstance(role_output.get(key), str):
                return role_output[key][:180]
    for key in ("summary", "exit_code", "job_id"):
        if key in payload:
            return f"{key}={payload[key]}"
    return "recorded"

