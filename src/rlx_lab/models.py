"""Value objects shared by the research graph and worker runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class JobMode(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class CampaignStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"


TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.BLOCKED, JobStatus.CANCELLED}
)


@dataclass(frozen=True)
class ResearchNode:
    id: str
    kind: str
    payload: Mapping[str, Any]
    content_hash: str
    created_at: float
    created_by: str


@dataclass(frozen=True)
class Job:
    id: str
    campaign_id: str | None
    operation: str
    provider: str
    role: str
    mode: JobMode
    payload: Mapping[str, Any]
    status: JobStatus
    priority: int
    dependencies: tuple[str, ...]
    attempt: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    not_before: float
    result_node_id: str | None
    error_class: str | None
    error_detail: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Campaign:
    id: str
    name: str
    question: str
    config: Mapping[str, Any]
    status: CampaignStatus
    created_at: float
    updated_at: float


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for hashes and persisted payloads."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(kind: str, payload: Mapping[str, Any]) -> str:
    material = canonical_json({"kind": kind, "payload": payload}).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
