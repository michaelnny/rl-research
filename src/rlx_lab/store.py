"""SQLite-backed research graph, queue, leases, and audit events."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from rlx_lab.artifacts import ArtifactRef
from rlx_lab.models import (
    Campaign,
    CampaignStatus,
    Job,
    JobMode,
    JobStatus,
    ResearchNode,
    canonical_json,
    content_hash,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    question TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL REFERENCES nodes(id),
    target_id TEXT NOT NULL REFERENCES nodes(id),
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, kind)
);

CREATE TABLE IF NOT EXISTS artifacts (
    sha256 TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS node_artifacts (
    node_id TEXT NOT NULL REFERENCES nodes(id),
    sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
    label TEXT NOT NULL,
    PRIMARY KEY (node_id, sha256, label)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    campaign_id TEXT REFERENCES campaigns(id),
    operation TEXT NOT NULL,
    provider TEXT NOT NULL,
    role TEXT NOT NULL,
    mode TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    dependencies_json TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    lease_owner TEXT,
    lease_expires_at REAL,
    not_before REAL NOT NULL,
    result_node_id TEXT REFERENCES nodes(id),
    error_class TEXT,
    error_detail TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_ready
ON jobs(status, not_before, priority DESC, created_at);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS controller_leases (
    campaign_id TEXT PRIMARY KEY REFERENCES campaigns(id),
    owner TEXT NOT NULL,
    expires_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class ResearchStore:
    """Transactional source of truth for research and orchestration state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def create_campaign(
        self,
        name: str,
        question: str,
        *,
        config: Mapping[str, Any] | None = None,
        campaign_id: str | None = None,
        now: float | None = None,
    ) -> str:
        now = time.time() if now is None else now
        campaign_id = campaign_id or f"campaign-{uuid.uuid4()}"
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO campaigns VALUES (?, ?, ?, ?, 'active', ?, ?)",
                (campaign_id, name, question, canonical_json(config or {}), now, now),
            )
            self._event(connection, "campaign", campaign_id, "created", {}, now)
        return campaign_id

    def get_campaign(self, campaign_id: str) -> Campaign:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return self._campaign_from_row(row)

    def list_campaigns(self) -> list[Campaign]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM campaigns ORDER BY created_at, id"
            ).fetchall()
        return [self._campaign_from_row(row) for row in rows]

    def set_campaign_status(
        self,
        campaign_id: str,
        status: CampaignStatus | str,
        *,
        now: float | None = None,
    ) -> Campaign:
        now = time.time() if now is None else now
        status = CampaignStatus(status)
        allowed = {
            CampaignStatus.ACTIVE: {
                CampaignStatus.PAUSED,
                CampaignStatus.STOPPED,
                CampaignStatus.COMPLETED,
                CampaignStatus.BUDGET_EXHAUSTED,
            },
            CampaignStatus.PAUSED: {CampaignStatus.ACTIVE, CampaignStatus.STOPPED},
        }
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if row is None:
                raise KeyError(campaign_id)
            current = CampaignStatus(row["status"])
            if status != current and status not in allowed.get(current, set()):
                raise ValueError(f"invalid campaign transition {current.value} -> {status.value}")
            connection.execute(
                "UPDATE campaigns SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, campaign_id),
            )
            if status in {
                CampaignStatus.STOPPED,
                CampaignStatus.COMPLETED,
                CampaignStatus.BUDGET_EXHAUSTED,
            }:
                cancelled = connection.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? "
                    "WHERE campaign_id = ? AND status = ?",
                    (JobStatus.CANCELLED.value, now, campaign_id, JobStatus.QUEUED.value),
                ).rowcount
            else:
                cancelled = 0
            self._event(
                connection,
                "campaign",
                campaign_id,
                "status_changed",
                {"from": current.value, "to": status.value, "cancelled_jobs": cancelled},
                now,
            )
            updated = connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
        assert updated is not None
        return self._campaign_from_row(updated)

    def acquire_controller_lease(
        self,
        campaign_id: str,
        owner: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        if lease_seconds <= 0.0:
            raise ValueError("controller lease_seconds must be positive")
        now = time.time() if now is None else now
        with self._transaction() as connection:
            campaign = connection.execute(
                "SELECT status FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(campaign_id)
            if campaign["status"] != CampaignStatus.ACTIVE.value:
                return False
            changed = connection.execute(
                """
                INSERT INTO controller_leases (campaign_id, owner, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(campaign_id) DO UPDATE SET
                    owner = excluded.owner,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                WHERE controller_leases.expires_at <= ? OR controller_leases.owner = ?
                """,
                (campaign_id, owner, now + lease_seconds, now, now, owner),
            ).rowcount
            if changed:
                self._event(
                    connection,
                    "campaign",
                    campaign_id,
                    "controller_leased",
                    {"owner": owner},
                    now,
                )
            return changed == 1

    def release_controller_lease(self, campaign_id: str, owner: str) -> bool:
        with self._transaction() as connection:
            changed = connection.execute(
                "DELETE FROM controller_leases WHERE campaign_id = ? AND owner = ?",
                (campaign_id, owner),
            ).rowcount
        return changed == 1

    def add_node(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        created_by: str,
        now: float | None = None,
    ) -> ResearchNode:
        now = time.time() if now is None else now
        digest = content_hash(kind, payload)
        node_id = f"node-{digest[:24]}"
        with self._transaction() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, kind, digest, canonical_json(payload), now, created_by),
            ).rowcount
            row = connection.execute("SELECT * FROM nodes WHERE content_hash = ?", (digest,)).fetchone()
            assert row is not None
            if inserted:
                self._event(connection, "node", node_id, "added", {"kind": kind}, now)
        return self._node_from_row(row)

    def get_node(self, node_id: str) -> ResearchNode:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        return self._node_from_row(row)

    def list_nodes(self, *, kind: str | None = None) -> list[ResearchNode]:
        with self.connect() as connection:
            if kind is None:
                rows = connection.execute("SELECT * FROM nodes ORDER BY created_at, id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM nodes WHERE kind = ? ORDER BY created_at, id", (kind,)
                ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def add_edge(self, source_id: str, target_id: str, kind: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._transaction() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)",
                (source_id, target_id, kind, now),
            ).rowcount
            if inserted:
                self._event(
                    connection,
                    "edge",
                    f"{source_id}:{kind}:{target_id}",
                    "added",
                    {},
                    now,
                )

    def register_artifact(
        self,
        ref: ArtifactRef,
        *,
        store_root: str | Path,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        relative = ref.path.resolve().relative_to(Path(store_root).resolve())
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (ref.sha256, ref.size, ref.media_type, str(relative), now),
            )

    def attach_artifact(
        self,
        node_id: str,
        ref: ArtifactRef,
        label: str,
        *,
        store_root: str | Path,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        self.register_artifact(ref, store_root=store_root, now=now)
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO node_artifacts VALUES (?, ?, ?)",
                (node_id, ref.sha256, label),
            )
            self._event(connection, "node", node_id, "artifact_attached", {"sha256": ref.sha256, "label": label}, now)

    def enqueue_job(
        self,
        *,
        operation: str,
        provider: str,
        role: str,
        mode: JobMode | str,
        payload: Mapping[str, Any],
        campaign_id: str | None = None,
        priority: int = 0,
        dependencies: Iterable[str] = (),
        max_attempts: int = 3,
        not_before: float | None = None,
        job_id: str | None = None,
        now: float | None = None,
    ) -> str:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        now = time.time() if now is None else now
        not_before = now if not_before is None else not_before
        job_id = job_id or f"job-{uuid.uuid4()}"
        mode = JobMode(mode)
        deps = tuple(dict.fromkeys(dependencies))
        if job_id in deps:
            raise ValueError("a job cannot depend on itself")
        with self._transaction() as connection:
            for dependency in deps:
                if connection.execute("SELECT 1 FROM jobs WHERE id = ?", (dependency,)).fetchone() is None:
                    raise KeyError(f"unknown dependency {dependency}")
            connection.execute(
                """
                INSERT INTO jobs (
                    id, campaign_id, operation, provider, role, mode, payload_json,
                    status, priority, dependencies_json, attempt, max_attempts,
                    lease_owner, lease_expires_at, not_before, result_node_id,
                    error_class, error_detail, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    job_id,
                    campaign_id,
                    operation,
                    provider,
                    role,
                    mode.value,
                    canonical_json(payload),
                    JobStatus.QUEUED.value,
                    priority,
                    canonical_json(deps),
                    max_attempts,
                    not_before,
                    now,
                    now,
                ),
            )
            self._event(connection, "job", job_id, "queued", {"dependencies": deps}, now)
        return job_id

    def claim_job(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        providers: Iterable[str] | None = None,
        campaign_id: str | None = None,
        now: float | None = None,
    ) -> Job | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time.time() if now is None else now
        provider_set = None if providers is None else frozenset(providers)
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = ? AND not_before <= ?
                ORDER BY priority DESC, created_at ASC
                """,
                (JobStatus.QUEUED.value, now),
            ).fetchall()
            selected = None
            for row in rows:
                if campaign_id is not None and row["campaign_id"] != campaign_id:
                    continue
                if provider_set is not None and row["provider"] not in provider_set:
                    continue
                if row["campaign_id"] is not None:
                    campaign = connection.execute(
                        "SELECT * FROM campaigns WHERE id = ?", (row["campaign_id"],)
                    ).fetchone()
                    if campaign is None or campaign["status"] != CampaignStatus.ACTIVE.value:
                        continue
                    if not self._campaign_budget_available(connection, campaign, row, now):
                        continue
                dependency_state = self._dependency_state(connection, json.loads(row["dependencies_json"]))
                if dependency_state == "failed":
                    connection.execute(
                        "UPDATE jobs SET status = ?, error_class = ?, error_detail = ?, updated_at = ? WHERE id = ?",
                        (
                            JobStatus.BLOCKED.value,
                            "dependency",
                            "one or more dependencies did not complete",
                            now,
                            row["id"],
                        ),
                    )
                    self._event(connection, "job", row["id"], "blocked", {}, now)
                    continue
                if dependency_state != "ready":
                    continue
                selected = row
                break
            if selected is None:
                return None
            expires = now + lease_seconds
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, attempt = attempt + 1, lease_owner = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    JobStatus.LEASED.value,
                    worker_id,
                    expires,
                    now,
                    selected["id"],
                    JobStatus.QUEUED.value,
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (selected["id"],)).fetchone()
            self._event(connection, "job", selected["id"], "leased", {"worker": worker_id}, now)
        assert row is not None
        return self._job_from_row(row)

    def start_job(self, job_id: str, worker_id: str, *, now: float | None = None) -> Job:
        now = time.time() if now is None else now
        with self._transaction() as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET status = ?, updated_at = ?
                WHERE id = ? AND status = ? AND lease_owner = ? AND lease_expires_at > ?
                """,
                (
                    JobStatus.RUNNING.value,
                    now,
                    job_id,
                    JobStatus.LEASED.value,
                    worker_id,
                    now,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"job {job_id} is not validly leased by {worker_id}")
            self._event(connection, "job", job_id, "started", {"worker": worker_id}, now)
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        return self._job_from_row(row)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._transaction() as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND lease_owner = ? AND status IN (?, ?)
                """,
                (
                    now + lease_seconds,
                    now,
                    job_id,
                    worker_id,
                    JobStatus.LEASED.value,
                    JobStatus.RUNNING.value,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"cannot heartbeat job {job_id}")

    def complete_job(
        self,
        job_id: str,
        worker_id: str,
        result_node_id: str,
        *,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._transaction() as connection:
            changed = connection.execute(
                """
                UPDATE jobs
                SET status = ?, result_node_id = ?, lease_owner = NULL,
                    lease_expires_at = NULL, error_class = NULL, error_detail = NULL,
                    updated_at = ?
                WHERE id = ? AND lease_owner = ? AND status = ?
                """,
                (
                    JobStatus.COMPLETED.value,
                    result_node_id,
                    now,
                    job_id,
                    worker_id,
                    JobStatus.RUNNING.value,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"cannot complete job {job_id}")
            self._event(connection, "job", job_id, "completed", {"result": result_node_id}, now)

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_class: str,
        detail: str,
        retryable: bool,
        retry_delay: float = 0.0,
        now: float | None = None,
    ) -> JobStatus:
        now = time.time() if now is None else now
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["lease_owner"] != worker_id or row["status"] != JobStatus.RUNNING.value:
                raise RuntimeError(f"cannot fail job {job_id}")
            should_retry = retryable and row["attempt"] < row["max_attempts"]
            status = JobStatus.QUEUED if should_retry else JobStatus.FAILED
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                    not_before = ?, error_class = ?, error_detail = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, now + retry_delay, error_class, detail, now, job_id),
            )
            self._event(
                connection,
                "job",
                job_id,
                "requeued" if should_retry else "failed",
                {"error_class": error_class},
                now,
            )
        return status

    def recover_expired(self, *, now: float | None = None) -> tuple[str, ...]:
        now = time.time() if now is None else now
        recovered: list[str] = []
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN (?, ?) AND lease_expires_at <= ?
                """,
                (JobStatus.LEASED.value, JobStatus.RUNNING.value, now),
            ).fetchall()
            for row in rows:
                status = JobStatus.QUEUED if row["attempt"] < row["max_attempts"] else JobStatus.FAILED
                connection.execute(
                    """
                    UPDATE jobs SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                        not_before = ?, error_class = ?, error_detail = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        now,
                        "expired_lease",
                        "worker lease expired",
                        now,
                        row["id"],
                    ),
                )
                self._event(connection, "job", row["id"], "lease_expired", {}, now)
                recovered.append(row["id"])
        return tuple(recovered)

    def get_job(self, job_id: str) -> Job:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_from_row(row)

    def list_jobs(self, *, campaign_id: str | None = None) -> list[Job]:
        with self.connect() as connection:
            if campaign_id is None:
                rows = connection.execute("SELECT * FROM jobs ORDER BY created_at, id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE campaign_id = ? ORDER BY created_at, id",
                    (campaign_id,),
                ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def job_counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def campaign_usage(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        jobs = self.list_jobs(campaign_id=campaign_id)
        provider_attempts = sum(job.attempt for job in jobs if job.operation == "agent")
        local_attempts = sum(job.attempt for job in jobs if job.operation == "execute")
        return {
            "provider_attempts": provider_attempts,
            "local_run_attempts": local_attempts,
            "jobs_total": len(jobs),
            "jobs_by_status": {
                status.value: sum(job.status is status for job in jobs) for status in JobStatus
            },
            "elapsed_seconds": max(0.0, time.time() - campaign.created_at),
        }

    def edges(self, *, target_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if target_id is None:
                rows = connection.execute("SELECT * FROM edges ORDER BY created_at").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM edges WHERE target_id = ? ORDER BY created_at", (target_id,)
                ).fetchall()
        return [dict(row) for row in rows]

    def artifact_links(self, node_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT na.label, a.sha256, a.size, a.media_type, a.relative_path
                FROM node_artifacts AS na
                JOIN artifacts AS a ON a.sha256 = na.sha256
                WHERE na.node_id = ? ORDER BY na.label, a.sha256
                """,
                (node_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def events(self, *, entity_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if entity_id is None:
                rows = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM events WHERE entity_id = ? ORDER BY sequence", (entity_id,)
                ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _dependency_state(connection: sqlite3.Connection, dependencies: list[str]) -> str:
        if not dependencies:
            return "ready"
        placeholders = ",".join("?" for _ in dependencies)
        rows = connection.execute(
            f"SELECT id, status FROM jobs WHERE id IN ({placeholders})", dependencies
        ).fetchall()
        statuses = {row["id"]: JobStatus(row["status"]) for row in rows}
        if len(statuses) != len(dependencies):
            return "failed"
        if any(status in {JobStatus.FAILED, JobStatus.BLOCKED, JobStatus.CANCELLED} for status in statuses.values()):
            return "failed"
        if all(status == JobStatus.COMPLETED for status in statuses.values()):
            return "ready"
        return "waiting"

    @classmethod
    def _campaign_budget_available(
        cls,
        connection: sqlite3.Connection,
        campaign: sqlite3.Row,
        job: sqlite3.Row,
        now: float,
    ) -> bool:
        config = json.loads(campaign["config_json"])
        policy = config.get("policy") if isinstance(config, dict) else None
        if not isinstance(policy, dict):
            return True
        wall_limit = policy.get("max_wall_seconds")
        operation_limit_key = (
            "max_provider_attempts" if job["operation"] == "agent" else "max_local_run_attempts"
        )
        operation_limit = policy.get(operation_limit_key)
        exhausted_reason = None
        if isinstance(wall_limit, (int, float)) and now - campaign["created_at"] >= wall_limit:
            exhausted_reason = "max_wall_seconds"
        elif isinstance(operation_limit, int):
            attempts = connection.execute(
                "SELECT COALESCE(SUM(attempt), 0) FROM jobs "
                "WHERE campaign_id = ? AND operation = ?",
                (campaign["id"], job["operation"]),
            ).fetchone()[0]
            if attempts >= operation_limit:
                exhausted_reason = operation_limit_key
        if exhausted_reason is None:
            return True
        connection.execute(
            "UPDATE campaigns SET status = ?, updated_at = ? WHERE id = ?",
            (CampaignStatus.BUDGET_EXHAUSTED.value, now, campaign["id"]),
        )
        cancelled = connection.execute(
            "UPDATE jobs SET status = ?, updated_at = ? "
            "WHERE campaign_id = ? AND status = ?",
            (JobStatus.CANCELLED.value, now, campaign["id"], JobStatus.QUEUED.value),
        ).rowcount
        cls._event(
            connection,
            "campaign",
            campaign["id"],
            "budget_exhausted",
            {"reason": exhausted_reason, "cancelled_jobs": cancelled},
            now,
        )
        return False

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        now: float,
    ) -> None:
        connection.execute(
            "INSERT INTO events (entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (entity_type, entity_id, event_type, canonical_json(payload), now),
        )

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> ResearchNode:
        return ResearchNode(
            id=row["id"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            created_by=row["created_by"],
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            campaign_id=row["campaign_id"],
            operation=row["operation"],
            provider=row["provider"],
            role=row["role"],
            mode=JobMode(row["mode"]),
            payload=json.loads(row["payload_json"]),
            status=JobStatus(row["status"]),
            priority=row["priority"],
            dependencies=tuple(json.loads(row["dependencies_json"])),
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            not_before=row["not_before"],
            result_node_id=row["result_node_id"],
            error_class=row["error_class"],
            error_detail=row["error_detail"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _campaign_from_row(row: sqlite3.Row) -> Campaign:
        return Campaign(
            id=row["id"],
            name=row["name"],
            question=row["question"],
            config=json.loads(row["config_json"]),
            status=CampaignStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
