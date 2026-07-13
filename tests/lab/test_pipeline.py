from __future__ import annotations

import subprocess
import sys

from rlx_lab.artifacts import ArtifactStore
from rlx_lab.brief import render_brief
from rlx_lab.models import JobStatus
from rlx_lab.providers import FakeProvider
from rlx_lab.store import ResearchStore
from rlx_lab.worker import Worker
from rlx_lab.worktrees import WorktreeManager


def git(cwd, *args):
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


def test_fake_campaign_runs_hypothesis_implementation_measurement_replication_and_finding(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.com")
    git(repository, "config", "user.name", "Test")
    (repository / "README.md").write_text("clean sheet\n")
    git(repository, "add", "README.md")
    git(repository, "commit", "-qm", "base")

    runtime = tmp_path / "runtime"
    store = ResearchStore(runtime / "state.db")
    artifacts = ArtifactStore(runtime / "artifacts")
    worktrees = WorktreeManager(repository, tmp_path / "worktrees")
    campaign = store.create_campaign("smoke", "does the evidence graph survive a complete cycle?")

    def responder(request):
        if request.role == "theorist":
            return {"mechanism": "trajectory contrast", "prediction": "effect > 0", "falsifier": "effect <= 0"}
        if request.role in {"implementer", "replicator"}:
            candidate = request.cwd / "candidates" / "probe.py"
            candidate.parent.mkdir()
            value = 0.21 if request.role == "implementer" else 0.19
            candidate.write_text(f"import json\nprint(json.dumps({{'effect': {value}}}))\n")
            return {"summary": f"{request.role} wrote an isolated probe"}
        if request.role == "analyst":
            assert "0.21" in request.prompt
            assert "0.19" in request.prompt
            return {"relation": "supports", "effect": 0.20, "replicated": True}
        raise AssertionError(request.role)

    worker = Worker(
        worker_id="pipeline-worker",
        repository=repository,
        store=store,
        artifacts=artifacts,
        providers={"fake": FakeProvider(responder)},
        worktrees=worktrees,
        lease_seconds=30,
    )

    hypothesis_schema = {
        "type": "object",
        "properties": {
            "mechanism": {"type": "string"},
            "prediction": {"type": "string"},
            "falsifier": {"type": "string"},
        },
        "required": ["mechanism", "prediction", "falsifier"],
        "additionalProperties": False,
    }
    implementation_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    finding_schema = {
        "type": "object",
        "properties": {
            "relation": {"enum": ["supports", "contradicts", "inconclusive"]},
            "effect": {"type": "number"},
            "replicated": {"type": "boolean"},
        },
        "required": ["relation", "effect", "replicated"],
        "additionalProperties": False,
    }

    hypothesis = store.enqueue_job(
        operation="agent", provider="fake", role="theorist", mode="read",
        payload={"prompt": "hypothesize", "schema": hypothesis_schema, "result_kind": "Hypothesis"},
        campaign_id=campaign, job_id="hypothesis",
    )
    common_write_payload = {
        "prompt": "implement the smallest probe",
        "schema": implementation_schema,
        "result_kind": "Implementation",
        "allowed_prefixes": ["candidates"],
        "protected_prefixes": ["src/rlx_bench", "src/rlx_lab"],
        "require_changes": True,
    }
    implementation = store.enqueue_job(
        operation="agent", provider="fake", role="implementer", mode="write",
        payload=common_write_payload, campaign_id=campaign, dependencies=(hypothesis,), job_id="implementation",
    )
    primary_run = store.enqueue_job(
        operation="execute", provider="local", role="evaluator", mode="execute",
        payload={
            "argv": [sys.executable, "candidates/probe.py"],
            "cwd_from_dependency_worktree": True,
            "result_kind": "Run",
        },
        campaign_id=campaign, dependencies=(implementation,), job_id="primary-run",
    )
    replication = store.enqueue_job(
        operation="agent", provider="fake", role="replicator", mode="write",
        payload=common_write_payload, campaign_id=campaign, dependencies=(hypothesis,), job_id="replication",
    )
    replication_run = store.enqueue_job(
        operation="execute", provider="local", role="evaluator", mode="execute",
        payload={
            "argv": [sys.executable, "candidates/probe.py"],
            "cwd_from_dependency_worktree": True,
            "result_kind": "Replication",
            "dependency_edge": "replicates",
        },
        campaign_id=campaign, dependencies=(replication,), job_id="replication-run",
    )
    finding = store.enqueue_job(
        operation="agent", provider="fake", role="analyst", mode="read",
        payload={"prompt": "analyze both measurements", "schema": finding_schema, "result_kind": "Finding"},
        campaign_id=campaign, dependencies=(primary_run, replication_run), job_id="finding",
    )

    processed = []
    while True:
        job_id = worker.run_once()
        if job_id is None:
            break
        processed.append(job_id)
    assert set(processed) == {hypothesis, implementation, primary_run, replication, replication_run, finding}
    states = {
        job.id: (job.status, job.error_class, job.error_detail)
        for job in store.list_jobs(campaign_id=campaign)
    }
    assert all(state[0] == JobStatus.COMPLETED for state in states.values()), "\n".join(
        f"{job_id}: {state}" for job_id, state in states.items()
    )
    finding_node = store.get_node(store.get_job(finding).result_node_id)
    assert finding_node.payload["role_output"] == {"relation": "supports", "effect": 0.2, "replicated": True}
    brief = render_brief(store, campaign_id=campaign)
    assert "Hypothesis" in brief and "Implementation" in brief and "Finding" in brief
    assert "failed: 0" not in brief
