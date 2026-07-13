from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from rlx_lab.artifacts import ArtifactStore
from rlx_lab.executor import ExecutionLimits, ExecutionSpec, LocalExecutor
from rlx_lab.models import JobMode, JobStatus
from rlx_lab.providers import FakeProvider, ProviderResult
from rlx_lab.store import ResearchStore
from rlx_lab.worker import Worker
from rlx_lab.worktrees import WorktreeManager


HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanism": {"type": "string", "minLength": 1},
        "prediction": {"type": "string", "minLength": 1},
        "falsifier": {"type": "string", "minLength": 1},
    },
    "required": ["mechanism", "prediction", "falsifier"],
    "additionalProperties": False,
}


def make_worker(tmp_path, responder, *, repository=None, worktrees=None):
    repository = Path(repository or tmp_path)
    store = ResearchStore(tmp_path / "state.db")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    worker = Worker(
        worker_id="worker-1",
        repository=repository,
        store=store,
        artifacts=artifacts,
        providers={"fake": FakeProvider(responder)},
        worktrees=worktrees,
        lease_seconds=10,
    )
    return store, artifacts, worker


def test_worker_builds_graph_from_structured_dependency_results(tmp_path):
    seen_prompts = []

    def responder(request):
        seen_prompts.append(request.prompt)
        if request.role == "theorist":
            return {"mechanism": "redistribute return", "prediction": "lower variance", "falsifier": "no variance change"}
        return {"relation": "contradicts", "reason": "effect disappears in the matched control"}

    store, artifacts, worker = make_worker(tmp_path, responder)
    parent = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="theorist",
        mode="read",
        payload={"prompt": "form a hypothesis", "schema": HYPOTHESIS_SCHEMA, "result_kind": "Hypothesis"},
        job_id="hypothesis-job",
    )
    child_schema = {
        "type": "object",
        "properties": {"relation": {"enum": ["supports", "contradicts"]}, "reason": {"type": "string"}},
        "required": ["relation", "reason"],
        "additionalProperties": False,
    }
    child = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="skeptic",
        mode="read",
        payload={"prompt": "test the parent", "schema": child_schema, "result_kind": "Finding"},
        dependencies=(parent,),
        job_id="finding-job",
    )

    assert worker.run_once() == parent
    assert worker.run_once() == child
    assert worker.run_once() is None
    hypothesis = store.get_node(store.get_job(parent).result_node_id)
    finding = store.get_node(store.get_job(child).result_node_id)
    assert hypothesis.kind == "Hypothesis"
    assert finding.kind == "Finding"
    assert finding.payload["role_output"]["relation"] == "contradicts"
    assert "DEPENDENCY RECORDS" in seen_prompts[1]
    assert store.edges(target_id=finding.id)[0]["source_id"] == hypothesis.id
    assert store.artifact_links(hypothesis.id)[0]["label"] == "provider_stdout"


def test_invalid_structured_output_fails_without_retry(tmp_path):
    store, _, worker = make_worker(tmp_path, lambda request: {"mechanism": "missing fields"})
    job_id = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="theorist",
        mode="read",
        payload={"prompt": "idea", "schema": HYPOTHESIS_SCHEMA},
        job_id="bad-job",
    )
    worker.run_once()
    job = store.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert job.attempt == 1
    assert job.error_class == "invalid_job_result"


def test_execute_job_records_crash_as_a_completed_run(tmp_path):
    store, _, worker = make_worker(tmp_path, lambda request: {})
    job_id = store.enqueue_job(
        operation="execute",
        provider="local",
        role="evaluator",
        mode="execute",
        payload={
            "argv": [sys.executable, "-c", "import sys; print('measurement'); raise SystemExit(7)"],
            "result_kind": "Run",
        },
        job_id="run-job",
    )
    worker.run_once()
    job = store.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    run = store.get_node(job.result_node_id)
    assert run.payload["exit_code"] == 7
    assert run.payload["run_status"] == "crashed"
    assert store.artifact_links(run.id)[0]["label"] == "stdout"


def test_execute_job_parses_and_validates_structured_stdout(tmp_path):
    store, _, worker = make_worker(tmp_path, lambda request: {})
    job_id = store.enqueue_job(
        operation="execute",
        provider="local",
        role="evaluator",
        mode="execute",
        payload={
            "argv": [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'effect': 0.2}))",
            ],
            "stdout_schema": {
                "type": "object",
                "properties": {"effect": {"type": "number"}},
                "required": ["effect"],
                "additionalProperties": False,
            },
        },
        job_id="structured-run",
    )

    worker.run_once()

    run = store.get_node(store.get_job(job_id).result_node_id)
    assert run.payload["run_status"] == "complete"
    assert run.payload["measurement"] == {"effect": 0.2}


def test_execute_job_records_invalid_measurement_without_retry(tmp_path):
    store, _, worker = make_worker(tmp_path, lambda request: {})
    job_id = store.enqueue_job(
        operation="execute",
        provider="local",
        role="evaluator",
        mode="execute",
        payload={
            "argv": [sys.executable, "-c", "print('not-json')"],
            "stdout_schema": {"type": "object"},
        },
        job_id="invalid-run",
    )

    worker.run_once()

    job = store.get_job(job_id)
    run = store.get_node(job.result_node_id)
    assert job.status == JobStatus.COMPLETED
    assert job.attempt == 1
    assert run.payload["run_status"] == "invalid_measurement"
    assert run.payload["measurement"] is None


def test_worker_shutdown_cancels_provider_group_and_requeues_job(tmp_path):
    class SlowProvider:
        def __init__(self) -> None:
            self.executor = LocalExecutor()

        def run(self, request):
            self.executor.run(
                ExecutionSpec(
                    argv=(sys.executable, "-c", "import time; time.sleep(30)"),
                    cwd=request.cwd,
                    limits=ExecutionLimits(timeout_seconds=60),
                )
            )
            return ProviderResult(output={"finished": True})

    provider = SlowProvider()
    store = ResearchStore(tmp_path / "state.db")
    worker = Worker(
        worker_id="shutdown-worker",
        repository=tmp_path,
        store=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        providers={"slow": provider},
        lease_seconds=10,
    )
    job_id = store.enqueue_job(
        operation="agent",
        provider="slow",
        role="reviewer",
        mode="read",
        payload={"prompt": "wait", "result_kind": "Analysis"},
        max_attempts=2,
    )
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    deadline = time.monotonic() + 2.0
    while provider.executor.active_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert worker.request_stop() == 1
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    job = store.get_job(job_id)
    assert job.status is JobStatus.QUEUED
    assert job.error_class == "worker_shutdown"


def test_write_job_isolated_in_worktree_and_cannot_touch_benchmark(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=repository, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=repository, check=True)
    (repository / "README.md").write_text("base\n")
    subprocess.run(("git", "add", "README.md"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=repository, check=True)
    manager = WorktreeManager(repository, tmp_path / "worktrees")

    def responder(request):
        target = request.cwd / "candidates" / "idea.py"
        target.parent.mkdir()
        target.write_text("MECHANISM = 'test'\n")
        return {"summary": "implemented probe"}

    store, _, worker = make_worker(
        tmp_path / "runtime",
        responder,
        repository=repository,
        worktrees=manager,
    )
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    job_id = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="implementer",
        mode=JobMode.WRITE,
        payload={
            "prompt": "implement",
            "schema": schema,
            "result_kind": "Implementation",
            "allowed_prefixes": ["candidates"],
            "protected_prefixes": ["src/rlx_bench"],
            "require_changes": True,
        },
        job_id="write-job",
    )
    worker.run_once()
    job = store.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    node = store.get_node(job.result_node_id)
    assert node.payload["provenance"]["changed_paths"] == ["candidates/idea.py"]
    assert not (repository / "candidates" / "idea.py").exists()
