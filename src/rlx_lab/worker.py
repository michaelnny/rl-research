"""Job worker joining providers, worktrees, execution, and the research graph."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from rlx_lab.artifacts import ArtifactRef, ArtifactStore
from rlx_lab.executor import ExecutionLimits, ExecutionSpec, LocalExecutor
from rlx_lab.models import Job, JobMode
from rlx_lab.providers import Provider, ProviderError, ProviderRequest
from rlx_lab.schema import SchemaValidationError, validate
from rlx_lab.secrets import CampaignSecretStore, SecretStoreError
from rlx_lab.store import ResearchStore
from rlx_lab.worktrees import Worktree, WorktreeError, WorktreeManager


class Worker:
    def __init__(
        self,
        *,
        worker_id: str,
        repository: str | Path,
        store: ResearchStore,
        artifacts: ArtifactStore,
        providers: Mapping[str, Provider],
        worktrees: WorktreeManager | None = None,
        secrets: CampaignSecretStore | None = None,
        runtime_root: str | Path | None = None,
        campaign_id: str | None = None,
        lease_seconds: float = 900.0,
    ) -> None:
        self.worker_id = worker_id
        self.repository = Path(repository).resolve()
        self.store = store
        self.artifacts = artifacts
        self.providers = dict(providers)
        self.worktrees = worktrees
        self.secrets = secrets
        self.runtime_root = None if runtime_root is None else Path(runtime_root).resolve()
        self.campaign_id = campaign_id
        self.lease_seconds = lease_seconds
        self.executor = LocalExecutor()
        self._stop_requested = False

    def request_stop(self) -> int:
        """Cancel active subprocess groups so supervisor shutdown cannot orphan them."""

        self._stop_requested = True
        executors = [self.executor]
        if self.worktrees is not None:
            executors.append(self.worktrees.executor)
        for provider in self.providers.values():
            provider_executor = getattr(provider, "executor", None)
            if isinstance(provider_executor, LocalExecutor):
                executors.append(provider_executor)
        return sum(executor.terminate_active() for executor in executors)

    def run_once(self) -> str | None:
        job = self.store.claim_job(
            self.worker_id,
            lease_seconds=self.lease_seconds,
            providers=self.providers.keys() | {"local"},
            campaign_id=self.campaign_id,
        )
        if job is None:
            return None
        self.store.start_job(job.id, self.worker_id)
        heartbeat = _Heartbeat(self.store, job.id, self.worker_id, self.lease_seconds)
        heartbeat.start()
        try:
            if job.operation == "agent":
                node_id = self._run_agent(job)
            elif job.operation == "execute":
                node_id = self._run_command(job)
            else:
                raise ValueError(f"unsupported operation {job.operation!r}")
            self.store.complete_job(job.id, self.worker_id, node_id)
            return job.id
        except ProviderError as exc:
            self.store.fail_job(
                job.id,
                self.worker_id,
                error_class=exc.error_class,
                detail=str(exc),
                retryable=exc.retryable,
                retry_delay=min(300.0, 5.0 * (2 ** max(job.attempt - 1, 0))),
            )
            return job.id
        except (
            SchemaValidationError,
            SecretStoreError,
            WorktreeError,
            ValueError,
            KeyError,
        ) as exc:
            if self._stop_requested:
                self.store.fail_job(
                    job.id,
                    self.worker_id,
                    error_class="worker_shutdown",
                    detail=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    retry_delay=0.0,
                )
                return job.id
            self.store.fail_job(
                job.id,
                self.worker_id,
                error_class="invalid_job_result",
                detail=f"{type(exc).__name__}: {exc}",
                retryable=False,
            )
            return job.id
        except Exception as exc:
            if self._stop_requested:
                self.store.fail_job(
                    job.id,
                    self.worker_id,
                    error_class="worker_shutdown",
                    detail=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    retry_delay=0.0,
                )
                return job.id
            self.store.fail_job(
                job.id,
                self.worker_id,
                error_class="harness_error",
                detail=f"{type(exc).__name__}: {exc}",
                retryable=False,
            )
            return job.id
        finally:
            heartbeat.stop()

    def _run_agent(self, job: Job) -> str:
        if job.provider not in self.providers:
            raise ValueError(f"provider {job.provider!r} is unavailable")
        payload = job.payload
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("agent job requires a non-empty prompt")
        schema = payload.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError("schema must be an object")
        timeout = float(payload.get("timeout_seconds", 1200.0))
        result_kind = str(payload.get("result_kind", job.role))
        worktree = self._prepare_worktree(job)
        cwd = self.repository if worktree is None else worktree.path
        contextual_prompt = self._with_dependency_context(job, prompt)
        result = self.providers[job.provider].run(
            ProviderRequest(
                role=job.role,
                prompt=contextual_prompt,
                cwd=cwd,
                mode=job.mode,
                schema=schema,
                timeout_seconds=timeout,
            )
        )
        if self._stop_requested:
            raise ProviderError(
                "worker shutdown interrupted provider execution",
                retryable=True,
                error_class="worker_shutdown",
            )
        if schema is not None:
            validate(result.output, schema)

        changed_paths: tuple[str, ...] = ()
        commit_sha: str | None = None
        if worktree is not None:
            assert self.worktrees is not None
            changed_paths = self.worktrees.changed_paths(worktree)
            allowed = tuple(str(item) for item in payload.get("allowed_prefixes", ()))
            protected = tuple(str(item) for item in payload.get("protected_prefixes", ()))
            self.worktrees.assert_paths_allowed(
                changed_paths,
                allowed_prefixes=allowed,
                protected_prefixes=protected,
            )
            if payload.get("require_changes", False) and not changed_paths:
                raise WorktreeError("write job produced no changed files")
            commit_sha = self.worktrees.commit_changes(worktree, changed_paths)

        node = self.store.add_node(
            result_kind,
            {
                "role_output": result.output,
                "provenance": {
                    "job_id": job.id,
                    "provider": job.provider,
                    "role": job.role,
                    "attempt": job.attempt,
                    "duration_seconds": result.duration_seconds,
                    "worktree": None if worktree is None else str(worktree.path),
                    "branch": None if worktree is None else worktree.branch,
                    "commit_sha": commit_sha,
                    "changed_paths": changed_paths,
                },
            },
            created_by=job.id,
        )
        self._link_dependencies(job, node.id)
        self._attach_stream(node.id, "provider_stdout", result.stdout)
        self._attach_stream(node.id, "provider_stderr", result.stderr)
        return node.id

    def _run_command(self, job: Job) -> str:
        payload = job.payload
        argv = payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("execute job requires argv as a non-empty string list")
        dependency_worktree = bool(payload.get("cwd_from_dependency_worktree", False))
        dependency_record: Worktree | None = None
        if dependency_worktree:
            dependency_record = self._dependency_worktree(job)
            cwd = dependency_record.path
        else:
            relative_cwd = payload.get("cwd", ".")
            if not isinstance(relative_cwd, str):
                raise ValueError("cwd must be a string")
            cwd = _resolve_within(self.repository, relative_cwd)
        limits = ExecutionLimits(
            timeout_seconds=float(payload.get("timeout_seconds", 300.0)),
            memory_bytes=_optional_int(payload.get("memory_bytes")),
            cpu_seconds=_optional_int(payload.get("cpu_seconds")),
            output_bytes=int(payload.get("output_bytes", 10_000_000)),
        )
        environment = payload.get("env", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
        ):
            raise ValueError("env must be a string mapping")
        environment = dict(environment)
        candidate_evaluation = bool(payload.get("candidate_evaluation", False))
        if candidate_evaluation:
            if not dependency_worktree:
                raise ValueError("candidate evaluation requires a dependency worktree")
            if job.campaign_id is None or self.secrets is None or self.runtime_root is None:
                raise ValueError("candidate evaluation requires campaign secret isolation")
            self.secrets.load(job.campaign_id)
            environment["RLX_FACTORLAB_SUITE_KEY_FILE"] = str(
                self.secrets.path_for(job.campaign_id)
            )
            environment["RLX_UNREADABLE_ROOTS"] = str(self.runtime_root)
            environment["RLX_UNWRITABLE_ROOTS"] = str(self.repository)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
        protected = tuple(str(item) for item in payload.get("protected_prefixes", ()))
        protected_snapshot = None
        if dependency_worktree and self.worktrees is not None and protected:
            protected_snapshot = self.worktrees.snapshot_paths(cwd, protected)
        result = self.executor.run(
            ExecutionSpec(
                argv=tuple(argv),
                cwd=cwd,
                limits=limits,
                env=environment,
                inherit_env=bool(payload.get("inherit_env", False)),
            )
        )
        if self._stop_requested:
            raise ProviderError(
                "worker shutdown interrupted local execution",
                retryable=True,
                error_class="worker_shutdown",
            )
        if dependency_worktree and self.worktrees is not None:
            if protected_snapshot is not None:
                after_snapshot = self.worktrees.snapshot_paths(cwd, protected)
                self.worktrees.assert_snapshot_unchanged(
                    protected_snapshot, after_snapshot
                )
            changed_paths = self.worktrees.changed_paths(
                Worktree(job_id=job.id, path=cwd, branch="dependency")
            )
            allowed = tuple(str(item) for item in payload.get("allowed_prefixes", ()))
            self.worktrees.assert_paths_allowed(
                changed_paths,
                allowed_prefixes=allowed,
                protected_prefixes=protected,
            )

        measurement = None
        validation_error = None
        stdout_schema = payload.get("stdout_schema")
        if stdout_schema is not None:
            if not isinstance(stdout_schema, dict):
                raise ValueError("stdout_schema must be an object")
            try:
                measurement = json.loads(result.stdout.decode("utf-8"))
                validate(measurement, stdout_schema)
            except (UnicodeDecodeError, json.JSONDecodeError, SchemaValidationError) as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                measurement = None
        run_status = "complete"
        if result.timed_out:
            run_status = "timed_out"
        elif result.exit_code != 0:
            run_status = "crashed"
        elif stdout_schema is not None and measurement is None:
            run_status = "invalid_measurement"
        cleanup_error = None
        if candidate_evaluation and dependency_record is not None and self.worktrees is not None:
            try:
                self.worktrees.remove(dependency_record)
            except WorktreeError as exc:
                cleanup_error = str(exc)
        node = self.store.add_node(
            str(payload.get("result_kind", "Run")),
            {
                "job_id": job.id,
                "argv": argv,
                "cwd": str(cwd),
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_seconds": result.duration_seconds,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
                "run_status": run_status,
                "measurement": measurement,
                "measurement_validation_error": validation_error,
                "worktree_cleanup_error": cleanup_error,
            },
            created_by=job.id,
        )
        self._link_dependencies(job, node.id)
        self._attach_stream(node.id, "stdout", result.stdout)
        self._attach_stream(node.id, "stderr", result.stderr)
        return node.id

    def _prepare_worktree(self, job: Job) -> Worktree | None:
        if job.mode != JobMode.WRITE:
            return None
        if self.worktrees is None:
            raise WorktreeError("write job requested but no worktree manager is configured")
        attempt_id = f"{job.id}-attempt-{job.attempt}"
        return self.worktrees.prepare(
            attempt_id, base_ref=str(job.payload.get("base_ref", "HEAD"))
        )

    def _with_dependency_context(self, job: Job, prompt: str) -> str:
        context = []
        for dependency_id in job.dependencies:
            dependency = self.store.get_job(dependency_id)
            if dependency.result_node_id is None:
                raise RuntimeError(f"dependency {dependency_id} has no result node")
            node = self.store.get_node(dependency.result_node_id)
            artifact_records = []
            for link in self.store.artifact_links(node.id):
                record = dict(link)
                if str(link["media_type"]).startswith("text/"):
                    data = self.artifacts.read_bytes(link["sha256"])
                    record["text_excerpt"] = data[:20_000].decode("utf-8", errors="replace")
                    record["excerpt_truncated"] = len(data) > 20_000
                artifact_records.append(record)
            context.append(
                {
                    "job_id": dependency_id,
                    "node_id": node.id,
                    "kind": node.kind,
                    "payload": node.payload,
                    "artifacts": artifact_records,
                }
            )
        if not context:
            return prompt
        return prompt + "\n\nDEPENDENCY RECORDS (immutable JSON):\n" + json.dumps(context, sort_keys=True)

    def _link_dependencies(self, job: Job, target_node_id: str) -> None:
        edge_kind = str(job.payload.get("dependency_edge", "informs"))
        for dependency_id in job.dependencies:
            dependency = self.store.get_job(dependency_id)
            if dependency.result_node_id is not None:
                self.store.add_edge(dependency.result_node_id, target_node_id, edge_kind)

    def _dependency_worktree(self, job: Job) -> Worktree:
        if self.worktrees is None:
            raise WorktreeError("dependency worktree requested without a worktree manager")
        candidates: list[Worktree] = []
        for dependency_id in job.dependencies:
            dependency = self.store.get_job(dependency_id)
            if dependency.result_node_id is None:
                continue
            node = self.store.get_node(dependency.result_node_id)
            provenance = node.payload.get("provenance", {})
            worktree = provenance.get("worktree") if isinstance(provenance, dict) else None
            branch = provenance.get("branch") if isinstance(provenance, dict) else None
            if isinstance(worktree, str) and isinstance(branch, str):
                path = Path(worktree).resolve()
                try:
                    path.relative_to(self.worktrees.root)
                except ValueError as exc:
                    raise WorktreeError(f"dependency worktree is outside managed root: {path}") from exc
                candidates.append(
                    Worktree(job_id=dependency_id, path=path, branch=branch)
                )
        if len(candidates) != 1:
            raise WorktreeError(f"expected exactly one dependency worktree, found {len(candidates)}")
        if not candidates[0].path.is_dir():
            raise WorktreeError(
                f"dependency worktree no longer exists: {candidates[0].path}"
            )
        return candidates[0]

    def _attach_stream(self, node_id: str, label: str, data: bytes) -> ArtifactRef | None:
        if not data:
            return None
        ref = self.artifacts.put_bytes(data, media_type="text/plain; charset=utf-8")
        self.store.attach_artifact(node_id, ref, label, store_root=self.artifacts.root)
        return ref


class _Heartbeat:
    def __init__(
        self,
        store: ResearchStore,
        job_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = max(0.1, self.lease_seconds / 3.0)
        while not self._stop.wait(interval):
            try:
                self.store.heartbeat(
                    self.job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception:
                return


def _resolve_within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative}") from exc
    return candidate


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
