"""Deterministic evidence-DAG controller for continuous research campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import socket
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from rlx_lab.models import CampaignStatus, Job, JobMode, JobStatus
from rlx_lab.store import ResearchStore


class SearchLane(StrEnum):
    EXPLOIT = "exploit"
    FALSIFY = "falsify"
    EXPLORE = "explore"
    TRANSFER = "transfer"
    SYNTHESIZE = "synthesize"


DEFAULT_PORTFOLIO = {
    SearchLane.EXPLOIT: 0.35,
    SearchLane.FALSIFY: 0.25,
    SearchLane.EXPLORE: 0.20,
    SearchLane.TRANSFER: 0.10,
    SearchLane.SYNTHESIZE: 0.10,
}

PROTECTED_PATHS = (
    "src/rlx_bench",
    "src/rlx_agents",
    "src/rlx_lab",
    "tests/bench",
    "tests/agents",
    "tests/lab",
    "campaigns/factorlab_long_v1",
    "campaigns/schemas",
    "design",
)


@dataclass(frozen=True)
class CampaignPolicy:
    benchmark_tier: str = "factorlab-long-5k-v1"
    concurrent_branches: int = 3
    max_branches: int = 24
    max_inflight_jobs: int = 12
    max_provider_attempts: int = 240
    max_local_run_attempts: int = 48
    max_wall_seconds: float = 7 * 24 * 3600
    primary_provider: str = "codex"
    independent_provider: str = "claude"
    portfolio: Mapping[SearchLane | str, float] = field(
        default_factory=lambda: dict(DEFAULT_PORTFOLIO)
    )
    synthesis_interval_findings: int = 4
    evaluation_horizon: int = 5000
    evaluation_factors: int = 12
    evaluation_levels_per_factor: int = 10
    evaluation_signal_dim: int = 16
    evaluation_context_dim: int = 8
    evaluation_state_dim: int = 8
    evaluation_teacher_hidden_dim: int = 16
    evaluation_training_episodes: int = 256
    evaluation_training_batch_size: int = 64
    evaluation_training_trials: int = 3
    evaluation_public_worlds: int = 32
    evaluation_heldout_worlds: int = 16
    evaluation_max_parameters: int = 2_000_000
    evaluation_wall_seconds: float = 14_400.0

    def __post_init__(self) -> None:
        integer_limits = (
            self.concurrent_branches,
            self.max_branches,
            self.max_inflight_jobs,
            self.max_provider_attempts,
            self.max_local_run_attempts,
            self.synthesis_interval_findings,
            self.evaluation_horizon,
            self.evaluation_factors,
            self.evaluation_levels_per_factor,
            self.evaluation_signal_dim,
            self.evaluation_context_dim,
            self.evaluation_state_dim,
            self.evaluation_teacher_hidden_dim,
            self.evaluation_training_episodes,
            self.evaluation_training_batch_size,
            self.evaluation_training_trials,
            self.evaluation_public_worlds,
            self.evaluation_heldout_worlds,
            self.evaluation_max_parameters,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("campaign policy integer limits must be positive")
        if self.concurrent_branches > self.max_branches:
            raise ValueError("concurrent_branches cannot exceed max_branches")
        if self.max_wall_seconds <= 0.0 or self.evaluation_wall_seconds <= 0.0:
            raise ValueError("campaign wall limits must be positive")
        if not self.primary_provider or not self.independent_provider:
            raise ValueError("campaign providers cannot be empty")
        if {self.primary_provider, self.independent_provider} - {"codex", "claude"}:
            raise ValueError("campaign providers must be codex or claude")
        if not self.benchmark_tier:
            raise ValueError("campaign benchmark_tier cannot be empty")
        normalized = {SearchLane(key): float(value) for key, value in self.portfolio.items()}
        if set(normalized) != set(SearchLane) or any(value <= 0.0 for value in normalized.values()):
            raise ValueError("portfolio must give every search lane positive weight")
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
            raise ValueError("portfolio weights must sum to one")
        object.__setattr__(self, "portfolio", normalized)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["portfolio"] = {lane.value: weight for lane, weight in self.portfolio.items()}
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CampaignPolicy:
        return cls(**dict(value))


@dataclass(frozen=True)
class ControllerTick:
    campaign_id: str
    scheduled_jobs: tuple[str, ...]
    incident_nodes: tuple[str, ...]
    active_branches: int
    completed_branches: int
    campaign_status: CampaignStatus
    lease_acquired: bool = True


def create_controlled_campaign(
    store: ResearchStore,
    name: str,
    question: str,
    *,
    policy: CampaignPolicy = CampaignPolicy(),
    campaign_id: str | None = None,
    now: float | None = None,
) -> str:
    return store.create_campaign(
        name,
        question,
        config={"controller_version": 1, "policy": policy.to_dict()},
        campaign_id=campaign_id,
        now=now,
    )


class CampaignController:
    def __init__(
        self,
        *,
        repository: str | Path,
        store: ResearchStore,
        owner: str | None = None,
        lease_seconds: float = 30.0,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.store = store
        self.owner = owner or f"{socket.gethostname()}-{id(self)}"
        self.lease_seconds = lease_seconds

    def tick(self, campaign_id: str) -> ControllerTick:
        if not self.store.acquire_controller_lease(
            campaign_id,
            self.owner,
            lease_seconds=self.lease_seconds,
        ):
            campaign = self.store.get_campaign(campaign_id)
            return ControllerTick(
                campaign_id, (), (), 0, 0, campaign.status, lease_acquired=False
            )
        scheduled: list[str] = []
        incidents: list[str] = []
        try:
            campaign = self.store.get_campaign(campaign_id)
            policy = self._policy(campaign.config)
            usage = self.store.campaign_usage(campaign_id)
            if self._budget_exhausted(policy, usage):
                campaign = self.store.set_campaign_status(
                    campaign_id, CampaignStatus.BUDGET_EXHAUSTED
                )
                return ControllerTick(campaign_id, (), (), 0, 0, campaign.status)

            jobs = self.store.list_jobs(campaign_id=campaign_id)
            capacity = max(
                0,
                policy.max_inflight_jobs
                - sum(job.status not in _TERMINAL for job in jobs),
            )

            def enqueue(**kwargs: Any) -> Job | None:
                nonlocal capacity
                if capacity <= 0:
                    return None
                job_id = kwargs["job_id"]
                try:
                    existing = self.store.get_job(job_id)
                except KeyError:
                    self.store.enqueue_job(campaign_id=campaign_id, **kwargs)
                    existing = self.store.get_job(job_id)
                    scheduled.append(job_id)
                    jobs.append(existing)
                    capacity -= 1
                return existing

            prior = _find_stage(jobs, "prior_art")
            if prior is None:
                enqueue(
                    **self._prior_art_job(
                        campaign_id, campaign.question, policy.primary_provider
                    )
                )
                prior = _find_stage(jobs, "prior_art")
            if prior is None or prior.status not in _TERMINAL:
                return self._result(campaign_id, scheduled, incidents, jobs)
            if prior.status is not JobStatus.COMPLETED:
                incident = self._incident(campaign_id, None, (prior,), "prior-art bootstrap failed")
                incidents.append(incident)
                self.store.set_campaign_status(campaign_id, CampaignStatus.PAUSED)
                return self._result(campaign_id, scheduled, incidents, jobs)

            branches = _group_branches(jobs)
            for branch_id in sorted(branches):
                branch_jobs = branches[branch_id]
                failure = self._branch_failure(branch_jobs)
                if failure:
                    incident = self._incident(campaign_id, branch_id, failure, "branch failed")
                    incidents.append(incident)
                    continue
                finding = branch_jobs.get("finding")
                if finding is not None and finding.status is JobStatus.COMPLETED:
                    continue
                self._expand_branch(
                    campaign_id,
                    campaign.question,
                    branch_id,
                    branch_jobs,
                    policy,
                    enqueue,
                )

            jobs = self.store.list_jobs(campaign_id=campaign_id)
            branches = _group_branches(jobs)
            completed = sum(_branch_completed(branch) for branch in branches.values())
            failed = sum(bool(self._branch_failure(branch)) for branch in branches.values())
            active = len(branches) - completed - failed
            while (
                capacity > 0
                and active < policy.concurrent_branches
                and len(branches) < policy.max_branches
            ):
                index = len(branches) + 1
                branch_id = f"branch-{index:04d}"
                lane = _choose_lane(branches, policy.portfolio)
                dependencies = self._branch_context(jobs, prior)
                hypothesis = enqueue(
                    **self._hypothesis_job(
                        campaign_id,
                        campaign.question,
                        branch_id,
                        lane,
                        dependencies,
                        policy,
                    )
                )
                if hypothesis is None:
                    break
                branches[branch_id] = {"hypothesis": hypothesis}
                active += 1

            jobs = self.store.list_jobs(campaign_id=campaign_id)
            self._schedule_synthesis(
                campaign_id, campaign.question, jobs, policy, enqueue
            )
            jobs = self.store.list_jobs(campaign_id=campaign_id)
            branches = _group_branches(jobs)
            completed = sum(_branch_completed(branch) for branch in branches.values())
            failed = sum(bool(self._branch_failure(branch)) for branch in branches.values())
            active = len(branches) - completed - failed
            nonterminal = [job for job in jobs if job.status not in _TERMINAL]
            if len(branches) >= policy.max_branches and not nonterminal:
                terminal_status = (
                    CampaignStatus.COMPLETED if completed > 0 else CampaignStatus.PAUSED
                )
                campaign = self.store.set_campaign_status(campaign_id, terminal_status)
            else:
                campaign = self.store.get_campaign(campaign_id)
            return ControllerTick(
                campaign_id,
                tuple(scheduled),
                tuple(dict.fromkeys(incidents)),
                active,
                completed,
                campaign.status,
            )
        finally:
            self.store.release_controller_lease(campaign_id, self.owner)

    @staticmethod
    def _policy(config: Mapping[str, Any]) -> CampaignPolicy:
        if config.get("controller_version") != 1 or not isinstance(config.get("policy"), dict):
            raise ValueError("campaign does not contain controller policy version 1")
        return CampaignPolicy.from_dict(config["policy"])

    @staticmethod
    def _budget_exhausted(policy: CampaignPolicy, usage: Mapping[str, Any]) -> bool:
        return (
            usage["provider_attempts"] >= policy.max_provider_attempts
            or usage["local_run_attempts"] >= policy.max_local_run_attempts
            or usage["elapsed_seconds"] >= policy.max_wall_seconds
        )

    def _schema(self, name: str) -> dict[str, Any]:
        path = self.repository / "campaigns" / "schemas" / f"{name}.schema.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _base_payload(
        self, *, stage: str, branch_id: str | None, lane: SearchLane | None
    ) -> dict[str, Any]:
        return {
            "scheduler": {
                "controller_version": 1,
                "stage": stage,
                "branch_id": branch_id,
                "lane": None if lane is None else lane.value,
            }
        }

    def _prior_art_job(
        self, campaign_id: str, question: str, provider: str
    ) -> dict[str, Any]:
        prompt = (
            "Map prior work for this research question using primary sources and direct URLs: "
            f"{question}\nRead design/00_foundations.md and design/10_benchmark_architecture.md. "
            "Separate direct mechanism collisions from adjacent ideas. Do not propose an algorithm."
        )
        payload = {
            **self._base_payload(stage="prior_art", branch_id=None, lane=None),
            "prompt": prompt,
            "schema": self._schema("prior_art"),
            "result_kind": "PriorArtClaim",
        }
        return _agent_job(
            campaign_id,
            "prior-art",
            provider=provider,
            role="mapper",
            mode=JobMode.READ,
            payload=payload,
        )

    def _hypothesis_job(
        self,
        campaign_id: str,
        question: str,
        branch_id: str,
        lane: SearchLane,
        dependencies: tuple[str, ...],
        policy: CampaignPolicy,
    ) -> dict[str, Any]:
        lane_instruction = {
            SearchLane.EXPLOIT: "extend a replicated or strongest observed mechanism",
            SearchLane.FALSIFY: "turn a leading assumption into a competing mechanistic hypothesis",
            SearchLane.EXPLORE: "choose a mechanism distant from active branches",
            SearchLane.TRANSFER: "predict a mechanism that should transfer across action renderings or families",
            SearchLane.SYNTHESIZE: "resolve a contradiction by proposing one discriminating mechanism",
        }[lane]
        prompt = (
            f"Research question: {question}\nSearch lane: {lane.value}; {lane_instruction}. "
            "Return one typed hypothesis with quantitative predictions and a real falsifier. "
            "It must target controlled Neural FactorLab axes and fit the published compact-neural "
            "consumer-GPU envelope. Do not write code."
        )
        payload = {
            **self._base_payload(stage="hypothesis", branch_id=branch_id, lane=lane),
            "prompt": prompt,
            "schema": self._schema("hypothesis"),
            "result_kind": "Hypothesis",
            "dependency_edge": "motivates",
        }
        return _agent_job(
            campaign_id,
            f"{branch_id}-hypothesis",
            provider=policy.primary_provider,
            role="theorist",
            mode=JobMode.READ,
            payload=payload,
            dependencies=dependencies,
        )

    def _expand_branch(
        self,
        campaign_id: str,
        question: str,
        branch_id: str,
        branch: dict[str, Job],
        policy: CampaignPolicy,
        enqueue: Any,
    ) -> None:
        hypothesis = branch.get("hypothesis")
        if hypothesis is None or hypothesis.status is not JobStatus.COMPLETED:
            return
        lane = SearchLane(hypothesis.payload["scheduler"]["lane"])
        if "skeptic" not in branch:
            payload = {
                **self._base_payload(stage="skeptic", branch_id=branch_id, lane=lane),
                "prompt": (
                    "Try to falsify the dependency hypothesis. Name confounds, prior-art collision, "
                    "and the cheapest failure tests. You cannot approve promotion."
                ),
                "schema": self._schema("falsification"),
                "result_kind": "Analysis",
                "dependency_edge": "tests",
            }
            enqueue(
                **_agent_job(
                    campaign_id,
                    f"{branch_id}-skeptic",
                    provider=policy.independent_provider,
                    role="skeptic",
                    mode=JobMode.READ,
                    payload=payload,
                    dependencies=(hypothesis.id,),
                )
            )
        if "probe" not in branch:
            payload = {
                **self._base_payload(stage="probe", branch_id=branch_id, lane=lane),
                "prompt": (
                    "Design the smallest matched FactorLab intervention that distinguishes the "
                    "hypothesis from at least one competing explanation. Respect fixed evaluator budgets."
                ),
                "schema": self._schema("probe_plan"),
                "result_kind": "ProbePlan",
                "dependency_edge": "tests",
            }
            enqueue(
                **_agent_job(
                    campaign_id,
                    f"{branch_id}-probe",
                    provider=policy.primary_provider,
                    role="probe_designer",
                    mode=JobMode.READ,
                    payload=payload,
                    dependencies=(hypothesis.id,),
                )
            )
        branch.update(_group_branches(self.store.list_jobs(campaign_id=campaign_id)).get(branch_id, {}))
        probe = branch.get("probe")
        if probe is None or probe.status is not JobStatus.COMPLETED:
            return
        for stage, provider, role, variant in (
            ("implementation", policy.primary_provider, "implementer", "primary"),
            ("replication_implementation", policy.independent_provider, "replicator", "replica"),
        ):
            if stage in branch:
                continue
            allowed = f"candidates/{campaign_id}/{branch_id}/{variant}"
            payload = {
                **self._base_payload(stage=stage, branch_id=branch_id, lane=lane),
                "prompt": (
                    f"Implement an RL candidate only under {allowed}/. Read "
                    "design/40_candidate_protocol.md. The program must implement a compact neural "
                    "RL policy, speak RLX batched JSONL v2, write a bounded binary checkpoint, "
                    "declare its exact neural model manifest, and use learner-visible data only. "
                    "Return candidate_argv exactly as ['python', '"
                    f"{allowed}/candidate.py']. Do not edit benchmark, evaluator, tests, schemas, or design."
                ),
                "schema": self._schema("implementation"),
                "result_kind": "Implementation",
                "allowed_prefixes": [allowed],
                "protected_prefixes": list(PROTECTED_PATHS),
                "require_changes": True,
                "dependency_edge": "implements",
            }
            enqueue(
                **_agent_job(
                    campaign_id,
                    f"{branch_id}-{stage}",
                    provider=provider,
                    role=role,
                    mode=JobMode.WRITE,
                    payload=payload,
                    dependencies=(hypothesis.id, probe.id),
                    max_attempts=2,
                )
            )
        branch.update(_group_branches(self.store.list_jobs(campaign_id=campaign_id)).get(branch_id, {}))
        for implementation_stage, run_stage in (
            ("implementation", "primary_run"),
            ("replication_implementation", "replication_run"),
        ):
            implementation = branch.get(implementation_stage)
            if (
                implementation is not None
                and implementation.status is JobStatus.COMPLETED
                and run_stage not in branch
            ):
                enqueue(
                    **self._evaluation_job(
                        campaign_id, branch_id, lane, implementation, run_stage, policy
                    )
                )
        branch.update(_group_branches(self.store.list_jobs(campaign_id=campaign_id)).get(branch_id, {}))
        skeptic = branch.get("skeptic")
        primary_run = branch.get("primary_run")
        replication_run = branch.get("replication_run")
        if (
            "finding" not in branch
            and skeptic is not None
            and skeptic.status is JobStatus.COMPLETED
            and _valid_measurement_job(self.store, primary_run)
            and _valid_measurement_job(self.store, replication_run)
        ):
            payload = {
                **self._base_payload(stage="finding", branch_id=branch_id, lane=lane),
                "prompt": (
                    f"Analyze the primary and independent replication for: {question}. "
                    "Use only recorded measurements. Report contradictions and uncertainty; "
                    "a negative or inconclusive result is valid scientific completion."
                ),
                "schema": self._schema("finding"),
                "result_kind": "Finding",
                "dependency_edge": "produces",
            }
            enqueue(
                **_agent_job(
                    campaign_id,
                    f"{branch_id}-finding",
                    provider=policy.primary_provider,
                    role="analyst",
                    mode=JobMode.READ,
                    payload=payload,
                    dependencies=(primary_run.id, replication_run.id, skeptic.id),
                )
            )

    def _evaluation_job(
        self,
        campaign_id: str,
        branch_id: str,
        lane: SearchLane,
        implementation: Job,
        stage: str,
        policy: CampaignPolicy,
    ) -> dict[str, Any]:
        output = _role_output(self.store, implementation)
        candidate_argv = output.get("candidate_argv")
        variant = "primary" if stage == "primary_run" else "replica"
        allowed = f"candidates/{campaign_id}/{branch_id}/{variant}"
        normalized_candidate = _validate_candidate_argv(candidate_argv, allowed)
        evaluator_argv = [
            sys.executable,
            "-m",
            "rlx_agents.evaluate",
            "--horizon",
            str(policy.evaluation_horizon),
            "--n-factors",
            str(policy.evaluation_factors),
            "--levels-per-factor",
            str(policy.evaluation_levels_per_factor),
            "--signal-dim",
            str(policy.evaluation_signal_dim),
            "--context-dim",
            str(policy.evaluation_context_dim),
            "--state-dim",
            str(policy.evaluation_state_dim),
            "--teacher-hidden-dim",
            str(policy.evaluation_teacher_hidden_dim),
            "--max-causal-lag",
            str(policy.evaluation_horizon),
            "--memory-lag",
            "0",
            "--reward-events",
            "1",
            "--conflict-strength",
            "0.75",
            "--terminal-state-weight",
            "1.0",
            "--training-episodes",
            str(policy.evaluation_training_episodes),
            "--training-batch-size",
            str(policy.evaluation_training_batch_size),
            "--training-trials",
            str(policy.evaluation_training_trials),
            "--public-worlds",
            str(policy.evaluation_public_worlds),
            "--heldout-worlds",
            str(policy.evaluation_heldout_worlds),
            "--wall-seconds",
            str(policy.evaluation_wall_seconds),
            "--max-parameters",
            str(policy.evaluation_max_parameters),
            "--suite-namespace",
            f"{campaign_id}-factorlab-long-v1-neural",
            "--candidate",
            *normalized_candidate,
        ]
        payload = {
            **self._base_payload(stage=stage, branch_id=branch_id, lane=lane),
            "argv": evaluator_argv,
            "cwd_from_dependency_worktree": True,
            "timeout_seconds": policy.evaluation_wall_seconds + 60.0,
            "output_bytes": 2_000_000,
            "env": {"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"},
            "candidate_evaluation": True,
            "stdout_schema": self._schema("candidate_measurement"),
            "result_kind": "Replication" if stage == "replication_run" else "Run",
            "dependency_edge": "replicates" if stage == "replication_run" else "produces",
            "allowed_prefixes": [allowed],
            "protected_prefixes": list(PROTECTED_PATHS),
        }
        return _execute_job(
            campaign_id,
            f"{branch_id}-{stage}",
            payload=payload,
            dependencies=(implementation.id,),
        )

    def _branch_failure(self, branch: Mapping[str, Job]) -> tuple[Job, ...]:
        failed = tuple(
            job
            for job in branch.values()
            if job.status in {JobStatus.FAILED, JobStatus.BLOCKED, JobStatus.CANCELLED}
        )
        if failed:
            return failed
        invalid_implementations: list[Job] = []
        for stage in ("implementation", "replication_implementation"):
            job = branch.get(stage)
            if job is None or job.status is not JobStatus.COMPLETED:
                continue
            allowed = job.payload.get("allowed_prefixes", [])
            try:
                if not isinstance(allowed, list) or len(allowed) != 1:
                    raise ValueError("implementation allowance is malformed")
                _validate_candidate_argv(
                    _role_output(self.store, job).get("candidate_argv"), allowed[0]
                )
            except (KeyError, ValueError):
                invalid_implementations.append(job)
        if invalid_implementations:
            return tuple(invalid_implementations)
        invalid_runs = tuple(
            job
            for stage, job in branch.items()
            if stage in {"primary_run", "replication_run"}
            and job.status is JobStatus.COMPLETED
            and not _valid_measurement_job(self.store, job)
        )
        return invalid_runs

    def _incident(
        self,
        campaign_id: str,
        branch_id: str | None,
        jobs: tuple[Job, ...],
        reason: str,
    ) -> str:
        node = self.store.add_node(
            "Incident",
            {
                "campaign_id": campaign_id,
                "branch_id": branch_id,
                "reason": reason,
                "jobs": [
                    {
                        "job_id": job.id,
                        "status": job.status.value,
                        "error_class": job.error_class,
                        "error_detail": job.error_detail,
                    }
                    for job in jobs
                ],
            },
            created_by=f"controller:{self.owner}",
        )
        return node.id

    @staticmethod
    def _branch_context(jobs: list[Job], prior: Job) -> tuple[str, ...]:
        syntheses = [
            job
            for job in jobs
            if (_stage(job) or "").startswith("synthesis-")
            and job.status is JobStatus.COMPLETED
        ]
        if syntheses:
            return (prior.id, syntheses[-1].id)
        findings = [
            job for job in jobs if _stage(job) == "finding" and job.status is JobStatus.COMPLETED
        ]
        return (prior.id, *(job.id for job in findings[-3:]))

    def _schedule_synthesis(
        self,
        campaign_id: str,
        question: str,
        jobs: list[Job],
        policy: CampaignPolicy,
        enqueue: Any,
    ) -> None:
        findings = [
            job for job in jobs if _stage(job) == "finding" and job.status is JobStatus.COMPLETED
        ]
        batch = len(findings) // policy.synthesis_interval_findings
        if batch < 1:
            return
        stage = f"synthesis-{batch:04d}"
        if _find_stage(jobs, stage) is not None:
            return
        dependencies = tuple(
            job.id for job in findings[-policy.synthesis_interval_findings :]
        )
        payload = {
            **self._base_payload(stage=stage, branch_id=None, lane=SearchLane.SYNTHESIZE),
            "prompt": (
                f"Synthesize the latest immutable findings for {question}. Identify contradictions, "
                "dead ends, replicated mechanisms, and the next highest-information uncertainties."
            ),
            "schema": self._schema("synthesis"),
            "result_kind": "Synthesis",
            "dependency_edge": "informs",
        }
        enqueue(
            **_agent_job(
                campaign_id,
                stage,
                provider=policy.primary_provider,
                role="synthesizer",
                mode=JobMode.READ,
                payload=payload,
                dependencies=dependencies,
            )
        )

    def _result(
        self,
        campaign_id: str,
        scheduled: list[str],
        incidents: list[str],
        jobs: list[Job],
    ) -> ControllerTick:
        branches = _group_branches(jobs)
        completed = sum(_branch_completed(branch) for branch in branches.values())
        failed = sum(bool(self._branch_failure(branch)) for branch in branches.values())
        return ControllerTick(
            campaign_id,
            tuple(scheduled),
            tuple(dict.fromkeys(incidents)),
            len(branches) - completed - failed,
            completed,
            self.store.get_campaign(campaign_id).status,
        )


_TERMINAL = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.BLOCKED,
    JobStatus.CANCELLED,
}


def _job_id(campaign_id: str, key: str) -> str:
    digest = hashlib.sha256(f"{campaign_id}|{key}".encode()).hexdigest()[:24]
    return f"job-{digest}"


def _agent_job(
    campaign_id: str,
    key: str,
    *,
    provider: str,
    role: str,
    mode: JobMode,
    payload: Mapping[str, Any],
    dependencies: tuple[str, ...] = (),
    max_attempts: int = 3,
) -> dict[str, Any]:
    return {
        "operation": "agent",
        "provider": provider,
        "role": role,
        "mode": mode,
        "payload": payload,
        "dependencies": dependencies,
        "max_attempts": max_attempts,
        "job_id": _job_id(campaign_id, key),
    }


def _execute_job(
    campaign_id: str,
    key: str,
    *,
    payload: Mapping[str, Any],
    dependencies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "operation": "execute",
        "provider": "local",
        "role": "evaluator",
        "mode": JobMode.EXECUTE,
        "payload": payload,
        "dependencies": dependencies,
        "max_attempts": 1,
        "job_id": _job_id(campaign_id, key),
    }


def _stage(job: Job) -> str | None:
    scheduler = job.payload.get("scheduler")
    return scheduler.get("stage") if isinstance(scheduler, dict) else None


def _find_stage(jobs: list[Job], stage: str) -> Job | None:
    return next((job for job in jobs if _stage(job) == stage), None)


def _group_branches(jobs: list[Job]) -> dict[str, dict[str, Job]]:
    branches: dict[str, dict[str, Job]] = {}
    for job in jobs:
        scheduler = job.payload.get("scheduler")
        if not isinstance(scheduler, dict):
            continue
        branch_id = scheduler.get("branch_id")
        stage = scheduler.get("stage")
        if isinstance(branch_id, str) and isinstance(stage, str):
            branches.setdefault(branch_id, {})[stage] = job
    return branches


def _branch_completed(branch: Mapping[str, Job]) -> bool:
    finding = branch.get("finding")
    return finding is not None and finding.status is JobStatus.COMPLETED


def _choose_lane(
    branches: Mapping[str, Mapping[str, Job]],
    portfolio: Mapping[SearchLane, float],
) -> SearchLane:
    counts = {lane: 0 for lane in SearchLane}
    for branch in branches.values():
        hypothesis = branch.get("hypothesis")
        if hypothesis is not None:
            counts[SearchLane(hypothesis.payload["scheduler"]["lane"])] += 1
    return min(SearchLane, key=lambda lane: (counts[lane] / portfolio[lane], list(SearchLane).index(lane)))


def _role_output(store: ResearchStore, job: Job) -> Mapping[str, Any]:
    if job.result_node_id is None:
        raise ValueError(f"job {job.id} has no result node")
    node = store.get_node(job.result_node_id)
    output = node.payload.get("role_output")
    if not isinstance(output, dict):
        raise ValueError(f"job {job.id} has no structured role output")
    return output


def _validate_candidate_argv(value: Any, allowed_prefix: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2 or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError("candidate_argv must be ['python', relative_script]")
    if PurePosixPath(value[0]).name not in {"python", "python3"}:
        raise ValueError("candidate executable must be python or python3")
    script = PurePosixPath(value[1])
    if script.is_absolute() or ".." in script.parts or script.suffix != ".py":
        raise ValueError("candidate script must be a safe relative Python path")
    prefix = PurePosixPath(allowed_prefix)
    if script != prefix and prefix not in script.parents:
        raise ValueError("candidate script is outside the branch allowance")
    return sys.executable, script.as_posix()


def _valid_measurement_job(store: ResearchStore, job: Job | None) -> bool:
    if job is None or job.status is not JobStatus.COMPLETED or job.result_node_id is None:
        return False
    node = store.get_node(job.result_node_id)
    measurement = node.payload.get("measurement")
    return (
        node.payload.get("run_status") == "complete"
        and isinstance(measurement, dict)
        and measurement.get("status") == "complete"
        and measurement.get("heldout_identifiers_exposed") is False
    )
