from __future__ import annotations

from pathlib import Path

from rlx_lab.campaign import CampaignController, CampaignPolicy, create_controlled_campaign
from rlx_lab.models import CampaignStatus, JobStatus
from rlx_lab.store import ResearchStore


REPOSITORY = Path(__file__).resolve().parents[2]


def _complete_next(store: ResearchStore, provider: str):
    job = store.claim_job(
        f"worker-{provider}", lease_seconds=30, providers=(provider,)
    )
    assert job is not None
    store.start_job(job.id, f"worker-{provider}")
    stage = job.payload["scheduler"]["stage"]
    if job.operation == "execute":
        payload = {
            "run_status": "complete",
            "measurement": {
                "status": "complete",
                "heldout_identifiers_exposed": False,
            },
        }
    elif stage in {"implementation", "replication_implementation"}:
        allowed = job.payload["allowed_prefixes"][0]
        payload = {
            "role_output": {
                "candidate_argv": ["python", f"{allowed}/candidate.py"],
                "summary": "independent candidate implementation for controlled evaluation",
                "files": [f"{allowed}/candidate.py"],
                "mechanism_invariants": ["uses learner-visible observations only"],
                "self_checks": ["protocol handshake passes"],
            }
        }
    else:
        payload = {"role_output": {"stage": stage, "valid": True}}
    node = store.add_node(
        str(job.payload.get("result_kind", "Run")), payload, created_by=job.id
    )
    store.complete_job(job.id, f"worker-{provider}", node.id)
    return job


def test_controller_expands_independent_evidence_dag_and_completes_campaign(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    policy = CampaignPolicy(
        concurrent_branches=1,
        max_branches=1,
        max_inflight_jobs=4,
        synthesis_interval_findings=10,
    )
    campaign = create_controlled_campaign(
        store,
        "campaign",
        "Can structured credit mechanisms survive terminal vector rewards?",
        policy=policy,
    )
    controller = CampaignController(repository=REPOSITORY, store=store, owner="controller")

    first = controller.tick(campaign)
    assert len(first.scheduled_jobs) == 1
    assert store.get_job(first.scheduled_jobs[0]).role == "mapper"
    _complete_next(store, "codex")

    second = controller.tick(campaign)
    hypothesis = store.get_job(second.scheduled_jobs[0])
    assert hypothesis.role == "theorist"
    _complete_next(store, "codex")

    third = controller.tick(campaign)
    assert {store.get_job(job_id).role for job_id in third.scheduled_jobs} == {
        "skeptic",
        "probe_designer",
    }
    skeptic = _complete_next(store, "claude")
    probe = _complete_next(store, "codex")

    fourth = controller.tick(campaign)
    implementation_jobs = [store.get_job(job_id) for job_id in fourth.scheduled_jobs]
    assert {job.role for job in implementation_jobs} == {"implementer", "replicator"}
    replicator = next(job for job in implementation_jobs if job.role == "replicator")
    assert set(replicator.dependencies) == {hypothesis.id, probe.id}
    assert all(job.role != "implementer" for job in (store.get_job(dep) for dep in replicator.dependencies))
    _complete_next(store, "codex")
    _complete_next(store, "claude")

    fifth = controller.tick(campaign)
    run_jobs = [store.get_job(job_id) for job_id in fifth.scheduled_jobs]
    assert len(run_jobs) == 2
    assert all(job.operation == "execute" for job in run_jobs)
    assert all(job.payload["candidate_evaluation"] is True for job in run_jobs)
    namespaces = {
        job.payload["argv"][job.payload["argv"].index("--suite-namespace") + 1]
        for job in run_jobs
    }
    assert namespaces == {f"{campaign}-factorlab-v0"}
    assert all("--training-trials" in job.payload["argv"] for job in run_jobs)
    _complete_next(store, "local")
    _complete_next(store, "local")

    sixth = controller.tick(campaign)
    finding = store.get_job(sixth.scheduled_jobs[0])
    assert finding.role == "analyst"
    assert set(finding.dependencies) == {
        *(job.id for job in run_jobs),
        skeptic.id,
    }
    _complete_next(store, "codex")

    final = controller.tick(campaign)
    assert final.completed_branches == 1
    assert final.active_branches == 0
    assert final.campaign_status is CampaignStatus.COMPLETED
    assert store.get_campaign(campaign).status is CampaignStatus.COMPLETED


def test_failed_branch_creates_incident_and_does_not_count_as_scientific_completion(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    campaign = create_controlled_campaign(
        store,
        "campaign",
        "Does the proposed mechanism survive falsification?",
        policy=CampaignPolicy(
            concurrent_branches=1,
            max_branches=1,
            max_inflight_jobs=2,
        ),
    )
    controller = CampaignController(repository=REPOSITORY, store=store, owner="controller")
    controller.tick(campaign)
    _complete_next(store, "codex")
    controller.tick(campaign)
    hypothesis = store.claim_job("dead", lease_seconds=30, providers=("codex",))
    store.start_job(hypothesis.id, "dead")
    store.fail_job(
        hypothesis.id,
        "dead",
        error_class="invalid_output",
        detail="no quantitative prediction",
        retryable=False,
    )

    result = controller.tick(campaign)

    assert result.completed_branches == 0
    assert result.incident_nodes
    assert store.get_node(result.incident_nodes[0]).kind == "Incident"
    assert result.campaign_status is CampaignStatus.PAUSED


def test_claim_enforces_hard_campaign_attempt_budget(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    campaign = create_controlled_campaign(
        store,
        "campaign",
        "Can the scheduler enforce provider budgets atomically?",
        policy=CampaignPolicy(max_provider_attempts=1),
    )
    for index in range(2):
        store.enqueue_job(
            operation="agent",
            provider="codex",
            role="mapper",
            mode="read",
            payload={},
            campaign_id=campaign,
            job_id=f"job-{index}",
        )

    assert store.claim_job("first", lease_seconds=30, providers=("codex",)) is not None
    assert store.claim_job("second", lease_seconds=30, providers=("codex",)) is None
    assert store.get_campaign(campaign).status is CampaignStatus.BUDGET_EXHAUSTED
    assert store.get_job("job-1").status is JobStatus.CANCELLED
