from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from rlx_lab.models import CampaignStatus, JobMode, JobStatus
from rlx_lab.store import ResearchStore


def test_nodes_are_content_addressed_and_edges_are_idempotent(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    first = store.add_node("Hypothesis", {"mechanism": "x"}, created_by="test", now=1.0)
    second = store.add_node("Hypothesis", {"mechanism": "x"}, created_by="other", now=2.0)
    finding = store.add_node("Finding", {"effect": 0.2}, created_by="test", now=3.0)
    assert first.id == second.id
    assert first.content_hash == second.content_hash
    store.add_edge(first.id, finding.id, "tests", now=4.0)
    store.add_edge(first.id, finding.id, "tests", now=5.0)
    assert len([event for event in store.events(entity_id=first.id) if event["event_type"] == "added"]) == 1
    edge_events = [
        event
        for event in store.events()
        if event["entity_type"] == "edge" and event["event_type"] == "added"
    ]
    assert len(edge_events) == 1


def test_graph_rejects_nonfinite_scientific_payloads(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    with pytest.raises(ValueError):
        store.add_node("Finding", {"effect": float("nan")}, created_by="test")


def test_dependencies_prevent_early_claim_and_then_release(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    parent = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="theorist",
        mode=JobMode.READ,
        payload={"prompt": "parent"},
        job_id="parent",
        now=10.0,
    )
    child = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="skeptic",
        mode=JobMode.READ,
        payload={"prompt": "child"},
        dependencies=(parent,),
        priority=100,
        job_id="child",
        now=10.0,
    )
    claimed = store.claim_job("worker", lease_seconds=30, now=11.0)
    assert claimed is not None and claimed.id == parent
    assert store.claim_job("other", lease_seconds=30, now=11.0) is None

    store.start_job(parent, "worker", now=12.0)
    result = store.add_node("Hypothesis", {"statement": "p"}, created_by=parent, now=13.0)
    store.complete_job(parent, "worker", result.id, now=14.0)
    claimed_child = store.claim_job("other", lease_seconds=30, now=15.0)
    assert claimed_child is not None and claimed_child.id == child


def test_atomic_claims_do_not_lease_the_same_job(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="mapper",
        mode="read",
        payload={},
        job_id="only-job",
        now=1.0,
    )

    def claim(worker):
        job = store.claim_job(worker, lease_seconds=30, now=2.0)
        return None if job is None else job.id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("a", "b")))
    assert results.count("only-job") == 1
    assert results.count(None) == 1


def test_expired_lease_is_recovered_and_attempts_are_bounded(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="mapper",
        mode="read",
        payload={},
        max_attempts=2,
        job_id="job",
        now=0.0,
    )
    assert store.claim_job("dead-1", lease_seconds=5, now=1.0) is not None
    assert store.recover_expired(now=7.0) == ("job",)
    assert store.get_job("job").status == JobStatus.QUEUED
    assert store.claim_job("dead-2", lease_seconds=5, now=8.0) is not None
    assert store.recover_expired(now=14.0) == ("job",)
    job = store.get_job("job")
    assert job.status == JobStatus.FAILED
    assert job.error_class == "expired_lease"


def test_retryable_failure_requeues_but_scientific_completion_does_not(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="analyst",
        mode="read",
        payload={},
        max_attempts=2,
        job_id="job",
        now=0.0,
    )
    store.claim_job("worker", lease_seconds=10, now=1.0)
    store.start_job("job", "worker", now=2.0)
    status = store.fail_job(
        "job",
        "worker",
        error_class="provider_timeout",
        detail="timed out",
        retryable=True,
        retry_delay=5.0,
        now=3.0,
    )
    assert status == JobStatus.QUEUED
    assert store.claim_job("worker", lease_seconds=10, now=7.0) is None
    assert store.claim_job("worker", lease_seconds=10, now=8.0) is not None


def test_failed_dependency_blocks_child(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    parent = store.enqueue_job(
        operation="agent",
        provider="fake",
        role="mapper",
        mode="read",
        payload={},
        max_attempts=1,
        job_id="parent",
        now=0.0,
    )
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="analyst",
        mode="read",
        payload={},
        dependencies=(parent,),
        job_id="child",
        now=0.0,
    )
    store.claim_job("worker", lease_seconds=10, now=1.0)
    store.start_job(parent, "worker", now=2.0)
    store.fail_job(
        parent,
        "worker",
        error_class="invalid_output",
        detail="bad",
        retryable=False,
        now=3.0,
    )
    assert store.claim_job("other", lease_seconds=10, now=4.0) is None
    assert store.get_job("child").status == JobStatus.BLOCKED


def test_campaign_pause_prevents_claim_and_resume_releases_work(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    campaign = store.create_campaign("test", "question", now=1.0)
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="mapper",
        mode="read",
        payload={},
        campaign_id=campaign,
        job_id="job",
        now=2.0,
    )

    paused = store.set_campaign_status(campaign, CampaignStatus.PAUSED, now=3.0)
    assert paused.status is CampaignStatus.PAUSED
    assert store.claim_job("worker", lease_seconds=10, now=4.0) is None
    resumed = store.set_campaign_status(campaign, CampaignStatus.ACTIVE, now=5.0)
    assert resumed.status is CampaignStatus.ACTIVE
    assert store.claim_job("worker", lease_seconds=10, now=6.0).id == "job"


def test_terminal_campaign_status_cancels_queued_jobs(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    campaign = store.create_campaign("test", "question", now=1.0)
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="mapper",
        mode="read",
        payload={},
        campaign_id=campaign,
        job_id="job",
        now=2.0,
    )

    stopped = store.set_campaign_status(campaign, CampaignStatus.STOPPED, now=3.0)

    assert stopped.status is CampaignStatus.STOPPED
    assert store.get_job("job").status is JobStatus.CANCELLED
    with pytest.raises(ValueError):
        store.set_campaign_status(campaign, CampaignStatus.ACTIVE, now=4.0)


def test_controller_lease_is_single_owner_and_recovers_after_expiry(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    campaign = store.create_campaign("test", "question", now=1.0)

    assert store.acquire_controller_lease(
        campaign, "controller-a", lease_seconds=10, now=2.0
    )
    assert not store.acquire_controller_lease(
        campaign, "controller-b", lease_seconds=10, now=3.0
    )
    assert store.acquire_controller_lease(
        campaign, "controller-b", lease_seconds=10, now=13.0
    )
    assert not store.release_controller_lease(campaign, "controller-a")
    assert store.release_controller_lease(campaign, "controller-b")


def test_campaign_usage_counts_actual_attempts(tmp_path):
    store = ResearchStore(tmp_path / "state.db")
    campaign = store.create_campaign("test", "question")
    store.enqueue_job(
        operation="agent",
        provider="fake",
        role="mapper",
        mode="read",
        payload={},
        campaign_id=campaign,
        job_id="job",
    )
    store.claim_job("worker", lease_seconds=10)

    usage = store.campaign_usage(campaign)

    assert usage["provider_attempts"] == 1
    assert usage["local_run_attempts"] == 0
    assert usage["jobs_by_status"]["leased"] == 1
