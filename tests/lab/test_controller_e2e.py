from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from rlx_lab.artifacts import ArtifactStore
from rlx_lab.campaign import CampaignController, CampaignPolicy, create_controlled_campaign
from rlx_lab.models import CampaignStatus
from rlx_lab.providers import FakeProvider
from rlx_lab.secrets import CampaignSecretStore
from rlx_lab.store import ResearchStore
from rlx_lab.worker import Worker
from rlx_lab.worktrees import WorktreeManager


SOURCE_REPOSITORY = Path(__file__).resolve().parents[2]


CANDIDATE = r"""
import hashlib
import json
import os
import sys

factors = 1
for line in sys.stdin:
    message = json.loads(line)
    kind = message["type"]
    if kind == "init":
        assert "RLX_FACTORLAB_SUITE_KEY_FILE" not in os.environ
        factors = len(message["task_spec"]["action_spec"]["factors"])
        checkpoint = message.get("checkpoint")
        if checkpoint:
            content = open(os.path.join(os.environ["RLX_CANDIDATE_SCRATCH"], checkpoint["artifact"]), "rb").read()
            assert hashlib.sha256(content).hexdigest() == checkpoint["sha256"]
        print(json.dumps({
            "type": "ready",
            "model_manifest": {
                "model_family": "neural_policy",
                "architecture": "integration_residual_policy",
                "framework": "fixture",
                "trainable_parameters": 32,
                "recurrent": False,
                "device": "cpu"
            }
        }), flush=True)
    elif kind == "act_batch":
        print(json.dumps({
            "type": "actions",
            "actions": [[0] * factors for _ in message["observations"]],
        }), flush=True)
    elif kind == "checkpoint":
        content = b"integration-neural-checkpoint"
        path = os.path.join(os.environ["RLX_CANDIDATE_SCRATCH"], "model.bin")
        open(path, "wb").write(content)
        print(json.dumps({
            "type": "checkpoint",
            "artifact": "model.bin",
            "sha256": hashlib.sha256(content).hexdigest()
        }), flush=True)
    elif kind == "close":
        print(json.dumps({"type": "closed"}), flush=True)
        break
"""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    for name in ("src", "campaigns", "design"):
        shutil.copytree(SOURCE_REPOSITORY / name, repository / name)
    shutil.copy2(SOURCE_REPOSITORY / "pyproject.toml", repository / "pyproject.toml")
    (repository / "README.md").write_text("controller integration fixture\n")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def test_controller_runs_real_isolated_primary_and_replication_evaluations(tmp_path) -> None:
    repository = _repository(tmp_path)
    runtime = tmp_path / "runtime"
    store = ResearchStore(runtime / "state.db")
    artifacts = ArtifactStore(runtime / "artifacts")
    worktrees = WorktreeManager(repository, tmp_path / "worktrees")
    secrets = CampaignSecretStore(runtime / "secrets")
    policy = CampaignPolicy(
        concurrent_branches=1,
        max_branches=1,
        max_inflight_jobs=4,
        synthesis_interval_findings=10,
        evaluation_horizon=8,
        evaluation_factors=2,
        evaluation_training_episodes=32,
        evaluation_training_trials=1,
        evaluation_public_worlds=4,
        evaluation_heldout_worlds=2,
        evaluation_wall_seconds=20,
    )
    campaign = create_controlled_campaign(
        store,
        "integration",
        "Can an isolated candidate reproduce a controlled FactorLab effect?",
        policy=policy,
    )
    secrets.ensure(campaign)

    def responder(request):
        if request.role == "mapper":
            return {
                "search_scope": "primary work on delayed credit and structured multiobjective control",
                "sources": [
                    {
                        "title": f"Source {index}",
                        "url": f"https://example.org/paper-{index}",
                        "year": 2020 + index,
                        "mechanism": "a directly testable credit assignment mechanism",
                        "relationship": "background",
                    }
                    for index in range(3)
                ],
                "crowded_mechanisms": ["return decomposition"],
                "unresolved_gaps": ["interaction with structured vector actions"],
                "queries_run": ["delayed credit vector reward structured action"],
            }
        if request.role == "theorist":
            return {
                "title": "Cue-aligned factorized control",
                "mechanism": "factor-local choices use public cues to avoid enumerating the joint action",
                "scope": ["FactorLab factored discrete terminal reward"],
                "predictions": [
                    {
                        "comparison": "candidate versus opposite cue policy",
                        "metric": "normalized objective zero",
                        "direction": "increase",
                        "minimum_effect": 0.2,
                    }
                ],
                "falsifier": "held-out normalized utility fails to exceed 0.7",
                "closest_known_work": ["branching action value functions"],
                "first_probe": "matched terminal-reward evaluation on two held-out worlds",
                "uncertainties": ["whether the cue mapping transfers"],
            }
        if request.role == "skeptic":
            return {
                "weakest_assumption": "the learner-visible cue contains sufficient factor information",
                "confounds": ["hard-coded cue semantics", "single preference evaluation"],
                "cheapest_falsifiers": ["flip the cue-action mapping"],
                "prior_art_collision": "possible branching-policy overlap",
                "recommendation": "proceed",
                "reason": "the held-out process boundary still provides a useful mechanics test",
            }
        if request.role == "probe_designer":
            return {
                "question": "Does cue-aligned factorization improve held-out terminal vector return?",
                "competing_explanations": ["factorization", "seed memorization"],
                "intervention": "restart from one checkpoint on separately generated worlds",
                "controls": ["fixed transition and episode budget"],
                "measurements": ["normalized component return"],
                    "budget": {
                        "environment_steps": 16,
                        "wall_seconds": 20,
                        "accelerator_seconds": 0,
                        "max_trainable_parameters": 1000,
                        "training_seeds": 1,
                },
                "decision_table": ["effect >= 0.2: support", "effect < 0.2: contradict"],
            }
        if request.role in {"implementer", "replicator"}:
            match = re.search(r"only under ([^/]+(?:/[^/]+){3})/", request.prompt)
            assert match is not None
            allowed = match.group(1)
            target = request.cwd / allowed / "candidate.py"
            target.parent.mkdir(parents=True)
            target.write_text(CANDIDATE)
            return {
                "summary": "implemented a standalone JSONL candidate for hidden-world evaluation",
                "candidate_argv": ["python", f"{allowed}/candidate.py"],
                "files": [f"{allowed}/candidate.py"],
                "model_manifest": {
                    "architecture": "integration_residual_policy",
                    "framework": "fixture",
                    "trainable_parameters": 32,
                    "recurrent": False,
                    "device": "cpu",
                },
                "mechanism_invariants": ["no evaluator metadata access"],
                "self_checks": ["JSONL messages are flushed"],
            }
        if request.role == "analyst":
            return {
                "statement": "primary and independent candidates complete the hidden-world probe",
                "relation": "supports",
                "effect": 1.0,
                "uncertainty": 0.0,
                "evidence_run_ids": ["primary-run", "replication-run"],
                "confounds": ["protocol fixture is a mechanism probe"],
                "next_information_needed": "replace fixture with a genuinely learned candidate",
            }
        raise AssertionError(request.role)

    worker = Worker(
        worker_id="integration-worker",
        repository=repository,
        store=store,
        artifacts=artifacts,
        providers={"codex": FakeProvider(responder), "claude": FakeProvider(responder)},
        worktrees=worktrees,
        secrets=secrets,
        runtime_root=runtime,
        lease_seconds=30,
    )
    controller = CampaignController(
        repository=repository, store=store, owner="integration-controller"
    )
    for _ in range(20):
        result = controller.tick(campaign)
        while worker.run_once() is not None:
            pass
        if result.campaign_status is CampaignStatus.COMPLETED:
            break
    else:
        raise AssertionError("campaign did not complete")

    assert store.get_campaign(campaign).status is CampaignStatus.COMPLETED
    measurements = [
        node.payload["measurement"]
        for node in store.list_nodes()
        if node.kind in {"Run", "Replication"}
    ]
    assert len(measurements) == 2
    assert all(measurement["status"] == "complete" for measurement in measurements)
    assert all(0.0 <= measurement["normalized_utility_mean"] <= 1.0 for measurement in measurements)
    assert len({measurement["suite_id"] for measurement in measurements}) == 1
    assert all(measurement["heldout_identifiers_exposed"] is False for measurement in measurements)
    assert list(worktrees.root.iterdir()) == []
    branches = subprocess.run(
        ("git", "branch", "--list", "rlx/job-*"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(branches) == 2
