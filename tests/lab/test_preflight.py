from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import rlx_lab.preflight as preflight
from rlx_bench.factorlab import FactorLabConfig
from rlx_bench.qualification import (
    REQUIRED_QUALIFICATION_CHECKS,
    CheckStatus,
    QualificationCheck,
    make_qualification_report,
)
from rlx_lab.campaign import create_controlled_campaign
from rlx_lab.secrets import CampaignSecretStore
from rlx_lab.store import ResearchStore


ROOT = Path(__file__).resolve().parents[2]


def _git(repository: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repository, check=True, capture_output=True)


def test_preflight_requires_a_committed_rebuild_and_valid_secret(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    for relative in (
        "src/rlx_lab/cli.py",
        "src/rlx_agents/evaluate.py",
        "src/rlx_bench/suite.py",
        "campaigns/schemas/example.json",
        "campaigns/factorlab_long_v1/definition.json",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# fixture\n")
    protocol = json.loads(
        (ROOT / "campaigns/factorlab_long_v1/qualification_protocol.json").read_text()
    )
    protocol_bytes = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    protocol_digest = hashlib.sha256(protocol_bytes).hexdigest()
    evidence_digest = "e" * 64
    qualification_dir = repository / "campaigns/factorlab_long_v1/qualification"
    qualification_dir.mkdir(parents=True)
    protocol_path = repository / "campaigns/factorlab_long_v1/qualification_protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    report_path = qualification_dir / "report.json"
    evidence_refs = (
        f"sha256:{evidence_digest}",
        f"protocol-sha256:{protocol_digest}",
    )
    anchor = dict(protocol["anchor_configuration"])
    anchor["levels_per_factor"] = (anchor["levels_per_factor"],)
    anchor["effects"] = tuple(anchor["effects"])
    qualification_report = make_qualification_report(
        task_id=FactorLabConfig(**anchor).task_id,
        suite_id="fixture-suite",
        benchmark_revision="factorlab-long-v1",
        checks=[
            QualificationCheck(name, CheckStatus.VERIFIED, {}, evidence_refs)
            for name in REQUIRED_QUALIFICATION_CHECKS
        ],
    )
    report_id = qualification_report.report_id
    report_path.write_text(json.dumps(qualification_report.to_dict()))
    (repository / "campaigns/factorlab_long_v1/definition.json").write_text(
        json.dumps(
            {
                "benchmark_revision": "factorlab-long-v1",
                "admitted_tiers": ["factorlab-long-5k-v1"],
                "qualification_reports": {
                    "factorlab-long-5k-v1": {
                        "report_path": "campaigns/factorlab_long_v1/qualification/report.json",
                        "protocol_path": "campaigns/factorlab_long_v1/qualification_protocol.json",
                        "report_id": report_id,
                        "protocol_sha256": protocol_digest,
                        "evidence_sha256": evidence_digest,
                        "admitted_scope": {
                            "objective_protocol": "preference_conditioned",
                            "preference": [1.0, 0.0],
                            "n_objectives": 2,
                            "action_mode": "factored_discrete",
                            "horizon": 5000,
                            "n_factors": 12,
                            "levels_per_factor": 10,
                            "signal_dim": 16,
                            "context_dim": 8,
                            "state_dim": 8,
                            "teacher_hidden_dim": 16,
                            "signal_target_scale": 0.25,
                            "context_target_scale": 2.0,
                            "state_target_scale": 0.25,
                            "max_causal_lag": 5000,
                            "memory_lag": 0,
                            "reward_events": 1,
                            "conflict_strength": 0.75,
                            "terminal_state_weight": 1.0,
                            "effects": ["additive", "dynamics"],
                            "training_episodes": 256,
                            "training_batch_size": 64,
                            "training_trials": 3,
                            "public_worlds": 32,
                            "heldout_worlds": 16,
                            "max_trainable_parameters": 2_000_000,
                            "wall_seconds_total": 14_400.0,
                        },
                        "reviewed_on": "2026-07-13",
                        "reviewed_by": "fixture-reviewer",
                    }
                },
                "status": "qualified",
            }
        )
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")

    runtime = tmp_path / "runtime"
    store = ResearchStore(runtime / "state.db")
    secret_store = CampaignSecretStore(runtime / "secrets")
    campaign = create_controlled_campaign(store, "test", "question")
    secret_store.ensure(campaign)

    real_command = preflight._command

    def fake_command(argv, cwd, timeout=15.0):
        if argv[0] == "codex":
            return subprocess.CompletedProcess(
                argv, 0, "--ask-for-approval --sandbox --search exec", ""
            )
        if argv[0] == "claude":
            return subprocess.CompletedProcess(
                argv, 0, "--safe-mode --json-schema --permission-mode", ""
            )
        return real_command(argv, cwd, timeout)

    monkeypatch.setattr(preflight, "_command", fake_command)
    monkeypatch.setattr(preflight.shutil, "which", lambda command: f"/mock/{command}")

    report = preflight.run_preflight(
        repository=repository,
        runtime=runtime,
        store=store,
        secrets=secret_store,
        campaign_id=campaign,
    )
    checks = {check.name: check for check in report.checks}
    assert checks["committed_snapshot"].passed
    assert checks["head_contains_rebuild"].passed
    assert checks["evaluator_key"].passed
    assert report.ready is (
        sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()
    )

    definition_path = repository / "campaigns/factorlab_long_v1/definition.json"
    definition_bytes = definition_path.read_bytes()
    changed_scope = json.loads(definition_bytes)
    changed_scope["qualification_reports"]["factorlab-long-5k-v1"]["admitted_scope"][
        "memory_lag"
    ] = 8
    definition_path.write_text(json.dumps(changed_scope))
    scope_tamper = preflight.run_preflight(
        repository=repository,
        runtime=runtime,
        store=store,
        secrets=secret_store,
        campaign_id=campaign,
    )
    assert not {check.name: check for check in scope_tamper.checks}[
        "qualified_benchmark_tier"
    ].passed
    definition_path.write_bytes(definition_bytes)

    report_bytes = report_path.read_bytes()
    changed_report = json.loads(report_bytes)
    changed_report["checks"][0]["measurements"]["unreviewed"] = True
    report_path.write_text(json.dumps(changed_report))
    report_tamper = preflight.run_preflight(
        repository=repository,
        runtime=runtime,
        store=store,
        secrets=secret_store,
        campaign_id=campaign,
    )
    assert not {check.name: check for check in report_tamper.checks}[
        "qualified_benchmark_tier"
    ].passed
    report_path.write_bytes(report_bytes)

    (repository / "dirty.txt").write_text("not committed\n")
    dirty = preflight.run_preflight(
        repository=repository,
        runtime=runtime,
        store=store,
        secrets=secret_store,
        campaign_id=campaign,
    )
    assert not {check.name: check for check in dirty.checks}["committed_snapshot"].passed
