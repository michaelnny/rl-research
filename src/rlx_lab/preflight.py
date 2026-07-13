"""Fail-closed production checks for autonomous campaign startup."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from rlx_bench.factorlab import BENCHMARK_REVISION, FactorLabConfig
from rlx_bench.qualification import REQUIRED_QUALIFICATION_CHECKS

from rlx_lab.executor import macos_deny_read_command
from rlx_lab.models import CampaignStatus, JobMode
from rlx_lab.providers import (
    ClaudeProvider,
    CodexProvider,
    ProviderError,
    ProviderRequest,
)
from rlx_lab.secrets import CampaignSecretStore, SecretStoreError
from rlx_lab.store import ResearchStore


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return {"ready": self.ready, "checks": [asdict(check) for check in self.checks]}


def _command(argv: tuple[str, ...], cwd: Path, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=None,
    )


def _strict_schema_errors(value: object, location: str = "$") -> list[str]:
    if not isinstance(value, dict):
        return []
    errors: list[str] = []
    if ("enum" in value or "const" in value) and "type" not in value:
        errors.append(f"{location}: enum/const requires an explicit type")
    if value.get("type") == "object":
        properties = value.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{location}: properties must be an object")
        else:
            if value.get("additionalProperties") is not False:
                errors.append(f"{location}: additionalProperties must be false")
            if set(value.get("required", ())) != set(properties):
                errors.append(f"{location}: every property must be required")
            for name, child in properties.items():
                errors.extend(_strict_schema_errors(child, f"{location}.{name}"))
    items = value.get("items")
    if items is not None:
        errors.extend(_strict_schema_errors(items, f"{location}[]"))
    return errors


def check_benchmark_admission(
    repository: Path, policy: Mapping[str, object]
) -> PreflightCheck:
    """Validate the immutable tier admission without starting external processes."""

    repository = repository.resolve()
    definition_path = repository / "campaigns" / "factorlab_long_v1" / "definition.json"
    expected_protocol_path = (
        repository / "campaigns" / "factorlab_long_v1" / "qualification_protocol.json"
    ).resolve()
    qualification_root = (
        repository / "campaigns" / "factorlab_long_v1" / "qualification"
    ).resolve()
    try:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        benchmark_tier = str(policy.get("benchmark_tier", ""))
        admitted = definition.get("admitted_tiers", [])
        qualification = definition.get("qualification_reports", {}).get(
            benchmark_tier, {}
        )
        report_path = (repository / str(qualification.get("report_path", ""))).resolve()
        protocol_path = (repository / str(qualification.get("protocol_path", ""))).resolve()
        if qualification_root not in report_path.parents:
            raise ValueError("qualification report is outside its versioned directory")
        if protocol_path != expected_protocol_path:
            raise ValueError("admission does not use the frozen long-horizon protocol")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_digest = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        evidence_ref = f"sha256:{qualification.get('evidence_sha256', '')}"
        protocol_ref = f"protocol-sha256:{protocol_digest}"
        report_checks = report.get("checks", [])
        scope = qualification.get("admitted_scope", {})
        expected_scope = {
            "objective_protocol": "preference_conditioned",
            "preference": [1.0, 0.0],
            "n_objectives": 2,
            "action_mode": "factored_discrete",
            "horizon": policy.get("evaluation_horizon"),
            "n_factors": policy.get("evaluation_factors"),
            "levels_per_factor": policy.get("evaluation_levels_per_factor"),
            "signal_dim": policy.get("evaluation_signal_dim"),
            "context_dim": policy.get("evaluation_context_dim"),
            "state_dim": policy.get("evaluation_state_dim"),
            "teacher_hidden_dim": policy.get("evaluation_teacher_hidden_dim"),
            "signal_target_scale": 0.25,
            "context_target_scale": 2.0,
            "state_target_scale": 0.25,
            "max_causal_lag": policy.get("evaluation_horizon"),
            "memory_lag": 0,
            "reward_events": 1,
            "conflict_strength": 0.75,
            "terminal_state_weight": 1.0,
            "effects": ["additive", "dynamics"],
            "training_episodes": policy.get("evaluation_training_episodes"),
            "training_batch_size": policy.get("evaluation_training_batch_size"),
            "training_trials": policy.get("evaluation_training_trials"),
            "public_worlds": policy.get("evaluation_public_worlds"),
            "heldout_worlds": policy.get("evaluation_heldout_worlds"),
            "max_trainable_parameters": policy.get("evaluation_max_parameters"),
            "wall_seconds_total": policy.get("evaluation_wall_seconds_total"),
        }
        report_payload = {
            "task_id": report.get("task_id"),
            "suite_id": report.get("suite_id"),
            "benchmark_revision": report.get("benchmark_revision"),
            "checks": report_checks,
        }
        expected_report_id = "flq-" + hashlib.sha256(
            json.dumps(
                report_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        anchor = dict(protocol["anchor_configuration"])
        anchor["levels_per_factor"] = (int(anchor["levels_per_factor"]),)
        anchor["effects"] = tuple(anchor["effects"])
        expected_task_id = FactorLabConfig(**anchor).task_id
        expected_check_names = list(REQUIRED_QUALIFICATION_CHECKS)
        evidence_digest = str(qualification.get("evidence_sha256", ""))
        reviewed_on = qualification.get("reviewed_on")
        reviewed_by = qualification.get("reviewed_by")
        tier_ok = (
            definition.get("status") == "qualified"
            and definition.get("benchmark_revision") == BENCHMARK_REVISION
            and benchmark_tier in admitted
            and report.get("qualified") is True
            and report.get("task_id") == expected_task_id
            and report.get("report_id") == qualification.get("report_id")
            and report.get("report_id") == expected_report_id
            and report.get("benchmark_revision") == BENCHMARK_REVISION
            and protocol_digest == qualification.get("protocol_sha256")
            and protocol.get("tier_id") == benchmark_tier
            and scope == expected_scope
            and isinstance(reviewed_on, str)
            and bool(reviewed_on.strip())
            and isinstance(reviewed_by, str)
            and bool(reviewed_by.strip())
            and len(evidence_digest) == 64
            and all(character in "0123456789abcdef" for character in evidence_digest)
            and [check.get("name") for check in report_checks] == expected_check_names
            and all(
                check.get("status") == "verified"
                and evidence_ref in check.get("evidence_refs", [])
                and protocol_ref in check.get("evidence_refs", [])
                for check in report_checks
            )
        )
        detail = (
            f"{benchmark_tier} admitted by {definition_path.relative_to(repository)}"
            if tier_ok
            else f"{benchmark_tier or '<missing>'} is not an admitted benchmark tier"
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        tier_ok = False
        detail = f"cannot validate benchmark admission: {exc}"
    return PreflightCheck("qualified_benchmark_tier", tier_ok, detail)


def run_preflight(
    *,
    repository: Path,
    runtime: Path,
    store: ResearchStore,
    secrets: CampaignSecretStore,
    campaign_id: str,
    live_providers: bool = False,
) -> PreflightReport:
    repository = repository.resolve()
    runtime = runtime.resolve()
    checks: list[PreflightCheck] = []

    root = _command(("git", "rev-parse", "--show-toplevel"), repository)
    git_ok = root.returncode == 0 and Path(root.stdout.strip()).resolve() == repository
    checks.append(PreflightCheck("git_repository", git_ok, root.stderr.strip() or root.stdout.strip()))

    status = _command(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"), repository
    )
    clean = status.returncode == 0 and not status.stdout
    detail = "clean" if clean else "commit or remove all working-tree changes before serving"
    checks.append(PreflightCheck("committed_snapshot", clean, detail))

    head_paths = ("src/rlx_lab/cli.py", "src/rlx_agents/evaluate.py", "src/rlx_bench/suite.py")
    missing = [
        path
        for path in head_paths
        if _command(("git", "cat-file", "-e", f"HEAD:{path}"), repository).returncode != 0
    ]
    checks.append(
        PreflightCheck(
            "head_contains_rebuild",
            not missing,
            "present" if not missing else f"missing from HEAD: {', '.join(missing)}",
        )
    )

    invalid_schemas: list[str] = []
    for path in sorted((repository / "campaigns" / "schemas").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("root is not an object")
            if path.name != "candidate_measurement.schema.json":
                invalid_schemas.extend(
                    f"{path.name}: {error}" for error in _strict_schema_errors(value)
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid_schemas.append(f"{path.name}: {exc}")
    checks.append(
        PreflightCheck(
            "campaign_schemas",
            not invalid_schemas,
            "valid" if not invalid_schemas else "; ".join(invalid_schemas),
        )
    )

    try:
        campaign = store.get_campaign(campaign_id)
        campaign_ok = campaign.status is CampaignStatus.ACTIVE
        campaign_detail = campaign.status.value
        policy = campaign.config.get("policy", {})
        required_providers = {
            str(policy.get("primary_provider", "codex")),
            str(policy.get("independent_provider", "claude")),
        }
    except KeyError:
        campaign_ok = False
        campaign_detail = "campaign not found"
        policy = {}
        required_providers = {"codex", "claude"}
    checks.append(PreflightCheck("active_campaign", campaign_ok, campaign_detail))

    checks.append(check_benchmark_admission(repository, policy))

    try:
        secrets.load(campaign_id)
        secret_ok, secret_detail = True, "32-byte key with owner-only permissions"
    except SecretStoreError as exc:
        secret_ok, secret_detail = False, str(exc)
    checks.append(PreflightCheck("evaluator_key", secret_ok, secret_detail))

    sandbox = Path("/usr/bin/sandbox-exec")
    sandbox_ok = sys.platform == "darwin" and sandbox.is_file()
    sandbox_detail = (
        str(sandbox) if sandbox_ok else "production isolation requires macOS sandbox-exec"
    )
    if sandbox_ok:
        with tempfile.TemporaryDirectory(prefix="rlx-preflight-sandbox-") as temporary:
            denied = Path(temporary) / "denied.txt"
            denied.write_text("secret", encoding="utf-8")
            probe = _command(
                macos_deny_read_command(
                    (sys.executable, "-c", f"open({str(denied)!r}).read()"),
                    (Path(temporary),),
                ),
                repository,
            )
            process_probe = _command(
                macos_deny_read_command(
                    ("/bin/ps", "eww", "-ax"),
                    (Path(temporary),),
                ),
                repository,
            )
            sandbox_ok = probe.returncode != 0 and process_probe.returncode != 0
            sandbox_detail = (
                "filesystem and process-information denials verified"
                if sandbox_ok
                else "sandbox denial probe failed"
            )
    checks.append(
        PreflightCheck(
            "process_sandbox",
            sandbox_ok,
            sandbox_detail,
        )
    )

    provider_requirements = {
        "codex": (
            ("codex", "--help"),
            ("--ask-for-approval", "--sandbox", "--search", "exec"),
        ),
        "claude": (("claude", "--help"), ("--safe-mode", "--json-schema", "--permission-mode")),
    }
    for provider in sorted(required_providers):
        if provider not in provider_requirements:
            checks.append(PreflightCheck(f"provider_{provider}", False, "unsupported provider"))
            continue
        argv, required = provider_requirements[provider]
        executable = shutil.which(argv[0])
        if executable is None:
            checks.append(PreflightCheck(f"provider_{provider}", False, "executable not found"))
            continue
        result = _command(argv, repository)
        output = result.stdout + result.stderr
        missing_flags = [flag for flag in required if flag not in output]
        checks.append(
            PreflightCheck(
                f"provider_{provider}",
                result.returncode == 0 and not missing_flags,
                f"{executable}; missing flags: {missing_flags}" if missing_flags else executable,
            )
        )

    if live_providers:
        schema = {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": sorted(required_providers)},
                "ok": {"type": "boolean", "const": True},
            },
            "required": ["provider", "ok"],
            "additionalProperties": False,
        }
        provider_types = {"codex": CodexProvider, "claude": ClaudeProvider}
        for provider in sorted(required_providers):
            if provider not in provider_types:
                checks.append(
                    PreflightCheck(f"provider_{provider}_live", False, "unsupported provider")
                )
                continue
            try:
                adapter = provider_types[provider](unreadable_roots=(runtime,))
                output = adapter.run(
                    ProviderRequest(
                        role="preflight",
                        prompt=(
                            "Return only the requested object with provider set to "
                            f"{provider!r} and ok set to true. Do not inspect files or use tools."
                        ),
                        cwd=repository,
                        mode=JobMode.READ,
                        schema=schema,
                        timeout_seconds=180.0,
                    )
                ).output
                passed = output == {"provider": provider, "ok": True}
                detail = "authenticated structured call passed" if passed else str(output)
            except (ProviderError, RuntimeError, OSError) as exc:
                passed, detail = False, f"{type(exc).__name__}: {exc}"
            checks.append(PreflightCheck(f"provider_{provider}_live", passed, detail))

    runtime_parent_ok = runtime.parent.is_dir()
    checks.append(
        PreflightCheck(
            "runtime_parent",
            runtime_parent_ok,
            str(runtime.parent),
        )
    )
    return PreflightReport(all(check.passed for check in checks), tuple(checks))
