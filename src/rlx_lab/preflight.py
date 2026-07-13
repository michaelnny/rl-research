"""Fail-closed production checks for autonomous campaign startup."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

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
        required_providers = {"codex", "claude"}
    checks.append(PreflightCheck("active_campaign", campaign_ok, campaign_detail))

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
