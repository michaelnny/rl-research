"""Operator CLI for the new research workflow kernel."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rlx_lab.artifacts import ArtifactStore
from rlx_lab.brief import render_brief
from rlx_lab.campaign import CampaignController, CampaignPolicy, create_controlled_campaign
from rlx_lab.models import CampaignStatus, JobMode
from rlx_lab.providers import ClaudeProvider, CodexProvider
from rlx_lab.preflight import run_preflight
from rlx_lab.secrets import CampaignSecretStore
from rlx_lab.store import ResearchStore
from rlx_lab.worker import Worker
from rlx_lab.worktrees import WorktreeManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlx-lab", description="Durable autonomous RL research workflow")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", type=Path, default=Path("runtime"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a campaign")
    init.add_argument("--name", required=True)
    init.add_argument("--question", required=True)
    init.add_argument("--concurrent-branches", type=int, default=3)
    init.add_argument("--max-branches", type=int, default=24)
    init.add_argument("--max-provider-attempts", type=int, default=240)
    init.add_argument("--max-local-run-attempts", type=int, default=48)
    init.add_argument("--max-wall-hours", type=float, default=168.0)
    init.add_argument("--primary-provider", choices=("codex", "claude"), default="codex")
    init.add_argument("--independent-provider", choices=("codex", "claude"), default="claude")

    enqueue = subparsers.add_parser("enqueue-agent", help="enqueue one structured model job")
    enqueue.add_argument("--campaign")
    enqueue.add_argument("--provider", choices=("codex", "claude"), required=True)
    enqueue.add_argument("--role", required=True)
    enqueue.add_argument("--mode", choices=("read", "write"), default="read")
    prompt_group = enqueue.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    enqueue.add_argument("--schema-file", type=Path)
    enqueue.add_argument("--result-kind", required=True)
    enqueue.add_argument("--depends-on", action="append", default=[])
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--allow", action="append", default=[])
    enqueue.add_argument(
        "--protect",
        action="append",
        default=[
            "src/rlx_bench",
            "src/rlx_agents",
            "src/rlx_lab",
            "tests/bench",
            "tests/agents",
            "tests/lab",
            "campaigns/factorlab_long_v1",
            "campaigns/schemas",
            "design",
        ],
    )
    enqueue.add_argument("--require-changes", action="store_true")

    execute = subparsers.add_parser("enqueue-run", help="enqueue a shell-free local argv")
    execute.add_argument("--campaign")
    execute.add_argument("--depends-on", action="append", default=[])
    execute.add_argument("--dependency-worktree", action="store_true")
    execute.add_argument("--timeout", type=float, default=300)
    execute.add_argument("argv", nargs=argparse.REMAINDER)

    worker = subparsers.add_parser("worker", help="run a worker")
    mode = worker.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--drain", action="store_true")
    mode.add_argument("--daemon", action="store_true")
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--lease-seconds", type=float, default=1800.0)
    worker.add_argument("--codex-model")
    worker.add_argument("--claude-model")
    worker.add_argument("--campaign", help="only claim jobs from one campaign")

    subparsers.add_parser("recover", help="recover expired worker leases")
    status = subparsers.add_parser("status", help="show queue or campaign status")
    status.add_argument("--campaign")
    subparsers.add_parser("campaigns", help="list campaigns")
    for command in ("pause", "resume", "stop"):
        lifecycle = subparsers.add_parser(command, help=f"{command} a campaign")
        lifecycle.add_argument("--campaign", required=True)
    controller = subparsers.add_parser("controller", help="expand an evidence DAG")
    controller.add_argument("--campaign", required=True)
    controller_mode = controller.add_mutually_exclusive_group(required=True)
    controller_mode.add_argument("--once", action="store_true")
    controller_mode.add_argument("--daemon", action="store_true")
    controller.add_argument("--poll-seconds", type=float, default=2.0)
    doctor = subparsers.add_parser("doctor", help="fail-closed production preflight")
    doctor.add_argument("--campaign", required=True)
    doctor.add_argument("--live-providers", action="store_true")
    serve = subparsers.add_parser("serve", help="run controller and supervised workers")
    serve.add_argument("--campaign", required=True)
    serve.add_argument("--workers", type=int, default=2)
    serve.add_argument("--poll-seconds", type=float, default=2.0)
    serve.add_argument("--lease-seconds", type=float, default=1800.0)
    serve.add_argument("--codex-model")
    serve.add_argument("--claude-model")
    brief = subparsers.add_parser("brief", help="render a disposable Markdown state view")
    brief.add_argument("--campaign")
    brief.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = args.repo.resolve()
    runtime = args.runtime if args.runtime.is_absolute() else repository / args.runtime
    runtime = runtime.resolve()
    store = ResearchStore(runtime / "state.db")
    secrets = CampaignSecretStore(runtime / "secrets")

    if args.command == "init":
        campaign_id = create_controlled_campaign(
            store,
            args.name,
            args.question,
            policy=CampaignPolicy(
                concurrent_branches=args.concurrent_branches,
                max_branches=args.max_branches,
                max_provider_attempts=args.max_provider_attempts,
                max_local_run_attempts=args.max_local_run_attempts,
                max_wall_seconds=args.max_wall_hours * 3600.0,
                primary_provider=args.primary_provider,
                independent_provider=args.independent_provider,
            ),
        )
        secrets.ensure(campaign_id)
        print(campaign_id)
        return 0
    if args.command == "enqueue-agent":
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
        schema = None if args.schema_file is None else json.loads(args.schema_file.read_text(encoding="utf-8"))
        payload = {
            "prompt": prompt,
            "schema": schema,
            "result_kind": args.result_kind,
            "allowed_prefixes": args.allow,
            "protected_prefixes": args.protect,
            "require_changes": args.require_changes,
        }
        print(
            store.enqueue_job(
                operation="agent",
                provider=args.provider,
                role=args.role,
                mode=JobMode(args.mode),
                payload=payload,
                campaign_id=args.campaign,
                dependencies=args.depends_on,
                priority=args.priority,
            )
        )
        return 0
    if args.command == "enqueue-run":
        command = list(args.argv)
        if command and command[0] == "--":
            command.pop(0)
        if not command:
            raise SystemExit("enqueue-run requires an argv after --")
        print(
            store.enqueue_job(
                operation="execute",
                provider="local",
                role="evaluator",
                mode=JobMode.EXECUTE,
                payload={
                    "argv": command,
                    "timeout_seconds": args.timeout,
                    "cwd_from_dependency_worktree": args.dependency_worktree,
                    "result_kind": "Run",
                },
                campaign_id=args.campaign,
                dependencies=args.depends_on,
            )
        )
        return 0
    if args.command == "recover":
        recovered = store.recover_expired()
        print(json.dumps({"recovered": recovered}))
        return 0
    if args.command == "doctor":
        report = run_preflight(
            repository=repository,
            runtime=runtime,
            store=store,
            secrets=secrets,
            campaign_id=args.campaign,
            live_providers=args.live_providers,
        )
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
        return 0 if report.ready else 2
    if args.command == "status":
        if args.campaign is None:
            print(json.dumps(store.job_counts(), sort_keys=True, indent=2))
        else:
            campaign = store.get_campaign(args.campaign)
            print(
                json.dumps(
                    {
                        "campaign": {
                            "id": campaign.id,
                            "name": campaign.name,
                            "question": campaign.question,
                            "status": campaign.status.value,
                        },
                        "usage": store.campaign_usage(args.campaign),
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
        return 0
    if args.command == "campaigns":
        print(
            json.dumps(
                [
                    {
                        "id": campaign.id,
                        "name": campaign.name,
                        "status": campaign.status.value,
                    }
                    for campaign in store.list_campaigns()
                ],
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if args.command in {"pause", "resume", "stop"}:
        target = {
            "pause": CampaignStatus.PAUSED,
            "resume": CampaignStatus.ACTIVE,
            "stop": CampaignStatus.STOPPED,
        }[args.command]
        campaign = store.set_campaign_status(args.campaign, target)
        print(json.dumps({"id": campaign.id, "status": campaign.status.value}))
        return 0
    if args.command == "controller":
        controller = CampaignController(repository=repository, store=store)
        if args.once:
            print(json.dumps(asdict(controller.tick(args.campaign)), default=str, sort_keys=True))
            return 0
        try:
            while True:
                result = controller.tick(args.campaign)
                print(json.dumps(asdict(result), default=str, sort_keys=True), flush=True)
                if result.campaign_status in {
                    CampaignStatus.STOPPED,
                    CampaignStatus.COMPLETED,
                    CampaignStatus.BUDGET_EXHAUSTED,
                }:
                    return 0
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            return 130
    if args.command == "serve":
        if args.workers < 1:
            raise SystemExit("--workers must be positive")
        report = run_preflight(
            repository=repository,
            runtime=runtime,
            store=store,
            secrets=secrets,
            campaign_id=args.campaign,
            live_providers=True,
        )
        if not report.ready:
            print(json.dumps(report.to_dict(), sort_keys=True, indent=2), file=sys.stderr)
            return 2
        logs = runtime / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        children: list[tuple[subprocess.Popen[bytes], object]] = []
        for index in range(args.workers):
            log_handle = (logs / f"worker-{index:02d}.log").open("ab", buffering=0)
            command = [
                sys.executable,
                "-m",
                "rlx_lab.cli",
                "--repo",
                str(repository),
                "--runtime",
                str(runtime),
                "worker",
                "--daemon",
                "--poll-seconds",
                str(args.poll_seconds),
                "--lease-seconds",
                str(args.lease_seconds),
                "--campaign",
                args.campaign,
            ]
            if args.codex_model:
                command.extend(("--codex-model", args.codex_model))
            if args.claude_model:
                command.extend(("--claude-model", args.claude_model))
            children.append(
                (
                    subprocess.Popen(
                        command,
                        cwd=repository,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    ),
                    log_handle,
                )
            )
        controller = CampaignController(repository=repository, store=store)
        exit_code = 0
        try:
            while True:
                dead = [process.pid for process, _ in children if process.poll() is not None]
                if dead:
                    print(json.dumps({"worker_processes_exited": dead}), file=sys.stderr)
                    exit_code = 1
                    break
                store.recover_expired()
                result = controller.tick(args.campaign)
                print(json.dumps(asdict(result), default=str, sort_keys=True), flush=True)
                if result.campaign_status in {
                    CampaignStatus.STOPPED,
                    CampaignStatus.COMPLETED,
                    CampaignStatus.BUDGET_EXHAUSTED,
                }:
                    break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            exit_code = 130
        finally:
            _terminate_children(children)
        return exit_code
    if args.command == "brief":
        text = render_brief(store, campaign_id=args.campaign)
        if args.output is None:
            print(text, end="")
        else:
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        return 0
    if args.command == "worker":
        artifacts = ArtifactStore(runtime / "artifacts")
        worktree_root = repository.parent / f".{repository.name}-rlx-worktrees"
        providers = {
            "codex": CodexProvider(
                model=args.codex_model, unreadable_roots=(runtime,)
            ),
            "claude": ClaudeProvider(
                model=args.claude_model, unreadable_roots=(runtime,)
            ),
        }
        worker_id = f"{socket.gethostname()}-{os.getpid()}"
        worker = Worker(
            worker_id=worker_id,
            repository=repository,
            store=store,
            artifacts=artifacts,
            providers=providers,
            worktrees=WorktreeManager(repository, worktree_root),
            secrets=secrets,
            runtime_root=runtime,
            lease_seconds=args.lease_seconds,
            campaign_id=args.campaign,
        )
        if args.once:
            job_id = worker.run_once()
            print(job_id or "idle")
            return 0
        if args.drain:
            while True:
                store.recover_expired()
                job_id = worker.run_once()
                if job_id is None:
                    break
                print(job_id, flush=True)
            return 0
        stop_requested = False

        def stop_worker(signum: int, frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True
            worker.request_stop()

        signal.signal(signal.SIGTERM, stop_worker)
        signal.signal(signal.SIGINT, stop_worker)
        try:
            while not stop_requested:
                store.recover_expired()
                job_id = worker.run_once()
                if job_id is None and not stop_requested:
                    time.sleep(args.poll_seconds)
                else:
                    if job_id is not None:
                        print(job_id, flush=True)
        except KeyboardInterrupt:
            return 130
        return 0
    raise AssertionError(args.command)


def _terminate_children(
    children: list[tuple[subprocess.Popen[bytes], object]],
) -> None:
    for process, _ in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    for process, handle in children:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        close = getattr(handle, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    sys.exit(main())
