"""Per-job Git worktree isolation and protected-path checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rlx_lab.executor import ExecutionLimits, ExecutionSpec, LocalExecutor


@dataclass(frozen=True)
class Worktree:
    job_id: str
    path: Path
    branch: str


class WorktreeError(RuntimeError):
    pass


class WorktreeManager:
    def __init__(self, repository: str | Path, root: str | Path) -> None:
        self.repository = Path(repository).resolve()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = LocalExecutor()

    def prepare(self, job_id: str, *, base_ref: str = "HEAD") -> Worktree:
        slug = re.sub(r"[^a-zA-Z0-9-]+", "-", job_id).strip("-")[:48]
        if not slug:
            raise ValueError("job_id does not contain a usable worktree name")
        path = self.root / slug
        branch = f"rlx/job-{slug}"
        if path.exists():
            raise WorktreeError(f"worktree path already exists: {path}")
        result = self.executor.run(
            ExecutionSpec(
                argv=("git", "worktree", "add", "-b", branch, str(path), base_ref),
                cwd=self.repository,
                limits=ExecutionLimits(timeout_seconds=60),
                inherit_env=True,
            )
        )
        if result.exit_code != 0:
            raise WorktreeError(result.stderr.decode("utf-8", errors="replace"))
        return Worktree(job_id=job_id, path=path, branch=branch)

    def changed_paths(self, worktree: Worktree) -> tuple[str, ...]:
        result = self.executor.run(
            ExecutionSpec(
                argv=("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
                cwd=worktree.path,
                limits=ExecutionLimits(timeout_seconds=30),
                inherit_env=True,
            )
        )
        if result.exit_code != 0:
            raise WorktreeError(result.stderr.decode("utf-8", errors="replace"))
        paths = []
        records = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if not record:
                continue
            status = record[:2]
            path = record[3:]
            if status[0] in {"R", "C"}:
                if index >= len(records):
                    raise WorktreeError("malformed git status rename record")
                path = records[index]
                index += 1
            paths.append(PurePosixPath(path).as_posix())
        return tuple(sorted(set(paths)))

    def commit_changes(self, worktree: Worktree, paths: tuple[str, ...]) -> str:
        """Persist validated job output on its isolated evidence branch."""

        if paths:
            added = self.executor.run(
                ExecutionSpec(
                    argv=("git", "add", "--", *paths),
                    cwd=worktree.path,
                    limits=ExecutionLimits(timeout_seconds=60),
                    inherit_env=True,
                )
            )
            if added.exit_code != 0:
                raise WorktreeError(added.stderr.decode("utf-8", errors="replace"))
            committed = self.executor.run(
                ExecutionSpec(
                    argv=(
                        "git",
                        "-c",
                        "user.name=RLX Research Harness",
                        "-c",
                        "user.email=rlx-harness@localhost",
                        "commit",
                        "--no-gpg-sign",
                        "-m",
                        f"rlx: preserve output for {worktree.job_id}",
                    ),
                    cwd=worktree.path,
                    limits=ExecutionLimits(timeout_seconds=60),
                    inherit_env=True,
                )
            )
            if committed.exit_code != 0:
                raise WorktreeError(committed.stderr.decode("utf-8", errors="replace"))
        resolved = self.executor.run(
            ExecutionSpec(
                argv=("git", "rev-parse", "HEAD"),
                cwd=worktree.path,
                limits=ExecutionLimits(timeout_seconds=30),
                inherit_env=True,
            )
        )
        if resolved.exit_code != 0:
            raise WorktreeError(resolved.stderr.decode("utf-8", errors="replace"))
        return resolved.stdout.decode().strip()

    @staticmethod
    def assert_paths_allowed(
        paths: tuple[str, ...],
        *,
        allowed_prefixes: tuple[str, ...],
        protected_prefixes: tuple[str, ...],
    ) -> None:
        for path in paths:
            if any(_under(path, prefix) for prefix in protected_prefixes):
                raise WorktreeError(f"job changed protected path {path}")
            if allowed_prefixes and not any(_under(path, prefix) for prefix in allowed_prefixes):
                raise WorktreeError(f"job changed path outside its allowance: {path}")

    @staticmethod
    def snapshot_paths(root: str | Path, prefixes: tuple[str, ...]) -> dict[str, str]:
        """Hash protected trees including ignored and untracked files."""

        root_path = Path(root).resolve()
        snapshot: dict[str, str] = {}
        for prefix in prefixes:
            target = (root_path / prefix).resolve()
            try:
                target.relative_to(root_path)
            except ValueError as exc:
                raise WorktreeError(f"protected prefix escapes worktree: {prefix}") from exc
            if not target.exists() and not target.is_symlink():
                continue
            paths = (target,) if not target.is_dir() else tuple(target.rglob("*"))
            for path in paths:
                if path.is_dir():
                    continue
                relative = path.relative_to(root_path).as_posix()
                if path.is_symlink():
                    material = f"symlink:{path.readlink()}".encode()
                else:
                    material = path.read_bytes()
                snapshot[relative] = hashlib.sha256(material).hexdigest()
        return snapshot

    @staticmethod
    def assert_snapshot_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
        if before == after:
            return
        changed = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
        excerpt = changed[:10]
        suffix = "" if len(changed) <= 10 else f" (+{len(changed) - 10} more)"
        raise WorktreeError(f"job mutated protected content: {excerpt}{suffix}")

    def remove(self, worktree: Worktree, *, force: bool = False) -> None:
        argv = ["git", "worktree", "remove"]
        if force:
            argv.append("--force")
        argv.append(str(worktree.path))
        result = self.executor.run(
            ExecutionSpec(
                argv=tuple(argv),
                cwd=self.repository,
                limits=ExecutionLimits(timeout_seconds=60),
                inherit_env=True,
            )
        )
        if result.exit_code != 0:
            raise WorktreeError(result.stderr.decode("utf-8", errors="replace"))


def _under(path: str, prefix: str) -> bool:
    clean = prefix.rstrip("/")
    return path == clean or path.startswith(clean + "/")
