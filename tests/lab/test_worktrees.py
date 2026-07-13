from __future__ import annotations

import subprocess

import pytest

from rlx_lab.worktrees import WorktreeError, WorktreeManager


def git(cwd, *args):
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


def test_worktree_isolation_and_path_protection(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.com")
    git(repository, "config", "user.name", "Test")
    (repository / "README.md").write_text("base\n")
    git(repository, "add", "README.md")
    git(repository, "commit", "-qm", "base")

    manager = WorktreeManager(repository, tmp_path / "worktrees")
    worktree = manager.prepare("job-123")
    assert worktree.path.is_dir()
    assert worktree.branch == "rlx/job-job-123"
    (worktree.path / "candidate.py").write_text("x = 1\n")
    assert manager.changed_paths(worktree) == ("candidate.py",)
    manager.assert_paths_allowed(
        ("candidate.py",),
        allowed_prefixes=("candidate.py",),
        protected_prefixes=("src/rlx_bench",),
    )
    with pytest.raises(WorktreeError):
        manager.assert_paths_allowed(
            ("src/rlx_bench/env.py",),
            allowed_prefixes=("src",),
            protected_prefixes=("src/rlx_bench",),
        )
    commit = manager.commit_changes(worktree, ("candidate.py",))
    assert manager.changed_paths(worktree) == ()
    assert len(commit) == 40
    manager.remove(worktree)
    assert not worktree.path.exists()
    assert subprocess.run(
        ("git", "show", f"{worktree.branch}:candidate.py"),
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout == b"x = 1\n"


def test_protected_snapshot_detects_ignored_runtime_mutation(tmp_path):
    root = tmp_path / "worktree"
    protected = root / "src" / "rlx_bench"
    protected.mkdir(parents=True)
    (protected / "module.py").write_text("VALUE = 1\n")
    before = WorktreeManager.snapshot_paths(root, ("src/rlx_bench",))
    (protected / "__pycache__").mkdir()
    (protected / "__pycache__" / "module.pyc").write_bytes(b"ignored mutation")
    after = WorktreeManager.snapshot_paths(root, ("src/rlx_bench",))

    with pytest.raises(WorktreeError, match="mutated protected content"):
        WorktreeManager.assert_snapshot_unchanged(before, after)
