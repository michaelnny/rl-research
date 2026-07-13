from __future__ import annotations

from rlx_lab.artifacts import ArtifactStore


def test_artifacts_are_content_addressed_and_verified(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.put_text("same")
    second = store.put_text("same")
    different = store.put_text("different")
    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert first.sha256 != different.sha256
    assert store.read_bytes(first.sha256) == b"same"
    assert store.verify(first)

