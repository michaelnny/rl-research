"""Content-addressed immutable artifact storage."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size: int
    media_type: str
    path: Path


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        target = self.root / digest[:2] / digest[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = target.read_bytes()
            if existing != data:
                raise RuntimeError(f"artifact hash collision at {target}")
        else:
            fd, temporary = tempfile.mkstemp(prefix="artifact-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ArtifactRef(digest, len(data), media_type, target)

    def put_text(self, text: str, *, media_type: str = "text/plain; charset=utf-8") -> ArtifactRef:
        return self.put_bytes(text.encode("utf-8"), media_type=media_type)

    def read_bytes(self, digest: str) -> bytes:
        return (self.root / digest[:2] / digest[2:]).read_bytes()

    def verify(self, ref: ArtifactRef) -> bool:
        data = ref.path.read_bytes()
        return len(data) == ref.size and hashlib.sha256(data).hexdigest() == ref.sha256

