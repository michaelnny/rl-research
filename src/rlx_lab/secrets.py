"""Evaluator key storage isolated from model-provider processes."""

from __future__ import annotations

import os
import re
import secrets
import stat
from pathlib import Path


class SecretStoreError(RuntimeError):
    pass


class CampaignSecretStore:
    """Create and load one high-entropy FactorLab suite key per campaign."""

    _SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def path_for(self, campaign_id: str) -> Path:
        if not self._SAFE_ID.fullmatch(campaign_id):
            raise SecretStoreError("campaign id is unsafe for secret storage")
        return self.root / f"{campaign_id}.key"

    def ensure(self, campaign_id: str) -> bytes:
        path = self.path_for(campaign_id)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self.load(campaign_id)
        key = secrets.token_bytes(32)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return key

    def load(self, campaign_id: str) -> bytes:
        path = self.path_for(campaign_id)
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            raise SecretStoreError(f"missing evaluator key for campaign {campaign_id}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise SecretStoreError(f"evaluator key is not a regular file: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise SecretStoreError(f"evaluator key permissions must be 0600: {path}")
        key = path.read_bytes()
        if len(key) != 32:
            raise SecretStoreError(f"evaluator key must contain exactly 32 bytes: {path}")
        return key
