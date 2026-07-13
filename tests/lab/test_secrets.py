from __future__ import annotations

import os

import pytest

from rlx_lab.secrets import CampaignSecretStore, SecretStoreError


def test_campaign_secret_is_stable_high_entropy_and_owner_only(tmp_path) -> None:
    store = CampaignSecretStore(tmp_path / "secrets")

    first = store.ensure("campaign-test")
    second = store.ensure("campaign-test")
    path = store.path_for("campaign-test")

    assert first == second
    assert len(first) == 32
    assert path.read_bytes() == first
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_campaign_secret_rejects_unsafe_ids_and_permissions(tmp_path) -> None:
    store = CampaignSecretStore(tmp_path / "secrets")
    with pytest.raises(SecretStoreError, match="unsafe"):
        store.ensure("../escape")

    store.ensure("campaign-test")
    os.chmod(store.path_for("campaign-test"), 0o644)
    with pytest.raises(SecretStoreError, match="0600"):
        store.load("campaign-test")
