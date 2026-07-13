from __future__ import annotations

import json

from rlx_agents.qualification_study import _bootstrap_mean_ci, load_protocol


def test_bootstrap_interval_is_deterministic_and_contains_sample_mean() -> None:
    first = _bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.5], draws=1000, seed=7, confidence=0.95)
    second = _bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.5], draws=1000, seed=7, confidence=0.95)

    assert first == second
    assert first[0] < 0.3 < first[1]


def test_protocol_digest_uses_canonical_content(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"z": 1, "a": [2, 3]}, indent=2), encoding="utf-8")

    protocol, digest = load_protocol(path)

    assert protocol == {"z": 1, "a": [2, 3]}
    assert len(digest) == 64
