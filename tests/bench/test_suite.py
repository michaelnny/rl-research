from __future__ import annotations

import json

import pytest

from rlx_bench.factorlab import FactorLabConfig
from rlx_bench.suite import EvaluatorWorldSuite, WorldBand, WorldSuiteSpec


def _key(index: int) -> bytes:
    return bytes([index]) * 32


def _suite(key_index: int = 123) -> EvaluatorWorldSuite:
    return EvaluatorWorldSuite(
        FactorLabConfig(horizon=8, n_factors=2, max_causal_lag=4),
        WorldSuiteSpec(
            namespace="factorlab-calibration",
            version=1,
            master_key=_key(key_index),
            public_worlds=2,
            tune_worlds=3,
            heldout_worlds=4,
            audit_worlds=2,
        ),
    )


def test_public_manifest_reveals_only_public_seeds() -> None:
    suite = _suite()
    public = suite.public_manifest().to_dict()
    serialized = json.dumps(public, sort_keys=True)

    assert public["public_seeds"] == suite.evaluator_seeds(WorldBand.PUBLIC)
    for band in (WorldBand.TUNE, WorldBand.HELDOUT, WorldBand.AUDIT):
        for seed in suite.evaluator_seeds(band):
            assert str(seed) not in serialized
    assert public["band_counts"]["heldout"] == 4
    assert len(public["band_commitments"]["heldout"]) == 64
    assert len(public["neural_kernel_commitment"]) == 64


def test_suite_and_world_derivation_are_reproducible() -> None:
    first = _suite()
    second = _suite()

    assert first.suite_id == second.suite_id
    assert first.evaluator_seeds(WorldBand.HELDOUT) == second.evaluator_seeds(
        WorldBand.HELDOUT
    )
    assert first.world(WorldBand.HELDOUT, 2).world_id == second.world(
        WorldBand.HELDOUT, 2
    ).world_id
    assert "master_key" not in repr(first.spec)


def test_different_master_key_changes_commitments_and_identity() -> None:
    first = _suite(123)
    second = _suite(124)

    assert first.suite_id != second.suite_id
    assert (
        first.public_manifest().band_commitments["heldout"]
        != second.public_manifest().band_commitments["heldout"]
    )


def test_replay_tokens_bind_suite_band_world_and_episode() -> None:
    suite = _suite()

    first = suite.replay_token(WorldBand.HELDOUT, 0, 0)
    assert first == suite.replay_token(WorldBand.HELDOUT, 0, 0)
    assert first != suite.replay_token(WorldBand.HELDOUT, 0, 1)
    assert first != suite.replay_token(WorldBand.HELDOUT, 1, 0)


def test_world_index_and_suite_spec_validation_are_strict() -> None:
    suite = _suite()
    with pytest.raises(IndexError):
        suite.world(WorldBand.PUBLIC, 2)
    with pytest.raises(ValueError):
        WorldSuiteSpec(namespace="bad name", version=0, master_key=b"x" * 32)
    with pytest.raises(ValueError, match="256 bits"):
        WorldSuiteSpec(namespace="valid", version=0, master_key=b"short")


def test_suite_neural_kernel_is_shared_and_hidden_from_the_manifest() -> None:
    suite = _suite()
    public_world = suite.world(WorldBand.PUBLIC, 0)
    heldout_world = suite.world(WorldBand.HELDOUT, 0)
    serialized = json.dumps(suite.public_manifest().to_dict(), sort_keys=True)

    assert public_world.task_kernel == heldout_world.task_kernel
    assert "encoder" not in serialized
    assert "objective_heads" not in serialized
    assert "kernel_key" not in serialized
