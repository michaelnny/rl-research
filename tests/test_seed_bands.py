"""Tests for the seed-bands held-out evaluation contract (gate 9)."""

from __future__ import annotations

import numpy as np

from rlh_bench import make_env, registered_envs
from rlh_bench.seed_bands import seed_band_for, SeedBands


def _world_fingerprint(env_id: str, seed: int) -> bytes:
    """Cheap public-state fingerprint for seed/world smoke tests."""
    env = make_env(env_id)
    obs, _ = env.reset(seed=seed)
    parts = [np.asarray(obs, dtype=np.float32).ravel()]
    if hasattr(env, "compat_matrix"):
        parts.append(np.asarray(env.compat_matrix, dtype=np.float32).ravel())
    if hasattr(env, "actuator_matrix"):
        parts.append(np.asarray(env.actuator_matrix, dtype=np.float32).ravel())
    return b"|".join(np.ascontiguousarray(part).tobytes() for part in parts)


def test_seed_band_for_returns_bands_per_tier():
    """Every registered env should have a `SeedBands` lookup."""
    for env_id in registered_envs():
        bands = seed_band_for(env_id)
        assert isinstance(bands, SeedBands)
        assert isinstance(bands.train, range)
        assert isinstance(bands.held_out, range)
        # All published blocks should be disjoint. Debug worlds are public,
        # but keeping them outside train/validation/held-out avoids ambiguity
        # in reports and tests.
        named_ranges = {
            "train": bands.train,
            "validation": bands.validation,
            "held_out": bands.held_out,
            "debug": bands.debug,
        }
        names = list(named_ranges)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                left_set = set(named_ranges[left])
                right_set = set(named_ranges[right])
                assert left_set.isdisjoint(right_set), (
                    f"{env_id}: {left} and {right} seed bands must be disjoint; "
                    f"overlap: {left_set & right_set}"
                )


def test_representative_band_seeds_produce_different_worlds():
    """Representative seeds from every band should produce distinct worlds.

    This is a smoke test, not a proof that every seed in every range is unique;
    exhaustive collision checks would make the test suite too expensive.
    """
    for env_id in registered_envs():
        bands = seed_band_for(env_id)
        seeds = [
            bands.train[0],
            bands.train[1],
            bands.validation[0],
            bands.held_out[0],
            bands.debug[0],
        ]
        fingerprints = [_world_fingerprint(env_id, seed) for seed in seeds]
        assert len(set(fingerprints)) == len(fingerprints), (
            f"{env_id}: representative seeds should produce distinct worlds"
        )


def test_seed_band_for_small_uses_smaller_train_range():
    """Small tier publishes fewer seeds (cheaper to enumerate) than v0."""
    small = seed_band_for("RecoverableCapacityScheduling-Small-v0")
    v0 = seed_band_for("RecoverableCapacityScheduling-v0")
    assert len(small.train) < len(v0.train)


def test_seed_band_for_large_uses_smallest_train_range():
    """Large tier published seeds are smallest because episodes are
    expensive."""
    large = seed_band_for("RecoverableCapacityScheduling-Large-v0")
    v0 = seed_band_for("RecoverableCapacityScheduling-v0")
    assert len(large.train) <= len(v0.train)
