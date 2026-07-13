"""Versioned world suites with a strict public/evaluator metadata split."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .factorlab import FactorLabConfig, FactorLabWorld, generate_world


class WorldBand(str, Enum):
    PUBLIC = "public"
    TUNE = "tune"
    HELDOUT = "heldout"
    AUDIT = "audit"


@dataclass(frozen=True)
class WorldSuiteSpec:
    namespace: str
    version: int
    master_key: bytes = field(repr=False)
    public_worlds: int = 8
    tune_worlds: int = 8
    heldout_worlds: int = 32
    audit_worlds: int = 8

    def __post_init__(self) -> None:
        if not self.namespace or any(character.isspace() for character in self.namespace):
            raise ValueError("suite namespace must be non-empty and contain no whitespace")
        if self.version < 0:
            raise ValueError("suite version cannot be negative")
        if not isinstance(self.master_key, bytes) or len(self.master_key) < 32:
            raise ValueError("master_key must contain at least 256 bits")
        counts = (
            self.public_worlds,
            self.tune_worlds,
            self.heldout_worlds,
            self.audit_worlds,
        )
        if any(count < 1 for count in counts):
            raise ValueError("every world band must contain at least one world")


@dataclass(frozen=True)
class PublicSuiteManifest:
    suite_id: str
    namespace: str
    version: int
    task_id: str
    public_seeds: tuple[int, ...]
    band_counts: dict[str, int]
    band_commitments: dict[str, str]
    neural_kernel_commitment: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _derive(spec: WorldSuiteSpec, purpose: str) -> bytes:
    payload = f"rlx-long-suite-v1|{spec.namespace}|{spec.version}|{purpose}".encode()
    return hmac.new(spec.master_key, payload, hashlib.sha256).digest()


def _derive_seed(spec: WorldSuiteSpec, band: WorldBand, index: int) -> int:
    return int.from_bytes(_derive(spec, f"world|{band.value}|{index}")[:8], "big") & (
        2**63 - 1
    )


def _commit_band(namespace: str, version: int, band: WorldBand, seeds: tuple[int, ...]) -> str:
    payload = {
        "namespace": namespace,
        "version": version,
        "band": band.value,
        "seed_commitments": [
            hashlib.sha256(f"{namespace}|{version}|{band.value}|{seed}".encode()).hexdigest()
            for seed in seeds
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class EvaluatorWorldSuite:
    """Owns hidden seeds; candidate jobs receive only ``public_manifest``."""

    def __init__(self, config: FactorLabConfig, spec: WorldSuiteSpec):
        self.config = config
        self.spec = spec
        self._kernel_key = _derive(spec, "neural-task-kernel-v1")
        counts = {
            WorldBand.PUBLIC: spec.public_worlds,
            WorldBand.TUNE: spec.tune_worlds,
            WorldBand.HELDOUT: spec.heldout_worlds,
            WorldBand.AUDIT: spec.audit_worlds,
        }
        self._seeds = {
            band: tuple(_derive_seed(spec, band, index) for index in range(count))
            for band, count in counts.items()
        }
        commitments = {
            band.value: _commit_band(spec.namespace, spec.version, band, seeds)
            for band, seeds in self._seeds.items()
        }
        kernel_commitment = hmac.new(
            spec.master_key,
            b"rlx-long-suite-v1|neural-kernel-commitment|" + hashlib.sha256(
                self._kernel_key
            ).digest(),
            hashlib.sha256,
        ).hexdigest()
        identity = {
            "namespace": spec.namespace,
            "version": spec.version,
            "task_id": config.task_id,
            "band_counts": {band.value: count for band, count in counts.items()},
            "band_commitments": commitments,
            "neural_kernel_commitment": kernel_commitment,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.suite_id = f"fls-{hashlib.sha256(canonical.encode()).hexdigest()}"
        self._public_manifest = PublicSuiteManifest(
            suite_id=self.suite_id,
            namespace=spec.namespace,
            version=spec.version,
            task_id=config.task_id,
            public_seeds=self._seeds[WorldBand.PUBLIC],
            band_counts={band.value: count for band, count in counts.items()},
            band_commitments=commitments,
            neural_kernel_commitment=kernel_commitment,
        )
        self._world_cache: dict[tuple[WorldBand, int], FactorLabWorld] = {}

    def public_manifest(self) -> PublicSuiteManifest:
        return self._public_manifest

    def evaluator_seeds(self, band: WorldBand | str) -> tuple[int, ...]:
        return self._seeds[WorldBand(band)]

    def world(self, band: WorldBand | str, index: int) -> FactorLabWorld:
        normalized = WorldBand(band)
        seeds = self._seeds[normalized]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(seeds):
            raise IndexError("world index is out of range")
        key = (normalized, index)
        world = self._world_cache.get(key)
        if world is None:
            world = generate_world(self.config, seeds[index], kernel_key=self._kernel_key)
            self._world_cache[key] = world
        return world

    def replay_token(self, band: WorldBand | str, index: int, episode: int) -> str:
        if episode < 0:
            raise ValueError("episode cannot be negative")
        normalized = WorldBand(band)
        world = self.world(normalized, index)
        payload = f"{self.suite_id}|{normalized.value}|{index}|{episode}|{world.world_id}"
        return f"flr-{hashlib.sha256(payload.encode()).hexdigest()}"
