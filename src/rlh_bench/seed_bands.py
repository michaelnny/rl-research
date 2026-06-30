"""World generation and seed-split contract for the substrate.

Every env family in this lab follows the same seed → world contract:

  * ``reset(seed=s)`` deterministically generates the same world and
    initial state for that tier. Same seed → same world. Different
    seeds → measurably different worlds (different topology, demand
    schedule, actuator matrix, etc.).
  * The env exposes a fixed registry tier (Small, v0, Large) whose
    generator parameters do not change across resets.
  * The seed varies *the world*: graph topology, demand profile,
    actuator matrix, key/seal placement, contract bundle structure.

Held-out evaluation is a first-class part of this substrate. Each
tier publishes train, validation, and held-out seed blocks. Algorithms
that train against the public train block must be reported against
the held-out block. A policy that hard-codes one world will fail the
held-out gate.

This module exists so each env family can declare its seed bands
without inventing a private convention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedBands:
    """Published seed ranges for one env tier.

    The contract is structural, not enforced — the env will accept any
    seed. These ranges exist so baseline scripts and candidate
    algorithms can declare which evaluation slice they used. A result
    reported against ``train`` is not substrate-valid for a novelty
    claim; results must be reported against ``held_out``.

    Conventions:

      * ``train``: seeds an algorithm may train against and tune on.
      * ``validation``: seeds an algorithm may probe during
        development, but should not be used as the final headline.
      * ``held_out``: seeds reserved for headline evaluation. These
        worlds the algorithm has never seen.
      * ``debug``: a small fixed set of seeds used by tests and
        examples. Worlds here are intentionally public and may be
        memorized.
    """

    train: range
    validation: range
    held_out: range
    debug: range

    def all_seeds(self) -> tuple[range, range, range, range]:
        return (self.train, self.validation, self.held_out, self.debug)


# Default seed bands per tier. Concrete env families may extend these
# bands or override them, but most should not need to.
DEFAULT_SMALL_BANDS = SeedBands(
    train=range(0, 100),
    validation=range(1000, 1050),
    held_out=range(10000, 10050),
    debug=range(0, 10),
)

DEFAULT_V0_BANDS = SeedBands(
    train=range(0, 1000),
    validation=range(1000, 1200),
    held_out=range(10000, 10200),
    debug=range(0, 10),
)

DEFAULT_LARGE_BANDS = SeedBands(
    train=range(0, 100),
    validation=range(1000, 1050),
    held_out=range(20000, 20050),
    debug=range(0, 5),
)


def seed_band_for(env_id: str) -> SeedBands:
    """Return the published seed bands for a tier name suffix.

    The convention is: an env ID ending in ``-Small-v0`` gets
    ``DEFAULT_SMALL_BANDS``, ``-v0`` gets ``DEFAULT_V0_BANDS``,
    ``-Large-v0`` gets ``DEFAULT_LARGE_BANDS``. Concrete env families
    that need different bands may register their own.

    Args:
        env_id: Registered environment ID.
    """

    if "Small" in env_id:
        return DEFAULT_SMALL_BANDS
    if "Large" in env_id:
        return DEFAULT_LARGE_BANDS
    return DEFAULT_V0_BANDS
