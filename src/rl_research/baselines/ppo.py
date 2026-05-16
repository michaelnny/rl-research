"""PPO baseline (frozen yardstick).

This is the **only** allowed RL algorithm implementation in the project. It is
NOT a candidate; it is the reference the Curator weighs candidates against.

Status: STUB — full implementation is the next deliverable.

Contract for the eventual implementation:

- Implements PPO from primitives. No `stable_baselines3` / `cleanrl` / `tianshou`
  imports. Uses only `torch`, `numpy`, and the env libraries.
- Exposes a `train(config: PPOConfig) -> dict` function that obeys
  `docs/contract.md` §train.py contract — accepts the same CLI flags via the
  config and produces the same TensorBoard scalars.
- Frozen hyperparameters per domain (Atari / continuous-control / multi-signal).
  The hyperparameter sets are documented in this file and never re-tuned per
  benchmark.
- Multi-signal handling: PPO is a scalar-credit method, so on `minecart-v0` the
  baseline runs with a fixed equal-weight scalarization of the reward vector.
  This is documented as a *known limitation* of the baseline — handling vector
  rewards natively is exactly what a third-family candidate is supposed to do
  better.

Hyperparameter sets (to be filled in when the implementation lands):

- Atari (Montezuma): TBD
- dm_control humanoid: TBD
- minecart (with equal-weight scalarization): TBD
- CartPole / Pendulum (sanity): TBD

Once written, results live under `lab/baselines/ppo/<env_id>/`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameter set. Values populated when the impl lands."""

    env_id: str
    seed: int
    total_env_steps: int
    max_wallclock_s: int
    logdir: str


def train(config: PPOConfig) -> dict:
    raise NotImplementedError(
        "PPO baseline is not yet implemented. "
        "See docs/charter.md §Hard rules for why we author this from primitives."
    )
