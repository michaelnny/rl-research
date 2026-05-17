"""Deterministic single-env evaluation, algorithm-agnostic.

The contract requires every ``train.py`` to log ``eval/return_mean`` and
``eval/return_std`` averaged over ≥10 episodes at ≥20 evals per run
(``docs/contract.md``). This module provides one canonical implementation so
candidates do not re-derive it.

The ``policy_fn`` is a callable ``(obs: np.ndarray) -> np.ndarray`` taking a
single un-batched observation and returning a single action. Its job is the
*deterministic* policy at eval time — typically argmax of logits for discrete
spaces or the action mean for Gaussian policies. We intentionally do NOT pass
it the network — eval should not care whether the algorithm has a network at
all. (Random search and ES candidates have no network, but still need to log
returns.)

Returns ``(mean, std, per_channel)`` — ``per_channel`` is an ``np.ndarray`` of
mean per-channel returns for ``minecart-v0`` and ``None`` otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .envs import (
    DMControlVecAdapter,
    adapter_family,
    make_atari_env,
    make_classic_env,
)

PolicyFn = Callable[[np.ndarray], np.ndarray]
ObsTransform = Callable[[np.ndarray], np.ndarray]


def evaluate(
    env_id: str,
    policy_fn: PolicyFn,
    *,
    seed: int,
    n_episodes: int = 10,
    obs_transform: ObsTransform | None = None,
    max_steps_per_episode: int = 10_000,
) -> tuple[float, float, np.ndarray | None]:
    """Roll ``policy_fn`` for ``n_episodes`` deterministic episodes; return
    ``(mean, std, per_channel)``.

    ``obs_transform`` is applied to each observation before passing it to
    ``policy_fn`` — used by PPO to normalize observations against frozen
    running stats at eval time.

    The eval seed is offset deterministically from ``seed`` so that the
    eval distribution is independent from the training-time env seeds:

      reset_seed = seed + 5000 + episode_index    (gym/mo)
      task_seed  = seed + 99                       (dm_control)
      action seed= seed + 10000                    (gym one-shot env)
    """
    fam = adapter_family(env_id)

    if fam == "gym-classic":
        env = make_classic_env(env_id)(seed + 10_000)
        return _evaluate_gym(env, policy_fn, seed, n_episodes, obs_transform, max_steps_per_episode)
    if fam == "gym-atari":
        env = make_atari_env(env_id)(seed + 10_000)
        return _evaluate_gym(
            env, policy_fn, seed, n_episodes, None, max_steps_per_episode, atari=True
        )
    if fam == "mo-minecart":
        import mo_gymnasium as mo_gym

        env = mo_gym.make("minecart-v0")
        return _evaluate_minecart(
            env, policy_fn, seed, n_episodes, obs_transform, max_steps_per_episode
        )
    if fam == "dm-control":
        from dm_control import suite

        env = suite.load(domain_name="humanoid", task_name="run", task_kwargs={"random": seed + 99})
        return _evaluate_dmcontrol(env, policy_fn, n_episodes, obs_transform, max_steps_per_episode)
    raise ValueError(f"no evaluator for env_id={env_id!r}")


def _evaluate_gym(
    env: Any,
    policy_fn: PolicyFn,
    seed: int,
    n_episodes: int,
    obs_transform: ObsTransform | None,
    max_steps: int,
    atari: bool = False,
) -> tuple[float, float, np.ndarray | None]:
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 5_000 + ep)
        obs = np.asarray(obs)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_for_net = obs if atari else (obs_transform(obs) if obs_transform else obs)
            action = policy_fn(obs_for_net)
            obs, r, term, trunc, _ = env.step(action)
            obs = np.asarray(obs)
            total += float(r)
            done = term or trunc
            steps += 1
        returns.append(total)
    if hasattr(env, "close"):
        env.close()
    return _summarize(returns, None)


def _evaluate_minecart(
    env: Any,
    policy_fn: PolicyFn,
    seed: int,
    n_episodes: int,
    obs_transform: ObsTransform | None,
    max_steps: int,
) -> tuple[float, float, np.ndarray | None]:
    returns: list[float] = []
    per_channel: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 5_000 + ep)
        obs = np.asarray(obs, dtype=np.float32)
        done = False
        ret_vec = np.zeros(3, dtype=np.float64)
        total_scalar = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_for_net = obs_transform(obs) if obs_transform else obs
            action = int(policy_fn(obs_for_net))
            obs, r_vec, term, trunc, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            r_vec = np.asarray(r_vec, dtype=np.float64)
            ret_vec += r_vec
            total_scalar += float(r_vec.sum())
            done = term or trunc
            steps += 1
        returns.append(total_scalar)
        per_channel.append(ret_vec.copy())
    if hasattr(env, "close"):
        env.close()
    return _summarize(returns, per_channel)


def _evaluate_dmcontrol(
    env: Any,
    policy_fn: PolicyFn,
    n_episodes: int,
    obs_transform: ObsTransform | None,
    max_steps: int,
) -> tuple[float, float, np.ndarray | None]:
    spec = env.action_spec()
    action_low = np.broadcast_to(spec.minimum, spec.shape).astype(np.float32)
    action_high = np.broadcast_to(spec.maximum, spec.shape).astype(np.float32)

    returns: list[float] = []
    for _ep in range(n_episodes):
        ts = env.reset()
        obs = DMControlVecAdapter._flatten(ts.observation)
        total = 0.0
        steps = 0
        while not ts.last() and steps < max_steps:
            obs_for_net = obs_transform(obs) if obs_transform else obs
            action = policy_fn(obs_for_net)
            a = np.clip(action, action_low, action_high).astype(np.float32)
            ts = env.step(a)
            obs = DMControlVecAdapter._flatten(ts.observation)
            total += float(ts.reward) if ts.reward is not None else 0.0
            steps += 1
        returns.append(total)
    return _summarize(returns, None)


def _summarize(
    returns: list[float], per_channel: list[np.ndarray] | None
) -> tuple[float, float, np.ndarray | None]:
    arr = np.asarray(returns, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else float("nan")
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    pc = np.stack(per_channel, axis=0).mean(axis=0) if per_channel else None
    return mean, std, pc


__all__ = ["PolicyFn", "evaluate"]
