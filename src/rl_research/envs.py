"""Vector environment adapters for the four families in our benchmark suite.

Each adapter exposes the same interface so a candidate ``train.py`` can write
its rollout loop once and run on any benchmark:

    adapter.reset(seed)              -> obs of shape (n_envs, *obs_shape)
    adapter.step(actions)            -> (next_obs, rewards, terminated, truncated, final_obs)
    adapter.pop_completed()          -> list of completed-episode returns
    adapter.close()
    adapter.single_observation_space, adapter.single_action_space
    adapter.n_envs

We do not use ``gymnasium.vector.VectorEnv`` because (a) we mix gymnasium,
``dm_control``, and ``mo_gymnasium``, (b) we want explicit auto-reset semantics
(``final_obs`` is always returned alongside ``next_obs`` so GAE can bootstrap
the truncated state), (c) Atari preprocessing varies and is easier to wire here.

Reward semantics differ slightly across adapters:
- ``GymVecAdapter`` and ``DMControlVecAdapter``: scalar rewards; ``pop_completed()``
  returns ``list[float]``.
- ``MinecartVecAdapter``: vector rewards; the algorithm receives a scalarized
  reward in ``step()`` (equal-weight by default) and ``pop_completed()`` returns
  ``list[np.ndarray]`` of per-channel completed returns.

References:
- Atari preprocessing: Mnih et al. 2015 "Human-level control..." Methods §Frame
  preprocessing — the (84x84 grayscale, 4-frame skip + max, 4-frame stack,
  noop-30 reset) recipe. Implemented via the canonical
  ``gymnasium.wrappers.AtariPreprocessing`` + ``FrameStackObservation``.
- dm_control: Tassa et al. 2018 "DeepMind Control Suite" arXiv:1801.00690 —
  observation is a dict of named ``ndarray``; we flatten by concatenating in
  the iteration order of the dict (insertion order, stable across resets).
- minecart-v0: Abels, Roijers, Lenaerts, Nowé, Steckelmacher 2019 "Dynamic
  Weights in Multi-Objective Deep Reinforcement Learning"
  (https://arxiv.org/abs/1809.07803). 3-channel reward
  ``(ore_1, ore_2, fuel_cost)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Env classification
# ---------------------------------------------------------------------------

DISCRETE_CLASSIC = frozenset({"CartPole-v1", "Acrobot-v1", "MountainCar-v0"})
CONTINUOUS_CLASSIC = frozenset({"Pendulum-v1", "MountainCarContinuous-v0"})


def adapter_family(env_id: str) -> str:
    """Pick the adapter family for an env id. Used by both the training loop
    and the eval helper to dispatch on env semantics."""
    if env_id in DISCRETE_CLASSIC or env_id in CONTINUOUS_CLASSIC:
        return "gym-classic"
    if env_id.startswith("ALE/"):
        return "gym-atari"
    if env_id == "minecart-v0":
        return "mo-minecart"
    if env_id == "humanoid.run":
        return "dm-control"
    raise ValueError(f"no adapter family registered for env_id={env_id!r}")


# ---------------------------------------------------------------------------
# Single-env factories (also used by evaluate.py for non-vectorized eval)
# ---------------------------------------------------------------------------


def make_classic_env(env_id: str) -> Callable[[int], Any]:
    """Build a thunk for a classic-control gymnasium env. ``ClipAction`` is
    applied to ``Box`` action spaces ([37D §5 continuous]; CleanRL
    ppo_continuous_action.py L94) — without it, unbounded Gaussian samples are
    silently clipped inside the env, creating a mismatch between executed
    action and the log-prob used in any importance ratio."""

    def _f(seed: int):
        import gymnasium as gym

        env = gym.make(env_id)
        if env.action_space.__class__.__name__ == "Box":
            env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed)
        return env

    return _f


def make_atari_env(env_id: str) -> Callable[[int], Any]:
    """Build a thunk for an ALE env with the canonical (Mnih 2015 / Schulman
    2017 Atari) preprocessing pipeline: noop-30 reset, frame_skip=4, 84x84
    grayscale, 4-frame stack. ``terminal_on_life_loss`` is left off because
    PPO and most third-family candidates train across full episodes."""

    def _f(seed: int):
        import ale_py
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

        gym.register_envs(ale_py)
        env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        env = FrameStackObservation(env, stack_size=4)
        env.action_space.seed(seed)
        return env

    return _f


# ---------------------------------------------------------------------------
# Vector adapters
# ---------------------------------------------------------------------------


class GymVecAdapter:
    """Thin synchronous vector wrapper around N gymnasium envs.

    On terminal/truncation the env is auto-reset; ``step()`` returns the
    pre-reset observation in ``final_obs`` so the caller can compute
    ``V(s_truncated)`` for GAE without losing it to the auto-reset.
    """

    def __init__(self, make_env: Callable[[int], Any], n_envs: int, seed: int):
        self.envs = [make_env(seed + i) for i in range(n_envs)]
        self.n_envs = n_envs
        self.completed: list[float] = []
        self._returns = np.zeros(n_envs, dtype=np.float64)

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space

    @property
    def single_action_space(self):
        return self.envs[0].action_space

    def reset(self, seed: int) -> np.ndarray:
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            obs_list.append(np.asarray(obs))
        self._returns[:] = 0.0
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        next_obs_list = []
        final_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, _ = env.step(actions[i])
            self._returns[i] += float(reward)
            rewards[i] = float(reward)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            if term or trunc:
                self.completed.append(float(self._returns[i]))
                self._returns[i] = 0.0
                final_obs_list.append(np.asarray(obs))
                obs, _ = env.reset()
            else:
                final_obs_list.append(np.asarray(obs))
            next_obs_list.append(np.asarray(obs))
        return (
            np.stack(next_obs_list, axis=0),
            rewards,
            terminated,
            truncated,
            np.stack(final_obs_list, axis=0),
        )

    def pop_completed(self) -> list[float]:
        out, self.completed = self.completed, []
        return out

    def close(self) -> None:
        for env in self.envs:
            env.close()


class MinecartVecAdapter:
    """``mo_gymnasium`` minecart-v0 with a configurable scalarization weight.

    The native ``step()`` returns a 3-vector reward; we scalarize for the
    inner update (PPO and other scalar-credit baselines) but expose the raw
    per-channel completed returns through ``pop_completed()`` so the eval
    layer can log them. Default scalarization is equal-weight.
    """

    def __init__(
        self,
        n_envs: int,
        seed: int,
        scalarization: np.ndarray | None = None,
    ):
        import mo_gymnasium as mo_gym

        self.envs = [mo_gym.make("minecart-v0") for _ in range(n_envs)]
        for i, env in enumerate(self.envs):
            env.action_space.seed(seed + i)
        self.n_envs = n_envs
        self.completed: list[np.ndarray] = []
        self._returns = np.zeros((n_envs, 3), dtype=np.float64)
        if scalarization is None:
            self._scalarization = np.array([1.0, 1.0, 1.0]) / 3.0
        else:
            self._scalarization = np.asarray(scalarization, dtype=np.float64)

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space

    @property
    def single_action_space(self):
        return self.envs[0].action_space

    def reset(self, seed: int) -> np.ndarray:
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            obs_list.append(np.asarray(obs, dtype=np.float32))
        self._returns[:] = 0.0
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        next_obs_list = []
        final_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            obs, reward_vec, term, trunc, _ = env.step(int(actions[i]))
            r_vec = np.asarray(reward_vec, dtype=np.float64)
            self._returns[i] += r_vec
            rewards[i] = float(np.dot(r_vec, self._scalarization))
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            if term or trunc:
                self.completed.append(self._returns[i].copy())
                self._returns[i] = 0.0
                final_obs_list.append(np.asarray(obs, dtype=np.float32))
                obs, _ = env.reset()
            else:
                final_obs_list.append(np.asarray(obs, dtype=np.float32))
            next_obs_list.append(np.asarray(obs, dtype=np.float32))
        return (
            np.stack(next_obs_list, axis=0),
            rewards,
            terminated,
            truncated,
            np.stack(final_obs_list, axis=0),
        )

    def pop_completed(self) -> list[np.ndarray]:
        out, self.completed = self.completed, []
        return out

    def close(self) -> None:
        for env in self.envs:
            env.close()


class DMControlVecAdapter:
    """Vector adapter for ``dm_control`` suite tasks (e.g. ``humanoid.run``).

    dm_control envs have no real terminal — every episode ends at the
    environment's time-limit. We surface that as ``truncated=True`` so a GAE
    implementation bootstraps with ``V(final_obs)`` instead of zero
    ([37D §"time-limits"]).

    Observations are dict-of-arrays; we flatten in the dict's insertion order
    (Python 3.7+ stable). We cache the last-emitted obs so ``reset()`` can be
    called multiple times without re-rolling the environment.
    """

    def __init__(self, domain: str, task: str, n_envs: int, seed: int):
        from dm_control import suite

        self.envs = [
            suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed + i})
            for i in range(n_envs)
        ]
        self.n_envs = n_envs
        self.completed: list[float] = []
        self._returns = np.zeros(n_envs, dtype=np.float64)

        ts = self.envs[0].reset()
        self._obs_dim = int(sum(np.prod(v.shape) for v in ts.observation.values()))
        spec = self.envs[0].action_spec()
        self._action_dim = int(np.prod(spec.shape))
        self._action_low = np.broadcast_to(spec.minimum, spec.shape).astype(np.float32)
        self._action_high = np.broadcast_to(spec.maximum, spec.shape).astype(np.float32)
        self._cached = np.stack(
            [self._flatten(self.envs[i].reset().observation) for i in range(n_envs)], axis=0
        )

    @staticmethod
    def _flatten(observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(v, dtype=np.float32).ravel() for v in observation.values()]
        )

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high

    def reset(self, seed: int) -> np.ndarray:
        return self._cached

    def step(self, actions: np.ndarray):
        next_obs_list = []
        final_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            action = np.clip(actions[i], self._action_low, self._action_high).astype(np.float32)
            ts = env.step(action)
            r = float(ts.reward) if ts.reward is not None else 0.0
            self._returns[i] += r
            rewards[i] = r
            if ts.last():
                truncated[i] = True
                self.completed.append(float(self._returns[i]))
                self._returns[i] = 0.0
                final_obs_list.append(self._flatten(ts.observation))
                ts = env.reset()
            else:
                final_obs_list.append(self._flatten(ts.observation))
            next_obs_list.append(self._flatten(ts.observation))
        out = np.stack(next_obs_list, axis=0)
        self._cached = out
        return out, rewards, terminated, truncated, np.stack(final_obs_list, axis=0)

    def pop_completed(self) -> list[float]:
        out, self.completed = self.completed, []
        return out

    def close(self) -> None:
        # dm_control envs hold MuJoCo simulation handles + viewer state. The
        # base API has no .close(), but each env wraps a Physics object whose
        # destructor releases the GLFW context. Drop our references explicitly
        # so this happens at close() time, not at GC. Safe to call twice.
        for env in self.envs:
            with contextlib.suppress(Exception):
                env._physics = None  # type: ignore[attr-defined]
        self.envs = []
        self._cached = None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def make_vec(env_id: str, n_envs: int, seed: int) -> Any:
    """Build the appropriate vector adapter for ``env_id``. Use this in
    candidate train loops to keep the adapter selection in one place. For
    ``minecart-v0`` the default equal-weight scalarization is applied; pass
    a custom scalarization by constructing ``MinecartVecAdapter`` directly."""
    fam = adapter_family(env_id)
    if fam == "gym-classic":
        return GymVecAdapter(make_classic_env(env_id), n_envs, seed)
    if fam == "gym-atari":
        return GymVecAdapter(make_atari_env(env_id), n_envs, seed)
    if fam == "mo-minecart":
        return MinecartVecAdapter(n_envs, seed)
    if fam == "dm-control":
        return DMControlVecAdapter("humanoid", "run", n_envs, seed)
    raise ValueError(f"unknown env family for {env_id!r}")


__all__ = [
    "CONTINUOUS_CLASSIC",
    "DISCRETE_CLASSIC",
    "DMControlVecAdapter",
    "GymVecAdapter",
    "MinecartVecAdapter",
    "adapter_family",
    "make_atari_env",
    "make_classic_env",
    "make_vec",
]
