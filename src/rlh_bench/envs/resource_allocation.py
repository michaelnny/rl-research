"""Recoverable deterministic resource-allocation environment.

This environment represents a structured large-action alternative to navigation.
At each step the agent allocates a continuous budget across multiple projects.
Projects have soft dependencies, so early allocation to downstream projects is
partly wasted but not fatal. The only feedback is at the terminal step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rlh_bench.core import RewardSpec, StepReturn, make_reward, zero_reward
from rlh_bench.spaces import Box, clip_to_box


@dataclass(frozen=True)
class ResourceAllocationConfig:
    """Configuration for :class:`RecoverableResourceAllocationEnv`.

    Args:
        horizon: Fixed number of allocation decisions.
        num_projects: Number of projects / action dimensions.
        budget: Maximum total allocation per step. Actions with larger sum are
            deterministically projected back to this budget.
        demand: Required terminal progress for each project. If ``None``, all
            demands are one.
        efficiency: Progress gained from one unit of ready allocation. If
            ``None``, efficiencies are mildly decreasing with project index.
        cost: Per-unit resource cost for each project. If ``None``, costs are
            mildly increasing with project index.
        deadlines: Soft project deadlines used only for terminal vector outcome.
            If ``None``, deadlines are spread across the horizon.
        min_readiness: Downstream projects are never completely locked. This is
            what makes bad early actions recoverable rather than fatal.
        safe_allocation: Per-project allocation above this value accumulates a
            terminal safety/constraint violation outcome.
        progress_cap_factor: Cap on overproduction relative to demand.
    """

    horizon: int = 100
    num_projects: int = 5
    budget: float = 1.0
    demand: tuple[float, ...] | None = None
    efficiency: tuple[float, ...] | None = None
    cost: tuple[float, ...] | None = None
    deadlines: tuple[int, ...] | None = None
    min_readiness: float = 0.08
    safe_allocation: float = 0.55
    progress_cap_factor: float = 1.25

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.num_projects <= 1:
            raise ValueError("num_projects must be at least 2")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if not (0.0 <= self.min_readiness <= 1.0):
            raise ValueError("min_readiness must be in [0, 1]")
        if self.safe_allocation < 0:
            raise ValueError("safe_allocation must be non-negative")
        if self.progress_cap_factor < 1.0:
            raise ValueError("progress_cap_factor must be >= 1")
        for field_name in ("demand", "efficiency", "cost", "deadlines"):
            value = getattr(self, field_name)
            if value is not None and len(value) != self.num_projects:
                raise ValueError(f"{field_name} must have length num_projects")

    def demand_array(self) -> np.ndarray:
        if self.demand is None:
            return np.ones(self.num_projects, dtype=np.float32)
        return np.asarray(self.demand, dtype=np.float32)

    def efficiency_array(self) -> np.ndarray:
        if self.efficiency is None:
            # Slightly decreasing efficiencies make late-stage recovery possible
            # but not free.
            return np.linspace(0.092, 0.078, self.num_projects, dtype=np.float32)
        return np.asarray(self.efficiency, dtype=np.float32)

    def cost_array(self) -> np.ndarray:
        if self.cost is None:
            return np.linspace(1.0, 1.35, self.num_projects, dtype=np.float32)
        return np.asarray(self.cost, dtype=np.float32)

    def deadline_array(self) -> np.ndarray:
        if self.deadlines is None:
            # Soft deadlines are not terminal conditions; they only affect the
            # terminal vector outcome.
            raw = np.linspace(0.28, 0.90, self.num_projects) * self.horizon
            return np.asarray(np.ceil(raw), dtype=np.int64)
        return np.asarray(self.deadlines, dtype=np.int64)


DEFAULT_RESOURCE_REWARD_SPEC = RewardSpec(
    names=("success", "service_level", "neg_cost", "neg_delay", "neg_safety_violation"),
    weights=(1.0, 0.65, 0.003, 0.10, 0.08),
)


class RecoverableResourceAllocationEnv:
    """Deterministic long-horizon allocation task with terminal vector feedback.

    Observation: ``[progress_ratio[K], readiness[K], last_allocation[K], t / H]``.

    Action: continuous allocation vector in ``[0, 1]^K``. If the sum exceeds the
    per-step budget, it is scaled down so the total equals the budget.

    Reward: zero until the fixed terminal step. The terminal vector is always
    available through ``info['reward_vector']``.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: ResourceAllocationConfig | None = None,
        reward_spec: RewardSpec = DEFAULT_RESOURCE_REWARD_SPEC,
        reward_mode: str = "scalar",
    ) -> None:
        self.config = ResourceAllocationConfig() if config is None else config
        self.reward_spec = reward_spec
        self.reward_dim = reward_spec.dim
        if reward_mode not in {"scalar", "vector"}:
            raise ValueError("reward_mode must be 'scalar' or 'vector'")
        self.reward_mode = reward_mode

        k = self.config.num_projects
        self.observation_space = Box(low=-1.0, high=1.5, shape=(3 * k + 1,), dtype=np.float32)
        self.action_space = Box(low=0.0, high=1.0, shape=(k,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._progress = np.zeros(k, dtype=np.float32)
        self._last_allocation = np.zeros(k, dtype=np.float32)
        self._total_cost = 0.0
        self._safety_violation = 0.0
        self._completion_times = np.full(k, -1, dtype=np.int64)
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)

        self._demand = self.config.demand_array()
        self._efficiency = self.config.efficiency_array()
        self._cost = self.config.cost_array()
        self._deadlines = self.config.deadline_array()

    @property
    def t(self) -> int:
        return self._t

    @property
    def progress(self) -> np.ndarray:
        return self._progress.copy()

    @property
    def completion_times(self) -> np.ndarray:
        return self._completion_times.copy()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        k = self.config.num_projects
        self._t = 0
        self._progress = np.zeros(k, dtype=np.float32)
        self._last_allocation = np.zeros(k, dtype=np.float32)
        self._total_cost = 0.0
        self._safety_violation = 0.0
        self._completion_times = np.full(k, -1, dtype=np.int64)
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)
        return self._observation(), self._info(np.zeros(self.reward_dim, dtype=np.float32))

    def step(self, action: np.ndarray) -> StepReturn:
        if self._terminated:
            raise RuntimeError("step() called after episode terminated; call reset() first")

        raw_action = clip_to_box(self.action_space, action)
        allocation = self._project_to_budget(raw_action)
        readiness = self._readiness()

        increment = self._efficiency * readiness * allocation
        cap = self._demand * self.config.progress_cap_factor
        self._progress = np.minimum(self._progress + increment, cap).astype(np.float32)
        self._last_allocation = allocation.astype(np.float32)
        self._total_cost += float(np.dot(allocation, self._cost))
        self._safety_violation += float(
            np.sum(np.square(np.maximum(allocation - self.config.safe_allocation, 0.0)))
        )
        self._t += 1

        newly_completed = (self._progress >= self._demand) & (self._completion_times < 0)
        self._completion_times[newly_completed] = self._t

        terminal = self._t >= self.config.horizon
        self._terminated = terminal
        if terminal:
            vector = self._terminal_reward_vector()
            reward = make_reward(self.reward_mode, self.reward_spec, vector)
            self._last_reward_vector = vector.copy()
        else:
            vector = np.zeros(self.reward_dim, dtype=np.float32)
            reward = zero_reward(self.reward_mode, self.reward_spec)

        return self._observation(), reward, terminal, False, self._info(vector)

    def render(self) -> str:
        ratios = np.round(self._progress / self._demand, 3).tolist()
        return (
            f"RecoverableResourceAllocation(t={self._t}, ratios={ratios}, "
            f"success={self.diagnostics()['success']})"
        )

    def diagnostics(self) -> dict[str, float | int | list[int]]:
        ratios = self._progress / self._demand
        service = float(np.mean(np.minimum(ratios, 1.0)))
        success = float(np.all(ratios >= 1.0))
        delay = self._terminal_delay()
        return {
            "t": int(self._t),
            "success": success,
            "service_level": service,
            "cost": float(self._total_cost),
            "delay": float(delay),
            "safety_violation": float(self._safety_violation),
            "completed_projects": int(np.sum(self._completion_times >= 0)),
        }

    def _project_to_budget(self, action: np.ndarray) -> np.ndarray:
        total = float(np.sum(action))
        if total > self.config.budget:
            return (action / total * self.config.budget).astype(np.float32)
        return action.astype(np.float32)

    def _readiness(self) -> np.ndarray:
        k = self.config.num_projects
        readiness = np.ones(k, dtype=np.float32)
        ratios = np.clip(self._progress / self._demand, 0.0, 1.0)
        for i in range(1, k):
            readiness[i] = self.config.min_readiness + (1.0 - self.config.min_readiness) * ratios[i - 1]
        return readiness

    def _observation(self) -> np.ndarray:
        ratios = self._progress / self._demand
        obs = np.concatenate(
            [
                np.clip(ratios, 0.0, self.config.progress_cap_factor),
                self._readiness(),
                self._last_allocation,
                np.asarray([self._t / self.config.horizon], dtype=np.float32),
            ]
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _terminal_delay(self) -> float:
        completion = self._completion_times.copy()
        # Incomplete projects count as completing at the horizon plus shortage
        # severity; this keeps the signal terminal-only but informative.
        incomplete = completion < 0
        completion[incomplete] = self.config.horizon
        base_delay = np.maximum(completion - self._deadlines, 0).astype(np.float32)
        shortage = np.maximum(1.0 - self._progress / self._demand, 0.0)
        return float((np.sum(base_delay) / self.config.horizon) + np.sum(shortage))

    def _terminal_reward_vector(self) -> np.ndarray:
        diag = self.diagnostics()
        return np.asarray(
            [
                diag["success"],
                diag["service_level"],
                -diag["cost"],
                -diag["delay"],
                -diag["safety_violation"],
            ],
            dtype=np.float32,
        )

    def _info(self, reward_vector: np.ndarray) -> dict[str, Any]:
        diag = self.diagnostics()
        diag.update(
            {
                "reward_vector": np.asarray(reward_vector, dtype=np.float32).copy(),
                "reward_names": self.reward_spec.names,
                "is_success": bool(diag["success"]),
                "readiness": self._readiness().copy(),
                "projected_allocation": self._last_allocation.copy(),
            }
        )
        return diag
