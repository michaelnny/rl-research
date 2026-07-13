"""Evaluator-owned resource accounting for comparable RL experiments."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


class BudgetExceeded(RuntimeError):
    def __init__(self, resource: str, limit: int | float):
        super().__init__(f"evaluation budget exceeded: {resource} limit is {limit}")
        self.resource = resource
        self.limit = limit


@dataclass(frozen=True)
class BudgetLimits:
    transitions: int
    episodes: int
    wall_seconds: float
    policies: int = 1
    preference_queries: int = 0

    def __post_init__(self) -> None:
        if self.transitions < 1 or self.episodes < 1 or self.wall_seconds <= 0.0:
            raise ValueError("transition, episode, and wall-time limits must be positive")
        if self.policies < 1 or self.preference_queries < 0:
            raise ValueError("policy limit must be positive and query limit non-negative")


@dataclass(frozen=True)
class BudgetUsage:
    transitions: int
    episodes: int
    wall_seconds: float
    policies: int
    preference_queries: int


class BudgetLedger:
    def __init__(
        self,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limits = limits
        self._clock = clock
        self._started_at = clock()
        self._transitions = 0
        self._episodes = 0
        self._policies = 0
        self._preference_queries = 0
        self._lock = threading.Lock()

    def _elapsed(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def _check_wall(self) -> None:
        if self._elapsed() >= self.limits.wall_seconds:
            raise BudgetExceeded("wall_seconds", self.limits.wall_seconds)

    def start_episode(self) -> None:
        with self._lock:
            self._check_wall()
            if self._episodes >= self.limits.episodes:
                raise BudgetExceeded("episodes", self.limits.episodes)
            self._episodes += 1

    def record_transition(self) -> None:
        with self._lock:
            self._check_wall()
            if self._transitions >= self.limits.transitions:
                raise BudgetExceeded("transitions", self.limits.transitions)
            self._transitions += 1

    def register_policy(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("policy count must be positive")
        with self._lock:
            self._check_wall()
            if self._policies + count > self.limits.policies:
                raise BudgetExceeded("policies", self.limits.policies)
            self._policies += count

    def record_preference_query(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("preference query count must be positive")
        with self._lock:
            self._check_wall()
            if self._preference_queries + count > self.limits.preference_queries:
                raise BudgetExceeded("preference_queries", self.limits.preference_queries)
            self._preference_queries += count

    def check(self) -> None:
        with self._lock:
            self._check_wall()

    def snapshot(self) -> BudgetUsage:
        with self._lock:
            return BudgetUsage(
                transitions=self._transitions,
                episodes=self._episodes,
                wall_seconds=self._elapsed(),
                policies=self._policies,
                preference_queries=self._preference_queries,
            )


class BudgetedEnv:
    """Count accepted resets and transitions without trusting candidate code."""

    def __init__(self, env: Any, ledger: BudgetLedger):
        self.env = env
        self.ledger = ledger
        self.action_spec = env.action_spec

    def reset(self, **kwargs: Any) -> Any:
        self.ledger.start_episode()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        self.ledger.record_transition()
        return self.env.step(action)
