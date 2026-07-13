"""Native structured action specifications for the replacement benchmark.

The benchmark reasons about a canonical vector in ``[-1, 1]^d``.  Action
specifications are renderings of that vector, not Gym-shaped containers.  This
lets FactorLab compare action representations while holding dynamics fixed.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class InvalidAction(ValueError):
    """Raised when an action does not satisfy its public specification."""


class ActionEnumerationError(ValueError):
    """Raised when a finite action space cannot be enumerated under a limit."""


@dataclass(frozen=True)
class EncodedAction:
    action: Any
    canonical: tuple[float, ...]


class ActionSpec(ABC):
    """Base contract retained by the native learner API."""

    kind: str
    canonical_dim: int

    @abstractmethod
    def decode(self, action: Any) -> tuple[float, ...]:
        """Validate and map a public action to the canonical decision vector."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one valid public action."""

    @abstractmethod
    def public_schema(self) -> dict[str, Any]:
        """Return a JSON-compatible schema with learner-visible structure."""

    def finite_size(self) -> int | None:
        return None

    def enumerate(self, limit: int = 100_000) -> Iterator[EncodedAction]:
        del limit
        raise ActionEnumerationError(f"{self.kind} is not a finite enumerable action space")


def _float_tuple(values: Sequence[Any], *, length: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise InvalidAction(f"{label} must contain exactly {length} values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise InvalidAction(f"{label} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise InvalidAction(f"{label} must contain finite values")
    return result


@dataclass(frozen=True)
class FactoredDiscreteActionSpec(ActionSpec):
    levels: tuple[tuple[float, ...], ...]
    kind: str = "factored_discrete"

    def __post_init__(self) -> None:
        if not self.levels or any(len(levels) < 2 for levels in self.levels):
            raise ValueError("each discrete factor needs at least two levels")
        for levels in self.levels:
            if tuple(sorted(set(levels))) != levels:
                raise ValueError("factor levels must be unique and sorted")
            if levels[0] < -1.0 or levels[-1] > 1.0:
                raise ValueError("canonical factor levels must lie in [-1, 1]")

    @property
    def canonical_dim(self) -> int:
        return len(self.levels)

    def finite_size(self) -> int:
        return math.prod(len(levels) for levels in self.levels)

    def decode(self, action: Any) -> tuple[float, ...]:
        if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
            raise InvalidAction("factored action must be a sequence of integer indices")
        if len(action) != self.canonical_dim:
            raise InvalidAction(f"factored action needs {self.canonical_dim} indices")
        canonical: list[float] = []
        for factor, (index, levels) in enumerate(zip(action, self.levels, strict=True)):
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise InvalidAction(f"factor {factor} must be an integer index")
            if index < 0 or index >= len(levels):
                raise InvalidAction(f"factor {factor} index is out of range")
            canonical.append(levels[int(index)])
        return tuple(canonical)

    def sample(self, rng: np.random.Generator) -> tuple[int, ...]:
        return tuple(int(rng.integers(len(levels))) for levels in self.levels)

    def public_schema(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "factors": [
                {"name": f"factor_{index}", "levels": list(levels)}
                for index, levels in enumerate(self.levels)
            ],
            "joint_choices": self.finite_size(),
        }

    def enumerate(self, limit: int = 100_000) -> Iterator[EncodedAction]:
        size = self.finite_size()
        if size > limit:
            raise ActionEnumerationError(f"{size} joint actions exceed enumeration limit {limit}")
        ranges = (range(len(levels)) for levels in self.levels)
        for action in itertools.product(*ranges):
            yield EncodedAction(action=action, canonical=self.decode(action))


@dataclass(frozen=True)
class FlatDiscreteActionSpec(ActionSpec):
    canonical_actions: tuple[tuple[float, ...], ...]
    kind: str = "flat_discrete"

    def __post_init__(self) -> None:
        _validate_catalog(self.canonical_actions)

    @property
    def canonical_dim(self) -> int:
        return len(self.canonical_actions[0])

    def finite_size(self) -> int:
        return len(self.canonical_actions)

    def decode(self, action: Any) -> tuple[float, ...]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise InvalidAction("flat discrete action must be an integer")
        if action < 0 or action >= len(self.canonical_actions):
            raise InvalidAction("flat discrete action is out of range")
        return self.canonical_actions[int(action)]

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(len(self.canonical_actions)))

    def public_schema(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n": len(self.canonical_actions),
            "canonical_dim": self.canonical_dim,
        }

    def enumerate(self, limit: int = 100_000) -> Iterator[EncodedAction]:
        if self.finite_size() > limit:
            raise ActionEnumerationError(
                f"{self.finite_size()} actions exceed enumeration limit {limit}"
            )
        for action, canonical in enumerate(self.canonical_actions):
            yield EncodedAction(action=action, canonical=canonical)


def _validate_catalog(canonical_actions: tuple[tuple[float, ...], ...]) -> None:
    if not canonical_actions:
        raise ValueError("an action catalog cannot be empty")
    width = len(canonical_actions[0])
    if width == 0 or any(len(action) != width for action in canonical_actions):
        raise ValueError("all canonical catalog actions must have the same non-zero width")
    if len(set(canonical_actions)) != len(canonical_actions):
        raise ValueError("canonical catalog actions must be unique")
    if any(not -1.0 <= value <= 1.0 for action in canonical_actions for value in action):
        raise ValueError("canonical catalog actions must lie in [-1, 1]")


@dataclass(frozen=True)
class EmbeddedCatalogActionSpec(ActionSpec):
    """An item-selection action whose features are public but dynamics are not."""

    canonical_actions: tuple[tuple[float, ...], ...]
    features: tuple[tuple[float, ...], ...]
    kind: str = "embedded_catalog"

    def __post_init__(self) -> None:
        _validate_catalog(self.canonical_actions)
        if len(self.features) != len(self.canonical_actions) or not self.features:
            raise ValueError("every catalog item needs a public feature vector")
        width = len(self.features[0])
        if width == 0 or any(len(feature) != width for feature in self.features):
            raise ValueError("catalog feature vectors must have a common non-zero width")
        if any(not math.isfinite(value) for feature in self.features for value in feature):
            raise ValueError("catalog features must be finite")

    @property
    def canonical_dim(self) -> int:
        return len(self.canonical_actions[0])

    @property
    def feature_dim(self) -> int:
        return len(self.features[0])

    def finite_size(self) -> int:
        return len(self.canonical_actions)

    def decode(self, action: Any) -> tuple[float, ...]:
        if not isinstance(action, Mapping) or set(action) != {"item"}:
            raise InvalidAction("catalog action must be {'item': integer_id}")
        item = action["item"]
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise InvalidAction("catalog item id must be an integer")
        if item < 0 or item >= len(self.canonical_actions):
            raise InvalidAction("catalog item id is out of range")
        return self.canonical_actions[int(item)]

    def sample(self, rng: np.random.Generator) -> dict[str, int]:
        return {"item": int(rng.integers(len(self.canonical_actions)))}

    def public_schema(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "selection": "one",
            "catalog_size": len(self.canonical_actions),
            "feature_dim": self.feature_dim,
            "items": [
                {"item": index, "features": list(features)}
                for index, features in enumerate(self.features)
            ],
        }

    def enumerate(self, limit: int = 100_000) -> Iterator[EncodedAction]:
        if self.finite_size() > limit:
            raise ActionEnumerationError(
                f"{self.finite_size()} items exceed enumeration limit {limit}"
            )
        for item, canonical in enumerate(self.canonical_actions):
            yield EncodedAction(action={"item": item}, canonical=canonical)


@dataclass(frozen=True)
class ContinuousActionSpec(ActionSpec):
    low: tuple[float, ...]
    high: tuple[float, ...]
    kind: str = "continuous"

    def __post_init__(self) -> None:
        if not self.low or len(self.low) != len(self.high):
            raise ValueError("continuous bounds need a common non-zero width")
        if any(not -1.0 <= low < high <= 1.0 for low, high in zip(self.low, self.high)):
            raise ValueError("continuous canonical bounds must satisfy -1 <= low < high <= 1")

    @property
    def canonical_dim(self) -> int:
        return len(self.low)

    def decode(self, action: Any) -> tuple[float, ...]:
        if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
            raise InvalidAction("continuous action must be a numeric sequence")
        values = _float_tuple(action, length=self.canonical_dim, label="continuous action")
        if any(
            value < low or value > high
            for value, low, high in zip(values, self.low, self.high, strict=True)
        ):
            raise InvalidAction("continuous action is outside its bounds")
        return values

    def sample(self, rng: np.random.Generator) -> tuple[float, ...]:
        return tuple(float(value) for value in rng.uniform(self.low, self.high))

    def public_schema(self) -> dict[str, Any]:
        return {"kind": self.kind, "low": list(self.low), "high": list(self.high)}


@dataclass(frozen=True)
class HybridBranch:
    name: str
    active_indices: tuple[int, ...]
    defaults: tuple[float, ...]


@dataclass(frozen=True)
class ConditionalHybridActionSpec(ActionSpec):
    low: tuple[float, ...]
    high: tuple[float, ...]
    branches: tuple[HybridBranch, ...]
    kind: str = "conditional_hybrid"

    def __post_init__(self) -> None:
        if not self.low or len(self.low) != len(self.high):
            raise ValueError("hybrid bounds need a common non-zero width")
        if not self.branches:
            raise ValueError("hybrid actions need at least one branch")
        names = {branch.name for branch in self.branches}
        if len(names) != len(self.branches):
            raise ValueError("hybrid branch names must be unique")
        for branch in self.branches:
            if len(branch.defaults) != len(self.low):
                raise ValueError("branch defaults must cover the canonical vector")
            if not branch.active_indices or len(set(branch.active_indices)) != len(
                branch.active_indices
            ):
                raise ValueError("branch active indices must be unique and non-empty")
            if any(index < 0 or index >= len(self.low) for index in branch.active_indices):
                raise ValueError("branch active index is out of range")
            if any(
                value < low or value > high
                for value, low, high in zip(
                    branch.defaults, self.low, self.high, strict=True
                )
            ):
                raise ValueError("branch default is outside canonical bounds")

    @property
    def canonical_dim(self) -> int:
        return len(self.low)

    def _branch(self, selector: Any) -> HybridBranch:
        if isinstance(selector, bool):
            raise InvalidAction("hybrid branch must be a name or integer index")
        if isinstance(selector, (int, np.integer)):
            if 0 <= selector < len(self.branches):
                return self.branches[int(selector)]
            raise InvalidAction("hybrid branch index is out of range")
        if isinstance(selector, str):
            for branch in self.branches:
                if branch.name == selector:
                    return branch
            raise InvalidAction("hybrid branch name is unknown")
        raise InvalidAction("hybrid branch must be a name or integer index")

    def decode(self, action: Any) -> tuple[float, ...]:
        if not isinstance(action, Mapping) or set(action) != {"branch", "parameters"}:
            raise InvalidAction("hybrid action needs branch and parameters fields")
        branch = self._branch(action["branch"])
        parameters = action["parameters"]
        if isinstance(parameters, (str, bytes)) or not isinstance(parameters, Sequence):
            raise InvalidAction("hybrid parameters must be a numeric sequence")
        values = _float_tuple(
            parameters, length=len(branch.active_indices), label="hybrid parameters"
        )
        canonical = list(branch.defaults)
        for value, index in zip(values, branch.active_indices, strict=True):
            if value < self.low[index] or value > self.high[index]:
                raise InvalidAction(f"hybrid parameter for dimension {index} is out of bounds")
            canonical[index] = value
        return tuple(canonical)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        branch = self.branches[int(rng.integers(len(self.branches)))]
        return {
            "branch": branch.name,
            "parameters": [
                float(rng.uniform(self.low[index], self.high[index]))
                for index in branch.active_indices
            ],
        }

    def public_schema(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "canonical_dim": self.canonical_dim,
            "branches": [
                {
                    "name": branch.name,
                    "parameters": [
                        {
                            "canonical_index": index,
                            "low": self.low[index],
                            "high": self.high[index],
                        }
                        for index in branch.active_indices
                    ],
                }
                for branch in self.branches
            ],
        }


def canonical_levels(count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("a discrete factor needs at least two levels")
    return tuple(float(value) for value in np.linspace(-1.0, 1.0, count))


def make_action_spec(
    mode: str,
    *,
    n_factors: int,
    levels_per_factor: int | Sequence[int] = 2,
    catalog_size: int | None = None,
    seed: int = 0,
    flat_limit: int = 100_000,
) -> ActionSpec:
    """Construct one rendering without consulting any legacy action classes."""

    if n_factors < 1:
        raise ValueError("n_factors must be positive")
    if isinstance(levels_per_factor, int):
        counts = (levels_per_factor,) * n_factors
    else:
        counts = tuple(int(count) for count in levels_per_factor)
        if len(counts) != n_factors:
            raise ValueError("levels_per_factor must match n_factors")
    levels = tuple(canonical_levels(count) for count in counts)

    if mode == "factored_discrete":
        return FactoredDiscreteActionSpec(levels=levels)
    if mode == "flat_discrete":
        size = math.prod(counts)
        if size > flat_limit:
            raise ValueError(f"flat rendering has {size} actions, above limit {flat_limit}")
        grid = tuple(tuple(values) for values in itertools.product(*levels))
        return FlatDiscreteActionSpec(canonical_actions=grid)
    if mode == "embedded_catalog":
        if catalog_size is None or catalog_size < 2:
            raise ValueError("embedded_catalog requires catalog_size >= 2")
        rng = np.random.default_rng(seed)
        joint = math.prod(counts)
        if joint <= flat_limit:
            full_grid = tuple(tuple(values) for values in itertools.product(*levels))
            indices = rng.choice(joint, size=min(catalog_size, joint), replace=False)
            canonical = tuple(full_grid[int(index)] for index in sorted(indices))
        else:
            seen: set[tuple[float, ...]] = set()
            while len(seen) < min(catalog_size, joint):
                seen.add(tuple(level[int(rng.integers(len(level)))] for level in levels))
            canonical = tuple(sorted(seen))
        projection = rng.normal(size=(n_factors, max(2, n_factors)))
        projected = np.asarray(canonical) @ projection
        scale = np.maximum(np.max(np.abs(projected), axis=0), 1e-12)
        features = tuple(tuple(float(value) for value in row / scale) for row in projected)
        return EmbeddedCatalogActionSpec(canonical_actions=canonical, features=features)
    if mode == "continuous":
        return ContinuousActionSpec(low=(-1.0,) * n_factors, high=(1.0,) * n_factors)
    if mode == "conditional_hybrid":
        even = tuple(range(0, n_factors, 2))
        odd = tuple(range(1, n_factors, 2))
        if not odd:
            odd = (0,)
        branches = (
            HybridBranch("even", even, (-1.0,) * n_factors),
            HybridBranch("odd", odd, (1.0,) * n_factors),
        )
        return ConditionalHybridActionSpec(
            low=(-1.0,) * n_factors,
            high=(1.0,) * n_factors,
            branches=branches,
        )
    raise ValueError(f"unknown action mode: {mode}")
