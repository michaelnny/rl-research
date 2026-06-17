"""Small NumPy space classes used to keep the environments self-contained.

The classes intentionally mirror the small subset of Gymnasium spaces needed by
this package: ``sample``, ``contains``, ``shape``, and ``dtype``. A Gymnasium
adapter is provided separately for projects that want true ``gymnasium.spaces``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class Space:
    """Minimal interface shared by all spaces in this package."""

    shape: tuple[int, ...]
    dtype: np.dtype

    def sample(self, rng: np.random.Generator | None = None):  # pragma: no cover - interface
        raise NotImplementedError

    def contains(self, x) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Box(Space):
    """Continuous interval space.

    Args:
        low: Scalar or array-like lower bound.
        high: Scalar or array-like upper bound.
        shape: Shape used when low/high are scalars.
        dtype: NumPy dtype for samples and validation.
    """

    low: np.ndarray | float
    high: np.ndarray | float
    shape: tuple[int, ...] | None = None
    dtype: np.dtype | type = np.float32

    def __post_init__(self) -> None:
        dtype = np.dtype(self.dtype)
        low_arr = np.array(self.low, dtype=dtype)
        high_arr = np.array(self.high, dtype=dtype)

        if self.shape is not None:
            target_shape = tuple(self.shape)
            if low_arr.shape == ():
                low_arr = np.full(target_shape, low_arr.item(), dtype=dtype)
            if high_arr.shape == ():
                high_arr = np.full(target_shape, high_arr.item(), dtype=dtype)
        else:
            target_shape = np.broadcast_shapes(low_arr.shape, high_arr.shape)
            low_arr = np.broadcast_to(low_arr, target_shape).astype(dtype, copy=True)
            high_arr = np.broadcast_to(high_arr, target_shape).astype(dtype, copy=True)

        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high must broadcast to the same shape")
        if np.any(low_arr > high_arr):
            raise ValueError("all low values must be <= high values")

        object.__setattr__(self, "low", low_arr)
        object.__setattr__(self, "high", high_arr)
        object.__setattr__(self, "shape", low_arr.shape)
        object.__setattr__(self, "dtype", dtype)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        sample = rng.uniform(self.low, self.high)
        return sample.astype(self.dtype)

    def contains(self, x) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)


@dataclass(frozen=True)
class Discrete(Space):
    """Integer space ``{0, ..., n - 1}``."""

    n: int
    dtype: np.dtype | type = np.int64

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        object.__setattr__(self, "shape", ())
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = np.random.default_rng() if rng is None else rng
        return int(rng.integers(0, self.n))

    def contains(self, x) -> bool:
        try:
            value = int(x)
        except (TypeError, ValueError):
            return False
        return 0 <= value < self.n


@dataclass(frozen=True)
class MultiDiscrete(Space):
    """Cartesian product of discrete spaces."""

    nvec: Sequence[int]
    dtype: np.dtype | type = np.int64

    def __post_init__(self) -> None:
        nvec = np.asarray(self.nvec, dtype=np.int64)
        if nvec.ndim != 1 or np.any(nvec <= 0):
            raise ValueError("nvec must be a 1D sequence of positive integers")
        object.__setattr__(self, "nvec", nvec)
        object.__setattr__(self, "shape", nvec.shape)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return np.asarray([rng.integers(0, high) for high in self.nvec], dtype=self.dtype)

    def contains(self, x) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return arr.shape == self.shape and np.all(arr >= 0) and np.all(arr < self.nvec)


def flatdim(space: Space) -> int:
    """Return the flattened dimensionality of a supported space."""

    if isinstance(space, Box):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiDiscrete):
        return int(np.sum(space.nvec))
    raise TypeError(f"unsupported space type: {type(space)!r}")


def clip_to_box(space: Box, x: Iterable[float] | np.ndarray) -> np.ndarray:
    """Clip an action or observation to a :class:`Box` and cast to its dtype."""

    arr = np.asarray(x, dtype=space.dtype)
    if arr.shape != space.shape:
        raise ValueError(f"expected shape {space.shape}, got {arr.shape}")
    return np.clip(arr, space.low, space.high).astype(space.dtype)
