"""World-generation helpers shared by the redesigned env families.

These are NumPy-only helpers used by ``RecoverableCapacityScheduling``
and ``RecoverableKeyFuelMaze`` to construct deterministic worlds from
a seed. They live in the substrate so the conventions are shared and
visible.

Three patterns recur:

  1. Multi-scale demand sketches: future demand summarized at several
     temporal resolutions, instead of either exact full calendar
     (planning oracle) or pure backlog (memorization-favoring).
  2. Setup graphs: discrete graphs over product families, with
     deterministic edge weights (setup-change costs). Sampled by seed.
  3. Compatibility matrices: which mode can produce which product, at
     what efficiency, sampled by seed.

The functions take an ``np.random.Generator`` and a few size
parameters and return deterministic NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def make_actuator_matrix(
    rng: np.random.Generator,
    *,
    action_dim: int,
    n_force_dims: int = 2,
    redundancy_bias: float = 0.6,
) -> np.ndarray:
    """Generate a deterministic actuator matrix ``A ∈ R^{n_force_dims × action_dim}``.

    The matrix maps the agent's continuous action into physical force.
    Two designs at the extremes are bad:

      * A small random matrix where every action direction is roughly
        equivalent → action_dim is syntactic only.
      * A matrix that ignores most action dims → the env claims
        D-dimensional control but only ``n_force_dims`` matter.

    What we want: a matrix where many action directions produce
    similar force but at *different* fuel/heat cost. The redundancy
    is real, but so is the cost-shaping. ``redundancy_bias`` controls
    how much each force direction is concentrated vs spread across
    actuators.

    Args:
        rng: NumPy generator (already seeded).
        action_dim: Number of actuator channels (>= n_force_dims).
        n_force_dims: Number of physical force dimensions (default 2).
        redundancy_bias: In ``[0, 1]``. Higher values give more
            concentrated columns; lower values spread the mapping.

    Returns:
        Array of shape ``(n_force_dims, action_dim)`` in approximately
        ``[-1, 1]``.
    """

    if action_dim < n_force_dims:
        raise ValueError(f"action_dim ({action_dim}) must be >= n_force_dims ({n_force_dims})")
    if not 0.0 <= redundancy_bias <= 1.0:
        raise ValueError("redundancy_bias must be in [0, 1]")

    # Start with a dense random matrix.
    dense = rng.uniform(-1.0, 1.0, size=(n_force_dims, action_dim)).astype(np.float32)

    # Concentrate columns toward a small number of physical directions
    # based on redundancy_bias. With high bias, each column is mostly
    # aligned with one force dim; with low bias, columns mix freely.
    concentration = redundancy_bias * rng.uniform(0.3, 1.0, size=action_dim).astype(np.float32)
    one_hot = np.zeros_like(dense)
    primary_dims = rng.integers(0, n_force_dims, size=action_dim)
    one_hot[primary_dims, np.arange(action_dim)] = rng.choice([-1.0, 1.0], size=action_dim)

    matrix = (1.0 - concentration) * dense + concentration * one_hot
    # Normalize so the spectral scale is bounded.
    norm = np.linalg.norm(matrix, axis=0, keepdims=True)
    matrix = matrix / np.maximum(norm, 1e-6)
    return matrix.astype(np.float32)


def make_actuator_costs(
    rng: np.random.Generator,
    *,
    action_dim: int,
    base_cost: float = 1.0,
    spread: float = 0.5,
) -> np.ndarray:
    """Per-actuator energy/heat cost weights.

    All positive. Multiplied elementwise against ``action ** 2`` to
    accumulate per-step actuator cost. Some actuator channels are
    cheaper than others, so an algorithm that picks the right basis
    in the redundant action space pays less.

    Args:
        rng: NumPy generator.
        action_dim: Number of actuators.
        base_cost: Center of the cost distribution.
        spread: Multiplicative spread; weights sit in
            ``[base_cost / (1+spread), base_cost * (1+spread)]``.

    Returns:
        Array of shape ``(action_dim,)``, positive.
    """

    factors = rng.uniform(1.0 / (1.0 + spread), 1.0 + spread, size=action_dim).astype(np.float32)
    return (base_cost * factors).astype(np.float32)


def make_demand_calendar(
    rng: np.random.Generator,
    *,
    num_projects: int,
    horizon: int,
    n_peaks_range: tuple[int, int] = (2, 5),
    peak_width_range: tuple[int, int] = (20, 80),
    regime: str = "smooth",
) -> np.ndarray:
    """Per-project demand calendar across the horizon.

    Each project receives several Gaussian-bump demand windows whose
    locations, widths, and amplitudes are sampled by seed. The
    integral of the calendar is normalized per project so total
    demand is roughly comparable across worlds.

    Long-horizon coupling does NOT come from the calendar alone — the
    calendar just shapes when demand arrives. Coupling comes from
    wear, setup, inventory, and bundles, which are computed by the
    env from this calendar plus its dynamics state.

    Args:
        rng: NumPy generator.
        num_projects: Number of projects (K).
        horizon: Episode horizon.
        n_peaks_range: Range of (min, max) demand peaks per project.
        peak_width_range: Range of (min, max) peak std-dev in steps.
        regime: One of {"smooth", "bursty", "front", "back"}. Shapes
            the temporal distribution.

    Returns:
        Array of shape ``(num_projects, horizon)``, non-negative.
    """

    if regime not in {"smooth", "bursty", "front", "back"}:
        raise ValueError(f"unknown regime: {regime!r}")

    calendar = np.zeros((num_projects, horizon), dtype=np.float32)
    t_grid = np.arange(horizon, dtype=np.float32)

    for k in range(num_projects):
        n_peaks = int(rng.integers(n_peaks_range[0], n_peaks_range[1] + 1))
        if regime == "bursty":
            n_peaks = max(1, n_peaks // 2)  # fewer, sharper
        for _ in range(n_peaks):
            if regime == "front":
                center = float(rng.uniform(0.0, horizon * 0.6))
            elif regime == "back":
                center = float(rng.uniform(horizon * 0.4, horizon - 1))
            else:
                center = float(rng.uniform(0.0, horizon - 1))
            if regime == "bursty":
                width = float(rng.uniform(peak_width_range[0] * 0.3, peak_width_range[0]))
            else:
                width = float(rng.uniform(*peak_width_range))
            amplitude = float(rng.uniform(0.5, 1.5))
            calendar[k] += amplitude * np.exp(-0.5 * ((t_grid - center) / width) ** 2)

    # Normalize so each project's total demand has comparable scale
    # across worlds (target: integral per project ≈ 1).
    integrals = calendar.sum(axis=1, keepdims=True)
    calendar = calendar / np.maximum(integrals, 1e-6)
    return calendar.astype(np.float32)


def demand_summary_at(
    calendar: np.ndarray,
    *,
    t: int,
    windows: tuple[int, ...] = (16, 64, 256),
) -> np.ndarray:
    """Multi-scale future demand sketch starting at time ``t``.

    The summary is a per-project sum of demand over each forward
    window. This is the observation contract for "future demand
    visible at multiple scales but not as a full calendar."

    Returns an array of shape ``(num_projects, len(windows))``.
    """

    num_projects, horizon = calendar.shape
    summaries = np.zeros((num_projects, len(windows)), dtype=np.float32)
    for i, w in enumerate(windows):
        end = min(t + w, horizon)
        if end > t:
            summaries[:, i] = calendar[:, t:end].sum(axis=1)
    return summaries


def make_compatibility_matrix(
    rng: np.random.Generator,
    *,
    num_projects: int,
    num_modes: int,
    min_modes_per_project: int = 1,
    max_modes_per_project: int | None = None,
) -> np.ndarray:
    """Project ↔ production-mode compatibility with efficiency weights.

    Each project is compatible with at least ``min_modes_per_project``
    modes (so every project has a feasible producer). Compatibility
    is sampled by seed; efficiency values within compatible pairs
    are drawn from ``[0.5, 1.0]``. Non-compatible pairs are exactly
    zero.

    Returns:
        Array of shape ``(num_projects, num_modes)``, non-negative.
    """

    if max_modes_per_project is None:
        max_modes_per_project = max(min_modes_per_project + 1, num_modes // 2)
    max_modes_per_project = min(max_modes_per_project, num_modes)
    if min_modes_per_project < 1:
        raise ValueError("min_modes_per_project must be >= 1")
    if max_modes_per_project < min_modes_per_project:
        raise ValueError("max_modes_per_project must be >= min_modes_per_project")

    matrix = np.zeros((num_projects, num_modes), dtype=np.float32)
    for k in range(num_projects):
        n = int(rng.integers(min_modes_per_project, max_modes_per_project + 1))
        modes = rng.choice(num_modes, size=n, replace=False)
        effs = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
        matrix[k, modes] = effs
    return matrix


def make_setup_graph(
    rng: np.random.Generator,
    *,
    num_families: int,
    base_setup_cost: float = 1.0,
    spread: float = 0.5,
) -> np.ndarray:
    """Pairwise setup-change costs over product families.

    Diagonal is zero (no setup cost to stay in the same family).
    Off-diagonal entries are positive, asymmetric (some transitions
    cost more than their reverse). Used to compute setup churn over
    a long episode — a policy that switches families too often pays
    in setup time and `neg_setup_churn`.

    Returns:
        Array of shape ``(num_families, num_families)``, non-negative.
    """

    matrix = rng.uniform(
        base_setup_cost / (1.0 + spread),
        base_setup_cost * (1.0 + spread),
        size=(num_families, num_families),
    ).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def make_bundles(
    rng: np.random.Generator,
    *,
    num_projects: int,
    n_bundles: int,
    bundle_size_range: tuple[int, int] = (2, 4),
) -> list[tuple[int, ...]]:
    """Sample ``n_bundles`` contract bundles over the project set.

    A bundle is a small subset of projects whose collective service
    determines a mandatory-completion check. Projects can appear in
    multiple bundles, but each project appears in at least one (if
    ``n_bundles`` is large enough). Used to compute
    ``mandatory_fill_rate`` in the scheduling family's terminal
    vector.

    Returns:
        List of tuples of project indices.
    """

    bundles: list[tuple[int, ...]] = []
    for _ in range(n_bundles):
        size = int(rng.integers(bundle_size_range[0], bundle_size_range[1] + 1))
        size = min(size, num_projects)
        members = tuple(sorted(int(x) for x in rng.choice(num_projects, size=size, replace=False)))
        bundles.append(members)
    return bundles
