"""Recoverable deterministic capacity-scheduling environment.

This is the redesigned allocation/scheduling family for the lab. It
replaces ``RecoverableResourceAllocation-*-v0`` from the original
substrate.

What the env tests
------------------

Continuous-action policies that must manage *long-lived* capacity
state — mode wear, setup inertia, heat, inventory perishability,
contract bundle priorities — across a long terminal-only horizon.
Greedy / earliest-deadline / focused policies are useful baselines
but should leave headroom in multiple terminal-vector components.

State per step
--------------

  * Per-project: cumulative service, current backlog, soft-deadline
    slack, priority.
  * Per-mode: capacity utilization recent average, wear, heat,
    maintenance debt, current setup mixture across product families.
  * Per-product: inventory level, age (perishability), reserved
    commitments.
  * Multi-scale future demand summaries (16/64/256-step windows).
  * Previous action aggregates (setup-churn signal).

Long-horizon coupling mechanisms (Codex's red-team highlighted that
a single "rolling demand" mechanism decomposes into independent
windows; this env has six):

  1. Current allocation → future mode wear and heat.
  2. Skipping maintenance early raises future production cost.
  3. Setup changes have multi-hundred-step consequences.
  4. Inventory built too early may perish; too late misses
     deadlines.
  5. Contract bundles make service order consequential.
  6. Project priorities differ → uniform fill-rate is dominated.

Reward
------

Terminal-only vector reward. Eleven components, all
larger-is-better. See ``DEFAULT_SCHEDULING_REWARD_SPEC`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlh_bench.core import RewardSpec, StepReturn, make_reward, zero_reward
from rlh_bench.spaces import Box, clip_to_box
from rlh_bench.world_gen import (
    demand_summary_at,
    make_bundles,
    make_compatibility_matrix,
    make_demand_calendar,
    make_setup_graph,
)


DEFAULT_SCHEDULING_REWARD_SPEC = RewardSpec(
    names=(
        "success",
        "weighted_fill_rate",
        "mandatory_fill_rate",
        "neg_lateness",
        "neg_shortfall_tail",
        "neg_wear",
        "neg_heat_violation",
        "neg_setup_churn",
        "neg_inventory_waste",
        "neg_energy",
        "resilience_margin",
    ),
    weights=(
        1.0,    # success
        0.40,   # weighted_fill_rate
        0.30,   # mandatory_fill_rate
        0.05,   # neg_lateness
        0.08,   # neg_shortfall_tail
        0.04,   # neg_wear
        0.04,   # neg_heat_violation
        0.02,   # neg_setup_churn
        0.03,   # neg_inventory_waste
        0.01,   # neg_energy
        0.05,   # resilience_margin
    ),
)


@dataclass(frozen=True)
class CapacitySchedulingConfig:
    """Configuration for :class:`RecoverableCapacitySchedulingEnv`.

    Defaults give the v0 tier (H=2000, K=48, M=8, P=8). Small and
    Large are constructed via the registry with smaller / larger
    values.

    See ``DESIGN.md`` for the env semantics; this dataclass is the
    full set of structural knobs.
    """

    horizon: int = 2000
    num_projects: int = 48
    num_modes: int = 8
    num_products: int = 8
    # Action layout is K + 3M + P = 48 + 24 + 8 = 80. The v0 default
    # action_dim of 96 leaves 16 trailing "free" dims that aren't
    # used by the env semantics; they exist so the registry can
    # advertise larger action spaces at higher tiers without changing
    # semantics. (Small / Large set their own action_dim that exactly
    # matches the layout for those tiers.)
    action_dim: int = 96
    n_bundles: int = 8
    bundle_size_range: tuple[int, int] = (2, 4)

    # Per-project demand
    demand_regime: str = "smooth"
    demand_peaks_range: tuple[int, int] = (2, 5)
    demand_peak_width_range: tuple[int, int] = (40, 120)

    # Per-mode capacity dynamics
    # NOTE: capacity is tuned to be modestly scarce. A competent policy
    # that maintains setup alignment with active projects and uses
    # compatible modes can serve most demand. A random policy averages
    # ~40-50% fill on Small. Total demand per project across the horizon is
    # 1.0 (normalized in make_demand_calendar); total demand for
    # K projects = K. Per-step max throughput per mode after alignment
    # and compatibility ≈ max_cap * 0.09. With M modes, target total
    # supply ≈ 2-3x demand to leave room for wear/setup churn losses.
    max_capacity_per_mode: float = 0.45
    wear_rate: float = 0.008            # wear gained per unit utilization
    wear_recovery_rate: float = 0.001    # per unit maintenance intensity
    max_wear: float = 1.0
    wear_capacity_penalty: float = 0.6   # capacity scales like (1 - α*wear)
    heat_buildup_rate: float = 0.008
    heat_decay_rate: float = 0.001
    heat_capacity_penalty: float = 0.7
    max_heat: float = 1.0

    # Setup dynamics
    setup_decay_rate: float = 0.02      # how fast setup mixture moves toward target
    setup_change_cost_scale: float = 1.0
    setup_base_cost: float = 1.0
    setup_capacity_penalty: float = 0.45

    # Inventory dynamics
    inventory_max_age: int = 250        # age beyond which inventory perishes
    inventory_capacity: float = 2.0     # per product
    inventory_release_rate: float = 1.0

    # Service / quality
    safe_allocation: float = 0.75       # per-project allocation soft cap
    quality_required: float = 0.55      # service fraction required per project
                                        # for bundle satisfaction
    success_fill_threshold: float = 0.55    # weighted_fill_rate threshold for success
    success_mandatory_threshold: float = 0.50  # mandatory_fill_rate threshold for success

    # Soft-deadline shaping
    deadline_relative: tuple[float, ...] | None = None  # in [0, 1], len num_projects

    # Project priorities — weights for weighted_fill_rate. Mandatory
    # bundles get priority elevated regardless of these.
    priorities: tuple[float, ...] | None = None

    # Compatibility generator knobs (forwarded)
    min_modes_per_project: int = 1
    max_modes_per_project: int | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.num_projects <= 1:
            raise ValueError("num_projects must be at least 2")
        if self.num_modes < 1:
            raise ValueError("num_modes must be at least 1")
        if self.num_products < 1:
            raise ValueError("num_products must be at least 1")
        # Action layout requires:
        #   project logits (K) + mode logits (M) + maint (M)
        #   + setup (M) + inventory release (P)
        # = K + 3M + P slots. Trailing dims beyond this layout are
        # ignored (they exist as a knob for future hierarchy).
        required = self.num_projects + 3 * self.num_modes + self.num_products
        if self.action_dim < required:
            raise ValueError(
                f"action_dim must be >= K + 3M + P = {required}; "
                f"got {self.action_dim}"
            )
        if self.n_bundles < 1:
            raise ValueError("n_bundles must be >= 1")

    def priorities_array(self, rng: np.random.Generator) -> np.ndarray:
        if self.priorities is not None:
            arr = np.asarray(self.priorities, dtype=np.float32)
            if arr.shape != (self.num_projects,):
                raise ValueError("priorities must have length num_projects")
            return arr
        return rng.uniform(0.5, 1.5, size=self.num_projects).astype(np.float32)

    def deadlines_array(self) -> np.ndarray:
        if self.deadline_relative is not None:
            arr = np.asarray(self.deadline_relative, dtype=np.float32)
            if arr.shape != (self.num_projects,):
                raise ValueError("deadline_relative must have length num_projects")
            return arr
        # Defaults: spread across the horizon to encourage non-trivial
        # service order.
        return np.linspace(0.35, 0.95, self.num_projects, dtype=np.float32)


class RecoverableCapacitySchedulingEnv:
    """Long-horizon capacity scheduling with terminal vector feedback.

    Action layout (length ``config.action_dim``):

        a[0 : K]                 -- project allocation logits
        a[K : K + M]             -- mode allocation logits
        a[K + M : K + 2M]        -- per-mode maintenance intensity
        a[K + 2M : K + 3M]       -- per-mode setup-change intensity
        a[K + 3M : K + 3M + P]   -- per-product inventory release rate

    Any trailing dimensions beyond this layout are ignored (they exist
    only to support tier-specific action_dim values without changing
    semantics; an honest algorithm should still be able to use them
    via the actuator-cost weighting in maze, or zero them in
    scheduling without penalty).

    Each logit block is softmax-projected to a simplex of intensities;
    maintenance / setup-change / inventory blocks are bounded to
    ``[0, 1]`` per element.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: CapacitySchedulingConfig | None = None,
        reward_spec: RewardSpec = DEFAULT_SCHEDULING_REWARD_SPEC,
        reward_mode: str = "scalar",
    ) -> None:
        self.config = CapacitySchedulingConfig() if config is None else config
        self.reward_spec = reward_spec
        self.reward_dim = reward_spec.dim
        if reward_mode not in {"scalar", "vector"}:
            raise ValueError("reward_mode must be 'scalar' or 'vector'")
        self.reward_mode = reward_mode

        c = self.config
        self.action_space = Box(low=-1.0, high=1.0, shape=(c.action_dim,), dtype=np.float32)

        # Observation layout (computed once for shape):
        #   per-project: cumulative_service, backlog, deadline_slack,
        #                priority                                = 4K
        #   per-mode: utilization_ema, wear, heat, maint_debt    = 4M
        #   per-mode setup mixture (P-dim per mode)              = P*M
        #   per-product: inventory, age_norm                     = 2P
        #   demand summary 3 scales per project                  = 3K
        #   previous action aggregates: total used, setup churn  = 2
        #   t/H                                                  = 1
        obs_dim = 4 * c.num_projects + 4 * c.num_modes + c.num_products * c.num_modes \
                  + 2 * c.num_products + 3 * c.num_projects + 2 + 1
        self.observation_space = Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self._seed: int | None = None
        # World tensors filled by reset()
        self._calendar: np.ndarray = np.zeros((c.num_projects, c.horizon), dtype=np.float32)
        self._priorities: np.ndarray = np.ones(c.num_projects, dtype=np.float32)
        self._deadlines: np.ndarray = np.zeros(c.num_projects, dtype=np.float32)
        self._compat: np.ndarray = np.ones((c.num_projects, c.num_modes), dtype=np.float32)
        self._setup_graph: np.ndarray = np.zeros((c.num_products, c.num_products), dtype=np.float32)
        self._bundles: list[tuple[int, ...]] = []
        # Per-project product association: which product family the
        # project belongs to (random assignment, deterministic by seed).
        self._project_product: np.ndarray = np.zeros(c.num_projects, dtype=np.int64)
        # Dynamic state
        self._t = 0
        self._cumulative_service = np.zeros(c.num_projects, dtype=np.float32)
        self._backlog = np.zeros(c.num_projects, dtype=np.float32)
        self._wear = np.zeros(c.num_modes, dtype=np.float32)
        self._heat = np.zeros(c.num_modes, dtype=np.float32)
        self._maint_debt = np.zeros(c.num_modes, dtype=np.float32)
        self._utilization_ema = np.zeros(c.num_modes, dtype=np.float32)
        self._setup_mixture = np.zeros((c.num_modes, c.num_products), dtype=np.float32)
        self._inventory = np.zeros(c.num_products, dtype=np.float32)
        self._inventory_age = np.zeros(c.num_products, dtype=np.float32)
        # Accumulators
        self._total_lateness = 0.0
        self._total_wear_accrued = 0.0
        self._total_heat_violation = 0.0
        self._total_setup_churn = 0.0
        self._total_inventory_waste = 0.0
        self._total_energy = 0.0
        self._prev_setup_mixture: np.ndarray = self._setup_mixture.copy()
        self._prev_action: np.ndarray = np.zeros(c.action_dim, dtype=np.float32)
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public properties

    @property
    def t(self) -> int:
        return self._t

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def cumulative_service(self) -> np.ndarray:
        return self._cumulative_service.copy()

    @property
    def backlog(self) -> np.ndarray:
        return self._backlog.copy()

    @property
    def wear(self) -> np.ndarray:
        return self._wear.copy()

    @property
    def heat(self) -> np.ndarray:
        return self._heat.copy()

    @property
    def inventory(self) -> np.ndarray:
        return self._inventory.copy()

    @property
    def compat_matrix(self) -> np.ndarray:
        return self._compat.copy()

    @property
    def bundles(self) -> list[tuple[int, ...]]:
        return list(self._bundles)

    # ------------------------------------------------------------------
    # Gym-like API

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        c = self.config
        if seed is None:
            seed = 0
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)

        # Sample world
        self._calendar = make_demand_calendar(
            self._rng,
            num_projects=c.num_projects,
            horizon=c.horizon,
            n_peaks_range=c.demand_peaks_range,
            peak_width_range=c.demand_peak_width_range,
            regime=c.demand_regime,
        )
        self._priorities = c.priorities_array(self._rng)
        self._deadlines = c.deadlines_array() * c.horizon
        self._compat = make_compatibility_matrix(
            self._rng,
            num_projects=c.num_projects,
            num_modes=c.num_modes,
            min_modes_per_project=c.min_modes_per_project,
            max_modes_per_project=c.max_modes_per_project,
        )
        self._setup_graph = make_setup_graph(
            self._rng,
            num_families=c.num_products,
            base_setup_cost=c.setup_base_cost,
        )
        self._bundles = make_bundles(
            self._rng,
            num_projects=c.num_projects,
            n_bundles=c.n_bundles,
            bundle_size_range=c.bundle_size_range,
        )
        self._project_product = self._rng.integers(0, c.num_products, size=c.num_projects).astype(np.int64)

        # Dynamic state
        self._t = 0
        self._cumulative_service = np.zeros(c.num_projects, dtype=np.float32)
        self._backlog = np.zeros(c.num_projects, dtype=np.float32)
        self._wear = np.zeros(c.num_modes, dtype=np.float32)
        # Modes start at heat 0 but with a small randomized initial
        # wear so worlds differ from the start.
        self._heat = np.zeros(c.num_modes, dtype=np.float32)
        self._maint_debt = np.zeros(c.num_modes, dtype=np.float32)
        self._utilization_ema = np.zeros(c.num_modes, dtype=np.float32)
        # Setup mixture starts uniform-ish, with small per-world
        # variation so initial state isn't identical across seeds.
        initial = self._rng.dirichlet(np.ones(c.num_products), size=c.num_modes).astype(np.float32)
        self._setup_mixture = initial
        self._inventory = np.zeros(c.num_products, dtype=np.float32)
        self._inventory_age = np.zeros(c.num_products, dtype=np.float32)
        # Accumulators
        self._total_lateness = 0.0
        self._total_wear_accrued = 0.0
        self._total_heat_violation = 0.0
        self._total_setup_churn = 0.0
        self._total_inventory_waste = 0.0
        self._total_energy = 0.0
        self._prev_setup_mixture = self._setup_mixture.copy()
        self._prev_action = np.zeros(c.action_dim, dtype=np.float32)
        self._terminated = False
        self._last_reward_vector = np.zeros(self.reward_dim, dtype=np.float32)

        return self._observation(), self._info(np.zeros(self.reward_dim, dtype=np.float32))

    def step(self, action: np.ndarray) -> StepReturn:
        if self._terminated:
            raise RuntimeError("step() called after episode terminated; call reset() first")

        c = self.config
        raw_action = clip_to_box(self.action_space, action)
        K, M, P = c.num_projects, c.num_modes, c.num_products

        # Decode action blocks
        proj_logits = raw_action[0:K]
        mode_logits = raw_action[K : K + M]
        maint_block = raw_action[K + M : K + 2 * M]
        setup_block = raw_action[K + 2 * M : K + 3 * M]
        inv_block = raw_action[K + 3 * M : K + 3 * M + P]

        # Project allocation: softmax over K, scaled by an aggregate
        # intensity. Negative logits naturally suppress projects.
        proj_alloc = _softmax_with_intensity(proj_logits)
        # Mode allocation: same shape over modes.
        mode_alloc = _softmax_with_intensity(mode_logits)
        # Maintenance/setup/inventory blocks: clipped to [0, 1].
        maint_intensity = np.clip(0.5 * (maint_block + 1.0), 0.0, 1.0).astype(np.float32)
        setup_intensity = np.clip(0.5 * (setup_block + 1.0), 0.0, 1.0).astype(np.float32)
        inv_release = np.clip(0.5 * (inv_block + 1.0), 0.0, 1.0).astype(np.float32)

        # Setup target from the current project allocation.  This is computed
        # before capacity so setup pressure can reduce same-step throughput;
        # otherwise random retargeting only pays a terminal bookkeeping cost.
        target_family_weights = np.zeros((M, P), dtype=np.float32)
        for k in range(K):
            target_family_weights[:, self._project_product[k]] += proj_alloc[k] * self._compat[k]
        target_norm = np.sum(target_family_weights, axis=1, keepdims=True)
        target_distribution = np.where(
            target_norm > 1e-6,
            target_family_weights / np.maximum(target_norm, 1e-6),
            self._setup_mixture,
        )
        setup_pressure = np.clip(
            np.sum(np.abs(target_distribution - self._setup_mixture), axis=1) * setup_intensity,
            0.0,
            1.0,
        ).astype(np.float32)

        # ---- Demand arrival ----
        self._backlog += self._calendar[:, self._t]

        # ---- Effective per-mode capacity ----
        # Each mode's raw step capacity is reduced by wear and heat,
        # then split by the agent's mode_alloc, then reduced further
        # by maintenance intensity (maintenance uses capacity).
        eff_capacity = c.max_capacity_per_mode * (
            1.0 - c.wear_capacity_penalty * self._wear
        ) * (1.0 - c.heat_capacity_penalty * self._heat)
        eff_capacity = np.maximum(eff_capacity, 0.0)
        # Maintenance and setup changes reserve productive time on the mode.
        # Setup pressure is a same-step operational loss; without it, random
        # retargeting only paid a small terminal bookkeeping cost.
        mode_capacity = (
            eff_capacity
            * mode_alloc
            * (1.0 - 0.65 * maint_intensity)
            * (1.0 - c.setup_capacity_penalty * setup_pressure)
        )

        # ---- Production ----
        # For each (mode m, project k), production contribution is
        # mode_capacity[m] * compat[k, m] * setup_alignment[m, family(k)] * proj_alloc[k].
        # Vectorize: alignment[m, family(k)] = setup_mixture[m, project_product[k]].
        alignment_per_project = self._setup_mixture[:, self._project_product]  # (M, K)
        # produced_km = mode_capacity[m] * alignment[m, k] * compat[k, m] * proj_alloc[k]
        produced_km = (
            mode_capacity[:, None]
            * alignment_per_project
            * self._compat.T  # (M, K)
            * proj_alloc[None, :]
        )
        produced = produced_km.sum(axis=0).astype(np.float32)  # (K,)

        # Cap per-project production at the current backlog (don't
        # overproduce beyond what's been demanded).
        produced = np.minimum(produced, self._backlog).astype(np.float32)
        # Each mode's actually-used capacity = sum over projects of
        # produced_km, scaled back from the cap that was hit. We
        # approximate by computing the ratio of produced/offered.
        offered_total = float(produced_km.sum()) + 1e-6
        used_total = float(produced.sum())
        mode_used = produced_km.sum(axis=1) * (used_total / offered_total) if offered_total > 1e-6 else np.zeros(M, dtype=np.float32)
        # Wear/heat are driven by committed operation, not only demand that
        # survives the backlog cap.  The previous version charged wear on
        # ``mode_used`` after capping production at tiny normalized demand, so
        # any over-capacity policy could run for 500 steps with ~zero wear.
        # We use pre-maintenance operating effort here: maintenance and setup
        # consume capacity but still exercise the line thermally/mechanically.
        mode_util = np.clip(
            mode_alloc * (1.0 - 0.30 * maint_intensity)
            + 0.05 * setup_intensity
            + 0.80 * setup_pressure,
            0.0,
            1.0,
        ).astype(np.float32)

        # ---- Inventory build/release ----
        # Inventory is built when production exceeds release; it is
        # consumed (perishes if too old). Inventory release does NOT
        # serve backlog directly; it's an alternative path that the
        # agent can use to smooth over future windows. Released
        # inventory effectively converts to extra production capacity
        # this step, weighted by inv_release.
        per_product_release = inv_release * self._inventory * c.inventory_release_rate
        per_product_release = np.minimum(per_product_release, self._inventory)
        # Distribute released inventory to compatible projects
        # proportionally (only projects in that product family).
        for p in range(P):
            mask = self._project_product == p
            if not np.any(mask):
                continue
            available = float(per_product_release[p])
            remaining_backlog = self._backlog[mask] - produced[mask]
            remaining_backlog = np.maximum(remaining_backlog, 0.0)
            denom = float(np.sum(remaining_backlog)) + 1e-6
            allocation = remaining_backlog / denom * min(available, float(np.sum(remaining_backlog)))
            produced[mask] += allocation
            self._inventory[p] -= float(np.sum(allocation))

        produced = np.minimum(produced, self._backlog).astype(np.float32)
        self._cumulative_service += produced
        self._backlog -= produced
        self._backlog = np.maximum(self._backlog, 0.0)

        # ---- Wear, heat, maintenance ----
        self._wear = np.clip(
            self._wear + c.wear_rate * mode_util - c.wear_recovery_rate * maint_intensity,
            0.0, c.max_wear
        ).astype(np.float32)
        self._total_wear_accrued += float(np.sum(c.wear_rate * mode_util))
        self._heat = np.clip(
            self._heat + c.heat_buildup_rate * mode_util - c.heat_decay_rate,
            0.0, c.max_heat
        ).astype(np.float32)
        self._total_heat_violation += float(np.sum(np.maximum(self._heat - 0.9, 0.0)))
        self._maint_debt = np.maximum(self._maint_debt + 0.5 * c.wear_rate * mode_util - maint_intensity * 0.01, 0.0)
        self._utilization_ema = 0.95 * self._utilization_ema + 0.05 * mode_util

        # ---- Setup mixture dynamics ----
        # setup_intensity per mode steers the mixture toward a target.
        # The target is the product family most-aligned with the chosen
        # project allocation (weighted by compat).
        # Movement scaled by setup_intensity AND a per-mode decay rate.
        movement_rate = c.setup_decay_rate * setup_intensity
        new_mixture = self._setup_mixture + (target_distribution - self._setup_mixture) * movement_rate[:, None]
        # Setup churn cost: per-mode L1 movement.  Terminal cost remains
        # normalized, while setup_pressure above makes churn operationally
        # visible through capacity/wear/heat.
        churn = float(np.sum(np.abs(new_mixture - self._setup_mixture)))
        self._total_setup_churn += churn * c.setup_change_cost_scale
        self._setup_mixture = new_mixture.astype(np.float32)

        # ---- Inventory aging + perishability ----
        # Inventory aging is uniform per step. Inventory production
        # is the difference between mode capacity AND backlog --
        # excess capacity goes into the inventory of the produced
        # product family.
        # Only capacity actually left after serving demand can become buffer
        # inventory.  Do not subtract the wear/heat utilization proxy here.
        excess_per_mode = np.maximum(mode_capacity - mode_used, 0.0)
        # Inventory is a weak smoothing mechanism, not a free second demand
        # channel: only a fraction of unconsumed capacity can be packed into
        # perishable buffers.
        excess_per_mode = 0.20 * excess_per_mode
        for m in range(M):
            if excess_per_mode[m] < 1e-6:
                continue
            # Distribute excess to product families in proportion to
            # setup_mixture[m].
            inventory_add = excess_per_mode[m] * self._setup_mixture[m]
            self._inventory += inventory_add
        # Cap inventory.
        overflow = np.maximum(self._inventory - c.inventory_capacity, 0.0)
        self._total_inventory_waste += float(np.sum(overflow))
        self._inventory = np.minimum(self._inventory, c.inventory_capacity)
        self._inventory_age += 1.0
        # Perish: if inventory is older than max_age, decay rapidly.
        too_old = self._inventory_age > c.inventory_max_age
        perish = self._inventory * too_old.astype(np.float32) * 0.05
        self._total_inventory_waste += float(np.sum(perish))
        self._inventory = np.maximum(self._inventory - perish, 0.0)
        # Reset age slightly if inventory was added this step.
        produced_inventory_mass = float(np.sum(excess_per_mode))
        if produced_inventory_mass > 1e-6:
            # Weighted reset: new mass dilutes age.
            inventory_mass_total = float(np.sum(self._inventory)) + 1e-6
            self._inventory_age = self._inventory_age * (1.0 - produced_inventory_mass / inventory_mass_total)

        # ---- Energy accumulator ----
        # Energy cost = action magnitude squared, integrated.
        self._total_energy += float(np.sum(np.square(raw_action)))

        # ---- Lateness shaping (terminal-only; this is just an
        # accumulator that becomes part of the terminal vector) ----
        # A project past its deadline with positive backlog incurs
        # per-step lateness. The accumulator is used in the terminal
        # vector only.
        deadline_step = self._deadlines.astype(np.int64)
        past_deadline = self._t >= deadline_step
        lateness_increment = past_deadline.astype(np.float32) * self._backlog
        self._total_lateness += float(np.sum(lateness_increment))

        self._prev_action = raw_action.copy()
        self._t += 1

        terminal = self._t >= c.horizon
        self._terminated = terminal
        if terminal:
            vector = self._terminal_reward_vector()
            reward = make_reward(self.reward_mode, self.reward_spec, vector)
            self._last_reward_vector = vector.copy()
        else:
            vector = np.zeros(self.reward_dim, dtype=np.float32)
            reward = zero_reward(self.reward_mode, self.reward_spec)

        return self._observation(), reward, terminal, False, self._info(vector)

    # ------------------------------------------------------------------
    # Observation / info / terminal vector

    def _observation(self) -> np.ndarray:
        c = self.config
        K, M, P = c.num_projects, c.num_modes, c.num_products
        # Per-project features
        total_demand_per_project = self._calendar.sum(axis=1) + 1e-6
        service_ratio = self._cumulative_service / total_demand_per_project
        deadline_slack = (self._deadlines - self._t) / c.horizon
        per_project = np.concatenate([service_ratio, self._backlog, deadline_slack, self._priorities])
        # Per-mode features
        per_mode = np.concatenate([self._utilization_ema, self._wear, self._heat, self._maint_debt])
        # Setup mixture flattened
        setup_flat = self._setup_mixture.flatten()
        # Per-product features
        inv_age_norm = self._inventory_age / max(c.inventory_max_age, 1)
        per_product = np.concatenate([self._inventory, inv_age_norm])
        # Multi-scale demand summary
        summary = demand_summary_at(self._calendar, t=self._t, windows=(16, 64, 256))
        # Aggregate previous action
        prev_total = float(np.sum(np.abs(self._prev_action)))
        prev_setup_churn = float(np.sum(np.abs(self._setup_mixture - self._prev_setup_mixture)))
        self._prev_setup_mixture = self._setup_mixture.copy()
        prev_aggregates = np.asarray([prev_total / max(c.action_dim, 1), prev_setup_churn], dtype=np.float32)
        # Time
        t_norm = np.asarray([self._t / c.horizon], dtype=np.float32)
        obs = np.concatenate(
            [per_project, per_mode, setup_flat, per_product, summary.flatten(), prev_aggregates, t_norm]
        ).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _terminal_reward_vector(self) -> np.ndarray:
        c = self.config
        total_demand = self._calendar.sum(axis=1)
        service_ratio = self._cumulative_service / (total_demand + 1e-6)
        service_ratio = np.clip(service_ratio, 0.0, 1.0)

        priority_weight = self._priorities / (np.sum(self._priorities) + 1e-6)
        weighted_fill_rate = float(np.sum(priority_weight * service_ratio))

        # Mandatory fill rate over bundles: a bundle is fulfilled if
        # every project in it meets quality_required.
        bundle_satisfaction: list[float] = []
        for b in self._bundles:
            members = np.asarray(b, dtype=np.int64)
            satisfied = float(np.all(service_ratio[members] >= c.quality_required))
            bundle_satisfaction.append(satisfied)
        mandatory_fill_rate = float(np.mean(bundle_satisfaction)) if bundle_satisfaction else 0.0

        success = float(
            weighted_fill_rate >= c.success_fill_threshold
            and mandatory_fill_rate >= c.success_mandatory_threshold
        )

        horizon = max(c.horizon, 1)
        num_projects = max(c.num_projects, 1)
        num_modes = max(c.num_modes, 1)
        num_products = max(c.num_products, 1)

        # Lateness is accumulated per project per step, so normalize by both
        # horizon and project count.  Without the project-count factor this
        # component grows with tier size even when per-project service quality
        # is comparable.
        normalized_lateness = self._total_lateness / (horizon * num_projects)
        shortfall = float(np.sum(np.maximum(c.quality_required - service_ratio, 0.0)))
        # Express shortfall as a fraction of the configured required service.
        # v0 intentionally uses a stricter quality threshold than Small/Large;
        # normalizing by that threshold keeps the component scale comparable
        # while preserving the harder success criterion.
        normalized_shortfall = shortfall / (num_projects * max(c.quality_required, 1e-6))
        normalized_wear = float(np.mean(self._wear))
        # Heat violation and setup churn sum over modes each step.  Report a
        # per-mode per-step average; cap persistent heat saturation so longer
        # tiers do not dominate scalar returns merely because they spend more
        # post-warmup time at the heat ceiling.
        heat_violation_cap = 0.5 * max(c.max_heat - 0.9, 0.0)
        normalized_heat = float(
            min(self._total_heat_violation / (horizon * num_modes), heat_violation_cap)
        )
        normalized_setup_churn = float(self._total_setup_churn / (horizon * num_modes))
        # Inventory waste is also an integrated flow.  Normalize by product
        # count and horizon so longer tiers do not accumulate a larger terminal
        # penalty solely because there were more opportunities to overflow or
        # perish buffers.
        normalized_inventory_waste = float(self._total_inventory_waste / (horizon * num_products))
        normalized_energy = float(self._total_energy / (horizon * max(c.action_dim, 1)))

        # Resilience margin: residual effective capacity averaged across
        # modes. Higher = more headroom at horizon end.
        resilience = float(np.mean(np.maximum(1.0 - self._wear - 0.5 * self._heat, 0.0)))

        return np.asarray(
            [
                success,
                weighted_fill_rate,
                mandatory_fill_rate,
                -normalized_lateness,
                -normalized_shortfall,
                -normalized_wear,
                -normalized_heat,
                -normalized_setup_churn,
                -normalized_inventory_waste,
                -normalized_energy,
                resilience,
            ],
            dtype=np.float32,
        )

    def _info(self, reward_vector: np.ndarray) -> dict[str, Any]:
        info = self.diagnostics()
        info.update(
            {
                "reward_vector": np.asarray(reward_vector, dtype=np.float32).copy(),
                "reward_names": self.reward_spec.names,
                "is_success": bool(info["success"]),
            }
        )
        return info

    def diagnostics(self) -> dict[str, Any]:
        c = self.config
        total_demand = self._calendar.sum(axis=1)
        service_ratio = self._cumulative_service / (total_demand + 1e-6)
        service_ratio = np.clip(service_ratio, 0.0, 1.0)
        priority_weight = self._priorities / (np.sum(self._priorities) + 1e-6)
        weighted_fill = float(np.sum(priority_weight * service_ratio))
        # Bundle satisfaction (matches _terminal_reward_vector)
        bundle_satisfaction: list[float] = []
        for b in self._bundles:
            members = np.asarray(b, dtype=np.int64)
            satisfied = float(np.all(service_ratio[members] >= c.quality_required))
            bundle_satisfaction.append(satisfied)
        mandatory_fill = float(np.mean(bundle_satisfaction)) if bundle_satisfaction else 0.0
        success_flag = float(
            weighted_fill >= c.success_fill_threshold
            and mandatory_fill >= c.success_mandatory_threshold
        )
        return {
            "t": int(self._t),
            "success": success_flag,
            "weighted_fill_rate": weighted_fill,
            "mandatory_fill_rate": mandatory_fill,
            "mean_service_ratio": float(np.mean(service_ratio)),
            "mean_wear": float(np.mean(self._wear)),
            "mean_heat": float(np.mean(self._heat)),
            "total_lateness": float(self._total_lateness),
            "total_setup_churn": float(self._total_setup_churn),
            "total_inventory_waste": float(self._total_inventory_waste),
            "total_energy": float(self._total_energy),
            "num_bundles": int(len(self._bundles)),
        }

    def render(self) -> str:
        diag = self.diagnostics()
        return (
            f"RecoverableCapacityScheduling(t={self._t}/{self.config.horizon}, "
            f"fill={diag['weighted_fill_rate']:.2f}, wear={diag['mean_wear']:.2f}, "
            f"heat={diag['mean_heat']:.2f})"
        )


def _softmax_with_intensity(logits: np.ndarray) -> np.ndarray:
    """Softmax-projected allocation with explicit intensity gating.

    The input is in ``[-1, 1]``. The output is a non-negative
    vector summing to a *gated* intensity:

      * If the maximum logit is <= 0, intensity is 0: no allocation.
      * If max_logit > 0, intensity scales linearly to 1.0 at
        max_logit = 1.
      * The simplex over the positive-relative logits is sharp but
        not one-hot (temperature ≈ 3).

    A zero action vector → zero allocation. A small positive action
    vector → small allocation. Negative actions stay off. This means
    a zero policy commits no capacity, and the agent must *actively*
    drive production.
    """

    max_logit = float(np.max(logits))
    intensity = float(np.clip(max_logit, 0.0, 1.0))
    if intensity < 1e-6:
        return np.zeros_like(logits, dtype=np.float32)

    # Simplex over relative logits, with moderate sharpness.
    shifted = logits - max_logit
    exp = np.exp(shifted * 3.0)
    simplex = exp / (np.sum(exp) + 1e-6)
    return (simplex * intensity).astype(np.float32)
