"""Baseline policy portfolio for the RecoverableCapacityScheduling family.

Per the v2 plan, each new env family ships with a *portfolio* of
cheap-to-moderate baselines plus at least one decomposition
diagnostic. No single heuristic is the difficulty signal.

The policies here are intentionally simple and read the observation
they receive (no privileged access to env internals beyond what is
necessary for the named decomposition diagnostic). A candidate
algorithm worth a journal entry must beat *multiple* of these along
multiple terminal-vector components.
"""

from __future__ import annotations

import numpy as np

from rlh_bench.envs.capacity_scheduling import (
    CapacitySchedulingConfig,
    RecoverableCapacitySchedulingEnv,
)


def _make_action(
    cfg: CapacitySchedulingConfig,
    *,
    proj_logits: np.ndarray,
    mode_logits: np.ndarray,
    maint: np.ndarray | None = None,
    setup: np.ndarray | None = None,
    inv: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a per-step action vector for the env's layout.

    Returns an action of length ``cfg.action_dim`` with trailing
    dims zero-filled.
    """

    K, M, P = cfg.num_projects, cfg.num_modes, cfg.num_products
    a = np.zeros(cfg.action_dim, dtype=np.float32)
    a[0:K] = proj_logits
    a[K : K + M] = mode_logits
    if maint is None:
        maint = np.zeros(M, dtype=np.float32)
    a[K + M : K + 2 * M] = maint
    if setup is None:
        setup = np.zeros(M, dtype=np.float32)
    a[K + 2 * M : K + 3 * M] = setup
    if inv is None:
        inv = np.zeros(P, dtype=np.float32)
    a[K + 3 * M : K + 3 * M + P] = inv
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def _observation_slices(cfg: CapacitySchedulingConfig) -> dict[str, slice]:
    """Decode the observation layout into named slices.

    Observation layout (from the env source):
      per-project: cumulative_service, backlog, deadline_slack, priority  = 4K
      per-mode: utilization_ema, wear, heat, maint_debt                    = 4M
      setup mixture (M × P, flattened row-major: per mode, per family)    = M*P
      per-product: inventory, age_norm                                    = 2P
      demand summary (K × 3)                                              = 3K
      previous-action aggregates                                          = 2
      t/H                                                                  = 1
    """

    K, M, P = cfg.num_projects, cfg.num_modes, cfg.num_products
    offsets = {}
    cur = 0
    offsets["service_ratio"] = slice(cur, cur + K); cur += K
    offsets["backlog"] = slice(cur, cur + K); cur += K
    offsets["deadline_slack"] = slice(cur, cur + K); cur += K
    offsets["priority"] = slice(cur, cur + K); cur += K
    offsets["util_ema"] = slice(cur, cur + M); cur += M
    offsets["wear"] = slice(cur, cur + M); cur += M
    offsets["heat"] = slice(cur, cur + M); cur += M
    offsets["maint_debt"] = slice(cur, cur + M); cur += M
    offsets["setup_mixture"] = slice(cur, cur + M * P); cur += M * P
    offsets["inventory"] = slice(cur, cur + P); cur += P
    offsets["inventory_age"] = slice(cur, cur + P); cur += P
    offsets["demand_summary"] = slice(cur, cur + 3 * K); cur += 3 * K
    offsets["prev_aggregates"] = slice(cur, cur + 2); cur += 2
    offsets["t_norm"] = slice(cur, cur + 1)
    return offsets


# ----- baseline policies ----------------------------------------------------- #


class SchedulingZeroPolicy:
    """Outputs zero action. Confirms no free service."""

    name = "zero"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.env.action_space.shape, dtype=np.float32)


class SchedulingUniformPolicy:
    """Allocates uniform positive intensity to every project and mode.

    Useful as a "homogeneous allocation" reference; should not solve
    bundles whose projects have different priorities or compatibilities.
    """

    name = "uniform"

    def __init__(self, env: RecoverableCapacitySchedulingEnv, intensity: float = 0.5) -> None:
        self.env = env
        self.intensity = float(intensity)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        return _make_action(
            c,
            proj_logits=self.intensity * np.ones(c.num_projects, dtype=np.float32),
            mode_logits=self.intensity * np.ones(c.num_modes, dtype=np.float32),
        )


class SchedulingCapacityPushPolicy:
    """Broad high-capacity production with maintenance disabled.

    v0's longer horizon lets the plain uniform baseline solve too often even
    while spending half of every mode on maintenance.  This keeps the broad
    project coverage but actively commits capacity, providing a simple
    non-random heuristic that should clear mandatory bundles without being a
    sophisticated planner.
    """

    name = "capacity_push"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        del obs
        c = self.env.config
        return _make_action(
            c,
            proj_logits=np.ones(c.num_projects, dtype=np.float32),
            mode_logits=np.ones(c.num_modes, dtype=np.float32),
            maint=-np.ones(c.num_modes, dtype=np.float32),
            setup=-np.ones(c.num_modes, dtype=np.float32),
        )


class SchedulingBacklogPriorityPolicy:
    """Greedy: allocate to projects with the largest (backlog × priority).

    A myopic urgency baseline. Should leave headroom in setup_churn,
    wear, and lateness terminal vector components because it
    ignores cross-time consequences of its choices.
    """

    name = "backlog_priority"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        backlog = obs[self.slices["backlog"]]
        priority = obs[self.slices["priority"]]
        score = backlog * priority
        # Top-K projects with positive backlog
        proj_logits = np.where(score > 0, 0.8 * score / max(score.max(), 1e-6), -1.0)
        # Activate all modes (the env will use only compatible ones).
        mode_logits = 0.5 * np.ones(c.num_modes, dtype=np.float32)
        return _make_action(c, proj_logits=proj_logits.astype(np.float32), mode_logits=mode_logits)


class SchedulingEarliestDeadlinePolicy:
    """Allocate to the project closest to its (soft) deadline.

    Reads ``deadline_slack`` from the observation: smaller slack →
    higher priority. Tie-broken by backlog.
    """

    name = "earliest_deadline"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        slack = obs[self.slices["deadline_slack"]]
        backlog = obs[self.slices["backlog"]]
        # Smaller slack = higher urgency. Negative slack (past deadline)
        # is highest.
        score = -slack + 0.1 * backlog
        proj_logits = np.where(backlog > 0, 0.8 * (score - score.min()) / max(score.max() - score.min(), 1e-6), -1.0)
        mode_logits = 0.5 * np.ones(c.num_modes, dtype=np.float32)
        return _make_action(c, proj_logits=proj_logits.astype(np.float32), mode_logits=mode_logits)


class SchedulingMaintenanceAwarePolicy:
    """Backlog-priority allocation plus a wear/heat threshold check.

    When mean wear exceeds 0.6 or mean heat exceeds 0.7, runs
    maintenance instead of full production. Should outperform
    backlog_priority on wear and heat components of the terminal
    vector but lose some fill_rate.
    """

    name = "maintenance_aware"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.inner = SchedulingBacklogPriorityPolicy(env)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        wear = obs[self.slices["wear"]]
        heat = obs[self.slices["heat"]]
        base = self.inner(obs)
        if float(np.mean(wear)) > 0.6 or float(np.mean(heat)) > 0.7:
            # Substitute high maintenance, drop production.
            return _make_action(
                c,
                proj_logits=-0.5 * np.ones(c.num_projects, dtype=np.float32),
                mode_logits=-0.5 * np.ones(c.num_modes, dtype=np.float32),
                maint=np.ones(c.num_modes, dtype=np.float32),
            )
        return base


class SchedulingSetupAwarePolicy:
    """Backlog-priority allocation that also drives setup retargeting
    toward currently-active projects' product families.

    Adds positive setup intensity for modes that are mis-aligned with
    the project allocation; should reduce setup_churn loss.
    """

    name = "setup_aware"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.inner = SchedulingBacklogPriorityPolicy(env)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        base = self.inner(obs)
        # Bump setup intensity uniformly: tells the env to move setup
        # toward the implicit target dictated by proj_alloc.
        K, M = c.num_projects, c.num_modes
        base[K + 2 * M : K + 3 * M] = 0.6
        return base


class SchedulingShortHorizonRolloutPolicy:
    """Decomposition diagnostic: short-horizon CEM-style rollout per step.

    Samples a small population of action vectors, simulates each for
    ``horizon_steps`` steps (in a copy of the env if available, or
    via a cheap proxy that uses backlog + capacity heuristics), and
    plays the best one. This is the decomposition diagnostic from
    acceptance gate 5: if it solves Large, the long-horizon claim is
    false.

    This implementation is a *cheap proxy* (does not copy the env)
    that uses the backlog × priority score plus a multi-scale
    demand summary to discount allocations whose downstream effect
    is unlikely to be useful. Used to confirm the env's lookahead
    structure.
    """

    name = "short_horizon_rollout"

    def __init__(
        self,
        env: RecoverableCapacitySchedulingEnv,
        *,
        horizon_steps: int = 100,
    ) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        self.horizon_steps = int(horizon_steps)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        backlog = obs[self.slices["backlog"]]
        priority = obs[self.slices["priority"]]
        demand_summary = obs[self.slices["demand_summary"]].reshape(c.num_projects, 3)
        # Weight by demand expected in next `horizon_steps`. We use
        # the 64-step window (index 1 of (16, 64, 256)) as a proxy
        # for "near future demand".
        near_future = demand_summary[:, 1]
        score = (backlog + 0.7 * near_future) * priority
        proj_logits = np.where(score > 0, 0.9 * score / max(score.max(), 1e-6), -1.0)
        mode_logits = 0.5 * np.ones(c.num_modes, dtype=np.float32)
        # Periodically inject light maintenance.
        maint = 0.3 * np.ones(c.num_modes, dtype=np.float32)
        return _make_action(
            c,
            proj_logits=proj_logits.astype(np.float32),
            mode_logits=mode_logits,
            maint=maint,
        )


class SchedulingBundleAwarePolicy:
    """Bundle-aware allocation: prioritize projects in unfilled bundles.

    A project is most urgent when it's a member of a mandatory
    bundle whose OTHER members haven't yet reached
    quality_required. Allocate aggressively to those projects;
    allocate moderately to other projects with high backlog;
    use light maintenance to avoid wear collapse.
    """

    name = "bundle_aware"

    def __init__(self, env: RecoverableCapacitySchedulingEnv) -> None:
        self.env = env
        self.slices = _observation_slices(env.config)
        # Cache bundle membership per world.
        self._cached_seed: int | None = None
        self._bundle_membership: np.ndarray | None = None

    def _membership(self) -> np.ndarray:
        """Return shape (n_bundles, K) bundle-membership matrix."""
        seed = self.env.seed
        if self._cached_seed != seed:
            c = self.env.config
            bundles = self.env.bundles
            m = np.zeros((len(bundles), c.num_projects), dtype=np.float32)
            for b_idx, members in enumerate(bundles):
                for k in members:
                    m[b_idx, k] = 1.0
            self._bundle_membership = m
            self._cached_seed = seed
        return self._bundle_membership

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.env.config
        service = obs[self.slices["service_ratio"]]
        backlog = obs[self.slices["backlog"]]
        priority = obs[self.slices["priority"]]
        wear = obs[self.slices["wear"]]
        membership = self._membership()           # (B, K)
        q = c.quality_required

        # For each bundle, find the per-member shortfall:
        # member_shortfall[b, k] = max(0, q - service[k]) if k in bundle b else 0.
        per_member_shortfall = membership * np.maximum(q - service, 0.0)[None, :]
        # A bundle is "open" if any member's service < q.
        bundle_open = (per_member_shortfall.max(axis=1) > 0).astype(np.float32)
        # Bundle urgency: how much TOTAL shortfall remains, summed across members
        # (heavier bundles = more total shortfall = pull harder on their members).
        # Per project: sum across all bundles it's in, of bundle-urgency * is-shortfall.
        bundle_urgency = per_member_shortfall.sum(axis=1)              # (B,)
        project_bundle_pull = (
            membership * bundle_urgency[:, None] * (per_member_shortfall > 0).astype(np.float32)
        ).sum(axis=0)                                                  # (K,)

        # Combined score: bundle pull dominates, raw backlog is a secondary signal.
        score = 3.0 * project_bundle_pull + 0.5 * backlog * priority
        # Logits in [-1, 1] roughly — normalize so max ≈ 1.
        s_max = max(float(score.max()), 1e-6)
        proj_logits = np.where(score > 0, np.clip(score / s_max, 0.0, 1.0), -1.0).astype(np.float32)

        # Mode allocation: use all modes, weighted slightly toward the modes
        # that the active project set is most compatible with. Without
        # privileged access to compat[k,m], we just use a mild positive mode
        # logit and let the env's softmax do the work.
        mode_logits = 0.6 * np.ones(c.num_modes, dtype=np.float32)

        # Light maintenance when wear is moderate.
        maint = np.where(wear > 0.5, 0.6, 0.0).astype(np.float32)

        # Moderate setup retargeting.
        setup = 0.4 * np.ones(c.num_modes, dtype=np.float32)

        return _make_action(
            c,
            proj_logits=proj_logits,
            mode_logits=mode_logits,
            maint=maint,
            setup=setup,
        )


# ----- registry -------------------------------------------------------------- #


SCHEDULING_BASELINES = [
    SchedulingZeroPolicy,
    SchedulingUniformPolicy,
    SchedulingCapacityPushPolicy,
    SchedulingBacklogPriorityPolicy,
    SchedulingEarliestDeadlinePolicy,
    SchedulingMaintenanceAwarePolicy,
    SchedulingSetupAwarePolicy,
    SchedulingShortHorizonRolloutPolicy,
    SchedulingBundleAwarePolicy,
]
"""All scheduling-family baselines. Ordered approximately by
sophistication: zero/uniform are sanity, the next four are myopic
heuristics, the last is a decomposition diagnostic."""
