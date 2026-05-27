"""Frozen harness for the rl-research substrate.

Modeled on autoresearch's `prepare.py`: provides the panel of environments,
the deterministic eval protocol, the time budget, and the frozen baseline
scores. The agent edits `train.py` only — never this file.

Public surface:

    PANEL_SMOKE        : list[str] — 5 envs the candidate iterates against (~30 min/sweep parallel).
    PANEL_HARD         : list[str] — 4 hard envs the candidate is judged against (separate sweep).
    PANEL              : list[str] — alias for PANEL_SMOKE (back-compat).
    PANEL_TYPE         : dict[str, str] — "scalar" or "vector" per env.
    BASELINES_PATH     : Path — JSON file storing frozen baseline scores (smoke).
    BASELINES_HARD_PATH: Path — JSON file storing hard-tier baselines (published_sota + our_baseline).
    TIME_BUDGET_SMOKE  : int seconds per smoke env (5 min, 5 envs in parallel).
    TIME_BUDGET_HARD   : int seconds per hard env (1 hour).
    TIME_BUDGET_PER_ENV: alias for TIME_BUDGET_SMOKE (back-compat).

    make_env(env_id, seed)        -> gymnasium.Env (with info['vector'] for vector envs)
    evaluate(policy_fn, env_id, *, seed, n_episodes) -> float
    load_baselines()              -> dict[env_id, {"random": float, "strong": float}]

`info['vector']` convention (vector envs only):
    Each `step()` returns `info` containing `'vector': np.ndarray` of shape
    `(n_channels,)` — the per-channel reward at this step. For scalar envs,
    `info` does NOT contain `'vector'`. Candidates wishing to consume vector
    reward natively read `info['vector']` directly; the scalar `reward`
    returned alongside is the env's default scalarization (used only by
    `evaluate` to compute hypervolume / panel score, NOT meant to be the
    candidate's training signal on vector envs).

Score semantics (what `evaluate` returns):
    Scalar envs: mean episode return over n_episodes.
    Vector envs: hypervolume of the Pareto front of episode return-vectors,
        against a per-env reference point. Higher is better in both cases.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Keep JAX off the GPU so it doesn't fight torch for VRAM. Craftax pulls JAX
# in transitively; on the linux GPU host JAX defaults to GPU and steals
# memory before torch can preallocate.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ---------------------------------------------------------------------------
# Panel: smoke tier (~30 min parallel) + hard tier (separate sweep).
# ---------------------------------------------------------------------------

PANEL_SMOKE: list[str] = [
    # Sparse long-horizon prerequisite-structure gridworlds.
    # DoorKey-8x8: 2-step item prereq (key → door → goal). Killed CARL #4 / TOP #11.
    # KeyCorridorS3R3: 3-step prereq with hidden subgoal (find key by exploring
    #   unlocked rooms, then unlock the target room). Killed KERNEL-RL #3 /
    #   OPP #7 / EOP-COP #8.
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    # Native vector reward (multi-objective).
    # deep-sea-treasure-CONCAVE-v0: concave Pareto front — Pareto-optimal
    #   treasures are unreachable by any linear scalarization wᵀr. The
    #   canonical detector for the "scalarized vector-reward" disqualifier.
    # minecart-v0: 3-channel ore-1 / ore-2 / fuel — non-trivial trade-offs.
    # mo-reacher-v4: 4-channel continuous-control trade-off across 4 targets.
    "deep-sea-treasure-concave-v0",
    "minecart-v0",
    "mo-reacher-v4",
]

PANEL_HARD: list[str] = [
    # Craftax-Symbolic-v1 (Matthews et al. 2024, ICML 2024): JAX-native open-world
    #   crafting+survival, 22 achievements over 4 difficulty tiers. The two
    #   hardest achievements remain unreached at 1B env steps under PPO; full
    #   completion is open. Sparse, long-horizon, scalar.
    "Craftax-Symbolic-v1",
    # MiniHack-Quest-Hard-v0: NetHack quest with prerequisite chain (find amulet,
    #   navigate corridor, defeat guard, retrieve quest item). RL from scratch
    #   gets near-zero; published baselines (DRL-Agent, IMPALA) reach <10%.
    "MiniHack-Quest-Hard-v0",
    # mo-halfcheetah-v4 (mo-gymnasium): native 2-channel reward (forward velocity,
    #   control-energy cost). Continuous-control vector-reward; PGMORL /
    #   Envelope MORL set the published Pareto-hypervolume frontier.
    "mo-halfcheetah-v4",
    # Humanoid-v5 (gymnasium-MuJoCo standard library): 17-dim continuous action,
    #   376-dim observation, whole-body humanoid balance + locomotion. Dense
    #   reward but high-DOF; included to test continuous-control breadth.
    "Humanoid-v5",
]

# Back-compat alias: legacy code (train.py default, run_panel.py without --hard)
# treats `PANEL` as the smoke tier.
PANEL: list[str] = PANEL_SMOKE

PANEL_TYPE: dict[str, str] = {
    # Smoke
    "MiniGrid-DoorKey-8x8-v0": "scalar",
    "MiniGrid-KeyCorridorS3R3-v0": "scalar",
    "deep-sea-treasure-concave-v0": "vector",
    "minecart-v0": "vector",
    "mo-reacher-v4": "vector",
    # Hard
    "Craftax-Symbolic-v1": "scalar",
    "MiniHack-Quest-Hard-v0": "scalar",
    "mo-halfcheetah-v4": "vector",
    "Humanoid-v5": "scalar",
}

# Per-env Pareto-hypervolume reference points for vector envs. Each ref point
# is "worst plausible per-channel return" — every Pareto point we care about
# must strictly dominate ref on every channel, otherwise it is filtered out
# and contributes 0 to hypervolume. Calibrated by inspecting random-policy
# episode returns: ref must be below random on every channel, otherwise
# random scores HV=0 and the "beat random" tier is trivial.
HV_REF: dict[str, np.ndarray] = {
    # Smoke
    "minecart-v0": np.array([-1.0, -1.0, -200.0], dtype=np.float64),
    "deep-sea-treasure-concave-v0": np.array([0.0, -100.0], dtype=np.float64),
    "mo-reacher-v4": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    # Hard
    # mo-halfcheetah: ch0 = forward velocity reward (positive when moving
    # forward, ~3-7 over 1000-step episode for a tuned policy; near zero for
    # random). ch1 = -energy cost (always negative; ~-0.5 to -50 over an
    # episode). Ref well below the random-policy mass.
    "mo-halfcheetah-v4": np.array([-100.0, -200.0], dtype=np.float64),
}

# Eval protocol
N_EVAL_EPISODES: int = 20

# Wallclock cap that train.py respects per env. Smoke tier: 5 min/env, 5 envs
# in parallel = ~5 min/sweep. Hard tier: 1 h/env, Option-B grouped = ~2 h/sweep.
TIME_BUDGET_SMOKE: int = 300  # seconds
TIME_BUDGET_HARD: int = 3600  # seconds

# Back-compat: legacy code used TIME_BUDGET_PER_ENV.
TIME_BUDGET_PER_ENV: int = TIME_BUDGET_SMOKE

ROOT: Path = Path(__file__).parent.resolve()
BASELINES_PATH: Path = ROOT / "baselines.json"
BASELINES_HARD_PATH: Path = ROOT / "baselines_hard.json"

# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def make_env(env_id: str, seed: int = 0):
    """Build a gymnasium.Env for `env_id`, seeded.

    For vector envs, the returned env's `step()` injects `info['vector']:
    np.ndarray` of shape `(k,)` containing the per-channel reward. For
    scalar envs, no such injection.
    """
    if env_id not in PANEL_TYPE:
        raise ValueError(f"unknown env_id: {env_id!r}")
    if env_id.startswith("MiniGrid-"):
        return _make_minigrid(env_id, seed)
    if env_id.startswith("MiniHack-"):
        return _make_minihack(env_id, seed)
    if env_id.startswith("Craftax-"):
        return _make_craftax(env_id, seed)
    if env_id == "Humanoid-v5":
        return _make_humanoid(env_id, seed)
    # Everything else is mo-gymnasium (registered IDs may or may not have a
    # 'mo-' prefix depending on the env — minecart-v0,
    # deep-sea-treasure-concave-v0, mo-reacher-v4, mo-halfcheetah-v4, etc.).
    return _make_mo(env_id, seed)


def _make_minigrid(env_id: str, seed: int):
    import gymnasium as gym
    import minigrid
    from minigrid.wrappers import ImgObsWrapper

    minigrid.register_minigrid_envs()  # idempotent; populates gymnasium.registry
    env = gym.make(env_id)
    env = ImgObsWrapper(env)  # strip dict obs to (H,W,3) image
    env.reset(seed=seed)
    return env


def _make_mo(env_id: str, seed: int):
    import gymnasium as gym
    import mo_gymnasium  # noqa: F401  (registers the mo- envs with gymnasium)

    env = gym.make(env_id)
    env = _VectorRewardWrapper(env)
    env.reset(seed=seed)
    return env


def _make_minihack(env_id: str, seed: int):
    import gymnasium as gym
    import minihack  # noqa: F401  (registers MiniHack-*-v0 with gymnasium)

    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def _make_humanoid(env_id: str, seed: int):
    import gymnasium as gym

    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def _make_craftax(env_id: str, seed: int):
    """Wrap Craftax's gymnax-flavored env into a minimal gymnasium-style API."""
    return _CraftaxAdapter(env_id, seed=seed)


# ---------------------------------------------------------------------------
# Wrapper that normalizes mo-gymnasium step() shape and injects info['vector'].
# ---------------------------------------------------------------------------


class _VectorRewardWrapper:
    """Wrap an mo-gymnasium env so step() returns (obs, scalar_r, term, trunc,
    info) with info['vector'] holding the per-channel reward."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        # mo-gymnasium returns reward as np.ndarray of shape (k,).
        vec = np.asarray(reward, dtype=np.float64)
        info = dict(info)
        info["vector"] = vec
        scalar_r = float(vec.sum())  # default scalarization for the panel score
        return obs, scalar_r, term, trunc, info

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Craftax adapter (gymnax → gymnasium-style API surface).
# ---------------------------------------------------------------------------


class _CraftaxAdapter:
    """Minimal gymnasium-style adapter over Craftax's gymnax-flavored env.

    Craftax exposes `step_env(rng, state, action, params) -> (obs, state, r,
    done, info)` with explicit state passing and JAX arrays. We hide that:
    keep `state` and `rng` internal, jit-compile step/reset, and convert all
    JAX arrays to numpy at the API boundary.
    """

    def __init__(self, env_id: str, *, seed: int = 0):
        import gymnasium as gym  # noqa: F401  (action_space type only)
        import jax
        from craftax.craftax_env import make_craftax_env_from_name
        from gymnasium import spaces

        self._env = make_craftax_env_from_name(env_id, auto_reset=False)
        self._params = self._env.default_params

        # Cache jit-compiled step/reset so the per-step Python overhead stays
        # bounded. seed → PRNGKey done lazily in reset().
        self._step_jit = jax.jit(self._env.step_env)
        self._reset_jit = jax.jit(self._env.reset_env)

        # gymnax action_space / observation_space → gymnasium spaces.
        a_space = self._env.action_space(self._params)
        o_space = self._env.observation_space(self._params)
        # Action: discrete integer 0..N-1.
        self.action_space = spaces.Discrete(int(a_space.n))
        # Obs: real-valued vector. gymnax exposes shape + dtype.
        obs_shape = tuple(int(s) for s in o_space.shape)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        self._rng = None
        self._state = None
        self._seed = int(seed)

    def reset(self, *, seed=None, options=None):
        import jax

        if seed is not None:
            self._seed = int(seed)
        self._rng = jax.random.PRNGKey(self._seed)
        self._rng, sub = jax.random.split(self._rng)
        obs, self._state = self._reset_jit(sub, self._params)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        import jax

        if self._state is None or self._rng is None:
            raise RuntimeError("reset() must be called before step().")
        self._rng, sub = jax.random.split(self._rng)
        obs, self._state, r, done, info = self._step_jit(
            sub, self._state, int(action), self._params
        )
        # Convert any jax arrays in info to numpy scalars/arrays so consumers
        # (eval loop, train.py) don't have to care about the JAX type.
        info_out: dict = {}
        for k, v in dict(info).items():
            try:
                info_out[k] = np.asarray(v)
            except Exception:
                info_out[k] = v
        return (
            np.asarray(obs, dtype=np.float32),
            float(r),
            bool(done),
            False,  # truncated — Craftax encodes time-limit in `done`.
            info_out,
        )

    def close(self):
        # JAX env has no resources to release.
        pass


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

PolicyFn = Callable[[np.ndarray], np.ndarray]


def evaluate(
    policy_fn: PolicyFn,
    env_id: str,
    *,
    seed: int = 0,
    n_episodes: int = N_EVAL_EPISODES,
) -> float:
    """Run `policy_fn` for n_episodes deterministic episodes; return
    a scalar score. Higher is better.

    Score:
      - scalar envs: mean episode return.
      - vector envs: Pareto-hypervolume of the per-episode return-vectors
        vs `HV_REF[env_id]`.

    Eval seeding is offset from `seed` to keep it disjoint from training
    seeds: env reset seed = seed + 5000 + episode_index.
    """
    env = make_env(env_id, seed=seed + 10_000)
    is_vector = PANEL_TYPE[env_id] == "vector"

    scalar_returns: list[float] = []
    vector_returns: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 5_000 + ep)
        obs = np.asarray(obs)
        done = False
        ret_scalar = 0.0
        ret_vector: np.ndarray | None = None
        steps = 0
        while not done and steps < 10_000:
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            obs = np.asarray(obs)
            ret_scalar += float(r)
            if is_vector:
                v = np.asarray(info["vector"], dtype=np.float64)
                ret_vector = v.copy() if ret_vector is None else ret_vector + v
            done = bool(term) or bool(trunc)
            steps += 1
        scalar_returns.append(ret_scalar)
        if is_vector:
            assert ret_vector is not None
            vector_returns.append(ret_vector)

    env.close()

    if is_vector:
        ref = HV_REF[env_id]
        return float(_hypervolume(np.stack(vector_returns, axis=0), ref))
    return float(np.mean(scalar_returns))


# ---------------------------------------------------------------------------
# Hypervolume (small-dim Pareto)
# ---------------------------------------------------------------------------


def _hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of the Pareto-optimal subset of `points` w.r.t. `ref`
    (every dim "higher is better"). Implements 2D sweep and a simple
    inclusion-exclusion for d=3. For d>3 returns the box-product over
    component-wise maxes (a coarse upper bound — fine for our 2/3-D panel).

    Negative values (point < ref on some dim) contribute 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != ref.shape[0]:
        raise ValueError("points/ref shape mismatch")
    # Filter to points that strictly dominate ref on every dim
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    # Pareto front (maximization)
    pts = _pareto_front(pts)
    d = ref.shape[0]
    if d == 2:
        # Sort by x descending, sweep area
        order = np.argsort(-pts[:, 0])
        sorted_pts = pts[order]
        hv = 0.0
        prev_y = ref[1]
        for x, y in sorted_pts:
            if y > prev_y:
                hv += (x - ref[0]) * (y - prev_y)
                prev_y = y
        return hv
    if d == 3:
        # Inclusion-exclusion over points (O(2^N), N small for our eval).
        n = pts.shape[0]
        total = 0.0
        # Use Klee's measure via sweep on z, dropping to 2D — cheaper.
        order = np.argsort(-pts[:, 2])
        sorted_pts = pts[order]
        prev_z = ref[2]
        for i in range(n):
            z = sorted_pts[i, 2]
            if z <= prev_z:
                continue
            # 2D HV of the front of the first i+1 points projected to xy
            sub = sorted_pts[: i + 1, :2]
            sub_hv = _hypervolume(sub, ref[:2])
            total += sub_hv * (z - prev_z)
            prev_z = z
        return total
    # d >= 4: box upper bound
    maxes = pts.max(axis=0)
    return float(np.prod(np.maximum(maxes - ref, 0.0)))


def _pareto_front(pts: np.ndarray) -> np.ndarray:
    """Return the Pareto-optimal subset of `pts` (every dim higher-is-better)."""
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # Anyone that dominates pts[i] kicks pts[i] out
        dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if dominated.any():
            keep[i] = False
    return pts[keep]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def load_baselines() -> dict[str, dict[str, float]]:
    """Return frozen baseline scores per env (smoke + hard).

    Schema:
        { env_id: {"random": float, "strong": float} }

    `random` is the uniform-action baseline; `strong` is the max score across
    the strong-baseline algorithms. Hard-tier baselines come from
    `baselines_hard.json` under `our_baseline.{random,strong}`. Returns
    zero-filled defaults for any missing entry so callers don't have to
    branch.
    """
    all_envs = list(PANEL_SMOKE) + list(PANEL_HARD)
    out: dict[str, dict[str, float]] = {env: {"random": 0.0, "strong": 0.0} for env in all_envs}

    if BASELINES_PATH.exists():
        with BASELINES_PATH.open() as f:
            raw = json.load(f)
        for env in PANEL_SMOKE:
            e = raw.get(env, {})
            out[env] = {
                "random": float(e.get("random", 0.0)),
                "strong": float(e.get("strong", 0.0)),
            }

    if BASELINES_HARD_PATH.exists():
        with BASELINES_HARD_PATH.open() as f:
            raw = json.load(f)
        for env in PANEL_HARD:
            e = raw.get(env, {}).get("our_baseline", {})
            out[env] = {
                "random": float(e.get("random", 0.0)),
                "strong": float(e.get("strong", 0.0)),
            }

    return out


__all__ = [
    "BASELINES_HARD_PATH",
    "BASELINES_PATH",
    "N_EVAL_EPISODES",
    "PANEL",
    "PANEL_HARD",
    "PANEL_SMOKE",
    "PANEL_TYPE",
    "TIME_BUDGET_HARD",
    "TIME_BUDGET_PER_ENV",
    "TIME_BUDGET_SMOKE",
    "evaluate",
    "load_baselines",
    "make_env",
]
