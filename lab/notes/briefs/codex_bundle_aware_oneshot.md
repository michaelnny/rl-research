# One-shot brief — implement SchedulingBundleAwarePolicy

Sole task: add ONE new policy class to
`src/rlh_bench/baselines/scheduling.py` and verify it succeeds on
Sched-v0. Do NOT do additional analysis, reviews, or notes beyond
what's listed in "Deliverable" below.

Read this as your full identity for this single invocation.

## Problem

Sched-v0 has no honest baseline that succeeds. The original
myopic heuristics all fail (you diagnosed this as bundle
all-of-N coverage failure). A bundle-aware policy should
succeed.

## What to do (in order)

### Step 1: Add `bundles` public property

In `src/rlh_bench/envs/capacity_scheduling.py`, find where other
public properties are defined (search for `@property`). Add:

```python
@property
def bundles(self) -> list[tuple[int, ...]]:
    """Read-only copy of the contract bundles for this world."""
    return [tuple(b) for b in self._bundles]
```

### Step 2: Add `SchedulingBundleAwarePolicy` to scheduling.py

After `SchedulingShortHorizonRolloutPolicy`, before
`SCHEDULING_BASELINES`, add:

```python
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
```

### Step 3: Add to SCHEDULING_BASELINES list

Find `SCHEDULING_BASELINES = [...]` near the end of
`scheduling.py`. Add `SchedulingBundleAwarePolicy` to that list.

### Step 4: Run the sweep + report

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/run_baselines.py --episodes 20 2>&1 | head -50
```

If the policy reaches succ ≥ 0.40 on Sched-v0, you're done.
If it doesn't, tune the score weights (line `score = 3.0 *
project_bundle_pull + 0.5 * backlog * priority`) — try
`score = 5.0 * project_bundle_pull + 1.0 * backlog * priority`,
or add a mode_logits bias from `env.compat_matrix` (public).

### Step 5: Write a 20-line outcome note

Write `lab/notes/codex_bundle_aware_outcome.md` with:

```
# Outcome

succ on Sched-v0: <number>
succ on Sched-Small: <number>
weighted_fill_rate v0: <number>
mandatory_fill_rate v0: <number>
neg_wear v0: <number>

Verdict: [PASS / FAIL]

If FAIL: what would need to change in the env to make this
solvable by a non-brute baseline.
```

## Rules

- DO NOT write extensive design notes. Just the code + the outcome.
- DO NOT modify other policies, env dynamics, or tests.
- Stop as soon as the criteria are met. If you have time after
  meeting them, you can add a quick test to
  `tests/test_capacity_scheduling_env.py` confirming the new
  policy reaches succ >= 0.3 on Sched-Small.

## How a session ends

When the policy class is in scheduling.py, baselines sweep
confirms succ ≥ 0.4 on Sched-v0, and the 20-line outcome note
is written. Do NOT spend more than 5 minutes on each step.
