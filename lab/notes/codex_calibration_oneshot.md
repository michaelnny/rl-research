# One-shot brief — diagnose capacity-scheduling difficulty calibration

You are NOT operating as the lab peer reviewer or journal-note
appender. You previously wrote the substrate redesign red-team
(`lab/notes/codex_redteam_substrate_redesign_2026-06-30.md`) and
counter-design (`lab/notes/codex_counterdesign_substrate_2026-06-30.md`).
The proposer is now implementing your counter-design and has hit a
specific calibration problem on the CapacityScheduling family. They
want concrete debug help, not a re-litigation of the design.

Read this brief as your full identity for this single invocation.

## The problem

The first cut of `src/rlh_bench/envs/capacity_scheduling.py` (the
v2-plan CapacityScheduling env) is written, compiles, runs, and is
deterministic. But its **difficulty calibration is degenerate**:
across a small panel of trivial policies on the Small tier
(H=500, K=16, M=4, P=4, action_dim=32, n_bundles=2):

```
policy        succ  fill  mand  wear  heat  churn  energy  late
zero          0.00  0.00  0.00  0.00  0.00     0       0   2222
ones          1.00  1.00  1.00  0.00  0.00    3.6  16000      0
half_pos      1.00  1.00  1.00  0.00  0.00    3.6   4000      0
neg_half      0.00  0.00  0.00  0.00  0.00    0.0   4000   2222
rand_pm       1.00  1.00  1.00  0.00  0.00   13.7   5335   0.01
rand_pos      1.00  1.00  1.00  0.00  0.00   12.8   5336   0.00
```

Read: ANY non-trivially-positive action solves the env at
success=1.00, fill=1.00, mand=1.00 with no wear, no heat, no
lateness. This violates your acceptance gates 6 (myopic-gap),
8 (action-complexity), and 7 (recoverability).

The proposer has burned ~3 retuning rounds and is now asking for
your eyes on the dynamics, because you already know what coupling
structure you intended. The substrate is the most important
artifact in the lab; getting this right matters.

## What you have access to

- The env source: `src/rlh_bench/envs/capacity_scheduling.py`
- The world-gen helpers: `src/rlh_bench/world_gen.py`
- Your earlier counter-design:
  `lab/notes/codex_counterdesign_substrate_2026-06-30.md`
- The v2 plan: `lab/notes/PLAN_substrate_redesign_v2_2026-06-30.md`
- The full repo. You can run Python.

You may modify `src/rlh_bench/envs/capacity_scheduling.py` directly.
You may also modify `src/rlh_bench/world_gen.py`, but prefer
changing only the env. You may add or modify a probe script under
`experiments/probes/` if it helps you diagnose. You may NOT add a
new env class or change `src/rlh_bench/__init__.py` (the env isn't
registered yet; that's intentional).

## What you should diagnose

1. **Why is "ones" already solving?** Total demand per project is
   normalized to 1.0 (so K=16 total demand units across H=500
   steps). Total mode capacity is M × cap × H = 4 × 0.3 × 500 = 600
   units of raw capacity. Even after compat-sparsity and
   alignment-fraction multipliers, supply ≈ 100×demand. Your
   counter-design said "scarce capacity"; this is glut. Either
   `max_capacity_per_mode` is way too high, or demand isn't being
   integrated against the right total.

2. **Why is wear/heat always 0?** The wear-rate is 0.08 per unit
   utilization per step. After 500 steps at any non-trivial util,
   wear should be substantial. Either `mode_util` is being computed
   wrong, or the cap on it is firing too soon.

3. **Is the long-horizon coupling actually wired?** Per your
   counter-design, the agent should pay measurable cost for setup
   churn, inventory waste, energy, wear, etc. Right now setup churn
   for "rand_pos" is 12.8 (small relative to fill rate). Inventory
   waste is not in this table — but the env tracks it. Is the
   trade-off real or pro-forma?

4. **Mandatory bundles solve at 1.0 for any positive action.** Bundle
   satisfaction requires every project in the bundle to be at
   service ≥ `quality_required`=0.85. With n_bundles=2 and
   bundle_size in (2,4), every bundle is being satisfied. Is
   bundle generation putting bundles only over easy-to-serve
   projects?

## What to deliver

Write a diagnosis + fix to
`lab/notes/codex_capacity_scheduling_calibration_2026-06-30.md`,
covering:

1. **What is structurally wrong** in the current env. Concrete
   pointers to lines.

2. **What knob settings would fix it**, ideally in a single coherent
   change rather than ad-hoc tweaks. The goal is a Small tier where:

   - `zero` action: succ ≈ 0, fill ≈ 0
   - `ones` action: succ ≈ 0.3–0.7, fill ≈ 0.5–0.8, some
     wear/heat accumulating (≥ 0.1)
   - `random_pm`: succ ≈ 0.0–0.3, fill ≈ 0.3–0.5
   - A *competent* policy (focused on currently-due bundles,
     compatible-mode selection, setup-aware): succ ≈ 0.6–0.9

   (The competent policy doesn't exist yet — but the env should
   leave room for it to differentiate.)

3. **If the right fix requires structural changes** to the env
   dynamics (not just constants), say what and implement them. The
   counter-design's coupling mechanisms are: wear ↔ capacity, setup
   inertia, heat decay, inventory perishability, bundles,
   priorities. Use the strongest 2–3, drop weaker ones if
   they're getting in the way.

4. **Apply your fixes directly** to
   `src/rlh_bench/envs/capacity_scheduling.py`. Then re-run the
   baseline panel below and report the new numbers.

## Repro probe

A one-shot probe you can copy into a file:

```python
# experiments/probes/capacity_scheduling_panel.py
import numpy as np
from rlh_bench.envs.capacity_scheduling import (
    RecoverableCapacitySchedulingEnv, CapacitySchedulingConfig
)

cfg = CapacitySchedulingConfig(
    horizon=500, num_projects=16, num_modes=4, num_products=4,
    action_dim=32, n_bundles=2,
)
N_SEEDS = 20

policies = [
    ('zero', lambda r, o: np.zeros(32, dtype=np.float32)),
    ('ones', lambda r, o: np.ones(32, dtype=np.float32)),
    ('half_pos', lambda r, o: 0.5 * np.ones(32, dtype=np.float32)),
    ('rand_pm', lambda r, o: r.uniform(-1, 1, size=32).astype(np.float32)),
    ('rand_pos', lambda r, o: r.uniform(0, 1, size=32).astype(np.float32)),
]

for name, pol in policies:
    fills, mand, success = [], [], []
    wears, heats, churn = [], [], []
    for seed in range(N_SEEDS):
        env = RecoverableCapacitySchedulingEnv(cfg, reward_mode='vector')
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        for _ in range(500):
            obs, r, term, trunc, info = env.step(pol(rng, obs))
            if term: break
        rv = info['reward_vector']
        fills.append(rv[1]); mand.append(rv[2])
        success.append(int(info['is_success']))
        wears.append(info['mean_wear']); heats.append(info['mean_heat'])
        churn.append(info['total_setup_churn'])
    print(f'{name:12s} succ={np.mean(success):.2f} fill={np.mean(fills):.2f} '
          f'mand={np.mean(mand):.2f} wear={np.mean(wears):.2f} '
          f'heat={np.mean(heats):.2f} churn={np.mean(churn):.1f}')
```

Run with `PYTHONPATH=src .venv/bin/python experiments/probes/capacity_scheduling_panel.py`.

## Rules

- You may modify the env source. Make the change minimal: tune
  constants and small dynamics fixes; do not rewrite the whole
  file.
- Do NOT introduce per-step shaping rewards. The env stays
  terminal-only.
- Do NOT change the action space layout
  (project logits / mode logits / maint / setup / inv release) or
  the terminal reward vector shape — downstream consumers will
  break.
- Keep determinism: same seed must give same world.
- Keep continuous-action only. No discrete branches.

## How a session ends

When you have a diagnosis + fixed env that passes the calibration
shape above. Document everything in the calibration note. Do not
commit; the proposer will review and commit.
