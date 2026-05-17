# Benchmark suite

Three primary benchmarks (one per pillar from [charter.md](charter.md)) plus two
sanity envs. All are *existing*, *well-known* problems — we do not invent envs.

## Sanity envs (Stage A)

Used by every candidate, regardless of primary benchmark. Pure correctness check.

| env_id          | type                  | obs    | action       | random return |
| --------------- | --------------------- | ------ | ------------ | ------------- |
| `CartPole-v1`   | classic-control       | 4      | `Discrete(2)`| ~22           |
| `Pendulum-v1`   | classic-control       | 3      | `Box(1)`     | ~-1200        |

Sanity budget: 50,000 env steps per env, 5-minute hard cap.
Pass criteria: see [loop.md](loop.md) §Stage A.

## Pillar 1 — sparse rewards over long horizons

| env_id                          | source     | obs                | action        |
| ------------------------------- | ---------- | ------------------ | ------------- |
| `ALE/MontezumaRevenge-v5`       | gymnasium + ale-py | (210, 160, 3) uint8 | `Discrete(18)` |

Why this one: the canonical sparse-reward benchmark. Vanilla policy gradients
score 0 on this in 200M frames. Solving it requires structural mechanisms beyond
scalar credit propagation — that is exactly the test of pillar 1.

Use the standard NoFrameskip wrapper with frameskip=4 and grayscale 84×84
preprocessing — implementations are responsible for their own preprocessing
inside `train.py`.

**Early-phase budget:** 2,000,000 env steps × 2 seeds, ≤ 2h wallclock.
**Mass-run budget (Curator-only):** 50,000,000 env steps × 5 seeds.

## Pillar 2 — long-horizon dense control

| env_id (dm_control)             | obs (dict, total dim) | action |
| ------------------------------- | --------------------- | ------ |
| `humanoid.run`                  | 67                    | 21     |

Why this one: dense reward but very long horizon (1000 steps × physics integration
of a 21-DoF humanoid). Tests credit propagation over thousands of steps. Standard
dm_control environment, well-studied.

Loaded via `dm_control.suite.load(domain_name="humanoid", task_name="run")`.

**Early-phase budget:** 1,000,000 env steps × 2 seeds, ≤ 2h wallclock.
**Mass-run budget:** 30,000,000 env steps × 5 seeds.

## Pillar 3 — multi-signal feedback (vector reward)

| env_id            | source        | obs | action        | reward dim |
| ----------------- | ------------- | --- | ------------- | ---------- |
| `minecart-v0`     | mo-gymnasium  | 7   | `Discrete(6)` | 3          |

Why this one: the canonical multi-objective RL benchmark. Reward is a 3-vector:
(ore_1_collected, ore_2_collected, fuel_cost). The agent drives a cart in 2D
continuous state, picks up ore from sites, and returns to base — there are
genuine trade-offs between the three reward channels with no "obvious" scalarization.

Note on framing: this is the **Multi-Objective RL (MORL)** framing, not the
constrained-MDP / safety framing. We intentionally chose MORL because:
- The maintained safety-gymnasium package is stale (2023, incompatible with
  gymnasium 1.x).
- "Multi-signal" in the charter refers to *vector rewards*, not constraint
  thresholds. MORL fits cleanly.

If the project later wants to test the CMDP framing too, OmniSafe is the
maintained alternative; add it as a fourth benchmark, do not replace minecart.

Loaded via `mo_gymnasium.make("minecart-v0")`. The env's `step()` returns a
3-vector reward, not a scalar.

**Early-phase budget:** 1,000,000 env steps × 2 seeds, ≤ 2h wallclock.
**Mass-run budget:** 10,000,000 env steps × 5 seeds.

## Per-pillar choice rule

Each candidate's `hypothesis.md` selects ONE primary benchmark — the one most
relevant to its claimed mechanism. Promotion to mass-run requires running on
the primary plus at least one additional benchmark from a different pillar.

| Hypothesis claims to address...  | Primary benchmark               |
| -------------------------------- | ------------------------------- |
| sparse-reward / exploration      | `ALE/MontezumaRevenge-v5`       |
| long-horizon credit assignment   | `humanoid.run`                  |
| multi-signal / vector rewards    | `minecart-v0`                   |

If a hypothesis claims to address two pillars, the Researcher picks the one for
which the evidence will be cleanest in the early phase.

## Random-policy baselines

Computed once and stored in `lab/baselines/random.json`. Used by the sanity gate
and as a sanity floor for primary benchmarks.

| env_id                          | random episodic return (mean over 100 eps) |
| ------------------------------- | ------------------------------------------ |
| `CartPole-v1`                   | 22.7                                       |
| `Pendulum-v1`                   | -1199                                      |
| `ALE/MontezumaRevenge-v5`       | 0.0                                        |
| `humanoid.run` (dm_control)     | 0.87                                       |
| `minecart-v0` (per channel)     | [0.36, 0.33, -10.58] (scalarized: -9.88)   |

The random baselines are computed by `scripts/build_baselines.py` and stored in
`lab/baselines/random.json` (which carries the per-env std/min/max as well).
Re-run that script if the suite changes.

## PPO baseline

The frozen PPO yardstick lives at `src/rl_research/baselines/ppo.py`. Frozen
hyperparameters per benchmark are documented inline in that file. PPO results
on each benchmark are stored in `lab/baselines/ppo/`. PPO is the only allowed
baseline; everything else is a candidate.
