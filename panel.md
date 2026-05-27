# panel.md

Rationale for the **two-tier** panel — a 5-env smoke tier the agent iterates
against, and a 4-env hard tier the agent is judged against. Each env is chosen
to detect a specific failure mode catalogued in `prior_attempts.md`. A
candidate that beats only one axis (sparse-long-horizon **or** vector-reward)
gets a partial score; only a candidate that handles both axes is interesting.

## Mission alignment

The mission (from `program.md`) is to find a behavior-improvement primitive
that simultaneously:

1. **Handles long-horizon sparse-reward problems** with prerequisite structure.
2. **Consumes vector reward** `r ∈ ℝᵏ` natively, without scalarization to `wᵀr`.
3. **Replaces what value *does*** — future compression, temporal composition,
   local improvement.

The smoke tier splits 2+3 along axes 1 and 2 (2 sparse gridworlds + 3 native
vector envs). The hard tier extends each axis to its open-research frontier
(Craftax, MiniHack, mo-halfcheetah, Humanoid). Axis 3 (value-role replacement)
is not a benchmark axis — it is a structural constraint on the algorithm,
checked by reading the commit description against the disqualifier-family
list in `prior_attempts.md`.

## Why two tiers

Iteration speed and selection pressure pull in opposite directions: the smoke
tier (5 envs × 300 s, parallel ⇒ ~5 min wallclock per sweep) lets the agent
do ~12 iterations/hour during active development; the hard tier (4 envs ×
3600 s, Option-B grouped ⇒ ~2 h wallclock per sweep) is reserved for
candidates that already cleared the smoke bar. The agent runs smoke
constantly; the hard tier runs only when the agent claims a serious result
or on a periodic checkpoint cadence (e.g. nightly).

## Smoke tier (5 envs, ~5 min/sweep parallel)

| Env | Type | Channels | Failure mode it detects | Why this variant / size |
| --- | --- | --- | --- | --- |
| **`MiniGrid-DoorKey-8x8-v0`** | scalar | 1 | Pure-exploration rebadges (frontier-graph, novelty, RND, count-bonus) — they find the key but do not learn that *picking-up-key-then-using-it-on-door* is the prereq. *Killed prior attempts CARL #4 (frontier-graph), KERNEL-RL #3 (passive correlation mining); TOP #11 was alive on DoorKey-5x5, weak on 6x6.* | 8x8 is the canonical "DoorKey is hard" size — random fails (~0% success); count-bonus gets it occasionally. Smaller variants (5x5, 6x6) are too easy and TOP-style methods already solved them. |
| **`MiniGrid-KeyCorridorS3R3-v0`** | scalar | 1 | Mining-on-passive-correlation (KERNEL-RL, OPP, EOP/COP) — key location varies by seed, so passively-correlated "actions before reward" is uninformative; the agent must actively explore unlocked rooms before unlocking the target. *Killed prior attempts #3 KERNEL-RL, #7 OPP, #8 EOP/COP.* | S3R3 (3 rooms × side-length 3) is the smallest size that has a hidden subgoal; smaller variants don't have the search-then-use structure. |
| **`deep-sea-treasure-concave-v0`** | vector | 2 | **Linear scalarization rebadge** (the strict version of generic scalarization). The CONVEX DST is solved by `wᵀr` for any sweep over `w` — every Pareto-optimal treasure is on the convex hull. The CONCAVE variant has Pareto-optimal treasures that are *unreachable by any linear scalarization* (they live in the concave region of the front). An algorithm that internally collapses to scalarization will miss those treasures. *Canonical detector for the disqualifier "Scalarized vector-reward maximization."* | Concave variant specifically. Picking the convex variant defeats the purpose. |
| **`minecart-v0`** | vector | 3 | Generic scalarization rebadge — 3 channels (ore-1, ore-2, fuel) with conflicting goals (mining ore worsens fuel). A scalarized agent picks a fixed weighting and produces a single point on the front; a true Pareto-aware agent produces a set covering the front. | Standard MO-Gym benchmark. 3 channels is the minimum for non-trivial trade-offs; episodes are 50–500 steps so it has some horizon. |
| **`mo-reacher-v4`** | vector | 4 | Single-objective collapse — 4 simultaneous channels, one per target. An algorithm that focuses on one channel and ignores the others ("local improvement on a fixed objective") gets one channel high and the rest low; a true vector-aware algorithm balances all four. *Disqualifier "Scalar-weighted log-prob update" (PPO/REINFORCE family) collapses multi-channel reward by construction.* | v4 is stable and standard. Continuous-control diversity in the panel — the only non-discrete-action smoke env. |

The smoke tier is what the agent iterates against. **It is strictly easier
than the hard tier.** Beating `strong` on every smoke env is the *threshold to
graduate to the hard tier*, not the success criterion for the project.

## Hard tier (4 envs, ~2 h/sweep with Option-B grouping)

The hard tier extends both axes to the open-research frontier, plus a
high-DOF continuous-control breadth env. Three are SOTA-resistant in their
own right; one is a vector-reward continuous-control benchmark with a
published Pareto-hypervolume frontier.

### Option-B scheduling (run_panel.py --hard)

- **Phase 1 (solo, ~1 h):** Craftax-Symbolic-v1 alone. The Craftax JAX kernel
  saturates the GPU; running it alongside another GPU-using job hurts both.
  Forced to CPU via `JAX_PLATFORMS=cpu` to leave VRAM free for torch in
  Phase 2.
- **Phase 2 (parallel, ~1 h):** MiniHack-Quest-Hard-v0 + mo-halfcheetah-v4 +
  Humanoid-v5 in parallel. All CPU-bound, small policies, fit in shared
  CPU + a sliver of VRAM each.

| Env | Type | Channels | Why this env, what it tests, sourced reference |
| --- | --- | --- | --- |
| **`Craftax-Symbolic-v1`** | scalar | 1 | **Crafter+NetHack hybrid, JAX-native, ICML 2024** (Matthews et al., *"Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning"*, arxiv 2402.16801). 22 achievements over 4 difficulty tiers; the two hardest tier-4 achievements (Collect-Diamond, Eat-Plant) remain unreached at 1B env steps under PPO + RND in the published paper. Sparse, long-horizon, open-ended. The standard Craftax-v1 reward is dense per-achievement; what makes it hard is the *prerequisite chain* — wood → workbench → stone-pickaxe → coal → iron-pickaxe → … . |
| **`MiniHack-Quest-Hard-v0`** | scalar | 1 | NetHack mini-quest with prerequisite chain (find amulet, navigate corridor, defeat guard, retrieve quest item). RL from scratch gets near-zero on this; published baselines (DRL-Agent in the MiniHack paper, Samvelyan et al. NeurIPS 2021 datasets-and-benchmarks; IMPALA) reach <10% completion. The hardest single env in the standard MiniHack suite for from-scratch RL. *Linux-only on PyPI — minihack 1.0.2 is sdist-only and the build requires cmake; macOS workstations skip this env at smoke-time and rely on the linux box for hard sweeps.* |
| **`mo-halfcheetah-v4`** | vector | 2 | mo-gymnasium 1.3.2 native 2-channel reward: (forward velocity reward, control-energy cost). Continuous-control vector-reward; PGMORL (Xu et al. ICML 2020, *"Prediction-Guided Multi-Objective Reinforcement Learning"*) and Envelope MORL (Yang et al. NeurIPS 2019, *"A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation"*) define the published Pareto-hypervolume frontier. Hard-tier vector-reward in continuous control. |
| **`Humanoid-v5`** | scalar | 1 | gymnasium-MuJoCo standard library humanoid: 17-dim continuous action, 348-dim observation, whole-body balance + locomotion. Dense reward but **high-DOF continuous control breadth** — the only non-RL-research-tradition env in the panel. Published baselines: SAC (Haarnoja et al. ICML 2018) reaches ~5500 return; TD3 ~5400; PPO ~3500 with careful tuning. A candidate that handles sparse-and-vector but collapses on high-DOF dense control is not the third-family primitive we want. |

### Why these four (and not others)

- **Safety-Gymnasium (constraint-satisfaction)** — dropped because
  `safety-gymnasium==1.0.0` hard-pins `gymnasium==0.28.1`, incompatible with
  the modern stack the rest of the substrate uses. Cost-channel-as-vector
  use case is partially covered by minecart's fuel channel and
  mo-halfcheetah's control-energy channel.
- **HumanoidBench** — considered as the high-DOF slot, dropped because the
  upstream package hard-pins `gymnasium==0.29.1`, `mujoco==3.1.6`, and
  `torch==2.3.1`, all incompatible with our stack. Vendoring + pin-relaxing
  was a 1–2 h debug rabbit hole; `Humanoid-v5` from the standard MuJoCo
  library covers the high-DOF breadth without the integration cost.
- **Atari Montezuma / Pitfall** — long-horizon-sparse classics, but pulling
  ALE + autorom into the lock for one extra signal isn't worth it when
  Craftax + MiniHack already cover that axis at the open-research frontier.
- **Crafter (scalar achievement-sum)** — strictly subsumed by Craftax-v1,
  which is faster (JAX) and harder (longer prerequisite chain).

## What the panel does NOT test (and we accept this)

- **DeepSea / Chain / Tree-MDP monotone-progress tasks**: explicitly excluded.
  `prior_attempts.md` cross-attempt failure mode #3: *"Passing
  DeepSea/Chain/Tree is not strong evidence."*
- **Custom-built toy benchmarks designed around the candidate**: explicitly
  excluded. *Cross-attempt failure mode #4: "all four sprint-1 candidates
  succeeded on a benchmark designed around the method; all failed on
  standard tasks (DoorKey, KeyCorridor)."*
- **A single env that tests sparse-long-horizon AND vector-reward
  simultaneously** — splitting the axes across distinct envs forces the
  candidate to be good at *both*, separately, which is what the mission
  demands. The 9-env tuple `(n_beat_random, n_beat_strong)` over the split
  panel is harder to game than a single composite score.

## Frozen baseline scores

`baselines.json` (smoke tier): two scores per env — `random` (uniform-action
floor) and `strong` (max across the strong baselines: `eps_greedy_q.py`,
`count_bonus.py`). Built once via `scripts/build_baselines.py` and committed.

`baselines_hard.json` (hard tier): two columns per env — `published_sota`
(literature reference number, sourced) and `our_baseline.{random, strong}`
(what we observed running our own random + strong baselines, since literature
SOTA is rarely reproducible on a single 24 GB GPU under a 1 h wallclock cap).
The agent's hard-tier score is judged against `our_baseline`; the
`published_sota` column is informational.

## Score semantics

- **Per env:** scalar envs → mean episode return over `N_EVAL_EPISODES = 20`;
  vector envs → Pareto hypervolume of episode return-vectors vs a fixed
  per-env reference point (`harness.HV_REF`).
- **Smoke tuple:** `(panel_n_beat_random, panel_n_beat_strong)` over the 5
  smoke envs.
- **Hard tuple:** `(panel_n_beat_random, panel_n_beat_strong)` over the 4
  hard envs, where `random` and `strong` are read from
  `baselines_hard.json`'s `our_baseline` column.

To clear the bar in `prior_attempts.md` ("good evidence"), a candidate must
beat strong on ≥ 2 smoke envs with at least one win on a vector env, *and*
beat random on ≥ 2 hard envs in a Curator-promoted hard sweep.
