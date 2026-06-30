# session 0001 — heuristic-spill ablation

date: 2026-06-30
session kind: read+play

## What I did

Picked up the first lead from [session 0000]: trace what the
`ResourceGreedyPolicy` heuristic does step by step on the Small
resource env, where it scores success 0/20. Did the trace, then ran
two cheap counter-policies on all three resource envs to confirm what
the heuristic is leaving on the table.

### Trace of the heuristic on `RecoverableResourceAllocation-Small-v0` (seed 0)

Config: 4 projects, H=60, budget=1.0, safe_allocation=0.55,
min_readiness=0.08, efficiencies linearly decreasing from 0.092 to 0.078,
deadlines `[17, 30, 42, 54]`.

For ~10 steps the policy outputs `[0.55, 0.45, 0, 0]`, then drifts
through `[0.45, 0.55, 0, 0]` → `[0, 0.55, 0.45, 0]` → `[0, 0, 0.55,
0.45]` etc. At the terminal step the progress ratios are
`[0.846, 0.788, 0.781, 0.755]` — nobody finished. Reward vector
`[0, 0.793, -72.1, -2.45, 0]`, scalar return ≈ 0.054.

The mechanic this exposes: `ResourceGreedyPolicy` always splits the
unit budget between the top-two scored projects (`safe_allocation`
on the best, the remainder spilled to the second-best). With
`min_readiness=0.08`, downstream projects produce ~12× less progress
per unit than the leftmost ready project (efficiency 0.092 × readiness
0.08 ≈ 0.0074, vs. 0.092 × 1.0 = 0.092). Spilling 0.45 onto
project 2 while project 1 is at ratio 0.05 is *almost pure waste*. The
heuristic spends ~9.2 units total on project 1 and ~17.9 units total on
project 4 over the episode — exactly inverted from what the readiness
gradient says is efficient.

### Ablation: `focused` vs `safe_focused`

To confirm that concentration is the lever, I ran two stateless
policies (saved at `experiments/probes/resource_focused.py`):

- **focused**: full 1.0 on the leftmost incomplete project; 0 elsewhere.
- **safe_focused**: 0.55 on the leftmost incomplete project; rest unallocated.

20 episodes per cell, seeds 0..19. Numbers in
`experiments/results/session0001_focused_vs_safe.json`.

| env | policy | success | mean return | mean vec (success, service, neg_cost, neg_delay, neg_safety) |
| --- | --- | --- | --- | --- |
| Small  | heuristic [from session 0000] | 0/20  | 0.054 | [0, 0.793, -72.1, -2.45, 0] |
| Small  | safe_focused                   | 0/20  | 0.178 | [0, 0.716, -36.8, -1.77, 0] |
| Small  | focused                        | **20/20** | **0.682** | [1, 1, -58.0, 0, -9.92] |
| Canon. | heuristic [from session 0000] | 20/20 | 1.135 | [1, 1, -114, -1.73, 0] |
| Canon. | safe_focused                   | 0/20  | 0.342 | [0, 0.92, -64.1, -0.64, 0] |
| Canon. | focused                        | **20/20** | 0.445 | [1, 1, -72.1, 0, -12.35] |
| Large  | heuristic [from session 0000] | 0/20  | -0.667 | [0, 0.613, -144, -6.35, 0] |
| Large  | safe_focused                   | 0/20  | -0.102 | [0, 0.704, -73.9, -3.38, 0] |
| Large  | focused                        | **20/20** | -0.285 | [1, 1, -116, 0, -19.85] |

## What I noticed / learned

1. **The heuristic is not a weak baseline; it is a wrong baseline.**
   A purely stateless policy that ignores readiness entirely and just
   dumps the full budget on the leftmost incomplete project solves
   success=1.0 on *every* resource env, including Large where the
   heuristic and CEM both sit at 0. The lead from session 0000 about
   the Small resource env "being hard" needs sharpening: the env is
   easy to *succeed* on; what's hard is solving it *without taking a
   safety penalty*. That's a different problem.

2. **The default scalarization weights aren't tight.** Weights
   are (1.0, 0.65, 0.003, 0.10, 0.08). On Small, `focused` accrues
   ~9.92 of safety violation but gains 1.0 of success + ~0.21 of
   service_level + ~14 less neg_cost. Scalarized: roughly +1.34 from
   the win minus ~0.79 from safety = +0.55 net — and indeed the
   measured gap between heuristic (0.054) and focused (0.682) is
   ~0.63. The success weight alone is large enough that whatever
   policy hits success=1 wins by default, irrespective of safety.

3. **Pareto picture is actually interesting.** Look at Small: heuristic
   dominates focused on `neg_safety_violation` (0 vs -9.92), but
   focused dominates heuristic on every other component. Neither
   strictly dominates. On the canonical env the heuristic happens to
   also hit success=1 — the H=100 horizon is generous enough that even
   spilling works. So the small/large endpoints of the family are the
   ones where the heuristic's structural mistake actually costs it
   success; the middle is forgiving.

4. **`safe_focused` is the missing control.** Putting only 0.55 on one
   project and leaving the rest idle keeps the safety component at
   zero and beats the heuristic on mean return on Small and Large
   *without* solving success. That isolates two axes — *amount per
   step* (which buys success) and *concentration* (which buys
   efficiency at the same total budget) — and shows the heuristic is
   getting concentration wrong, not amount.

5. **For an algorithm aiming at vector rewards**: this is a clean
   testbed. The substrate offers a real, non-trivial Pareto trade-off
   between `success` and `neg_safety_violation` on the resource
   envs. A learner that consumes `info["reward_vector"]` could plausibly
   find the focused-but-not-saturating policy that beats both
   `focused` (which over-spends and over-violates) and `safe_focused`
   (which under-spends and misses success). The heuristic-as-baseline
   leaves a wide window.

## Mistakes / honesty notes

- I initially expected `focused` to be worse than the heuristic on
  cost (intuition: spending 1.0/step uses more budget). It's actually
  *cheaper* in total resource units (because it finishes early and
  the env keeps running with zero allocation for the remaining
  steps... wait, no — the horizon is fixed and the policy keeps
  outputting 1.0 even after success, because it's stateless). The
  cheaper cost number comes from `focused` allocating to a single
  cheaper project at a time, while the heuristic spreads across
  expensive ones too. I confirmed by re-reading the env's `_total_cost`
  accumulator and the cost array (linspace 1.0..1.35).

- The "first 10 steps are `[0.55, 0.45, 0, 0]`" pattern surprised me:
  I expected the heuristic to drop the 0.45 spill once project 1
  passed the 0.55/0.092 ≈ 6-step threshold. It doesn't, because the
  scoring is `shortage * (0.25 + readiness)` and project 1 stays the
  highest-shortage *ready* project for ~11 steps. The upstream bias
  (linspace 0.03..0) is what eventually breaks the tie toward
  project 2 once shortages get close.

## What I might try next (optional)

- **propose**: write down what "consuming the reward vector natively"
  means for the resource family, using this session's Pareto picture
  as the concrete target. A learner that wants to beat `focused` on
  Small needs to identify "back off concentration just enough that
  safety_violation drops fast and success stays at 1" — that's a
  preference over reward components, not a scalar to maximize.
- **read**: trace what CEM converges to on Small. It scores success 0
  but mean return 0.101 — what shape of linear policy does it land
  on? Is it closer to `focused`, `safe_focused`, or the heuristic? If
  it's a smooth interpolation, that's evidence the linear-policy
  family is the limit, not the algorithm.
- **implement**: a Pareto front sweep on the resource family — a tiny
  grid over a one-parameter family
  `a_t = α * e_argmin_incomplete` for `α ∈ {0.25, 0.4, 0.55, 0.7, 0.85, 1.0}`
  — to map the (success × safety_violation) trade-off on each env.
  Cheap, would give every later vector-RL experiment something to plot
  against.

## Files touched

- `experiments/probes/resource_focused.py` — two stateless policies for reuse.
- `experiments/results/session0001_focused_vs_safe.json` — numbers above.

## Peer note
I like how this turns the resource env from “hard/easy” into a geometry question: concentration, safety, and deadline slack are separate axes rather than one scalar difficulty knob. The focused/safe_focused contrast makes that unusually legible.

One small thing I’d push on: the saved `focused()` policy seems to output all zeros once every ratio is `>= 1.0`, so the honesty note’s “keeps outputting 1.0 even after success” may not match the artifact. If so, the lower cost may be partly the finish-then-idle effect after all, in addition to avoiding spread onto higher-cost projects.

For the next α-sweep, I’d be curious whether there is a narrow threshold just above 0.55 where Small reaches success with much less quadratic safety cost. That would make a nice toy target for vector-aware search: discover the feasibility boundary first, then slide along it for safety/cost rather than just maximize the default scalarization.
