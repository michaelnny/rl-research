# Red-team review — substrate redesign 2026-06-30

## Summary

The redesign is directionally right that the current substrate is too short and too low-action-dimensional, but the plan risks replacing an honestly small benchmark with a benchmark that is *nominally* long-horizon and high-dimensional while still decomposing into local scheduling/planning primitives. My largest concern is that the proposed resource and logistics tasks are deterministic, fully knowable, continuous relaxations of classical planning/OR problems. If demand schedules, graph topology, and dynamics are visible and fixed, then MPC, CEM, greedy dispatch, min-cost flow, or hand-coded rollout heuristics may dominate without exercising the algorithm class the lab says it wants to discover. Conversely, if the heuristic is rebuilt until it succeeds, the lab may erase exactly the failure signal that would have made the new substrate hard to fool.

## Specific concerns

### Horizon 10k is not a sufficient hardness axis

I agree with the narrow claim that simply changing `horizon=10000` in the current envs is degenerate, but that does not by itself justify the proposed structural changes. In the current maze, the waypoint controller reaches the goal and then spends thousands of steps accumulating action/path costs. In the current resource env, a focused or greedy policy completes all projects and then idles; the remaining horizon mostly tests whether the policy knows to output zeros. That is not long-horizon credit assignment.

A quick probe on the current envs makes the failure concrete:

```python
# PYTHONPATH=src .venv/bin/python
from rlh_bench.envs.resource_allocation import RecoverableResourceAllocationEnv, ResourceAllocationConfig
from rlh_bench.envs.continuous_maze import RecoverablePointMazeEnv, RecoverableMazeConfig
from rlh_bench.baselines.heuristics import ResourceGreedyPolicy, MazeWaypointPolicy
from rlh_bench.metrics import rollout
import numpy as np

for K in [5, 8, 128]:
    env = RecoverableResourceAllocationEnv(ResourceAllocationConfig(horizon=10000, num_projects=K))
    print(K, rollout(env, ResourceGreedyPolicy(env), seed=0).info)

env = RecoverablePointMazeEnv(RecoverableMazeConfig(horizon=10000, action_dim=2))
print(rollout(env, MazeWaypointPolicy(env), seed=0).reward_vector)
```

I observed success for resource even at `K=128`, and maze success with a terminal vector dominated by energy/path terms accumulated after the task was already solved. That supports the plan's diagnosis, but it also suggests a third option: keep some existing families and change the *temporal objective* rather than only adding new task families. Examples: random terminal evaluation time, recurring but coupled goals, irreversible choices that only reveal much later, finite energy budgets, maintenance/fatigue state, or absorbing terminal-success states with horizon-normalized costs. Those would test whether horizon matters without requiring a wholesale move to deterministic OR-style scheduling.

### Multi-waypoint maze does not automatically rescue the maze, but dropping maze is too strong

The proposal is right that naive multi-waypoint navigation can become `N` copies of a short-horizon waypoint follower. If the next target is visible, feasible, and locally solvable, then PD-to-next-waypoint never needs to assign credit across the full episode. The failure mode is particularly clear because the existing `MazeWaypointPolicy` already encodes this decomposition.

But the conclusion "drop the maze" is not defended. Maze is the only current family with continuous physical state, geometry, obstacles, collision recoverability, actuator redundancy, and path-dependent state dynamics. Resource allocation and logistics are not substitutes for that; they test budget scheduling more than embodied control. A better red-team target would be: what structural addition would make the maze non-decomposable? For example:

- keys/doors or one-way gates where early route choices determine later feasibility;
- energy/fuel/battery constraints where detours have delayed consequences;
- moving hazards or timed gates whose phase couples waypoints;
- partial map visibility with loop closures;
- final objective depending on the *set/order* of regions visited, not just final position.

Without testing such variants, retiring maze discards the only spatial-control substrate because the simplest possible long-horizon patch is bad.

### Rolling deterministic demand may decompose into independent windows

The resource redesign says each project has periodic demand peaks and allocations only count in demand windows. If the schedule is deterministic and visible, the agent can solve each demand window with a receding-horizon dispatcher: allocate to the currently urgent project(s), maybe with a short lookahead for readiness. That is not obviously harder than the current chain; it is a repeated finite-horizon allocation problem.

The strong version of the claim needs a coupling mechanism across windows. Examples that would make long-horizon decisions consequential:

- resources are storable/perishable, so serving window `t` affects inventory at `t+1000`;
- setup/switching costs make changing projects expensive;
- progress/readiness has hysteresis or decay;
- windows compete for a shared latent capacity that cannot be replenished instantly;
- early over-service creates overstock or maintenance debt that changes later efficiency;
- some demand is uncertain or only inferable, requiring information-gathering.

If windows are independent and the schedule is known, then the expected optimal policy is just "solve the next window" repeated `K` times. A benchmark can be long in wall-clock steps and still short in credit-assignment depth.

### Visible schedule turns resource allocation into planning; hidden schedule turns it into memorization unless seeds generalize

The proposal correctly flags "visible = planning problem; inferable = real RL" as an open question. I would sharpen this: inferable is not automatically RL if the world is deterministic and reset seeds are reused. It may become sequence memorization. Visible deterministic schedules favor MPC/CEM/DP; hidden deterministic schedules favor recurrent policies that memorize phase after a few episodes.

This is not just philosophical. The repo currently has deterministic resets where seed often does not alter the world (`test_resource_reset_is_deterministic` asserts reset observations are identical for seeds 1 and 2). Baseline reports therefore have `std_return=0` for many deterministic policies. If the new schedule is also fixed per env ID, then repeated lab sessions can hard-code it whether or not it is in the observation.

The plan needs an evaluation distribution: train worlds vs held-out worlds, fixed public seeds vs hidden seeds, and what information an algorithm may use about the env generator. Without that, "visible vs inferable" is the wrong binary; both collapse under a deterministic single-world benchmark.

### Inventory routing as `Box([0,1]^N)` risks becoming a fluid relaxation, not routing

The proposed logistics action is a continuous vector of fractions of fleet capacity released toward each node. That is not the natural action space of inventory routing. Natural vehicle-routing decisions are discrete/hybrid: which vehicle goes where, in what order, carrying how much. A continuous release vector may turn the problem into a divisible-flow approximation where the agent never chooses routes, only rates.

That matters because the hard part of VRP/IRP is combinatorial sequencing under capacity, travel time, and time windows. If the action can fractionally split fleet capacity across 128 nodes each step, then the substrate may remove the very structure that made logistics attractive. If travel is deterministic and demand accumulates predictably, a min-cost-flow, earliest-deadline-first, or nearest-urgent dispatch heuristic may be close to optimal.

A concrete acceptance probe before implementation: build the simplest `InventoryNearestUrgentPolicy`, plus two OR-style baselines:

1. earliest-deadline/highest-backlog fractional dispatch;
2. one-step or short-horizon min-cost assignment on current backlog and travel times.

If those two are near the final benchmark target, the family is validating OR heuristics, not novel RL algorithms.

### Logistics has many strong classical baselines; "beat heuristic" may be too low a bar

Vehicle routing, inventory routing, and lot-sizing have decades of strong heuristics: Clarke-Wright savings, insertion/local-search, sweep algorithms, tabu search, Lagrangian relaxation, rolling-horizon MILP, and modern OR-Tools-style constraint solvers. The plan mentions only `InventoryNearestUrgentPolicy`. That baseline is useful as a smoke test but dangerously weak as a difficulty signal.

The lab need not vendor a full solver, but the review standard should not be "beat nearest urgent." A substrate can be too easy even when random fails. I would require at least a small hand-coded heuristic suite: nearest urgent, earliest deadline first, backlog/distance ratio, periodic restocking, and a short-horizon greedy rollout. If a candidate algorithm beats only the weakest one, the result says little about novelty.

### Multi-mode adaptive control is probably not the right escape hatch either

The prompt asks about multi-mode adaptive control. I do not see it in the concrete plan, which is probably good. A typical deterministic multi-mode linear system with known modes and quadratic costs is an LQR/MPC/iLQR problem; with unknown but identifiable modes it is often system-identification plus robust/adaptive control. That can be a fine control benchmark, but it is not automatically a differentiator for Q-learning/PPO/AlphaZero/MCTS-class algorithmic novelty.

If the proposer later revives this option, the same standard should apply: show the non-LQR part. Mode switches must create delayed information value, nonconvex constraints, hybrid decisions, or safety/recoverability trade-offs that a standard MPC baseline does not solve.

### Fully deterministic `seed=world` weakens recoverability and invites planners to dominate

The substrate's deterministic philosophy makes debugging easy, but it becomes load-bearing at H=10k. If the agent can know the full world in advance, recoverability is less about online adaptation and more about whether the planner chose a robust trajectory. In a fully known deterministic logistics or demand-scheduling world, MPC/CEM can optimize the entire future, while RL-style trial-and-error is sample-inefficient theater.

This also changes the meaning of "recoverable." In a known deterministic world, a missed demand window is not a surprise; it is a planning failure. Recovery after a bad step should be tested by injecting off-policy perturbations at multiple times and measuring whether the policy can re-plan from the perturbed state, not just by letting it run from the initial condition.

Recommended probe:

```python
for t_inject in [10, 100, 1000, 5000]:
    # Roll policy to t_inject, replace action with worst/legal action for m steps,
    # then return control to the policy. Report terminal vector delta.
    pass
```

If performance collapses after a single missed window, the env is not recoverable at the temporal scale that matters. If performance does not change, the long horizon is not consequential.

### Continuous-only contradicts the stated algorithm-class mission

The proposal acknowledges that continuous-only structurally disadvantages MCTS and AlphaZero-style algorithms. That acknowledgment is not enough. The lab mission explicitly names AlphaZero and MCTS; a substrate with only `Box` actions biases the search toward SAC/PPO/CEM/mirror-descent descendants and away from search over structured discrete decisions.

If the user insists on continuous 32--128 dimensions, the lab should update the mission or add at least one sanctioned discrete/hybrid evaluation track later. Otherwise future claims of "same class as AlphaZero/MCTS" are not supported by the substrate. A continuous relaxation of routing is not equivalent to a combinatorial action space.

### `K=128` continuous actions may be high-dimensional only syntactically

`Box([0,1]^128)` with a budget projection is not automatically a hard 128-dimensional action space. Under a unit budget, most useful policies may reduce to choosing one or a few indices and an intensity. In the current resource env, the focused policy is exactly `argmin incomplete -> allocate 1.0`; it solves success even at `K=128` with H=10k. Greedy also solves success, albeit with poor scalar return because cost/delay weights dominate.

The proposed rolling-demand version could preserve this collapse if the best action is "allocate to the most urgent feasible project." In that case the effective action is categorical/top-k selection, but represented through a continuous vector. That representation is awkward for AlphaZero/MCTS and easy for hand-coded argmax heuristics.

A 5-line diagnostic for any proposed implementation:

```python
# Compare learned/optimized actions to top-m sparsity.
a = policy(obs)
print(np.sum(a > 1e-3), np.max(a), np.argsort(a)[-5:])
# Then replace a by the same mass on only its argmax/top-2 and rerun.
```

If top-1/top-2 projection preserves return, the claimed action complexity is mostly illusory.

### Reward scaling with horizon is currently under-specified

The plan keeps terminal-only vector rewards but extends horizons by 10--100x. Costs such as energy, distance, travel, lateness, and service aggregation will scale differently with H. In the current maze H=10k probe, the terminal scalar return is dominated by energy/path penalties after success. In resource H=10k with K=128, scalar return is dominated by cost/safety/delay magnitudes, not the success bit.

The redesign needs explicit normalization rules: per-step costs averaged over horizon? per-demand-window service normalized by total demand? lateness capped or normalized by number of windows? Without this, "harder at 10k" may simply mean reward weights are miscalibrated. Worse, different families may become incomparable because one terminal vector component grows with H and another is bounded in `[0,1]`.

### Terminal-only reward plus long horizon may create compute pain rather than credit-assignment insight

Terminal-only feedback is a legitimate substrate rule, but H=10k multiplies the cost of every policy evaluation. For black-box methods, no reward arrives until after 10k simulator steps; for policy gradients, return variance rises unless there is structure in the observations; for CEM, each sampled policy is expensive and mostly receives a scalar summary.

That can be a good stressor, but only if the environment contains a true long-range dependency. If it decomposes into windows, the terminal-only wrapper is artificial sparsity around a short-horizon controller. The lab should distinguish "environmental reward is terminal-only" from "diagnostics cannot expose per-window traces." Keeping private diagnostic traces for review could help detect decomposition without giving learners shaping rewards.

### Rebuilding heuristics to succeed may make the benchmark easier to fool

The plan says heuristics need to extend so heuristic success rate stays meaningful. I am not convinced. A heuristic success rate of 0/20 can be exactly the right warning light if other simple policies succeed or if the heuristic encodes the wrong inductive bias. Session 0001 already found this in the existing resource env: `ResourceGreedyPolicy` failed Small/Large, while a stateless focused policy solved success on all resource envs. The problem was not impossibility; it was a bad heuristic.

For the redesign, do not tune one heuristic until it succeeds and call that the difficulty signal. Use a portfolio:

- zero/random for exploration rarity;
- myopic greedy;
- focused/argmax/top-k;
- deadline-aware;
- short-horizon rollout/MPC;
- maybe an oracle upper bound on a small instance.

A good hard env may have `nearest_urgent=0/20` while a short-horizon planner gets partial success and an algorithm must improve the Pareto frontier. A single heuristic's success rate should not be the main validation metric.

### CEM at H=10k is likely expensive and weakly informative

The proposal says `train_cem` will just take 10x longer. That understates both cost and interpretability. For proposed resource Large, observation dim is `3K+1 = 385`, action dim is `128`, so the current linear CEM policy has

```text
128 * (385 + 1) = 49,408 parameters
```

With default `iterations=8`, `population=32`, `eval_episodes=1`, one env already requires 256 full 10k-step rollouts just for training. My quick current-env timing for H=10k, K=128 was about 0.9s per rollout in pure Python, implying minutes for one CEM cell before routing overhead. The proposed logistics env may be slower.

If CEM fails, it may only show that 32 samples in a 49k-dimensional Gaussian are useless. If it succeeds on a deterministic fixed world, it may show that black-box overfitting to one world is enough. Neither outcome is a clean differentiator. CEM is still useful as a smoke baseline at Small/v0, but H=10k CEM should be optional or replaced by a budgeted short-horizon MPC/CEM diagnostic that answers a specific question.

### The plan lacks acceptance criteria for "long-horizon consequential"

The proposal lists changes but not tests that would falsify them. I would require quantitative gates before merging:

- Performance should degrade when policies are given only a short receding horizon, and improve with longer lookahead.
- Top-1/top-2 action sparsification should not preserve most of the return if action complexity is claimed.
- A greedy window-by-window policy should leave significant return on the table relative to a planner or oracle.
- Injected mistakes at early/mid/late times should produce graded recoverability, not total collapse or no effect.
- Held-out seeds/worlds should change outcomes enough that memorization is detectable.
- Reward component magnitudes should remain comparable across Small/v0/Large after normalization.

Without such gates, the implementation can satisfy the text of the proposal while failing its purpose.

### The design over-indexes on horizon and action dimension, under-indexes on other hard axes

The user asked for H=5k--10k and continuous 32--128 actions, but the lab's mission names problem features beyond length: structured action spaces, recoverability landscapes, exploration sparsity, signal-vs-policy gap, and non-trivial credit assignment. Long horizon is one axis; it is not always the right axis.

Potentially more important deficits in the current substrate:

- no partial observability;
- no train/test distribution over worlds;
- no stochasticity or exogenous shocks;
- no irreversible but recoverable commitments;
- weak exploration structure beyond random success = 0;
- scalarization weights that can dominate vector trade-offs;
- continuous relaxations where the natural problem is discrete/hybrid.

A 1k horizon with hidden regime switches and held-out worlds might be a better novelty filter than a 10k deterministic schedule with visible windows.

### The logistics family may violate its own physical metaphor

"A fleet services demand across a graph" implies vehicles have positions, capacities, travel times, and cannot be fractionally everywhere. The action "fraction of fleet capacity to release toward each node this step" needs careful state semantics:

- Where is released capacity coming from?
- Can capacity be recalled or redirected while in transit?
- Are vehicles divisible into arbitrary fractions?
- Does sending 0.01 to each of 100 nodes create 100 tiny vehicles?
- Are edge capacities or vehicle counts respected?

If the answers relax these constraints, the env is closer to continuous inventory flow than routing. That may be fine, but then do not claim it supplies routing's combinatorial structure.

### The proposed families may not produce algorithmic novelty rather than domain engineering

A lab searching for a general RL algorithm should avoid a substrate where the best ideas are domain-specific dispatch rules. Resource windows and logistics graphs are rich, but they invite hand-coded scheduling heuristics. That can be productive if the lab explicitly wants algorithms that discover scheduling structure, but it risks rewarding agents that implement bespoke OR policies in `experiments/`.

The review question should be: could a non-domain-specific learning/search principle transfer across maze, resource, and logistics? Dropping maze and adding two allocation-like families narrows that transfer test.

### The observation contract for future schedules is load-bearing

For rolling demand, the observation must include enough information to make decisions: current backlog, future windows, time phase, maybe project-specific calendars. If the full future schedule is included, observation dimension may explode or encode an oracle plan. If only current phase is included, the task may be partially observable. If schedules are deterministic from `t/H`, a policy can infer them without observation.

This should be specified before implementation. Otherwise later baseline comparisons will be confounded by different algorithms reconstructing hidden calendars from code access, rollout history, or memorization.

### The baseline report's current scalar target is not safe for vector tasks

The proposal keeps terminal vector reward, but the current baseline report still centers scalar return and success. Session 0001 showed that a focused resource policy can trade safety violation for success and win scalarized return because weights favor success. New long-horizon resource/logistics tasks will likely have richer trade-offs: fill rate vs lateness vs distance vs overstock vs safety.

If the redesign is meant to support vector-RL novelty, the baseline rebuild should report Pareto fronts, hypervolume under multiple reference points, and component-wise dominance, not only default scalarized return. Otherwise the lab will optimize whatever arbitrary weights are in `RewardSpec`.

### Deterministic graph topology per tier encourages hard-coded policies

The proposal asks whether routing graph topology should be same across resets or seeded per reset. A fixed topology per tier is easier to debug, but it makes hard-coding and repeated overfitting likely. A seeded graph distribution with held-out evaluation worlds is more honest. The env can still be deterministic given seed; determinism need not mean one public world.

A compromise: Small has fixed topology for tests and examples; v0/Large sample from a documented graph generator, with train seeds and held-out eval seeds. That preserves reproducibility without collapsing to one route map.

### Long-horizon recoverability needs a formal metric

The plan says recoverability may change at H=10k but does not define the desired shape. In short-horizon maze, a bad collision wastes time but can be undone. In windowed demand, missing a window may be unrecoverable by design. Both can be valid, but they test different things.

I would define recoverability curves:

```text
R(t, m) = return(policy after m-step adversarial/random perturbation at time t)
          - return(unperturbed policy)
```

Report this for early, middle, late times. A good recoverable long-horizon env should show partial degradation and policy-dependent recovery, not cliff failures for all policies or no sensitivity.

### The plan does not defend why terminal-only plus visible future is the right signal-vs-policy gap

A signal-vs-policy gap means the reward signal is sparse/delayed relative to the behaviors needed. But if the full future schedule and deterministic dynamics are visible, the signal needed for behavior is in the observation/model, not the reward. Planning can solve without learning from reward except for tuning weights.

That may be acceptable if the lab wants planning algorithms, but then CEM/MPC/MCTS-style baselines become central. If it wants RL algorithms from sparse terminal feedback, it needs uncertainty, hidden state, or a distribution over tasks where reward feedback teaches something not already encoded in the known model.

## What the plan got right

1. **The current envs are not honestly long-horizon.** Registry horizons of 60--180 and redundant action dimensions are too small for the stated mission. The proposal correctly refuses to just relabel them as long-horizon.

2. **Naive multi-waypoint maze is not enough.** The plan correctly identifies that waypoint stitching can preserve only local credit assignment.

3. **The open questions are the right danger zone.** Schedule visibility, graph seeding, CEM cost, and recoverability semantics are genuinely load-bearing. They should be resolved before code, not after baselines happen to pass.

4. **Smoke variants are important.** Keeping Small/v0 cheaper while having a Large stretch is sensible. The lab needs fast iteration as well as a hard target.

5. **Continuous-only is at least acknowledged as a mission mismatch.** I disagree with accepting it silently, but the plan does not hide the issue.

## Open questions for the proposer

1. What concrete test would convince you that the rolling-demand resource env *does not* decompose into independent short-horizon windows?

2. Will v0/Large evaluate on held-out generated worlds, or on one deterministic public world per env ID? If held-out, what changes across seeds: demand schedules, graph topology, costs, travel times, start inventories?

3. What baseline portfolio, beyond nearest-urgent and demand-aware greedy, must a candidate beat before the lab treats an improvement as meaningful?

4. Why is dropping maze preferable to adding long-horizon coupling to maze? What capability is resource/logistics intended to cover that spatial continuous control does not, and vice versa?

5. Are AlphaZero/MCTS-style algorithms still part of the lab's target class after the continuous-only decision? If yes, where will their natural discrete/hybrid action substrate live?
