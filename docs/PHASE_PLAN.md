# Phase Plan

## Overall goal

Build a small benchmark suite for recoverable long-horizon sparse-feedback RL. The benchmark should let researchers study exploration and credit assignment under terminal scalar or vector-valued environmental outcomes without requiring heavy compute.

## Phase 1: problem validation

### Objective

Determine whether the proposed environment families isolate the desired research phenomenon.

### Questions

1. Are the tasks genuinely long-horizon?
2. Are rewards terminal-only?
3. Is random success rare enough to create exploration difficulty?
4. Are early mistakes recoverable rather than instantly fatal?
5. Does the terminal vector provide meaningful outcome dimensions?
6. Does the action space scale beyond tiny toy actions?
7. Can all experiments run cheaply on CPU or a single low-end GPU?

### Deliverables

- Environment acceptance checklist.
- Candidate environment scorecard.
- Random and heuristic baseline reports.
- Initial difficulty curves over horizon and action dimension.
- Decision on which family becomes the primary benchmark track.

### Recommended experiments

For each environment ID:

```text
RecoverablePointMaze-Small-v0
RecoverablePointMaze-v0
RecoverablePointMaze-HD-v0
RecoverableResourceAllocation-Small-v0
RecoverableResourceAllocation-v0
RecoverableResourceAllocation-Large-v0
```

run:

```text
random policy, 100-1000 episodes
heuristic policy, 10 episodes
CEM baseline, small budget
optional REINFORCE baseline, small budget
```

Record:

```text
success rate
mean scalar return
mean terminal vector
first success episode
collision/cost/delay/safety outcomes
runtime per episode
```

## Phase 2: benchmark prototype

### Objective

Turn the most promising environment families into a reproducible benchmark with controlled difficulty knobs.

### Tasks

1. Freeze one or two canonical configurations.
2. Add benchmark-report scripts.
3. Add recovery-test protocols, such as forced early mistakes.
4. Add vector-reward evaluation scripts.
5. Add regression tests for benchmark metrics.
6. Compare small standard baselines under equal compute.

### Expected output

A benchmark paper or internal technical report claiming:

> Existing small RL environments do not cleanly isolate recoverable long-horizon exploration and terminal vector credit assignment. RLH Bench provides deterministic, cheap-to-simulate environments with controlled difficulty knobs and natural terminal outcome vectors.

## Phase 3: algorithm research, if desired

Only after the problem definition is validated should the project move to algorithm design.

Potential directions:

- terminal-vector credit assignment;
- hindsight relabeling for vector outcomes;
- trajectory-level exploration objectives;
- preference-conditioned policies over terminal vectors;
- recovery-aware exploration metrics.
