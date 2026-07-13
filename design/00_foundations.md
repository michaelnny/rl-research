# Foundations for the rebuild

Status: design baseline, 2026-07-13

This document defines the project without inheriting requirements from the
existing implementation. The old repository is evidence about a failed attempt;
it is not the product specification.

## Mission

Build a compute-light scientific instrument and an autonomous research system
for discovering reinforcement-learning mechanisms that remain useful when all
of the following are present:

- decisions have causal consequences hundreds or thousands of steps later;
- actions are high-dimensional, very numerous, combinatorial, conditional, or
  some mixture of those forms;
- external feedback is sparse and may be terminal-only; and
- outcomes are vectors with genuine conflicts, not a scalar reward split into
  cosmetic components.

The goal is not to maximize one benchmark score. The goal is to identify which
algorithmic mechanisms help which capabilities, compose those mechanisms, and
produce reproducible evidence strong enough to support a novelty investigation.

## Formal target

Each task is drawn from a procedural family of finite-horizon POMDPs

`M(z, theta) = (S, O, A, P, R, H)`,

where `z` is a world instance and `theta` is a declared difficulty vector.
Terminal return is `G in R^m`, with `m >= 2`. The suite varies these independent
axes:

1. **causal span**: the lag between an action and the outcomes it can change;
2. **action structure**: flat discrete, embedded catalog, factored discrete,
   continuous vector, or conditional hybrid;
3. **feedback density**: terminal-only through sparse periodic feedback;
4. **objective protocol**: known preference, preference supplied at deployment,
   Pareto coverage, or constrained/lexicographic objectives;
5. **memory demand**: fully observed through controlled partial observability;
6. **world shift**: same dynamics, parameter shift, and compositional shift; and
7. **recoverability**: whether an early error is correctable at a measurable
   opportunity cost.

Episode length alone is not evidence of long-horizon credit assignment. Action
vector width alone is not evidence of a large action space. A reward vector
alone is not evidence of multi-objective learning. The benchmark must measure
the corresponding causal span, joint choice count or geometry, and policy-set
trade-offs directly.

## Three vector-objective protocols

“Vector reward” covers materially different research problems, so the suite
will not collapse them into one mode.

### Preference-conditioned control

A preference or utility descriptor is supplied to the agent. Evaluation tests
interpolation and extrapolation to unseen preferences and unseen worlds.

### Unknown-preference policy coverage

The utility is unknown during training. The algorithm returns a bounded policy
set or a conditional policy. Evaluation measures how well that set covers a
fixed utility panel under one shared interaction budget.

### Constrained control

Some vector components are constraints or lexicographic requirements rather
than exchangeable utility. Evaluation reports feasibility first and trade-offs
inside the feasible region second.

A fixed weighted sum is permitted as a baseline. It is never the definition of
the task and never the sole headline metric.

## What the project will not assume

- It will not be continuous-action-only.
- It will not force every environment to have terminal-only feedback.
- It will not equate deterministic simulation with benchmark validity.
- It will not equate a scripted heuristic with evidence that an RL learner can
  learn the task.
- It will not ban established libraries merely to make baseline code local.
- It will not call an environment “real-world” because its variables have
  scheduling or navigation names.
- It will not make prose, a journal, an LLM review score, or a paper draft the
  research product.
- It will not ask one scalar leaderboard to choose the research direction.
- It will not let candidate code alter benchmark definitions, hidden evaluation
  instances, metrics, or reference results.

## Scientific products

The durable outputs are:

1. versioned environment generators and evaluation protocols;
2. baseline fingerprints showing which capability each tier stresses;
3. typed hypotheses with quantitative predictions and falsifiers;
4. immutable experiment manifests, code revisions, raw results, and analyses;
5. independent replications and ablations; and
6. sourced prior-art maps separating “new here” from plausible field novelty.

Human-readable reports and journals are views generated from these records.

## Design evidence

The capability-first approach follows the motivation of the
[Behaviour Suite for Reinforcement Learning](https://arxiv.org/abs/1908.03568),
which uses small, scalable experiments to expose specific agent behaviors.
[MO-Gymnasium and MORL-Baselines](https://proceedings.neurips.cc/paper_files/paper/2023/file/4aa8891583f07ae200ba07843954caeb-Paper-Datasets_and_Benchmarks.pdf)
provide useful API and reproducibility precedents, while recent work on
[generalization in MORL](https://arxiv.org/abs/2503.00799) shows why a scalar
formulation and a single fixed environment are insufficient.

For the research system, [MLAgentBench](https://proceedings.mlr.press/v235/huang24y.html)
finds large variation and persistent long-term planning/hallucination failures.
[RE-Bench](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/)
reports that agents can run trials much faster than humans but often fail to
react to new evidence or build on progress over longer budgets. Recent work
formalizes research agents as search policies and shows that the
[operator set and search policy must be designed together](https://arxiv.org/abs/2507.02554).
These findings motivate the evidence graph, deterministic scheduler, and
explicit search policy in this rebuild.

