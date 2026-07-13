# Benchmark architecture

Status: 5,000--20,000-step neural architecture; long-tier qualification pending

The implemented v1 slice lives in `src/rlx_bench/`. It uses continuous
procedural observations and an evaluator-owned nonlinear task/dynamics kernel
shared across suite worlds. It includes native structured actions, procedural
causal truth, vector protocols, exact evaluator-only tiny-world solvers,
intervention audit, evaluator budgets, protocol-specific metrics, and committed
public/held-out suite boundaries. Preference-conditioned factored-discrete
evaluation is executable through the batched neural candidate protocol v2.

The former 64-step experiment is rejected as research evidence and exists only
in Git history.
The frozen `factorlab-long-5k-v1` study uses a 5,000-step terminal-return anchor
and a mandatory 20,000-step scaling contrast. Until its reviewed report passes
all gates, no tier is admitted. Memory-lag variants, world shifts, other
objective/action protocols, and applied families remain separate scope.

## Suite shape

The benchmark has two layers. The first layer contains controlled diagnostic
kernels with known causal structure. The second contains procedural applied
simulators that compose several kernels. An algorithmic claim must first appear
as a predicted fingerprint in the diagnostics and then survive at least two
applied families.

This avoids both common failures:

- an applied simulator is too entangled to identify why a method works; and
- a synthetic diagnostic is too narrow to show that the mechanism composes.

## Layer 1: FactorLab

`FactorLab` is a generator, not one environment. A task configuration selects
one value on each difficulty axis while retaining a small state representation
and a known planning solution on tiny instances.

### Dynamics

An episode contains continuous signals, a procedural context, observed dynamic
state, and controllable delay spans. A suite-shared hidden neural kernel maps
signals, context, and state to conflicting objective targets. Actions affect
both delayed reward contributions and later state. The generator also supports
threshold effects, pairwise interactions, and ordered prerequisites.

Signals are not drawn from a reusable finite dictionary. Public, tune,
held-out, and audit bands have distinct trajectories and contexts while sharing
only the hidden task kernel. Consequently, successful held-out control requires
function approximation and representation transfer; a cue-keyed table cannot
cover the state space.

This gives ground truth for:

- which action-times can affect each terminal component;
- the true causal delay distribution;
- the minimal memory state;
- whether action factors interact; and
- the Pareto set on small configurations.

The same underlying dynamics can be rendered with flat discrete, embedded
catalog, factored discrete, continuous, or hybrid actions. That permits matched
counterfactual experiments instead of comparing unrelated domain stories.

### Required diagnostic sweeps

- research horizon: 5,000, 10,000, and 20,000 steps;
- maximum causal lag: 5,000 through the full 20,000-step horizon;
- joint discrete choices: `10^2` through at least `10^12` via factors;
- continuous dimensions: 4, 16, 64;
- reward events: terminal only, 2, 4, and 8 sparse events;
- objectives: 2, 4, and 8 with controlled conflict strength;
- memory lag: 0, 128, 512, and 2,048; and
- world shift: parameter interpolation and unseen composition.

Neural width/depth, recurrent state, parameter count, and accelerator use are
reported as experimental factors, not hidden implementation details.

Not every Cartesian combination becomes a benchmark task. A fractional design
selects configurations that identify main effects and important interactions
without turning the suite into a compute sink.

## Layer 2: applied families

### SlateMarket

A low-cost simulator of repeated recommendation/allocation decisions. Each step
selects a slate from a catalog containing hundreds or thousands of items. Item
features are observed; item identities and world parameters vary procedurally.
The selected slate changes a slowly evolving population state, so short-term
click-like gains can reduce terminal retention, diversity, trust, or safety.

Action variants include top-k selection, ordered slates, budgeted slates, and
conditional continuous parameters. Reward protocols cover known preference,
unknown-preference coverage, and constraints. Small worlds admit dynamic
programming or exhaustive bounds; larger worlds use a privileged model-predictive
controller only as a feasibility bound.

### GraphOps

A network operations simulator covering routing, maintenance, activation, and
resource allocation. Components degrade, demand changes, and interventions have
delayed effects. Each action contains discrete selections plus allocation
parameters, creating a conditional hybrid action space. Terminal outcomes cover
service, cost, energy, fairness, and resilience to a late shock.

Worlds are generated from small graphs with controlled topology and causal
depth. Tiny graphs admit exact planning. Larger graphs remain CPU-cheap and test
whether an algorithm exploits locality and factorization rather than memorizing
node identities.

### AssemblyLab

A procedural workflow with prerequisites, shared resources, rework, and late
quality tests. Actions assign multiple resources and optional process parameters
at once. Early shortcuts change the probability and cost of rework much later.
Terminal outcomes cover completion, quality, lateness, resource cost, and risk.

AssemblyLab differs from GraphOps in its irreversible prerequisite structure
and from SlateMarket in its finite consumable jobs. Together the three families
prevent one domain-specific representation from defining success.

## Environment interface

The interface must expose ordinary learner information and privileged benchmark
metadata through separate capabilities.

Learner API:

- `reset(world_ref, preference=None) -> observation, public_info`
- `step(action) -> observation, reward_vector, terminated, truncated, public_info`
- declared observation and structured action specifications;
- no access to causal graphs, hidden dynamics, oracle state, or held-out world
  generators.

Evaluator API:

- immutable task and protocol IDs;
- world-suite version and cryptographic manifest;
- objective names, orientation, constraint semantics, and fixed normalization;
- episode accounting and deterministic replay tokens; and
- privileged diagnostics available only to environment validation jobs.

Gymnasium adapters are useful, but the native action specification must retain
catalog features, masks, factor dependencies, and hybrid parameter schemas.
Flattening everything into a `Box` or enumerated `Discrete` space would erase
the property being studied.

## Evaluation units

An evaluation unit is `(algorithm revision, configuration, task protocol,
training seed, world suite, budget)`. The budget simultaneously caps:

- environment transitions;
- complete episodes;
- wall time;
- accelerator time and peak device memory;
- trainable learner parameters; and
- number of returned policies or preference queries.

All resource usage is measured by the evaluator. Self-reported costs are not
accepted.

## Vector metrics

Component values are normalized using fixed task semantics, never the observed
min/max of candidate runs. Reports include:

- per-component distributions across worlds and training seeds;
- feasibility and violation magnitude for constrained protocols;
- expected utility over a versioned preference panel;
- worst-panel and tail preference regret against a reference coverage set;
- hypervolume or an equivalent set-coverage metric when mathematically
  appropriate; and
- policy-set size and total interaction budget.

No single number is used across all objective protocols. Aggregate plots show a
capability fingerprint, not a winner rank.

## Qualification experiments

A benchmark tier is not qualified by unit tests. It is qualified by a study.

1. **Mechanics:** replay, API invariants, numerical bounds, and throughput.
2. **Causal audit:** interventions recover the generator's declared influence
   graph and effective horizon.
3. **Feasibility:** exact or privileged planners establish a ceiling.
4. **Learnability:** at least one compact neural learner improves reproducibly
   over random under the published transition and consumer-GPU budget.
5. **Headroom:** the best reference learner remains materially below the ceiling
   on at least one intended capability.
6. **Factor sensitivity:** changing the target factor changes baseline rankings
   or slopes in the predicted direction.
7. **Specificity:** changing a non-target factor does not dominate the result.
8. **Generalization:** public-world tuning is separated from held-out worlds and
   unseen compositions.
9. **Statistics:** results span worlds and independent training seeds with
   uncertainty intervals and failure counts.
10. **Independent audit:** a second implementation path checks oracle values,
    metrics, and at least one baseline fingerprint.
11. **Neural admissibility:** continuous held-out observations are disjoint,
    the task kernel is nonlinear, a state-key lookup is outside the finite
    budget by construction, and removing neural representation capacity
    materially degrades the intended fingerprint.

Until these experiments pass, a tier is **not admitted** and campaign preflight
rejects it. There is no ambiguous half-valid benchmark state.

The ten evidence records remain the qualification state-machine keys;
`neural_admissibility` is mandatory evidence attached to mechanics,
learnability, and generalization rather than an eleventh software status.

## Neural reference algorithm matrix

The suite needs contrasting mechanisms, not a long list of fashionable names:

- compact MLP and recurrent Monte Carlo/TD actor-critics;
- PPO and SAC with compact neural encoders where their action assumptions apply;
- neural DQN plus a branching value architecture;
- CEM or another trajectory-level black-box optimizer;
- a return-decomposition/reward-redistribution method such as a minimal RUDDER
  reference;
- a recurrent and a memoryless version of the same learner;
- scalarized, preference-conditioned, and policy-coverage MORL references; and
- privileged planners reported strictly as evaluator-only ceilings.

Tabular and cue-keyed policies are not reference algorithms in this project.
Architecture ablations with insufficient representation capacity may be used
as negative controls, but they are labeled ablations rather than candidate
methods.

Implementations may use established libraries when this reduces baseline bugs.
Research candidates remain local and inspectable. Every reference has one
frozen config-selection procedure; per-environment hand tuning is recorded as a
separate, non-comparable result.
