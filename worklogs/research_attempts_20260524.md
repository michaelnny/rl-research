# Failed Attempts Toward a Novel Third Family of RL

**Purpose.** This report records the attempted algorithmic directions explored so far, the core math behind each one, why each initially looked promising, what was tested, why it failed, and what constraint it adds for the next research phase.

**Important status.** None of these attempts should be treated as the final novel RL family. The useful output is the set of falsifications and design constraints.

---

## Executive Summary

The target was a new RL algorithmic family for real-world long-horizon sparse-reward problems. The desired family should preserve the RL spirit of trial-and-error reward maximization, but should not have Bellman backup, TD error, value functions, policy-gradient optimization, scalarized reward-vector optimization, imitation of better trajectories, CEM/ES elite fitting, or verifier/planner stacks as its core identity.

Four serious directions were explored:

| Attempt | Core primitive | Improvement operator | Initial appeal | Final status |
|---|---|---|---|---|
| **FROST** | Vector repair certificate | Vector-cone trajectory repair projection | Native vector feedback and safety | Rejected: too repair/verifier-centered; not reward-native; weak for terminal-only reward. |
| **BRIC** | Reward-intervention bracket | Constraint insertion from terminal intervention tests | Terminal-reward causal credit assignment | Rejected: too expensive; requires extra counterfactual trials; awkward for robotics/LLMs. |
| **KERNEL-RL / RSK** | Reward-support behavior kernel | Condition policy on statistically stable reward-support atoms | Passive, cheap, sparse-reward native | Rejected as final: favorable custom benchmark; weak on deep exploration. |
| **Frontier/CARL direction** | Reproducible controllability cell | Expand controllable frontier and compose routes | Strong for hard exploration | Not accepted: too close to Go-Explore/model-based graph exploration unless substantially reinvented. |

The largest lesson is that a serious next attempt must be verified on established hard-exploration and sparse-reward benchmarks before any claims of novelty or dominance. Custom favorable tasks are useful only for debugging, not validation.

---

## Non-Negotiable Constraints for the Next Research Phase

A future candidate must satisfy all of the following:

1. **Reward-native.** It must learn from ordinary reward experience. Terminal-only reward must be a special case, not the main assumption.
2. **Long-horizon sparse-reward capable.** It must handle deep exploration where useful reward correlation may not exist until late.
3. **No hidden value machinery.** No `Q`, `V`, advantage, return-to-go, Bellman backup, TD error, or reward model as the central primitive.
4. **No hidden policy-gradient machinery.** No scalar-weighted log-prob update, PPO/TRPO/GRPO-style objective, or advantage estimator as the core update.
5. **No elite cloning.** It must not be CEM/ES or “clone high-return trajectories” under another name.
6. **Efficient verification.** It must not require expensive counterfactual environment rollouts for every proposed edit.
7. **Benchmark-first.** It must be tested on existing hard tasks: DeepSea, MiniGrid sparse tasks, Montezuma-like exploration, sparse robotics manipulation, and terminal-verifier LLM/web-agent tasks.
8. **Simple mathematical identity.** The primitive and improvement operator should fit on one page.

---

# Attempt 1: FROST — Feasible Residual Operators for Sequential Trajectories

## One-Sentence Idea

Instead of learning values or policy gradients, use structured vector feedback to compute a minimal behavior repair that reduces active defects while preserving protected constraints.

## Core Primitive

A rollout is

\[
\tau=(o_0,a_0,o_1,a_1,\ldots,o_H).
\]

The environment or verifier returns structured vector feedback:

\[
F(\tau)=
[\text{success},\text{safety}_{0:H},\text{energy},\text{latency},\text{smoothness},\text{validity},\ldots].
\]

Define a defect vector against budgets or constraints:

\[
d(\tau)=[F(\tau)-b]_+.
\]

FROST's proposed primitive was a **vector repair certificate**:

\[
C(\tau)=\{d(\tau),G(\tau),\mathcal{E},\mathcal{K}\},
\]

where

\[
G_{i,t}(\tau) \approx \frac{\partial F_i(\tau)}{\partial e_t}
\]

or a counterfactual finite-difference influence from editing behavior element \(e_t\).

## Improvement Operator

Given active violated channels \(A\), protected channels \(P\), and editable behavior elements \(e_t\), solve:

\[
\Delta e^*=
\arg\min_{\Delta e}\|L\Delta e\|^2
\]

subject to

\[
G_A\Delta e \le -\rho d_A,
\]

\[
G_P\Delta e \le 0,
\]

\[
\Delta e\in\mathcal{E},\qquad \|\Delta e_t\|\le \epsilon_t.
\]

The policy is then distilled toward the repaired behavior:

\[
\theta \leftarrow \arg\min_\theta
\sum_t D\big(\pi_\theta(h_t), e_t+\Delta e_t^*\big).
\]

The intended distinction was that the policy is not updated by reward-weighted log-probability or by value backup; it is updated toward a vector-feasible repair of its own failed behavior.

## Prototype and Result

Toy setting: 2-D point robot with a goal, circular obstacle, energy budget, and smoothness budget. Horizon \(H=96\). Feedback was vector-valued and event-local for obstacle safety.

Observed summary:

| Method | Success rate | Median iterations | Mean final distance | Mean safety defect | Mean energy defect |
|---|---:|---:|---:|---:|---:|
| FROST-event policy | 1.00 | 86 | 0.136 | 0.000 | 0.000 |
| FROST-event plan-only | 0.67 | 12 | 0.148 | 0.00027 | 0.000 |
| Scalar repair, safety-heavy | 0.00 | 70 | 0.304 | 0.000 | 0.385 |
| Scalar repair, goal-heavy | 0.00 | 45 | 0.147 | 0.000 | 0.951 |
| REINFORCE scalar | 0.00 | 50 | 1.274 | 0.118 | 0.000 |

This showed that event-local vector repair can beat scalarized repair on a specially constructed safety task.

## Why It Failed as a Candidate Family

FROST is not reward-native enough. It assumes structured feedback channels and local repair influence estimates. It is closer to constrained repair/control than general sparse-reward RL.

The core dependency

\[
G_{i,t}\approx \partial F_i/\partial e_t
\]

is also heavy for real robotics, LLM agents, and terminal-only tasks. When only terminal reward exists, FROST has no natural primitive unless a verifier or dense defect model is introduced, which violates the desired RL philosophy.

## Lesson Learned

Native vector feedback is valuable, but **repair cannot be the core identity**. The next candidate must start from trial-and-error reward maximization, not from externally supplied constraint defects.

---

# Attempt 2: BRIC — Bracketed Reward-Intervention Control

## One-Sentence Idea

Use terminal reward only, but assign credit by testing whether replacing one segment of an anchor trajectory with a donor segment improves the terminal outcome.

## Core Primitive

Given an anchor trajectory \(\tau^a\), a donor trajectory \(\tau^b\), and editable interval \(I\), define the patched trial:

\[
\tilde{\tau}=\tau^a[I\leftarrow \tau^b_I].
\]

The primitive was the **reward-intervention bracket**:

\[
B(\tau^a,\tau^b,I)
=\operatorname{sign}\left(R(\tau^a[I\leftarrow\tau^b_I])-R(\tau^a)\right).
\]

If

\[
R(\tau^a[I\leftarrow\tau^b_I])>R(\tau^a),
\]

then the donor segment is accepted in that anchor context:

\[
\tau^b_I \succ_{\tau^a,I} \tau^a_I.
\]

## Improvement Operator

An accepted bracket creates a behavior clause:

\[
(c_I,\tau^a_I)\rightarrow \tau^b_I,
\]

where \(c_I\) is the local context around the segment.

The policy update is a minimal projection onto verified clauses:

\[
\pi_{k+1}=\arg\min_{\pi\in\Pi}D(\pi,\pi_k)
\]

subject to

\[
\Pr_\pi(\tau^b_I\mid c_I)\ge 1-\epsilon.
\]

No scalar-weighted log-probability update is used. Reward only determines whether a concrete intervention bracket is accepted.

## Prototype and Result

Toy setting: terminal-only binary sequence task, where hidden chunks of length 8 produce terminal reward only when the whole chunk is correct.

Summary:

| Task | Method | Seeds | Success rate | Median solve evals | Mean best reward |
|---|---:|---:|---:|---:|---:|
| H=128 | BRIC-seg | 10 | 1.00 | 7,228.5 | 1.000 |
| H=128 | CEM | 10 | 0.00 | — | 0.700 |
| H=128 | REINFORCE | 10 | 0.00 | — | 0.169 |
| H=256 | BRIC-seg | 10 | 1.00 | 15,856.5 | 1.000 |
| H=256 | CEM | 10 | 0.00 | — | 0.644 |
| H=256 | REINFORCE | 10 | 0.00 | — | 0.122 |

A bad-grammar ablation failed: when the edit grammar used length-6 segments while the true chunks were length 8, BRIC solved 0/5 runs and reached mean best reward around 0.283.

## Why It Failed as a Candidate Family

The intervention primitive is elegant in a causal sense, but it is not efficient enough for real domains. It requires extra environment or simulator trials of the form

\[
R(\tau^a[I\leftarrow\tau^b_I]),
\]

which is expensive for robotics and LLM/web agents. In real robotics, every bracket test may require a physical rollout or high-fidelity simulation. In LLM agents, each bracket may require a full tool-using episode or external verifier call.

It also depends heavily on a useful edit grammar. Without the right segment boundaries, the method fails.

## Lesson Learned

The next candidate must be **passive** or at least near-passive: it should extract credit from ordinary trajectories, not require many counterfactual environment verifications.

---

# Attempt 3: KERNEL-RL / RSK — Reward-Support Kernel Conditioning

## One-Sentence Idea

Passively mine compact behavior atoms that statistically separate reward-bearing experience from non-reward-bearing experience, then condition the policy only on the stable atoms while leaving everything else exploratory.

## Core Primitive

Let \(z\in\mathcal{Z}\) be a behavior atom: context-action pair, chunk, skill, tool call, code edit, plan step, or latent primitive.

For a reward event \(e\), define the reward-support contrast:

\[
\Delta_e(z)
=
\mathbb{E}[\psi_e(\tau)\mid z\in\tau]
-
\mathbb{E}[\psi_e(\tau)\mid z\notin\tau].
\]

For terminal-only reward:

\[
\psi(\tau)=R(\tau)-\bar{R}_{\text{batch}}.
\]

The **reward-support kernel** is

\[
K_e=
\{z:\operatorname{LCB}(\Delta_e(z))>\lambda
\;\land\;
 z\text{ confirms across independent batches}\}.
\]

## Improvement Operator

Condition the policy on the kernel:

\[
\pi_{k+1}=\operatorname{Cond}(\pi_k,K).
\]

More explicitly:

\[
\pi_{k+1}
=
\arg\min_{\pi}D_{\mathrm{KL}}(\pi\,\|\,\pi_k)
\]

subject to

\[
\Pr_\pi(z\mid c_z)
\ge 1-\epsilon
\qquad \forall z\in K,
\]

and leave the policy unchanged where no kernel atom applies.

This was intended to avoid both policy gradient and elite cloning: the algorithm does not update every action in high-return trajectories, only statistically confirmed atoms.

## Prototype and Result

Toy setting: terminal-only sparse sequence task with hidden length-8 chunks. The terminal score is the number of exactly correct hidden chunks; no chunk identity or process reward is given.

Summary:

| Task | Method | Seeds | Success rate | Median evals solved | Mean final score |
|---|---:|---:|---:|---:|---:|
| H=512 | KERNEL-RL | 3 | 1.00 | 12,288 | 64 / 64 |
| H=512 | REINFORCE | 3 | 1.00 | 114,688 | 64 / 64 |
| H=512 | CEM | 3 | 0.00 | — | 53.3 / 64 |
| H=1024 | KERNEL-RL | 3 | 1.00 | 12,288 | 128 / 128 |
| H=1024 | REINFORCE | 3 | 0.00 | — | 118 / 128 |
| H=1024 | CEM | 3 | 0.00 | — | 79.7 / 128 |

The initial accumulator version failed by promoting false atoms too early. The revised version required independent confirmation before promotion.

## Why It Failed as a Candidate Family

The validation was unfair. The task was invented and decomposable in exactly the way the algorithm needed. It showed that reward-support mining can work when reward-relevant atoms are statistically visible, but it did not prove capability on established hard exploration tasks.

A DeepSea-style probe exposed the weakness: when reward support does not exist until a deep unrewarded path is intentionally traversed, RSK/KERNEL-style mining has nothing to mine.

Observed DeepSea probe:

| N | Random solved | Q-learning solved | RSK-context solved | Frontier-Graph solved |
|---:|---:|---:|---:|---:|
| 12 | 0/5 | 0/5 | 0/5 | 5/5 |
| 20 | 0/5 | 0/5 | 0/5 | 5/5 |
| 30 | 0/5 | 0/5 | 0/5 | 5/5 |

This means passive reward-support mining is insufficient for deep exploration.

## Lesson Learned

Association mining is not enough. Long-horizon sparse reward needs a mechanism for **intentionally reaching new temporally deep experience before reward correlations exist**.

---

# Attempt 4: Frontier-Graph / CARL Direction — Controllability-First RL

## One-Sentence Idea

Before optimizing reward, learn what parts of the environment are reproducibly reachable; expand the controllable frontier, compress reliable routes into skills, and attach reward events only after they are encountered.

## Core Primitive

The proposed primitive is a controllability cell:

\[
\mathcal{C}=\{(z,\pi_z,\rho_z)\},
\]

where:

- \(z\) is an abstract reachable situation;
- \(\pi_z\) is a policy fragment that reliably reaches \(z\);
- \(\rho_z\) is a reproducibility certificate.

A learned abstraction would map history to controllable cells:

\[
z=f_\phi(h).
\]

A cell is accepted only if it is reproducibly reachable:

\[
P_{\pi_z}(f_\phi(h_t)=z)\ge 1-\epsilon.
\]

## Improvement Operator

The high-level operator is frontier expansion:

\[
\operatorname{Expand}(\mathcal{C})
=
\arg\max_{z,a}
\text{frontier-novelty}(T(z,a))
\]

subject to reproducible reachability of the source cell:

\[
z\in\mathcal{C}.
\]

When reward is observed, it marks reachable cells rather than becoming a value backup:

\[
z\mapsto \text{reward event record}.
\]

Routes are composed from learned fragments:

\[
\pi_{z_0\rightarrow z_k}
=
\pi_{z_{k-1}\rightarrow z_k}
\circ\cdots\circ
\pi_{z_0\rightarrow z_1}.
\]

The appeal is a possible complexity shift: under a correct compact abstraction, graph expansion can discover reachable structure in roughly

\[
O(|\mathcal{Z}||\mathcal{A}|)
\]

interaction tests, instead of random discovery of a length-\(H\) rewarding action string, which can require

\[
\Omega(|\mathcal{A}|^H)
\]

episodes in the unstructured worst case.

## Probe Result

A simple frontier graph solved the DeepSea probe where random, Q-learning, and RSK-context failed:

| DeepSea size | Frontier-Graph solved | Median solved episode |
|---:|---:|---:|
| N=12 | 5/5 | 25 |
| N=20 | 5/5 | 41 |
| N=30 | 5/5 | 59 |

## Why It Was Not Accepted as the Final Candidate

The raw frontier-graph idea is too close to existing exploration and model-based graph-search families, especially Go-Explore-like logic: remember reachable states, return to promising states, then explore outward. Without a genuinely new abstraction-learning and improvement principle, calling this a new family would be dishonest.

The open possibility is not the tabular frontier graph. The potential new family would need to be something stronger:

\[
\textbf{CARL: Controllability-Abstraction Reinforcement Learning}
\]

where the central invention is a learned abstraction optimized for reproducible reachability, not reward prediction or value estimation.

## Lesson Learned

The next promising direction should probably be **controllability-first**, but it must avoid merely reinventing Go-Explore, options, hierarchical RL, model-based search, or count-based exploration.

---

# Cross-Attempt Failure Analysis

## 1. The primitive must create information before reward correlation exists

FROST needs dense diagnostics. BRIC needs expensive interventions. KERNEL-RL needs statistical reward support. All three fail or become inefficient when the agent has not yet reached informative reward-bearing states.

Deep exploration requires a primitive that can operate before reward appears. Controllability/reachability is one such candidate, but it risks overlapping with existing methods.

## 2. Terminal-only reward cannot be the whole design target

Terminal-only reward is a hard special case, but the final algorithm must handle:

\[
r_t\neq 0\quad\text{occasionally},
\]

\[
r_H\text{ only},
\]

\[
\text{vector rewards or constraints},
\]

\[
\text{delayed sparse subgoal rewards},
\]

and noisy/partial success signals.

A terminal-only benchmark alone is too narrow.

## 3. Custom toy tasks are insufficient

The sequence/chunk tasks were useful for debugging algorithmic mechanics, but they were not fair tests. The next research cycle must start with existing benchmarks before custom tasks are used.

Minimum benchmark set:

- DeepSea and stochastic DeepSea variants;
- MiniGrid DoorKey, FourRooms, MultiRoom, KeyCorridor;
- Montezuma-style hard exploration;
- sparse MuJoCo/robotics manipulation;
- Safety Gym or constraint-heavy robotics;
- LLM/web-agent terminal-verifier tasks.

## 4. Simplicity matters

The better candidate should have the following structure:

\[
\boxed{
\text{primitive} \rightarrow \text{operator} \rightarrow \text{policy update}
}
\]

with no hidden stack of special cases. FROST and BRIC were too operationally complex. KERNEL-RL was simpler but not powerful enough.

## 5. Novelty must be defended against nearby families

Any next candidate must be explicitly distinguished from:

- value RL;
- policy-gradient RL;
- actor-critic;
- CEM/ES/CMA-ES;
- Go-Explore;
- count-based exploration;
- RND/curiosity;
- options and hierarchical RL;
- model-based planning;
- GFlowNets;
- reward machines;
- hindsight relabeling;
- decision transformers / conditional imitation;
- reward-model RLHF and DPO-like preference optimization.

---

# Recommended Next Research Direction

The most promising high-level direction left by these failures is:

\[
\boxed{
\textbf{Controllability-first RL with learned reproducible abstractions}
}
\]

But it should not begin as another named algorithm. It should begin as a falsification-driven research question:

> Can we learn an abstraction whose cells are defined by reproducible reachability, and use frontier expansion plus route composition to solve sparse long-horizon reward tasks without value backup, policy gradient, or elite trajectory cloning?

A serious next phase should implement only the minimal tabular/procedural version first, compare against Go-Explore and count-based exploration, then add neural abstraction learning only after the primitive proves nontrivial on existing benchmarks.

## Proposed Next Experiment Ladder

1. **DeepSea deterministic.** Verify polynomial frontier behavior and expose overlap with Go-Explore.
2. **DeepSea stochastic.** Test robustness of reproducibility certificates.
3. **MiniGrid DoorKey / MultiRoom.** Test abstraction beyond tabular states.
4. **MiniGrid with pixel observations.** Force representation learning.
5. **Sparse manipulation simulator.** Test continuous control and contact.
6. **LLM/web-agent terminal tasks.** Treat tool calls, browser states, and code patches as controllable cells.

Only after passing these should a new algorithmic family name be introduced.

---

# Artifact Inventory from This Research Sprint

The following files were produced during the exploration. They are useful as debugging records, not as final validation:

- `frost_event_prototype.py` — FROST event-local repair prototype.
- `frost_event_summary.csv` — FROST prototype summary.
- `frost_event_trajectory.png` — Representative repaired trajectory.
- `bric_research_prototype.py` — BRIC intervention-bracket prototype and baselines.
- `bric_research_results.json` — BRIC experiment records.
- `bric_h128_learning_curve.png` — BRIC learning curve.
- `kernel_rl_prototype.py` — KERNEL-RL / RSK prototype.
- `kernel_rl_summary.csv` — KERNEL-RL summary.
- `kernel_rl_learning_curve_H512.png`, `kernel_rl_learning_curve_H1024.png` — KERNEL-RL curves.
- `deepsea_probe.py` — DeepSea sanity probe.
- `deepsea_probe_summary.csv` — DeepSea results summary.
- `deepsea_probe_plot.png` — DeepSea probe plot.

---

# Final Takeaway

No explored attempt is good enough to be the target algorithmic family.

The useful lesson is sharper:

\[
\text{A next-generation RL family must create controllable exploration structure before rewards are informative, while still using reward experience for final behavioral selection.}
\]

The next research cycle should therefore start from controllability, reachability, and abstraction learning — but must prove novelty and superiority against Go-Explore, hierarchical RL, model-based planning, and curiosity methods on established benchmarks before being treated as a serious candidate.
