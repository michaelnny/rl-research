---
verdict: failed-structural
nearest_prior_or_disqualifier: attempt-22 / scalarized-vector-reward-maximization (disqualifier family)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction (partial).** The lag-asymmetry tensor `Λ` and its derived precedence DAG P are genuinely novel objects — not Q/V/advantage, not obs-hash buckets, not trajectory-pair matching. The reviewer correctly affirmed structural novelty at the hypothesis stage. However, the hypothesis's own falsifier (item in §8 of the hypothesis) was confirmed in practice: on Deep Sea Treasure the DAG produces exactly one non-trivial edge (step-penalty precedes treasure), so the residual channel set R collapses to a singleton {step-penalty} at every non-terminal step. A Pareto comparison restricted to a single channel R={j} is identical to scalar maximization of E[v[j] | c, a] — a direct instance of the "scalarized vector-reward maximization" disqualifier. On Resource Gathering, the gold/gem channel symmetry averages `Λ[gold,gem]` toward zero, so the DAG is near-empty and R = full channel set, reducing the operator to FED-style obs-cluster-indexed Pareto front — the FED/CEC family (#15, #18).

- **Primitive vs stack.** The primitive (Λ, P, R) is one coherent object. The improvement operator (Pareto-vote over μ[c,a,R]) is clean. No problematic stacking — this is a genuine one-primitive-plus-one-operator candidate. The stack criterion is not the failure mode here.

- **Evidence quality.** beat_random=0, beat_strong=0 on both vector envs. Deep Sea Treasure: 99.0 vs random 194.0 (below random). Resource Gathering: 0.121 vs random 1.331 (below random). The below-random result on DST indicates the operator is actively counterproductive: once R collapses to {step-penalty}, the Pareto-vote steers toward maximizing step-penalty accumulation (i.e., staying alive longer while avoiding the terminal-reward-bearing state), which directly opposes reaching the treasure. This confirms the structural collapse described above.

- **Failure-mode informativeness.** The failure rules out the following family: any primitive that constructs a temporal-ordering / precedence graph over vector channels and uses it to gate which channels drive action selection will collapse on substrates where (a) one or more channels are terminal-only, forcing non-trivial DAG edges to be trivially asymmetric, and (b) the resulting residual set is a singleton or the empty set for most of the trajectory. This extends the CHX/PICAV/CRP ruling to DAG-based temporal-ordering primitives.

## Lesson for the next iteration

Temporal precedence primitives that gate on channel-firing-order (whether as a DAG, automaton, or partial order) collapse on substrates with terminal-only reward channels because the non-trivial DAG edges are trivially asymmetric and the residual channel set degenerates to a single channel, reducing Pareto comparison to scalar maximization of that channel.
