---
id: 23
slug: tcp-temporal-channel-precedence
status: failed
sprint: 2026-06-06
verdict_in_one_line: "DAG-gated residual channel set collapses to singleton {step-penalty} on terminal-only-reward substrates, reducing Pareto vote to scalar channel maximization (disqualifier)."
side_information: [vector diagnostics, transition geometry]
nearest_prior: "15-fed / scalarized-vector-reward-maximization"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  0
  hard_n_beat_strong:  0
  commit: 6d63fc18c5efff7e4995604078eb25326728ff6a
---

# 23 — TCP (Temporal Channel Precedence)

## One-sentence idea

Build a cross-trajectory channel-pair lag-asymmetry tensor `Λ` to derive a precedence DAG P over vector channels; at each decision step restrict Pareto-vote logit nudges to the "residual" channels (not yet fired, with all DAG predecessors already fired in the current trajectory prefix).

## Core primitive

The lag-asymmetry tensor `Λ ∈ [−1,+1]^{k×k}` with `Λ[j,k] = E_τ[sign(t_first(j;τ) − t_first(k;τ))]`, accumulated as a running sample mean over full trajectories. Asymmetry above threshold τ_DAG induces a directed edge `k→j` in a precedence DAG P over the k vector channels. At each decision step the residual channel set `R(τ_{:t}) = { j : j not yet fired AND all predecessors of j in P have fired }` is computed from the live trajectory prefix.

## Improvement operator

Logit nudge `Δ_logit(c,a) = α · ( #{a': μ[c,a,R] ≻_Pareto μ[c,a',R]} − #{a': μ[c,a',R] ≻_Pareto μ[c,a,R]} )` where `μ[c,a,m]` is the empirical mean of channel-m's next-step firing indicator conditioned on observation cluster c and action a, and the Pareto comparison is restricted to R-coordinates. Both `Λ` and `μ` update from on-policy trajectory replay; no critic, no scalar collapse.

## Why it looked promising

- The lag-asymmetry tensor `Λ` fills as soon as any two channels fire at distinct times in any trajectory — much weaker threshold than obs-hash bucket accumulation (FED/CEC bootstrap wall).
- The residual channel set R is a trajectory-prefix-dependent object, not a state-revisit structure — structurally distinct from FED/CEC family.
- Reviewer confirmed structural novelty: not Q/V/advantage, not obs-hash buckets, not trajectory-pair matching, not reward machine (no product MDP, no value backup over automaton states).
- Monotonic improvement claim was stated under explicit assumptions (stationary DAG, non-degenerate R).

## What was tested

Stage: vector. Envs: deep-sea-treasure-concave-v0, resource-gathering-v0. Budget: 120s. No retries.
- deep-sea-treasure-concave-v0: score=99.0, random=194.0, strong=285.0 — **below random**
- resource-gathering-v0: score=0.121, random=1.331, strong=1.331 — **below random**
- beat_random=0, beat_strong=0

Commit: 6d63fc18c5efff7e4995604078eb25326728ff6a

## Why it failed

The hypothesis's own falsifier was confirmed. On Deep Sea Treasure:
- The step-penalty channel fires every step; the treasure channel fires only at the terminal step.
- This makes `Λ[treasure, step-penalty]` approach +1 trivially (treasure always fires after step-penalty).
- The DAG gets one dominant edge: step-penalty → treasure.
- At every non-terminal step, step-penalty has fired but treasure has not, so `R = {treasure}`.
- A Pareto comparison restricted to the single coordinate R={treasure} is identical to scalar maximization of E[v[treasure] | c, a] — the "scalarized vector-reward maximization" disqualifier.
- The resulting nudge steers toward actions that maximize next-step treasure-firing probability, which — since treasure only fires at the terminal step of the correct trajectory — actively steers the agent away from the long-horizon path needed to reach treasure. Hence the below-random score (99.0 vs 194.0 random).

On Resource Gathering:
- Gold/gem channels have near-symmetric precedence across episodes (both orderings occur), so `Λ[gold,gem] ≈ 0` and the DAG drops that edge.
- R collapses to the full channel set, reducing the operator to FED/CEC-style Pareto front over all channels indexed by observation cluster — the bootstrap-wall family.

Cross-attempt failure mode: "within-trajectory signal-geometry primitive collapses on substrates with terminal-only vector channels" (extends CHX/PICAV/CRP ruling to DAG-based temporal-ordering primitives).

## Lesson / constraint added

Any primitive that constructs a temporal-ordering or precedence graph over vector channels and uses the residual-eligible set to gate action selection will produce a singleton residual set on substrates with terminal-only reward channels, collapsing Pareto comparison to scalar channel maximization. Future candidates must either (a) not rely on residual-set non-degeneracy, or (b) explicitly handle the terminal-only channel case by excluding step-penalty channels from DAG construction.

## Nearest neighbors in the literature

- **Reward machines** (Icarte et al.): hand-specified automaton over symbolic events; TCP uses empirically-discovered asymmetry DAG, but the terminal-only channel collapse makes the DAG behave like a trivial one-edge automaton.
- **FED #15 / CEC #18**: Pareto-front over vector outcomes indexed by obs-hash; TCP with degenerate R={full set} reduces to this family.
- **GVFs / successor features**: multi-channel future prediction; TCP's `μ[c,a,m]` is a one-step variant; the channel-precedence gating is novel but fails when R is degenerate.
- **Intrinsic motivation / count-based**: the scalar-channel-maximization collapse makes TCP with singleton R behave like a count or frequency maximizer on the step-penalty channel.

## Artifacts

- Run: `worklogs/runs/20260606-08-auto/`
- train.py: `worklogs/runs/20260606-08-auto/train.py`
- panel output: `worklogs/runs/20260606-08-auto/panel.txt`
- result: `worklogs/runs/20260606-08-auto/result.json`
- commit: 6d63fc18c5efff7e4995604078eb25326728ff6a
