---
id: 11
slug: top
status: alive
sprint: 2026-05-26
verdict_in_one_line: "Solves DoorKey-5x5 (eval 0.98 but worse than Q); DoorKey-6x6 with more episodes; KeyCorridor unstable across seeds. Hand-event-lens dependent and close to GVFs / successor features."
side_information: [event traces, object state]
nearest_prior: "GVFs / successor features / multi-objective RL"
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 11 — TOP (Temporal Outcome Profiles)

## One-sentence idea

Stop avoiding value; replace scalar value with a structured future-
consequence object — first-hit times to consequential events plus vector
outcomes — and use dominance over those profiles for local improvement.

## Core primitive

For event/status atoms `i`, the **temporal outcome profile** at step `t`:

\[
\Theta_t = (T_{t,1},\ldots,T_{t,m},\, Y,\, C_t)
\]

with first-hit composition

\[
T_{t,i} = 0 \text{ if event }i\text{ true at }t,\quad T_{t,i} = 1 + T_{t+1,i}\ \text{otherwise}.
\]

Context-action profile sets are nondominated frontiers:

\[
\mathcal P_k(c,a) = \operatorname{ND}\{ \Theta_t : \rho(h_t)=c,\ a_t=a \}.
\]

## Improvement operator

KL projection over dominance between context-action temporal profiles.

## Why it looked promising

- Genuinely tries to replace value's *role* (per #10 reset), not its
  vocabulary.
- Has a real composition law (first-hit recursion) that is value-like:
  compresses future consequences and composes over time.
- Native vector consumption: `Y` and `C_t` are vector channels; the
  dominance order is partial.

## What was tested

- DoorKey-5x5: eval success 0.98 but **worse than Q-learning**.
- DoorKey-6x6: improved with more episodes.
- KeyCorridor: one strong seed, one unstable seed.

## Why it failed

Three problems converge:

1. Unstable across seeds.
2. Hand-event-lens-dependent — TOP works only after exposing status/event
   atoms (carrying-key, door-open, picked-up-ball, success, fluents). The
   event lens is *side information* that must be declared in the
   problem assumption, not pretended free.
3. Under inspection it sits inside the family of GVFs / successor
   features / multi-objective RL / reward machines. The structural
   distinction from those is not yet articulated.

It is **alive**, not failed — it is the only surviving research lead
out of attempts #01–#11 — but the structural distinction is the unfinished
work, and the panel evidence is null because it has not yet been
implemented inside the substrate's `train.py`.

## Lesson / constraint added

If a future-consequence object has a real composition law, it is worth
chasing. But event-lens side information must be declared explicitly
(per `prior_attempts.md` cross-attempt mode "hand-engineered event lenses
are side information"), and the structural distinction from GVFs /
successor features must be argued before any panel claim.

## Nearest neighbors in the literature

- General Value Functions (Sutton et al. 2011).
- Successor features (Barreto et al. 2017).
- Multi-objective RL frontier methods (Roijers et al., Yang et al.,
  Xu et al. PGMORL).
- Reward machines (Toro Icarte et al. 2022).

## Artifacts

_n/a — prototype only in pre-substrate notebooks._
