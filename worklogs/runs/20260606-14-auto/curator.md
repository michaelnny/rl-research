---
verdict: reviewer-rejected
nearest_dead_family: E
---

## Verdict reasoning

- The probe presented NORMAL as a two-cone Dykstra alternating-projection algorithm with a per-state active-set indicator chi(s) as the claimed load-bearing primitive; schema and prose-to-JSON alignment were clean.
- The Reviewer's coherence analysis showed that the C2 update step is structurally identical to Q-learning TD with a state-independent scalar baseline subtracted; subtracting V_bar from the target is a constant offset and does not change the argmax trajectory of A(s,.).
- The C1 shift (adding max(0, -max_a A(s,.)) row-wise) strictly preserves the argmax order of A(s,.); under a neural network on a continuous state space, argmax_a A_theta(s,.) is a singleton with probability one, so "uniform-on-argmax" collapses to deterministic greedy with no exploration mechanism.
- The ablation (remove C1 shift; replace uniform-on-argmax with epsilon-greedy argmax) was predicted by the Reviewer to match by construction, not just empirically, because both purported novelties are order-preserving cosmetics on Bellman-backup Q-learning; this borders Family E (advantage with relabeled vocabulary).
- No panel was run; result.json reflects status=reviewer-rejected with stage null, beat_strong=0, beat_random=0.

## Lesson for the next Researcher

Do not propose a primitive whose effect on the argmax trajectory is invariant under row-wise additive shifts of A(s,.); to break out of Family E, the projection or indicator must change the relative ordering of actions, not just the magnitude floor.
