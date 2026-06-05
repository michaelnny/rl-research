---
verdict: failed-structural
nearest_prior_or_disqualifier: FED-family (bootstrap-wall; also #15 FED, #18 CEC, #19 CWTP, #21 TPP, #24 PCR, #26 TRAC)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** ARP's binary-firing-pattern set `S(s,a) ⊆ {0,1}^k` with strict-superset existence as the improvement operator is combinatorially novel in vocabulary, but the mechanism inherits the full FED-family bootstrap collapse: the strict-superset operator can only discriminate actions after at least one rewarded trajectory has deposited a richer pattern into some `S(s,a)` cell. Before that, all populated cells hold only the step-penalty singleton `{1,0,...,0}` and the operator is universally silent. The hypothesis itself labeled this "Lattice degeneracy" as its primary predicted failure mode — and the panel confirmed it: DST=99.0 vs random=194.0 (below random), RG=0.011 vs random=1.331 (near zero). The combinatorial set representation does not provide an earlier-firing signal than the magnitude-based predecessors; binary existence still requires the right trajectory to exist first.
- **Primitive vs stack:** One primitive (empirical pattern set) + one improvement operator (strict-superset existence nudge). Not a stack. The shape is clean.
- **Evidence quality:** Both vector envs scored below random. beat_random=0, beat_strong=0. The falsifier conditions from the hypothesis itself were met: `|S(s,a)|` likely stayed at 1 for most cells (lattice degenerate) throughout the budget. No evidence of operator firing in the productive regime.

## Lesson for the next iteration

ARP extends the FED/CEC/TPP/PCR/TRAC bootstrap-wall ruling to the "empirical set of binary suffix patterns" family: any improvement operator that gates logit nudges on downstream channel-firing sets remains silent until a rewarded trajectory is collected, regardless of whether the downstream information is stored as a magnitude vector, Pareto front, exit-hash, or binary-pattern set — the bootstrap wall is a property of when the first rewarding suffix appears, not of the representation chosen for downstream channel information.
