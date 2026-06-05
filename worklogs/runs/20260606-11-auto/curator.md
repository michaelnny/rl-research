---
verdict: failed-structural
nearest_prior_or_disqualifier: alive-candidate-FFTV (fftv-first-firing-time-variance-lattice.md)
side_information: [vector diagnostics, transition geometry]
---

## Verdict reasoning

- **Structural distinction:** ACS is structurally an F-dimensional extension of FFTV (single-band → F log-spaced bands per channel). The structural distinction from FFTV is real in principle, but the extension itself is the cause of failure: expanding the Pareto comparison from k dimensions (FFTV) to k·F dimensions (ACS, k=2 channels × F=4 bands = 8 dims) pushes the Pareto-non-dominated set toward |A| (failure mode d, predicted by the hypothesis). With 8-dimensional coordinate-wise Pareto over a typical discrete action set, virtually all actions are non-dominated — the nudge becomes symmetric (random). FFTV scored 1382 on DST vs strong 285; ACS scored 0.0 vs random 194.0 on the same env. This is not degradation — it is operator blindness consistent with Pareto-front saturation. The structural collapse confirms: simply extending FFTV's dimension count without a front-compression mechanism is a disqualifying rebadge of random policy.
- **Primitive count:** One primitive (S[s,a,m,f] spectral tensor) + one improvement operator (Pareto-non-dominance logit nudge). Passes the count gate, but the interaction between primitive dimensionality and the improvement operator creates the collapse.
- **Evidence quality:** score 0.0 / 0.0 / 0.0 / 0.001 vs random 0.137 / 0.0 / 194.0 / 1.331. Beat random on 0 of 4 envs; beat strong on 0 of 4. Notably DoorKey (KeyCorridor is ties at 0.0 vs 0.0 random), Deep Sea Treasure (0.0 vs 194.0), and RG (0.001 vs 1.331) all failed. DST's 0.0 vs 194.0 is the strongest counter-evidence: FFTV already showed DST is solvable with 1-band Pareto; the 4-band expansion destroyed the signal. Status is `completed` (no crash), ruling out an implementation abort.

## Lesson for the next iteration

The (k·F)-dimensional Pareto comparison in ACS causes front-saturation when F > 1 on these small action-set envs; future multi-band / multi-resolution spectrum candidates must include a front-compression mechanism (e.g., lexicographic ordering by frequency tier, or a strict-margin dominance test) before the Pareto operator is informative — or simply keep the comparison space at k dimensions by aggregating across bands before the dominance test.
