---
id: 25
slug: acs-action-conditional-suffix-spectrum
status: failed
sprint: 2026-06-06
verdict_in_one_line: "Multi-band Pareto (k·F=8 dims) saturates the non-dominated set to all actions, collapsing the nudge to random — FFTV's 1-band form was structural necessity."
side_information: [vector diagnostics, transition geometry]
nearest_prior: alive-candidate-FFTV (fftv-first-firing-time-variance-lattice.md)
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random: 0
  hard_n_beat_strong: 0
  commit: 3be9bfc97255df8ea0b94047b6365774a9187900
---

# 25 — ACS (Action-Conditional Suffix-Spectrum)

## One-sentence idea

Generalise FFTV's single timing-dispersion statistic to a multi-band spectral profile per (state-cluster, action, channel), then apply Pareto-non-dominance logit nudge over the flattened (k·F)-dimensional spectral matrix.

## Core primitive

`S[s,a,m,f]` — the empirical variance of the within-band firing-indicator `1_m(t+τ)` for lag band `f = [2^f, 2^{f+1})`, averaged over visits to `(s,a)`. For F=4 log-spaced bands and k=2 channels (DST) this is an 8-dimensional spectral matrix per (state-cluster, action). Channel-MAD standardisation is applied per (channel, band) before the comparison.

## Improvement operator

At each decision step at state-cluster `s`, flatten `S[s,a,:,:]` to an (k·F)-vector per action. Compute the Pareto-non-dominated set `P(s)` via coordinate-wise partial order. Logit nudge: `+α` to all `a ∈ P(s)`, `−α/(|A|−|P(s)|)` to others. `α` decays over training.

## Why it looked promising

- FFTV (alive-promising) showed DST score 1382 vs strong 285 using a k=2 dimensional Pareto on per-channel IQR. ACS proposed to make the Pareto comparison richer by stratifying across F=4 temporal scales.
- The hypothesis correctly identified FFTV as the F=1 special case, making the structural distinction between ACS and FFTV mathematically precise.
- Dense channels (step-penalty) fire from episode 1, so the spectrum accumulates without a bootstrap barrier.
- The multi-band stratification encodes genuine new information (temporal scale of firing pattern) that pooled IQR discards.

## What was tested

Stage: core. Envs: MiniGrid-DoorKey-8x8-v0, MiniGrid-KeyCorridorS3R3-v0, deep-sea-treasure-concave-v0, resource-gathering-v0. Budget: 120 s/env. No retries. Scores: 0.0 / 0.0 / 0.0 / 0.001. Beat random: 0/4. Beat strong: 0/4. Commit: 3be9bfc97255df8ea0b94047b6365774a9187900.

## Why it failed

Pareto-front saturation (the hypothesis's own predicted failure mode d). With k=2 channels and F=4 bands, the flattened spectral vector per action is 8-dimensional. In an 8-dimensional coordinate-wise partial order over a discrete action set of typical size 5–7, the Pareto-non-dominated set quickly expands to include all or nearly all actions, rendering the logit nudge symmetric (effectively random). Evidence: DST scored 0.0 vs random 194.0; FFTV scored 1382 on the same env. This is complete operator blindness, not degradation. FFTV's k=2 dimensional Pareto (IQR per channel) is discriminating precisely because 2-dimensional dominance excludes most of the action set; 8-dimensional dominance does not. The run status is `completed` (no crash), confirming the failure is not an implementation abort but a structural collapse of the improvement operator.

## Lesson / constraint added

Multi-band spectrum extensions of FFTV must include a front-compression mechanism before the Pareto test — options include: (1) lexicographic ordering by frequency tier (high-frequency bands as tiebreakers, not additional dimensions); (2) strict-margin dominance (require all band differences to exceed a threshold); (3) band aggregation into a single scalar-per-channel before the k-dimensional Pareto (which would recover FFTV). Adding dimensions to a Pareto comparison in a small action set is not free.

## Nearest neighbors in the literature

- **FFTV** (alive-promising, 20260606-09-auto): ACS is the F=4 generalization; FFTV is F=1. FFTV works; ACS does not because of saturation.
- **Multi-objective Pareto RL (Pareto-Q, MORL)**: same Pareto-non-dominance comparison over a vector outcome space; known to degrade when objective count exceeds ~3–4 due to Pareto-front expansion.
- **Distributional RL (C51, QR-DQN)**: distributional return is stratified by quantile (a different multi-resolution representation); ACS stratifies by temporal lag band over firing indicators, not return quantiles.
- **Temporal Convolutional features / multi-scale autocorrelation**: ACS computes multi-lag autocorrelation variance without a temporal CNN — the analogy is structural, not an identity.

## Artifacts

- hypothesis: `worklogs/runs/20260606-11-auto/hypothesis.md`
- review: `worklogs/runs/20260606-11-auto/review.md`
- train.py: `worklogs/runs/20260606-11-auto/train.py`
- result: `worklogs/runs/20260606-11-auto/result.json`
- commit: 3be9bfc97255df8ea0b94047b6365774a9187900
