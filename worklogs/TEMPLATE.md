---
id: NN
slug: kebab-case-name
status: failed | alive | rebadge | negative-closure
sprint: YYYY-MM-DD
verdict_in_one_line: "<120 chars; what this attempt actually showed>"
nearest_dead_family: null      # A | B | C | D | E | F | G | H | null
mode: probe-v1
candidate_json:
  update_family: null
  memory: null
  nearest_disqualifier: null
  ablation_plan: null
panel_evidence:
  stage: null
  envs: []
  beat_random: 0
  beat_strong: 0
  scores: {}
  ablation_scores: {}
  ablation_delta: {}
  confirmation_seeds: []
  commit: null
---

# NN - <Human Name>

## One-sentence idea

<One sentence; the principle or mechanism being recorded.>

## Core primitive

<Single mathematical object. One paragraph.>

## Improvement operator

<Single update rule or the reason no update should be run.>

## Why it looked promising

- <Up to 5 bullets.>

## What was tested

<Run id, candidate.json summary, stage, envs, random seeds, time budget,
candidate panel result, ablation panel result, and confirmation result.
For a negative closure, state that no panel run was warranted and cite the
theorem or counterexample.>

## Why it failed

<Specific failure mode. State whether this was an ablation failure, null
result, implementation failure, structural failure, or rebadge. Cite
`prior_attempts.md` family names when applicable.>

## Lesson / constraint added

<One sentence: what this rules out for future probes.>

## Nearest neighbors in the literature

<2-4 named methods with explicit overlap and distinction.>

## Artifacts

<Run directory, panel file, result JSON, commit SHA, relevant corpus rows.>
