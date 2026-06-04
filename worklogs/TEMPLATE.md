---
id: NN
slug: kebab-case-name
status: failed | alive | rebadge
sprint: YYYY-MM-DD
verdict_in_one_line: "<≤120 chars; what this attempt actually showed, not what it tried to show>"
side_information: []           # list from the canonical channel set in prior_attempts.md
nearest_prior: null            # id (e.g. "07") of nearest prior attempt, OR a disqualifier-family name
panel_evidence:                # required (this is an attempts/ file). null on individual fields if a sweep crashed
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# NN — <Human Name>

## One-sentence idea

<one sentence; the elevator pitch>

## Core primitive

<single mathematical object. One paragraph. Use display math if you must,
but stay under ~6 lines.>

## Improvement operator

<single update rule. One paragraph.>

## Why it looked promising

- <≤ 5 bullets>

## What was tested

<envs, seeds, time-budget, what passed, what failed. If this ran on the
substrate, cite the commit SHA and `results.tsv` row.>

## Why it failed

<the specific failure mode. Cite the cross-attempt failure modes in
`prior_attempts.md` by name when applicable.>

## Lesson / constraint added

<one sentence — what does this kill for future candidates?>

## Nearest neighbors in the literature

<2–4 named methods; explicit overlap statement so future candidates can
audit against the rebadge boundary.>

## Artifacts

<file paths, commit SHAs, `results.tsv` rows, prototype scripts.>
