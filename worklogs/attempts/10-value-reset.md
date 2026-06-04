---
id: 10
slug: value-reset
status: failed
sprint: 2026-05-26
verdict_in_one_line: "Not an algorithm — the conceptual turning point that surfaced WHY the prior nine attempts all failed: they avoided value vocabulary without replacing what value DOES."
side_information: []
nearest_prior: null
panel_evidence:
  smoke_n_beat_random: null
  smoke_n_beat_strong: null
  hard_n_beat_random:  null
  hard_n_beat_strong:  null
  commit: null
---

# 10 — Value-function-first reset (meta-correction)

## One-sentence idea

The original constraint "the primitive must not be Q/V/advantage/Bellman/
TD/policy-gradient" was over-applied as "avoid value vocabulary"; the
correct reading is "replace what value *does* — future compression,
temporal composition, support for local improvement."

## Core primitive

_n/a — this entry records a research-direction correction, not a
primitive. The other entries (#01–#09 before, #11–#14 after) record
attempted primitives._

## Improvement operator

_n/a_

## Why it looked promising

This is the corrective insight, not a candidate algorithm. It explains
the systematic failure of attempts #1–#9: each one invented an object
designed to avoid using the words `Q`, `V`, advantage, return, Bellman,
or TD — but none of them seriously addressed value's *role*. Value is
not an implementation detail. It is the central compression that turns
long-horizon future consequences into a local object usable for policy
improvement:

\[
V^\pi(s) = \mathbb E_\pi[\text{future utility}\mid s].
\]

Any serious replacement must answer: *what replaces value's role of
future compression, temporal composition, and support for local
improvement?*

## What was tested

_n/a — conceptual._ The follow-on test was the design of the next
attempt (#11 TOP), which tried to be a structured-future-consequence
object.

## Why it failed

This is not an attempt that failed; it is the moment the search frame
was reset. The reason it appears in the worklog at all is that
`prior_attempts.md` enumerates it as #10 to keep the index numbering
aligned with the sprint-2 narrative.

## Lesson / constraint added

**"Avoid value vocabulary" is not a research direction. Value's role —
future compression, temporal composition, local improvement — is what
needs replacing, not its name.** This sentence is reproduced verbatim in
`prior_attempts.md` cross-attempt failure modes.

## Nearest neighbors in the literature

_n/a_

## Artifacts

_n/a_
