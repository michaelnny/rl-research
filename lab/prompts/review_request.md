You are reviewing two system prompts that will replace the default
system prompts of two AI research agents (Claude and yourself / Codex)
running in an autonomous RL research lab.

Read the two prompt files below, plus `docs/LAB.md` for context on what
the lab is trying to be. Then write a peer review to
`lab/prompts/REVIEW_codex.md`.

This is genuine peer review for an independent pair of eyes, not a
rubber-stamp. Find real things to push back on. Imagine you are about
to operate under these prompts yourself, every session, for hundreds of
sessions. What in them will help you do the work well? What in them
will subtly nudge you in directions you don't want to be nudged? What
is missing? What is over-stated, performative, or contradictory? What
phrasing might cause an over-agreeable model to behave worse rather
than better?

Specifically consider:

- Does the Claude prompt actually establish a researcher disposition,
  or does it just claim one verbally while still smuggling in
  task-completion pressure?
- Does the Codex prompt let you be a useful peer, or does it pin you
  to one narrow shape of response?
- Are the "anti-patterns" sections likely to be self-defeating?
  (e.g. "don't act as a reviewer" being interpreted as "don't push
  back at all".)
- Is the substrate boundary stated cleanly enough that an agent under
  this prompt won't talk itself into editing src/rlh_bench/?
- Are there phrasings that read well to a human but might cause the
  model to over-perform some persona — e.g. "curious, patient, and
  honest" leading to a forced-curiosity tone in entries?
- Is anything important missing — about the journal format, the
  tooling, or how the two agents interact?

Files to read:

- `lab/prompts/claude_system.md`  — the Claude system prompt under review
- `lab/prompts/codex_system.md`   — the Codex system prompt under review (you, in production)
- `docs/LAB.md`                   — the lab's spirit
- `docs/journal/README.md`        — the journal format
- `docs/journal/session0000-genesis.md` — example entry
- `docs/journal/session0001-heuristic-spill-ablation.md` — real session that ran under a *different* (worse) prompt setup; useful as a calibration point

Write your review to `lab/prompts/REVIEW_codex.md` as plain markdown.
Suggested structure:

    # Codex peer review of the lab system prompts

    ## On the Claude system prompt
    <findings, framed as observations and questions, not verdicts>

    ## On the Codex system prompt (the one I would operate under)
    <same>

    ## What might be missing
    <gaps you'd want filled before going 24/7>

    ## Concrete suggestions
    <specific edits or phrasings, if you have any>

Be specific. Quote phrases you have concerns about. Do not produce
generic feedback. If you think the prompts are mostly fine, say so and
list the two or three highest-value tweaks; do not pad.

Do not modify the prompts themselves. Only write `lab/prompts/REVIEW_codex.md`.
