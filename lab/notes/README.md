# Lab notes

Working notes and historical artifacts about the lab itself — design
decisions, prompt reviews, post-mortems. Distinct from
`docs/journal/`, which is the research journal the loop produces;
`lab/notes/` is meta about the lab.

## Contents

- `codex_review_of_prompts_2026-06-30.md` — independent peer review of
  the production system prompts (`lab/prompts/claude_system.md` and
  `lab/prompts/codex_system.md`) by Codex/jelly. Caught real issues
  that were folded back into the prompts before the loop went live.
- `codex_reviewer_oneshot_used_for_prompt_review.md` — the one-shot
  system prompt used to put Codex in "reviewer" mode for the above.
  Kept for reproducibility; not used by the production loop.

Add notes here when:

- A lab-design decision is made and the reasoning is worth preserving.
- An outside reviewer (Codex, a human) gives feedback on the lab
  itself.
- Something about the loop's behavior needs to be documented out-of-
  band of the journal.

Do not put per-session research findings here — those go in
`docs/journal/`.
