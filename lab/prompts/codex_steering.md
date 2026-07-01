Your lab steering system prompt has set the context for this session.

Read the most recent 8-12 files in `docs/journal/` (sort by filename,
newest last), including any `## Peer note` sections. Look for repeated
directions, missing falsification, unexamined benchmark assumptions, or
places where the journal is doing local search instead of exploring.

Write one Codex-authored steering memo to:

    __JOURNAL_ENTRY_PATH__

Use this format:

```markdown
# session __SESSION_NUMBER__ — codex steering

date: YYYY-MM-DD
session kind: steer
author: Codex

## Pattern I see

<short, concrete read of the recent journal trajectory>

## Leads for Claude

### Lead 1 — <short name>

<what Claude should do next, why it may unlock a different algorithmic
idea, what would falsify it, and closest known algorithm family>

### Lead 2 — <short name>

<same structure>

### Lead 3 — <short name>

<optional; include only if it is genuinely distinct>

## Recommended next session

<pick one lead and explain why in evidence/tradeoff terms>
```

Do not write a peer note. Do not modify any other file. Do not commit.

---

SESSION NUMBER FOR THIS STEERING MEMO: __SESSION_NUMBER__
STEERING CADENCE: after about __STEERING_INTERVAL__ regular Claude sessions
