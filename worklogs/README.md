# worklogs

Research memory for the autonomous RL-algorithm-discovery loop. Bounded,
templated, append-only.

## Why this exists

Each iteration of the experiment loop (see `program.md`) produces evidence
that the next iteration must read. The scale we are building for —
weeks-long unattended autonomous research, ~100 smoke sweeps per night —
will produce hundreds of attempts. A single growing prose file does not
scale: it cannot be reliably appended to under template discipline, it
gets re-read in full every time, and it loses index structure.

So the worklogs are split into:

```
worklogs/
├── README.md          ← this file
├── TEMPLATE.md        ← the fixed per-attempt template
└── attempts/          ← database: one bounded file per attempt
```

The curated index — one paragraph per attempt, with verdict and the
nearest disqualifier family — lives at the repo root in
[`prior_attempts.md`](../prior_attempts.md). That file is what the agent
reads between iterations. The per-attempt files in `attempts/` are the
backing database; the agent only opens them when it needs the math, the
prototype detail, or the cross-attempt comparison.

## When to write what

The loop is single-track: hypothesis → edit `train.py` → commit → run
panel → log. There is no separate "propose" step that lives outside
that loop. Every file the *autonomous loop* writes under `attempts/`
corresponds to a `train.py` commit that ran against the panel; its
`panel_evidence` block is non-null. An idea the agent considered but
did not implement is not a file — it is a sentence in the next commit
message, or a row in `results.tsv` if it crashed.

Entries 01–14 currently in `attempts/` are pre-substrate derivations
imported from prior offline research sprints — they have `panel_evidence:
null` and are kept here as the structured negative space the agent reads
between iterations. New entries the autonomous loop produces will have
non-null `panel_evidence`; that is the going-forward rule.

## What if a human wants to seed an idea

A human-supplied research direction (e.g. from offline exploration) is
delivered at session start by including it in the launch prompt or
`/iterate` invocation, not by checking a file into the repo. The agent
treats it the same as any hypothesis it formed itself: implement,
commit, run panel, log under `attempts/`. There is no inbox.

## Template discipline

Every entry under `attempts/` follows [`TEMPLATE.md`](TEMPLATE.md)
exactly: same frontmatter keys, same section headings in the same order.
This is non-negotiable — the autonomous agent appends by template
match. If a section is empty, write `_n/a_`; do not delete the heading.

Target length: **80–200 lines per file**. Above 250 lines means the
attempt should be split or the prose should be tightened.

If a new attempt needs a section the template does not have, add the
section to the template first (and update existing entries to include the
empty heading). Do not let entries diverge.

## Indexing rules

- File name: `NN-<slug>.md` where `NN` is a zero-padded sequential id
  matching the `id:` frontmatter field. Ids are global across the project,
  not per-sprint. The next attempt id is one greater than the highest id
  currently under `attempts/`.
- `slug:` matches the file-name slug exactly.
- `nearest_prior:` is either another attempt id (e.g. `"07"`) or a
  disqualifier-family name from `prior_attempts.md`'s list (e.g.
  `"Bellman backup"`, `"count-based exploration"`).

## Provenance

The two original offline sprint dumps —
`research_attempts_20260524.md` and `research_attemps_20260526.md` —
were the source for entries 01–11. Entries 12–14 came from a separate
offline exploration batch on 2026-05-29. Those raw dumps are not kept
in the repo; the per-attempt files under `attempts/` are the canonical
record going forward.
