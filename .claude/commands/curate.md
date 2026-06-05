---
description: Standalone Curator pass — synthesize any uncurated runs (e.g. after an interrupted iteration).
allowed-tools: Agent, Read, Bash
---

Find every `worklogs/runs/<run_id>/` that has a `result.json` but no
`curator.md`, and spawn the `curator` subagent for each in chronological
order (oldest first).

## Discovery

```bash
for dir in worklogs/runs/*/; do
  run_id=$(basename "$dir")
  if [[ -f "$dir/result.json" && ! -f "$dir/curator.md" ]]; then
    echo "$run_id"
  fi
done
```

## For each uncurated run_id

Spawn `curator` with this exact prompt:

> Synthesize iteration `<run_id>`. Read hypothesis.md, review.md,
> result.json, panel.txt (if present), then write `curator.md`,
> append to `worklogs/ledger.jsonl`, and update the corpus per the
> verdict-conditional outputs in your agent definition.

After each subagent returns, verify `curator.md` was written and the
ledger has a matching line. Do not retry — if a curator run failed,
surface it to the user.

## Final summary

List the run_ids that were curated, each with its assigned verdict.
