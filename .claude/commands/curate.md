---
description: Standalone Curator pass - synthesize any uncurated probe-first runs after an interrupted iteration.
allowed-tools: Agent, Read, Bash
---

Find every `worklogs/runs/<run_id>/` that has a `result.json` but no
`curator.md`, and spawn the `curator` subagent for each in chronological
order.

## Discovery

```bash
for dir in worklogs/runs/*/; do
  run_id=$(basename "$dir")
  if [[ -f "$dir/result.json" && ! -f "$dir/curator.md" ]]; then
    echo "$run_id"
  fi
done
```

## For each run_id

Spawn `curator` with this prompt:

> Synthesize iteration `<run_id>`. Read hypothesis.md, candidate.json (if
> present), review.md (if present), result.json, panel-*.txt or panel.txt
> (if present), and fix/blocker notes (if present). Write curator.md,
> append to worklogs/ledger.jsonl, and update corpus files per your agent
> definition. If the verdict is `proven-on-substrate`, write
> worklogs/HALT_REQUESTED.md.

After each Curator returns, verify `curator.md` exists and the ledger has
one matching line. Do not retry more than once; surface failures to the
user.

## Final Summary

List curated run_ids with verdict, status, and stage.
