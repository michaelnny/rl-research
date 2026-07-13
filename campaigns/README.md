# Versioned campaign definitions

This directory contains immutable scientific definitions, schemas, and
qualification protocols. Mutable queues, keys, model transcripts, checkpoints,
worktrees, and raw runs belong under ignored `runtime/` storage.

`factorlab_long_v1/` is the active 5,000--20,000-step neural benchmark campaign.
`factorlab_v1/` is retained as a retired 64-step smoke record. Automated
algorithm research may run only active task tiers named in `admitted_tiers` and
backed by a verified qualification report. A software test pass alone cannot
add a tier.
