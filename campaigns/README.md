# Versioned campaign definitions

This directory contains immutable scientific definitions, schemas, and
qualification protocols. Mutable queues, keys, model transcripts, checkpoints,
worktrees, and raw runs belong under ignored `runtime/` storage.

`factorlab_v1/` is the neural benchmark campaign. Automated algorithm research
may run only task tiers named in its `admitted_tiers` list and backed by a
verified qualification report. A software test pass alone cannot add a tier.
