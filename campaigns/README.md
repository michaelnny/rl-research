# Versioned campaign definitions

This directory contains immutable scientific definitions, schemas, and
qualification protocols. Mutable queues, keys, model transcripts, checkpoints,
worktrees, and raw runs belong under ignored `runtime/` storage.

`factorlab_long_v1/` is the only active benchmark campaign. Automated algorithm
research may run only tiers named in `admitted_tiers` and backed by a reviewed,
verified qualification report. The rejected 64-step experiment remains only in
Git history. A software test pass alone cannot add a tier.
