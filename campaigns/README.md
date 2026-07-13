# Campaign definitions

This directory contains versioned prompts, role schemas, and campaign
configuration. Mutable databases, worktrees, transcripts, and experiment
artifacts belong under ignored runtime storage.

Role schemas are scientific contracts. Their required predictions and evidence
identifiers are intentionally stricter than ordinary agent final messages.
Changing a schema during a campaign creates a new campaign protocol version.

`factorlab_v0/` is the first benchmark-calibration campaign. It deliberately
contains no held-out master seed and remains `under_calibration` until every
qualification gate has immutable, independently audited evidence.
