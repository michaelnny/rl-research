# Lab notes

Working notes and historical artifacts about the lab itself — design
decisions, peer reviews, calibration outcomes. Distinct from
`lab/journal/`, which is the research journal the loop produces;
`lab/notes/` is meta about the lab.

## Top-level

The two files that describe the current substrate state:

- **`acceptance_gates_audit_2026-06-30.md`** — per-env status of
  the 12 acceptance gates from the v2 substrate plan, kept in sync
  with the registry.
- **`strict_registry_outcome_2026-06-30.md`** — why only the two
  Small-tier envs are registered; per-env reasoning for the
  v0/Large removals; re-registration policy.

## planning/

Substrate redesign artifacts. Read in order:

1. `PLAN_substrate_redesign_2026-06-30.md` — first plan I wrote
   (Claude). Wrong on several axes; kept as historical record.
2. `codex_redteam_substrate_redesign_2026-06-30.md` — Codex's
   adversarial review that destroyed the first plan.
3. `codex_counterdesign_substrate_2026-06-30.md` — Codex's
   counter-design (KeyFuelMaze + CapacityScheduling families, 12
   acceptance gates, held-out evaluation as a first-class property).
4. `PLAN_substrate_redesign_v2_2026-06-30.md` — the consolidated
   plan that the implementation followed.

## reviews/

Codex review and audit outputs across the substrate redesign work.
Each one is the *output* of a Codex pass; the corresponding *input*
brief is in `briefs/`.

- `codex_review_of_prompts_2026-06-30.md` — independent review of
  the production system prompts.
- `codex_capacity_scheduling_calibration_2026-06-30.md` — fix for
  the initial CapacityScheduling Small calibration.
- `codex_v0_calibration_2026-06-30.md` — v0 calibration outcome.
- `codex_v0_calibration_review_2026-06-30.md` — Codex's
  adversarial review of its own v0 fixes (caught the oracle-as-
  baseline issue).
- `codex_reward_normalization_audit_2026-06-30.md` — gate 10
  cross-tier normalization fixes.
- `codex_held_out_review_2026-06-30.md` — gate 9 review.
- `codex_final_review_2026-06-30.md` — pre-merge review; flagged
  the three blockers (trailing action dims, route_efficiency=1.0
  for zero policy, stale operator docs).
- `codex_cleanup_review_2026-06-30.md` — final cleanup pass
  outcome.

## briefs/

The one-shot system prompts used to put Codex into each
specialized role (red-team, counter-designer, calibrator,
reviewer, etc). Kept for reproducibility — anyone can re-run a
review by re-using the corresponding brief.

Files match 1:1 with `reviews/` outputs (e.g.
`codex_v0_calibration_oneshot.md` → `codex_v0_calibration_2026-06-30.md`).

## Adding new notes

Put notes here when:

- A lab-design decision is made and the reasoning is worth
  preserving.
- An outside reviewer (Codex, a human) gives feedback on the lab
  itself.
- Something about the loop's behavior needs to be documented
  out-of-band of the journal.

Do not put per-session research findings here — those go in
`lab/journal/`.
