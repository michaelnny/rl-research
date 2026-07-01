# Lab — operator notes

This directory holds everything needed to run the lab loop. The
*lab itself* is described in `docs/LAB.md`; read that first. This file
is the dry operator's manual.

## Files

- `prompts/claude_system.md` — Claude's lab system prompt (the source
  of truth for how Claude should behave in a session). Loaded via
  `--bare --system-prompt-file`, which fully replaces Claude Code's
  ~37k-token default engineering prompt and disables auto-memory.
- `prompts/claude_session.md` — thin per-iteration user prompt for
  Claude; carries only the session number and a pointer to begin.
- `prompts/codex_system.md` — Codex's lab system prompt (peer
  reviewer disposition). Loaded via
  `-c model_instructions_file=...`, which replaces the default
  `base_instructions` for that one invocation.
- `prompts/codex_peer.md` — thin per-iteration user prompt for Codex;
  carries only the journal entry path and a pointer to begin.
- `run_lab.sh` — the dumb loop that wires the two together and
  commits.
- `logs/` — per-iteration logs (`iter-NNNN/{claude,codex}_{stdout,stderr,prompt}.txt`)
  plus a `run.log` heartbeat. Gitignored.
- `notes/` — lab-meta artifacts. Organized into:
  - `notes/planning/` — substrate redesign artifacts.
  - `notes/reviews/` — Codex review/audit outputs.
  - `notes/briefs/` — one-shot prompts used to launch each
    review pass (kept for reproducibility).
  - Top-level: `acceptance_gates_audit_*.md` and
    `strict_registry_outcome_*.md` document the current testbed
    state.
  Not research findings; those go in `docs/journal/`.

## Running

Forever, detached:

```bash
mkdir -p lab/logs
nohup bash lab/run_lab.sh > lab/logs/console.log 2>&1 &
```

In a tmux session (easier to detach/reattach):

```bash
tmux new -s rl-lab 'bash lab/run_lab.sh'
# detach: Ctrl-b d ; reattach: tmux attach -t rl-lab ; stop: Ctrl-c inside tmux
```

The loop auto-branches off `master` to `lab/auto` on first run so the
main branch is never auto-committed to.

## Stopping

```bash
kill $(cat lab/logs/run.pid)
```

or `Ctrl-C` in the tmux. The loop catches `SIGINT`/`SIGTERM` and exits
cleanly between iterations.

## Watching

```bash
tail -f lab/logs/run.log
tail -f lab/logs/console.log   # only needed for shell-level stderr/stdout
ls -lat docs/journal/ | head        # newest entries first
git log --oneline lab/auto | head   # commits the loop has made
```

## What "good" looks like

You are not waiting for an algorithm. You are waiting for a journal
that, read end-to-end after a few hundred sessions, sounds like a real
research group thinking. A run is healthy when:

- Session kinds vary: `read`, `play`, `propose`, `implement`,
  `synthesize`, `tool-build` all appear over time.
- Peer notes engage with the entry rather than rubber-stamping it.
- Bad ideas appear honestly and get picked up or refuted later.
- The substrate (`src/rlh_bench/`) is unchanged across the run.

A run is unhealthy if:

- Every entry is `implement`, or every entry is `propose`. The harness
  doesn't enforce balance; if the journal monocultures, intervene by
  hand.
- Peer notes start scoring or verdict-tagging. Tighten the Codex
  prompt or stop the loop.
- Files under `src/rlh_bench/` start changing. Stop the loop. Revert.
  Re-read `CLAUDE.md` together with the agent next time.

## Disposition reminder

Running the loop is not the same as producing an algorithm. The loop
produces a journal. If a novel algorithm emerges, it will be downstream
of the journal being honest.
