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
- `prompts/codex_steering_system.md` — Codex's steering prompt for
  periodic research-direction memos.
- `prompts/codex_steering.md` — thin steering-session user prompt;
  carries the target journal path and steering cadence.
- `run_lab.sh` — the dumb loop that wires the two together and
  commits.
- `journal/` — append-only research journal, one markdown file per
  session. Tracked.
- `logs/` — process-level logs (`run.log`, `console.log`) and
  `run.pid`. Gitignored.
- `runs/` — per-session raw artifacts
  (`iter-NNNN/{claude,codex}_{stdout,stderr,prompt}.txt`). Gitignored.
- `notes/` — lab-meta artifacts. Organized into:
  - `notes/planning/` — substrate redesign artifacts.
  - `notes/reviews/` — Codex review/audit outputs.
  - `notes/briefs/` — one-shot prompts used to launch each
    review pass (kept for reproducibility).
  - Top-level: `acceptance_gates_audit_*.md` and
    `strict_registry_outcome_*.md` document the current testbed
    state.
  Not research findings; those go in `lab/journal/`.

## Running

Forever, detached:

```bash
# Refuses to start unless the worktree is clean and HAI_OPENAI_API_KEY
# is exported in this shell.
git status --short

mkdir -p lab/logs
nohup bash lab/run_lab.sh > lab/logs/console.log 2>&1 &
```

In a tmux session (easier to detach/reattach):

```bash
tmux new -s rl-lab 'bash lab/run_lab.sh'
# detach: Ctrl-b d ; reattach: tmux attach -t rl-lab ; stop: Ctrl-c inside tmux
```

The loop auto-branches off `master` to `lab/auto` on first run so the
main branch is never auto-committed to. It refuses to start from a
dirty worktree; commit or stash human edits before launching it.

## Stopping

```bash
kill $(cat lab/logs/run.pid)
```

or `Ctrl-C` in the tmux. The loop catches `SIGINT`/`SIGTERM`, stops
the active agent process tree if one is running, and removes
`lab/logs/run.pid`.

## Watching

```bash
tail -f lab/logs/run.log
tail -f lab/logs/console.log   # only needed for shell-level stderr/stdout
ls -lat lab/runs/ | head       # newest raw run artifacts first
ls -lat lab/journal/ | head    # newest journal entries first
git log --oneline lab/auto | head   # commits the loop has made
```

## What "good" looks like

You are not waiting for an algorithm. You are waiting for a journal
that, read end-to-end after a few hundred sessions, sounds like a real
research group thinking. A run is healthy when:

- Session kinds vary: `read`, `play`, `propose`, `implement`,
  `synthesize`, `tool-build` all appear over time.
- Peer notes engage with the entry rather than rubber-stamping it.
- `sessionNNNN-codex-steering.md` appears roughly every few regular
  sessions, and the following Claude entry engages with one of its
  leads.
- Bad ideas appear honestly and get picked up or refuted later.
- The substrate (`src/rlh_bench/`) is unchanged across the run.

A run is unhealthy if:

- Every entry is `implement`, or every entry is `propose`. The harness
  now inserts steering memos, but if the journal still monocultures,
  intervene by hand.
- Peer notes start scoring or verdict-tagging. Tighten the Codex
  prompt or stop the loop.
- Files under `src/rlh_bench/` start changing. The runner refuses to
  commit substrate edits, but if you see them in a stopped worktree,
  inspect and revert by hand. Re-read `CLAUDE.md` together with the
  agent next time.

## Disposition reminder

Running the loop is not the same as producing an algorithm. The loop
produces a journal. If a novel algorithm emerges, it will be downstream
of the journal being honest.
