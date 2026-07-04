# The lab

This file is what every agent in the lab reads first, every session. It is
short on purpose.

## What this lab is

This is a research lab for finding a novel **continuous-action** RL
algorithm — same class as PPO, SAC, CEM, mirror descent, GAE-style
credit assignment, or trajectory-level vector-reward methods. The
substrate the algorithm has to work on is `rlh_bench` (vendored under
`src/rlh_bench/`): deterministic, recoverable, long-horizon,
terminal-only sparse feedback, with an optional terminal vector reward,
and continuous action spaces throughout. AlphaZero / MCTS-class
algorithms need a discrete/structured-search action substrate that
this lab does not currently provide; see
`lab/notes/planning/PLAN_substrate_redesign_v2_2026-06-30.md` for the
rationale.

The lab has two members:

- **Claude** — works one session at a time, writes a journal entry.
- **Codex** (profile `jelly`) — reads Claude entries afterwards as a
  peer and adds a `## Peer note` section. Every few regular sessions,
  Codex also writes a short steering memo with fresh research leads.

Neither agent reviews the other. There is no gatekeeper, no verdict,
no scoring.

## What this lab is not

It is not a piecework factory. Sessions are not graded. There are no
outcome tags like `empty-hand` or `reviewer-rejected`. An entry that
says "I read the resource-allocation env carefully and noticed X" is a
good entry. An entry that proposes a half-baked idea with no
implementation is a good entry. An entry whose implementation runs and
shows nothing useful is a good entry — the result is the lesson.

If the loop runs for a hundred sessions and no algorithm is found, but
the journal is full of thoughtful entries that map the problem and the
ideas that don't work and why, that is a success.

## The product is the journal

The product is **not** "a novel RL algorithm." The product is
`lab/journal/` — a research journal that, accumulated over time,
makes a novel algorithm likely. The algorithm, if it emerges, will be
a downstream consequence of the journal being honest and varied.

## How a session works

A regular session is one `claude -p` invocation followed by one
`codex -a never exec -p jelly` peer-note invocation. The Claude side
does whatever a thoughtful researcher in this lab would do right now —
the disposition and the menu of session kinds are in
`lab/prompts/claude_system.md` (loaded as Claude's system prompt via
`--bare --system-prompt-file`). The Codex peer side reads what just got
written and reacts as a colleague would — its disposition is in
`lab/prompts/codex_system.md`.

Every few regular sessions, the loop instead asks Codex to write
`sessionNNNN-codex-steering.md`: a short memo with 2-3 next-session
leads and one recommended lead for Claude. The following Claude prompt
explicitly tells Claude to use that memo rather than continuing by
inertia.

## The only hard rules

These are about protecting the substrate, not about gating ideas:

1. Do not edit anything under `src/rlh_bench/` to make an idea work.
2. Do not add per-step reward shaping back into the env. Terminal-only
   is load-bearing.
3. Do not pull in baseline RL libraries (stable-baselines3, RLlib,
   cleanrl, Tianshou). NumPy and optional PyTorch are the bar.
4. Vector reward learning means consuming `info["reward_vector"]`. If
   the learner flattens to a weighted scalar internally, it is
   scalarization, not vector RL. Call it what it is in the journal.

Everything else is the agent's call. Speculate freely.

## Pointers

- `docs/SUBSTRATE_MAP.md` — what the substrate offers, in one page.
- `docs/AGENT_GUIDE.md` — how a candidate algorithm plugs into the
  evaluation runner, when an agent decides to implement something.
- `docs/baseline_report.md` — honest baseline portfolio numbers
  on the currently registered Small-tier envs. v0/Large tiers were
  removed pending validation.
- `lab/journal/` — every session's entry, append-only.
- `lab/prompts/claude_system.md` — Claude's lab system prompt (the
  source of truth for how Claude should behave in a session).
- `lab/prompts/codex_system.md` — Codex's peer-note system prompt.
- `lab/prompts/codex_steering_system.md` — Codex's steering-memo
  system prompt.
- `lab/prompts/{claude_session,codex_peer,codex_steering}.md` — thin
  per-iteration user prompts; the system prompts above carry the
  substance.
- `lab/run_lab.sh` — the dumb loop.
- `lab/README.md` — operator's manual (start/stop/watch).
- `CLAUDE.md` — project-level rules of engagement.
