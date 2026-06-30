# The lab

This file is what every agent in the lab reads first, every session. It is
short on purpose.

## What this lab is

This is a research lab for finding a novel RL algorithm — same class as
Q-learning, PPO, AlphaZero, mirror descent, SAC, MCTS, GAE. The substrate
the algorithm has to work on is `rlh_bench` (vendored under
`src/rlh_bench/`): deterministic, recoverable, long-horizon,
terminal-only sparse feedback, with an optional terminal vector reward.

The lab has two members:

- **Claude** — works one session at a time, writes a journal entry.
- **Codex** (profile `jelly`) — reads the entry afterwards as a peer
  and adds a `## Peer note` section to the same entry.

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
`docs/journal/` — a research journal that, accumulated over time,
makes a novel algorithm likely. The algorithm, if it emerges, will be
a downstream consequence of the journal being honest and varied.

## How a session works

A session is one `claude -p` invocation followed by one
`codex exec -p jelly` invocation. The Claude side does whatever a
thoughtful researcher in this lab would do right now — the menu is in
`lab/prompts/claude_session.md`. The Codex side reads what just got
written and reacts as a colleague would. Then the loop commits the
journal entry with a descriptive (not verdictive) message and starts
the next session.

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
- `docs/baseline_report.md` — honest baseline numbers (random,
  heuristic, CEM) across all six env IDs.
- `docs/journal/` — every session's entry, append-only.
- `lab/prompts/` — exactly what each agent is told per session.
- `lab/run_lab.sh` — the dumb loop.
- `CLAUDE.md` — project-level rules of engagement.
