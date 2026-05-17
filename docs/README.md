# docs/

The anti-divergence specs. Every role and every iteration of the loop reads
these. Changes ripple — make them deliberately, in PRs the user reviews.

Read in this order if you are new to the project:

1. **[charter.md](charter.md)** — mission, hard rules, disqualifiers.
   The single source of truth for "what is this project trying to do."
2. **[loop.md](loop.md)** — the four roles and how an iteration proceeds.
3. **[contract.md](contract.md)** — the run artifact contract that makes the
   corpus comparable across runs.
4. **[benchmarks.md](benchmarks.md)** — the three primary benchmarks (one per
   pillar) plus sanity envs.
5. **[sota.md](sota.md)** — published SOTA references per benchmark and the
   audit of our own PPO yardstick against canonical PPO numbers.

Role prompts live under [roles/](roles/) and are loaded into the corresponding
Claude Code agent at iteration start:

- [roles/researcher.md](roles/researcher.md)
- [roles/reviewer.md](roles/reviewer.md)
- [roles/engineer.md](roles/engineer.md)
- [roles/curator.md](roles/curator.md)

Operational artifacts (the actual corpus the agents read and write) live under
`/lab/`.
