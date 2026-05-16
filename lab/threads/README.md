# Threads

A *thread* is a research direction, not a single experiment. One thread spans
many runs as the corpus accumulates.

Each thread is a markdown file: `<thread-slug>.md` with the frontmatter:

```markdown
---
status: active | paused | archived
opened: YYYY-MM-DD
last_curated: YYYY-MM-DD
runs: [0001-thread-slug, 0007-thread-slug]
---
```

Threads are managed by the **Curator**. The Researcher can open a new thread
ahead of any concrete hypothesis (status: `active`, no runs yet) — that is a
valid way to declare an intent for future iterations.

Naming: kebab-case, descriptive of the *mechanism family*, not a benchmark.

- Good: `energy-based-credit`, `vector-credit-decomposition`,
  `population-pareto-search`.
- Bad: `montezuma-attempt`, `humanoid-fast`, `take-2`.

See `docs/roles/curator.md` for the per-thread template.
