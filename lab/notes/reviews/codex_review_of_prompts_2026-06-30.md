# Codex peer review of the lab system prompts

## On the Claude system prompt

Overall, this is much better than a normal task-completion prompt for the kind of lab described in `docs/LAB.md`. The strongest parts are the repeated reframing of the journal as the product, the explicit acceptance of partial/speculative sessions, and the substrate rules. I would expect this to produce more honest entries than the setup that led to the old verdict dynamic.

A few places still seem worth tightening:

- The prompt mostly establishes a researcher disposition, but it still ends with a little task-pressure: "If you accidentally produced no journal entry, the loop will record that and continue. So write the entry." I understand the operational reason, but the phrasing pushes toward producing *something* even when the session has gone sideways. A softer version could say that if the session collapses, the entry can be just a short honest account of that collapse. That preserves the journal requirement without implying "always manufacture an entry."

- "You are curious, patient, and honest" reads well to a human, but it is also the kind of phrase that can make a model perform a personality. The later operational clauses are more useful than the adjectives. I would rather see less identity-language and more behavioral guidance like: "Prefer concrete observations over polished-sounding synthesis; when unsure, say what would need to be checked."

- "There are no verdicts in this lab" is good and probably necessary, but the anti-verdict language is very strong. It might accidentally suppress useful internal judgments such as "this idea is probably not worth more implementation until X is checked." The prompt says entries stand on their own, but researchers still need to prioritize. I would distinguish *no public outcome tags / no gatekeeping* from *it is okay to make local prioritization judgments and explain them as reasons, not verdicts*.

- The session menu is helpful, but the `implement` bullet has a subtle completion pull: implement under `experiments/algorithms/<slug>.py`, evaluate, save JSON. That is right for real algorithm attempts, but for exploratory probes like session 0001, `experiments/probes/` was a better shape. The prompt allows `play`, yet the implement path is the only explicit file path. I would add that quick probes/scripts can live under `experiments/probes/` and do not need to satisfy the full `Algorithm` protocol unless they are candidate algorithms.

- The substrate boundary is stated clearly enough in the hard rules. I do not think an agent carefully reading this prompt should talk itself into editing `src/rlh_bench/`. The one possible leak is "You may create additional files under `experiments/` or `lab/` if your session calls for it". `lab/` includes prompts and runner machinery; that is probably intended for tool-build sessions, but it may be too broad for an autonomous loop. If prompt/tooling edits are allowed, say so explicitly; otherwise narrow this to `experiments/` plus perhaps `lab/tools/` or a named scratch area.

- The prompt says "You do not have a `TodoWrite` tool." That is brittle as a system-prompt statement because the actual tool surface may change. If the runtime exposes a planner anyway, this creates an unnecessary conflict. The real point is "do not let process-management dominate the session"; I would phrase that without naming absent tools.

- "You will not produce that algorithm in this session" is directionally healthy, but absolute. A session might stumble on a surprisingly strong simple algorithm, as session 0001 already overturned the initial hardness assumption with a stateless policy. Maybe: "Do not measure the session by whether it produces the algorithm." That avoids ruling out lucky discoveries.

## On the Codex system prompt (the one I would operate under)

This prompt would make me a useful low-friction colleague in many cases. The coffee-peer framing, the short-note constraint, and the ban on verdicts directly address the failure mode described in `docs/LAB.md`. I especially like "one concrete question or one real connection" as the unit of contribution.

The main risk is that it may overcorrect from reviewer/gatekeeper into being too gentle or too narrow:

- "not as a reviewer, not as a gatekeeper, not as a judge" plus "Do not act as a gate" is correct, but an over-agreeable model may hear "do not review" as "do not push back." The later examples include pushback, but I would make the distinction explicit: "You should still challenge claims, arithmetic, novelty, or framing when the challenge would help; just do not turn that challenge into a verdict."

- The line "What a peer note is not: ... a contradiction of the entry's narrative" worries me. If the entry's narrative is contradicted by its own numbers, prior journal entries, or the substrate, the peer note should absolutely be allowed to say so. The undesirable thing is probably "do not rewrite the session into the story you wish Claude had told." I would replace this bullet. As written, it could suppress exactly the kind of useful correction a peer should provide.

- "Do not tag it 'good' or 'weak' or 'rejected' or 'promising'" is slightly too broad. I agree about avoiding verdict tags, but "promising" is also ordinary research language for a lead worth following. If the word itself is banned, the model may contort around useful prioritization. Better: do not use those as labels/verdicts; if something seems worth following, explain the concrete reason.

- Five to fifteen lines and "No bullet salad" is good discipline, but it pins Codex to one response shape even when the entry is long, technical, or contains multiple independent issues. I would keep "default to short" but allow exceptions: "Usually 5-15 lines; go longer only if needed to surface a concrete error or important connection." Otherwise Codex may under-respond to an entry like session 0001, where checking the cost/safety interpretation could merit more than one question.

- The prompt says most notes need only the entry, and optionally docs / one or two earlier entries. I would add an explicit nudge to inspect referenced artifacts when the entry rests on them: result JSON, probe scripts, or prior peer notes. Not to audit everything, but to avoid making coffee-note comments based only on prose when the evidence is one file away.

- The ending behavior "If the entry is missing entirely, do nothing and exit" is operationally clean but silent. If the loop permits it, a tiny diagnostic somewhere would be useful; but since this prompt forbids writing outside the entry, silence may be the only safe behavior. The runner should probably catch and log this outside the model.

## What might be missing

- **A clearer distinction between verdicts and prioritization.** Both prompts rightly reject scoring and acceptance/rejection, but researchers still need to say "I would not spend another session here until X is true" or "this seems like the best lead because Y." The prompts should explicitly allow reasoned prioritization without outcome labels.

- **A standard optional `## Files touched` section.** Session 0001 includes it, and it is useful. `lab/journal/README.md` and the Claude prompt do not list it in the template. I would add it as optional, especially for `play`, `implement`, and `tool-build` sessions.

- **Guidance for reading peer notes.** Claude is told Codex appends notes and may pick them up, but I would make this more operational: when reading recent entries, include their peer notes; if a peer note raises a concrete question, either follow it or explicitly choose another lead. That would make the two-agent loop more cumulative.

- **A scratch/probe convention.** The lab already has `experiments/algorithms/` for protocol-compliant candidates, but the real research flow will often need small probes, one-off traces, and notebooks/scripts. Naming `experiments/probes/` or `experiments/scratch/` as acceptable would prevent everything from being prematurely framed as an algorithm.

- **Novelty/prior-work humility.** Codex has an honesty note about not inventing papers. Claude could use a parallel instruction: if an idea resembles known families, say what it resembles and what might be new; do not market it as novel just because it is locally new to the journal.

- **Tooling reality around network and dependencies.** The Claude prompt says no network calls unless there is a specific reason, and no baseline RL libraries. It might also say whether installing small utilities is allowed, and where dependency changes should be recorded. For a 24/7 loop, uncontrolled dependencies can become a hidden source of drift.

## Concrete suggestions

1. Replace "So write the entry" with something like: "If the session goes badly, write the smallest honest entry that records what happened and where the next session should restart."

2. In both prompts, add one sentence after the no-verdict rule: "You may still make reasoned prioritization judgments; phrase them as evidence and tradeoffs, not as labels or gatekeeping."

3. In the Codex prompt, replace "a contradiction of the entry's narrative" with: "a replacement narrative that ignores what the entry actually did." Add: "If the entry's claim seems contradicted by its evidence, say that plainly and specifically."

4. Add optional journal sections to the template:
   - `## Files touched` for code/results created or modified.
   - Possibly `## Open question` for the single question the session leaves behind.

5. Narrow or clarify Claude's write permissions under `lab/`: either reserve `lab/` edits for explicit tool-build sessions, or name a safer subdirectory. Keep the `src/rlh_bench/` prohibition exactly as strong as it is.

6. Add a light Codex evidence-check instruction: "If the entry cites a new result file or probe script and your note depends on that claim, skim the artifact before commenting."
