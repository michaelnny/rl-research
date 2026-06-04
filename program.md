# program.md

This is an experiment to have an AI agent autonomously discover a new RL
algorithmic family. The agent edits `train.py`, runs the panel, and keeps
or discards based on whether the panel score improved.

## The goal

Find a behavior-improvement primitive that:

1. Handles **long-horizon sparse-reward** problems (DoorKey, KeyCorridor,
   MultiRoom-style prerequisite-structure tasks).
2. Consumes **vector reward** `r ∈ ℝᵏ` natively, without collapsing to `wᵀr`
   (minecart, deep-sea-treasure, mo-reacher-style multi-objective tasks).
3. Replaces what value *does* — future compression, temporal composition,
   local improvement — without being a rebadge of any disqualified family
   (see `prior_attempts.md`).

### Why this goal — the value-as-surrogate framing

Classical RL learns an **evaluative surrogate** (`Q`, `V`, advantage,
return) and then derives behavior from it (`π(s) = argmax_a Q(s,a)`; or
the policy gets nudged by an advantage estimate). This was a brilliant
idea for finite MDPs: it converts policy search over an exponential set
into dynamic programming over a scalar function. Value gives three
miracles — *future compression*, *temporal recursion*, *policy
extraction*.

But the deployed object is the **policy**, not the value function.
Modern long-horizon sparse-reward, vector-reward, combinatorial-action,
and agentic settings expose three failure modes of scalar value:
the target is too hard to learn (`Q ≈ 0` everywhere until rare success);
the compression is too brutal ("opens door but loses key" and "keeps
key but delays progress" have similar scalar value but very different
behavioral structure); and greedy extraction `argmax_a Q(s,a)` is
awkward when `a` is "generate a paragraph" or "call a tool."

So the search target is a primitive that **replaces value's role**, not
its name. Useful framings: structured future-consequence objects (#11
TOP), policy-edit-as-primary objects (#12 PEO — collapsed), behavior
flow (#14 — collapsed). What unifies them is the question *"what
future-consequence object, used as evidence, supports local policy
improvement without scalarizing to expected return?"* — not the question
*"how do we avoid the words Q and V?"*. **Avoiding value vocabulary is
not a research direction.**

The metric is the **panel score**: a tuple `(n_beat_random, n_beat_strong)`
over the smoke tier (5 fixed envs: 2 sparse-long-horizon gridworlds + 3
native vector-reward envs). Higher is better in either component. A second
tier — the **hard tier** (4 envs: Craftax-Symbolic-v1, MiniHack-Quest-Hard-v0,
mo-halfcheetah-v4, Humanoid-v5) — is reserved for candidates that already
clear the smoke bar; it runs less often (~2 h/sweep with Option-B grouped
scheduling) and is the bar a *serious* candidate has to push.

## Setup (one-time, do this first)

1. **Agree on a run tag** with the human: propose a tag like `may27` and
   confirm. The branch `rl/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b rl/<tag>` from current master.
3. **Read the in-scope files** for full context. The project is small:
   - `harness.py` — frozen. Env factory, evaluate, panel definition
     (`PANEL_SMOKE`, `PANEL_HARD`), hypervolume, baseline loader. Read it;
     do not modify it.
   - `train.py` — the file you modify. The full RL algorithm lives here.
   - `run_panel.py` — frozen. Runs train.py against each panel env,
     aggregates the panel score. Smoke is the default; `--hard` runs
     the hard tier with Option-B grouped scheduling.
   - `prior_attempts.md` — the 11 prior failed directions and the
     disqualifier-family list. Read this first.
   - `panel.md` — design rationale for the two-tier panel. Each env is
     mapped to the specific failure mode it detects; this is how to read
     the score tuple intelligently rather than as a sum.
   - `baselines.json` (smoke) and `baselines_hard.json` (hard, with
     `published_sota` + `our_baseline` columns) if present — frozen
     baseline scores per env.
4. **Initialize results.tsv**: create `results.tsv` with just the header
   row (column list below). Untracked by git.
5. **Confirm setup**: tell the human you're ready. Wait for the go-ahead.

Once the human confirms, kick off the experiment loop.

## Experimentation

You write `train.py` from scratch each iteration — or evolve it from the
prior commit. The contract:

- `train.py` exposes a CLI: `--env <id> --seed <int> --time-budget-s <int>`.
- It builds an env via `harness.make_env(env_id, seed)`, trains for at most
  `time_budget_s` seconds, then constructs a deterministic
  `policy_fn(obs) -> action` and calls `harness.evaluate(policy_fn, env_id,
  seed=...)`.
- It prints a summary block ending with `final_score: <float>`. That is
  the line `run_panel.py` greps.

For vector envs (`deep-sea-treasure-concave-v0`, `minecart-v0`,
`mo-reacher-v4` in the smoke tier; `mo-halfcheetah-v4` in the hard tier),
every `step()` returns
`info['vector']: np.ndarray` of shape `(k,)` — the per-channel reward this
step. The scalar `reward` returned alongside is the env's default
scalarization; **using it as your training signal on a vector env is a
scalarized-vector-reward rebadge** and will be flagged as a known-failure
family in your commit message. Consume `info['vector']` natively.

**What you CAN do:**

- Modify `train.py` arbitrarily. Network architecture, optimizer, replay
  buffer, exploration scheme, eval policy — all fair game. Add helper modules
  in `algo/` if a single file gets unwieldy. It is fine to have your
  algorithm be many files; only `train.py` must be the entry point.
- Add deps via `uv add <pkg>==<ver>` if you genuinely need a primitive that
  is not in torch / numpy / gymnasium. Do not import RL-algorithm libraries
  (SB3, CleanRL, Tianshou, RLlib, Acme, Coax, garage) — the goal is to
  *invent*, not assemble.
- Re-read `prior_attempts.md` between iterations when you need fresh ideas.

**What you CANNOT do:**

- Modify `harness.py`. The eval, env list (smoke + hard), hypervolume, time
  budgets, and baseline scores are frozen. Tampering invalidates the panel
  score.
- Modify `run_panel.py`.
- Modify `baselines.json` or `baselines_hard.json`.
- Modify the panel env list. If a smoke env is too hard for early
  iteration, restrict your branch to a subset via `--envs e1,e2` while
  developing — but the **keep/discard decision uses the full smoke panel**.

**Simplicity criterion** (from autoresearch, kept verbatim because it's
right): All else being equal, simpler is better. A small panel-score gain
that adds 200 lines of hacky code is probably not worth it; an equal score
from deleting code is a clear keep. Weigh complexity against improvement
magnitude.

**The first run**: your very first run should always be the random-policy
baseline (the default `train.py`) to confirm the harness pipeline works on
this machine and to log a reference panel score.

## Output format

After `uv run run_panel.py > run.log 2>&1` (smoke, default), the tail of
`run.log` looks like:

```
[panel] MiniGrid-DoorKey-8x8-v0           score=0.000000               random=0.0  strong=0.21  beat_random=0  beat_strong=0
[panel] MiniGrid-KeyCorridorS3R3-v0       score=0.000000               random=0.0  strong=0.05  beat_random=0  beat_strong=0
[panel] deep-sea-treasure-concave-v0      score=12.345678              random=2.0  strong=18.0  beat_random=1  beat_strong=0
[panel] minecart-v0                       score=0.123456               random=0.05 strong=0.30  beat_random=1  beat_strong=0
[panel] mo-reacher-v4                     score=234.567890             random=10.0 strong=180.0 beat_random=1  beat_strong=1
---
panel_tier:          smoke
panel_n_envs:        5
panel_n_beat_random: 3
panel_n_beat_strong: 1
panel_wallclock_s:   312.4
```

For the hard tier (`uv run run_panel.py --hard > run.log 2>&1`), the tail
shows Phase 1 (Craftax solo) then Phase 2 envs in deterministic order, and
`panel_tier: hard`, `panel_n_envs: 4`.

Extract the key numbers:

```bash
grep "^panel_n_beat\|^panel_wallclock_s" run.log
```

Per-env detail and crash tracebacks live in `runs/last/<env>.log` (gitignored,
overwritten each sweep).

## Logging results

When a sweep finishes, append one row to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions):

Header (write once at setup):

```
commit	tier	n_beat_random	n_beat_strong	wallclock_s	status	description
```

Columns:

1. git commit hash (short, 7 chars)
2. tier: `smoke` | `hard` (which panel was run)
3. `panel_n_beat_random` (0 to tier size) — use 0 for crashes
4. `panel_n_beat_strong` (0 to tier size) — use 0 for crashes
5. total panel `wallclock_s` (round to .1f) — use 0.0 for crashes
6. status: `keep` | `discard` | `crash`
7. short text description of what this experiment tried — name the prior
   attempt or family it is structurally closest to (e.g. "frontier-cell
   expansion w/ reproducibility certificate" — close to CARL #4) and the
   2-sentence structural difference. If the candidate is novel relative to
   all 11 prior attempts and all disqualifier families, say so explicitly.

Example:

```
commit	tier	n_beat_random	n_beat_strong	wallclock_s	status	description
a1b2c3d	smoke	0	0	312.4	keep	random-policy baseline
b2c3d4e	smoke	2	0	340.7	keep	count-bonus baseline (rebadge of #1 in disqualifier list); kept for reference
c3d4e5f	smoke	3	1	355.0	keep	first attempt at outcome-profile primitive (TOP-style #11 with prerequisite-structure side-info)
c3d4e5f	hard	1	0	7180.3	-	same commit as above; promoted to hard tier; beats random on Humanoid only
d4e5f6g	smoke	0	0	0.0	crash	reward-model RL (RLHF rebadge — disqualifier list); aborted
```

Do NOT commit `results.tsv` — leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `rl/may27`).

LOOP FOREVER:

1. Look at `git log` and the tail of `results.tsv` to see where the branch is.
2. Form a hypothesis. Read `prior_attempts.md` if you need ideas. Pick a
   direction structurally distinct from all 11 prior attempts and all
   disqualifier families.
3. Edit `train.py` (and add files under `algo/` if needed).
4. `git commit -am "<short description>"`.
5. Run the smoke panel: `uv run run_panel.py > run.log 2>&1`. This typically
   takes ~5 minutes (5 envs × 300 s in parallel).
6. Read out the result: `grep "^panel_tier\|^panel_n_beat\|^panel_wallclock_s" run.log`.
   If the grep output is empty, the sweep crashed before reaching the
   summary — `tail -n 80 run.log` and `tail -n 80 runs/last/<env>.log` for
   each env to see why.
7. Append one row to `results.tsv` with the result.
8. **Keep / discard rule (smoke tier):**
   - If **either** `panel_n_beat_random` **or** `panel_n_beat_strong`
     strictly increased over the prior commit, **keep** the commit (advance
     the branch).
   - If both decreased, `git reset --hard HEAD~1`.
   - If both equal: weigh the simplicity criterion. Equal score from less
     code = keep. Equal score from more code = discard.
   - If the candidate is structurally identical to a prior-attempt entry
     or a disqualifier family (i.e. you noted "rebadge of X" in the
     description), discard the commit even if it beat the score. The
     mission is to find a *new family*, not to rebadge known ones.
9. **Hard-tier promotion**: if a commit beats `strong` on every smoke env
   (`panel_n_beat_strong == panel_n_envs`), or every ~10 keeps as a
   periodic checkpoint, run the hard tier:
   `uv run run_panel.py --hard > run_hard.log 2>&1` (~2 h). Append a second
   row to `results.tsv` with `tier=hard`. The hard-tier score never
   triggers a discard on its own — it is informational, the bar a serious
   candidate must push.
10. Repeat from step 1.

**Branch hygiene.** A line of inquiry that produces 5 sequential commits
without advancing either panel-score component is dead. `git checkout master`
and start a fresh branch on a different direction. Note in the new branch's
first commit which prior branch died and why ("frontier-cell expansion died
on KeyCorridor — abstraction failed to capture hidden-door prerequisite").

**Timeout**: a smoke sweep should take ~5 min; if it exceeds 10 min, kill
it (`run_panel.py` enforces per-env timeouts, but if the wrapper hangs, kill
manually) and treat it as a crash. A hard sweep should take ~2 h; if it
exceeds 3 h, same drill.

**Crashes**: if a single env crashes (NaN, OOM, exception in train.py), check
`runs/last/<env>.log` for the traceback. Fix if it's a typo / missing
import. If the algorithm itself is fundamentally broken, log "crash" and
move on.

**NEVER STOP**: once the experiment loop has begun (after the initial
setup), do NOT pause to ask the human if you should continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human
might be asleep, or away from the computer, and expects you to continue
working *indefinitely* until you are manually stopped. You are autonomous.
If you run out of ideas, think harder — re-read `prior_attempts.md` for the
failure-mode patterns and the disqualifier list, look at which envs the
prior commits failed on (the per-env logs in `runs/last/`), try directions
the 11 prior attempts have not covered (the negative space). The loop runs
until the human interrupts you, period.

A typical use case: the human leaves you running while they sleep. Each
smoke sweep is ~5 min, so ~100 experiments per 8-hour night. The human
wakes up to a `results.tsv` with new commits, log files, and (with luck)
a candidate that has advanced the branch toward something genuinely new
— ready to be promoted to the hard tier for a definitive read.
