# Run artifact contract

Every run is a directory `lab/runs/<run_id>/` containing exactly the files specified
below. Any deviation is a contract violation; the Operator must refuse to record
runs that violate it.

This contract is the basis for the corpus being navigable, comparable across runs,
and parseable by future Curator iterations.

## Directory layout

```
lab/runs/<run_id>/
  hypothesis.md     # Researcher-authored, see template
  review.md         # Reviewer-authored verdict
  train.py          # Researcher-authored, self-contained, runs end-to-end
  config.json       # Operator-written; CLI args + env info + library versions
  result.json       # Operator-written; see schema below
  stdout.log        # captured during run
  stderr.log        # captured during run
  tb/               # TensorBoard event files written by train.py
  sanity/           # Stage A artifacts (per-env subdirs with logs)
  fix-N.md          # Operator-written; one per debug retry on Stage A
```

## run_id format

`NNNN-thread-slug` — zero-padded 4-digit sequence number, then a kebab-case fragment
naming the research thread.

Examples: `0001-energy-credit`, `0042-vector-decomp`, `0107-trajectory-em`.

The Operator allocates the run_id at iteration start by inspecting `lab/ledger.jsonl`.

## hypothesis.md (template)

```markdown
---
thread: energy-based-credit
primary_benchmark: ALE/MontezumaRevenge-v5
sanity_envs: [CartPole-v1, Pendulum-v1]   # default; override only with justification
pillar: sparse-long-horizon                # one of: sparse-long-horizon | long-horizon-dense | multi-signal
seeds: [42, 43]                            # ≥2 in early phase
---

# Hypothesis: <short title>

## Claim
2-4 sentences. What is the algorithmic mechanism, and why might it address the named pillar?

## How it differs structurally from PPO and Q-learning
Be specific. What mathematical object is the optimization target?
What signal flows where? Why is this *not* ∇ log π · A or a Bellman update?

## Implementation sketch
5-15 lines of pseudocode. Identify the novel primitives.

## What success would look like
Qualitative. NOT "X% more return." Examples that are acceptable:
- "Training curve is non-monotone, consistent with the multi-modal credit signal."
- "Compute scales sub-linearly in episode length."
- "On multi-signal envs, learns Pareto-distinct policies under different scalarizations."

## Falsification
What observation would convince you this idea is wrong?
```

## train.py contract

`train.py` MUST:

- Be self-contained. Allowed imports: `torch`, `numpy`, `gymnasium`, `dm_control`,
  `mo_gymnasium`, `ale_py`, `tensorboard`, anything in `src/rl_research/`.
- **Forbid**: `stable_baselines3`, `cleanrl` (or copies), `tianshou`, `ray.rllib`,
  `acme`, `coax`, `garage`. Reviewer/Operator block on import detection.
- Accept these CLI flags (parsed by `argparse`):
  - `--env <env_id>` — required
  - `--seed <int>` — required
  - `--total-env-steps <int>` — required
  - `--logdir <path>` — required (where TB event files + `result.json` go)
  - `--max-wallclock-s <int>` — required (Operator passes the budget)
- Honor `--max-wallclock-s` by checking elapsed wallclock at every eval and exiting
  cleanly (writing `result.json`) before the SIGTERM grace deadline.
- Log the required TensorBoard scalars (below).
- Write `result.json` (per-seed) at exit.

## Required TensorBoard scalars

Logged during training, at minimum every `total_env_steps / 100` steps:

- `eval/return_mean` — mean episodic return averaged over ≥10 eval episodes.
- `eval/return_std`
- `eval/return_per_channel/<i>` — only for multi-signal benchmarks; one tag per
  reward channel.
- `train/loss` — primary training loss (whatever the algorithm calls it).
- `progress/env_steps`
- `progress/wallclock_s`

Eval cadence: at least 20 evals over the run.

## result.json schema

The operational schema lives at `lab/result.schema.json`. Conceptually:

```json
{
  "run_id":            "0001-energy-credit",
  "stage":             "A+B",
  "status":            "completed",
  "primary_benchmark": "ALE/MontezumaRevenge-v5",
  "pillar":            "sparse-long-horizon",
  "thread":            "energy-based-credit",
  "seeds":             [42, 43],
  "env_steps":         2000000,
  "wallclock_s":       5398.2,
  "best_return":       145.0,
  "final_return":      138.7,
  "by_seed": {
    "42": {"best_return": 150.0, "final_return": 142.5, "env_steps": 2000000},
    "43": {"best_return": 140.0, "final_return": 135.0, "env_steps": 2000000}
  },
  "sanity": {
    "passed": true,
    "by_env": {
      "CartPole-v1": {"return_random": 22.1, "return_final": 187.4, "passed": true},
      "Pendulum-v1": {"return_random": -1463.0, "return_final": -612.0, "passed": true}
    },
    "retries": 0
  },
  "git_sha":    "fc98d81abc...",
  "started_at": "2026-05-16T11:00:00Z",
  "ended_at":   "2026-05-16T12:30:00Z",
  "deps_lock":  "uv.lock@<sha>",
  "notes":      ""
}
```

`stage` ∈ {`A-only`, `A+B`}. `status` ∈ {`sanity-failed`, `benchmark-failed`,
`killed-budget`, `killed-error`, `completed`}.

## Ledger

After `result.json` is written, the Operator appends one line to `lab/ledger.jsonl`:

```json
{"run_id":"0001-energy-credit","thread":"energy-based-credit","pillar":"sparse-long-horizon","primary_benchmark":"ALE/MontezumaRevenge-v5","status":"completed","best_return":145.0,"wallclock_s":5398,"verdict_curator":null,"hypothesis_path":"lab/runs/0001-energy-credit/hypothesis.md"}
```

`verdict_curator` is `null` until the Curator processes the run.

## Validation

Runtime helpers in `src/rl_research/contract.py`:

- `validate_result_json(path) -> None` — schema-checks and raises on any deviation.
- `append_to_ledger(result_path) -> None` — atomically appends a ledger line.
- `next_run_id(thread_slug) -> str` — allocates the next NNNN sequence number.

The Operator MUST call `validate_result_json` before appending to the ledger.
