# Operations manual

This is the day-to-day runbook for operating the autonomous research loop. It
assumes the framework specs (charter, loop, contract) are already familiar; if
not, read those first.

The loop is designed to run **unattended for weeks**. The substrate's job is to
keep going while producing a navigable corpus; your job during the run is to
glance at `make status` once or twice a day and intervene only when the
substrate flags a problem.

---

## Starting the loop

```bash
# From the repo root
make loop                    # creates tmux session 'loop'
tmux attach -t loop          # watch the live output (^B d to detach)
```

The loop runs `scripts/loop.sh`, which:

1. **Rotates `iterations.log`** if it exceeds `LOOP_LOG_MAX_MB` (default 100 MB),
   keeping `LOOP_LOG_RETAIN_N` (default 5) rotations.
2. **Runs `scripts/preflight.sh`** before every iteration. Refuses to start the
   iteration if any check fails.
3. **Backs up the corpus** (ledger + lessons + threads) to
   `lab/.backups/<timestamp>/`, retaining the most recent 14.
4. **Spawns one `claude -p '/iterate'`** under `timeout LOOP_ITER_MAX_S`
   (default 4h) — a hung iteration cannot freeze the loop forever. Default
   covers Stage A (5min) + Stage B (2h) + agent overhead.
5. **Cleans up orphan `train.py` processes** between iterations
   (SIGTERM, then SIGKILL after 5s grace).
6. **Refreshes `lab/CORPUS_STATS.md`** so the next Researcher has a stable
   corpus surface (mode-collapse alerts, status histogram, per-thread state).
7. **Updates `lab/.heartbeat`** at every phase boundary so external watchdogs
   can detect a stuck loop without parsing the log.
8. **Tracks consecutive failures.** After `LOOP_MAX_CONSECUTIVE_FAILS`
   back-to-back failed iterations the wrapper auto-writes
   `lab/HALT_REQUESTED.md` and exits — a runaway broken loop can't burn
   credits indefinitely.
9. **Sleeps `LOOP_COOLDOWN_S`** (default 60) and repeats, until either the
   daily wallclock cap (`LOOP_DAILY_HOURS`, default 16h) is hit or
   `lab/HALT_REQUESTED.md` exists.

Knobs (set via env vars before `make loop`):

| var                         | default | what                                                  |
| --------------------------- | ------- | ----------------------------------------------------- |
| `LOOP_DAILY_HOURS`          | 16      | total wallclock cap before the wrapper exits          |
| `LOOP_COOLDOWN_S`           | 60      | sleep between iterations                              |
| `LOOP_ITER_MAX_S`           | 14400   | hard kill an iteration that exceeds this (default 4h) |
| `LOOP_LOG_MAX_MB`           | 100     | rotate `iterations.log` past this size                |
| `LOOP_LOG_RETAIN_N`         | 5       | rotated logs to keep                                  |
| `LOOP_PREFLIGHT_RETRIES`    | 3       | preflight retries before declaring loop dead          |
| `LOOP_PREFLIGHT_BACKOFF_S`  | 300     | sleep between preflight retries                       |
| `LOOP_MAX_CONSECUTIVE_FAILS`| 5       | auto-halt after N back-to-back failed iterations      |
| `PREFLIGHT_MIN_GB`          | 20      | min free GB on the lab/ partition                     |
| `PREFLIGHT_MAX_INODE_PCT`   | 90      | fail if inode usage exceeds this                      |
| `PREFLIGHT_MIN_MEM_MB`      | 2048    | min available RAM in MB                               |
| `PREFLIGHT_GPU_MAX_MEM`     | 4096    | max MB of GPU memory in use to be considered idle     |
| `BACKUP_RETAIN_N`           | 14      | number of corpus backups to keep                      |

---

## Stopping the loop

Three options, ordered by graceful → hard:

1. **`make stop`** — writes `lab/HALT_REQUESTED.md`. The current iteration
   finishes, then the wrapper exits. The Curator can also write this file
   from within an iteration if it judges the loop has stopped producing signal.
2. **`tmux kill-session -t loop`** — terminates the wrapper between
   iterations. Any `train.py` from a *currently-running* iteration keeps going
   under the orphaned `claude -p` process.
3. **`pkill -f scripts/loop.sh`** — same as above, but force.

After stopping, you can resume with `make loop` once whatever caused the stop
is fixed. The loop has no resume state — it just picks up the next iteration
from the corpus state on disk.

To re-enable after a halt, delete `lab/HALT_REQUESTED.md` first.

---

## Monitoring

### `make status` (snapshot)

The one-screen dashboard. Run it any time, even while the loop is active:

```
== rl-research status ==
2026-05-17T12:34:56+00:00

Loop
  tmux: session 'loop' is RUNNING (attach with: tmux attach -t loop)
  claude -p: PID=12345 (iteration in flight)
  train.py:
    12678 uv run python lab/runs/0042-foo/train.py --env humanoid.run --seed 0 ...

Corpus
  total runs: 23
  status: {'completed': 12, 'sanity-failed': 7, 'killed-error': 4}
  verdict: {'pending': 8, 'dead-end': 9, 'promising': 4, 'inconclusive': 2}
  last 20 threads: {'energy-credit': 5, 'vector-decomp': 4, ...}

Last 5 ledger entries
  0019-foo                 status=completed         verdict=promising  wall=5398s
  ...

GPU
  NVIDIA GeForce RTX 3090 Ti, 11321 MiB, 24564 MiB, 87 %, 73

Disk
  287G available / 961G total (70% used) on /
  lab/runs: 12G
  lab/baselines: 123M
  lab/.backups: 4.2M
```

### `lab/CORPUS_STATS.md` (auto-refreshed, machine-friendly)

Refreshed at the end of every iteration. Read by the Researcher and Curator
prompts via the corpus reading step. Surfaces:

- Total runs, status histogram, pillar histogram, verdict distribution.
- Per-thread tallies + last status + Curator-assigned `verdict_curator`.
- **Mode-collapse warning** when one thread accounts for ≥60% of the last 20
  runs. The Researcher should propose in a different thread; the Curator
  should consider halting if this persists.
- Recent failures grouped by class.
- Last 10 runs with verdicts.

### `tail -f lab/iterations.log`

Per-iteration JSON output from `claude -p` plus all preflight/backup/cleanup
log lines. Rotated at `LOOP_LOG_MAX_MB`.

---

## Checking in: a daily ritual

```bash
make status                                   # corpus + GPU + disk
tail -50 lab/iterations.log                   # what the loop actually did
ls -1 lab/.backups | tail -3                  # backups happening?
cat lab/CORPUS_STATS.md | head -40            # mode-collapse alerts?
```

Healthy signs:

- `total runs` increases roughly in line with elapsed days × iterations/day.
- `status` distribution has at least *some* `completed` (not all
  sanity-failed).
- `verdict` distribution has fewer `pending` than recent total — Curator is
  keeping up.
- No mode-collapse warning at the top of `CORPUS_STATS.md`.
- `lab/.backups/` has fresh entries.
- GPU memory < 24 GB used, utilization >0% during a run.

Worry signs (intervene):

- All recent runs `sanity-failed` → bug in framework primitives or shared
  drift in the Researcher prompt; check the most recent `fix-N.md` files.
- `verdict_curator` stuck at `pending` for many runs → Curator failing
  silently; check the last few iterations in `iterations.log` for errors.
- Mode-collapse warning persisting across multiple iterations → Researcher
  isn't reading the corpus stats; consider whether the prompt needs a nudge,
  or write `lab/HALT_REQUESTED.md` and re-anchor.
- Disk free dropping fast → `lab/runs/<run_id>/sanity/` and `tb/` events can
  be huge for failed runs; archive or delete after Curator review (manual).
- `claude -p` exiting non-zero on every iteration → check
  `lab/iterations.log` tail for auth/rate-limit/permission issues.

---

## In-flight monitoring

Each Stage B run gets an external sidecar (`rl_research.run_monitor`) that
watches the run's TB events and kills `train.py` if any of these fire:

| signal                     | default threshold                             | resulting status   |
| -------------------------- | --------------------------------------------- | ------------------ |
| TB writer silent           | no event for 1200s (20 min) past 5min grace   | `killed-stalled`   |
| `progress/param_checksum`  | unchanged for 3 consecutive checks            | `killed-stalled`   |
| NaN / Inf in any scalar    | 2 consecutive checks                          | `killed-diverged`  |

The 5-min grace period and 20-min stall threshold are deliberately loose:
slow-but-real progress is never killed before its 2h budget expires. The
monitor never writes `result.json` itself — it writes a single-word verdict
to `lab/runs/<run_id>/.monitor_verdict` (`stalled` or `diverged`) and the
Engineer reads that to pick the correct `status`.

Per-run JSON-line postmortem at `lab/runs/<run_id>/monitor.log`:

```
{"ts":1700000000.0,"event":"start","pid":12345,"thresholds":{...}}
{"ts":1700000060.5,"event":"check","action":"continue","tb_age_s":12.3,...}
{"ts":1700000900.7,"event":"check","action":"kill","reason":"stalled:..."}
{"ts":1700000900.8,"event":"kill","verdict":"stalled"}
```

To loosen / tighten the monitor for a specific run, the Engineer passes
flags: `--tb-stale-kill-s`, `--max-frozen-checks`, `--max-nan-checks`,
`--grace-period-s`, `--check-interval-s`.

---

## Common failure modes and recovery

### Preflight keeps failing on disk

```
preflight: FAIL — only 12 GB free on lab/ partition; need >= 20 GB
```

Free up space:

```bash
# Inspect biggest run dirs
du -sh lab/runs/* | sort -h | tail -20

# Archive completed runs' TB events (keep result.json/hypothesis.md/train.py)
for d in lab/runs/<run_ids_to_archive>; do
  rm -rf "$d/tb" "$d/sanity"
done

# Or move whole runs to cold storage
mkdir -p ~/rl-research-cold
mv lab/runs/<run_ids> ~/rl-research-cold/
```

### Preflight reports orphan train.py

```
preflight: FAIL — orphan train.py processes detected (kill them or wait):
  18234 uv run python lab/runs/0019-foo/train.py --env ALE/MontezumaRevenge-v5 ...
```

The previous iteration's `train.py` outlived `claude -p`. Either wait for the
2h cap to fire, or kill manually:

```bash
pkill -TERM -f 'lab/runs/.*/train\.py'
sleep 5
pkill -KILL -f 'lab/runs/.*/train\.py'
```

The loop wrapper does this automatically between iterations, but if `claude -p`
is killed mid-iteration the next preflight may catch the orphan first.

### Preflight reports halt

```
preflight: HALT — lab/HALT_REQUESTED.md present (first line: ...)
```

The Curator (or `make stop`) requested a halt. Read the file:

```bash
cat lab/HALT_REQUESTED.md
```

Decide whether to fix the underlying issue and resume, or to leave the loop
stopped. To resume:

```bash
rm lab/HALT_REQUESTED.md
make loop
```

### `claude -p` exits non-zero immediately

Check the tail of `iterations.log`. Common causes:

- **Auth expired** — `claude login` then resume.
- **Rate-limited** — `claude -p` retries internally; if it surfaces it, the
  daily quota is exhausted. Wait for reset.
- **Permission denied on a tool** — a sub-agent tried something not in its
  `allowed-tools`. Check `.claude/agents/<role>.md` and the operator's
  permission settings.

### Iteration succeeds but the run dir is empty

Likely the Researcher crashed mid-write. The orchestrator's
`lab/.run_id.in_progress` mechanism + the `stale state` gate in `/iterate`
catches this on the next iteration: it logs the abandoned run_id and
proceeds.

If you see a `0042-foo` directory with `hypothesis.md` but no `result.json`
that has been there for >2h, the iteration was killed during the Engineer's
Stage B. Manually:

```bash
# Either delete the dir (clean re-allocation will re-use the slot)
rm -rf lab/runs/0042-foo

# Or write an abandoned result.json and ledger line
uv run python -c "
from rl_research.contract import write_result, append_to_ledger
from datetime import datetime, UTC
import subprocess
git_sha = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
now = datetime.now(UTC)
p = write_result(
    run_id='0042-foo', stage='A-only', status='killed-error',
    primary_benchmark='CartPole-v1', pillar='sparse-long-horizon', thread='foo',
    seeds=[0], env_steps=0, wallclock_s=0.0,
    best_return=0.0, final_return=0.0,
    by_seed={'0': {'best_return': 0.0, 'final_return': 0.0, 'env_steps': 0}},
    sanity={'passed': False, 'by_env': {}, 'retries': 0},
    git_sha=git_sha, started_at=now, ended_at=now,
    notes='manually closed: iteration killed mid-flight',
)
append_to_ledger(p)
"
```

### Ledger has a malformed line

```
preflight: FAIL — lab/ledger.jsonl has malformed lines:
line 47: Expecting value: line 1 column 1 (char 0)
```

Restore from backup:

```bash
ls -1 lab/.backups | tail -3                          # find the latest good backup
cp lab/.backups/<timestamp>/ledger.jsonl lab/        # restore
```

If the malformed line is the last one (most likely — write was interrupted),
you can also just truncate it:

```bash
head -n46 lab/ledger.jsonl > lab/ledger.jsonl.fixed
mv lab/ledger.jsonl.fixed lab/ledger.jsonl
```

### GPU stuck (utilization 0% during a run, OOM in `nvidia-smi`)

```bash
# Inspect
nvidia-smi
fuser -k /dev/nvidia*    # nuclear option — only if no real work is in flight

# Or kill the specific train.py
pkill -KILL -f 'lab/runs/.*/train\.py'
```

The 2h wallclock cap inside `train.py` should catch most hangs, but
not all (e.g., a deadlocked dataloader). After kill, the next preflight
will pass and the next iteration starts clean.

---

## Backups

`scripts/ledger_backup.sh` writes to `lab/.backups/<timestamp>/`:

- `ledger.jsonl`
- `lessons.md`
- `threads/`

Retains the most recent `BACKUP_RETAIN_N` (default 14). Called between
iterations.

To restore:

```bash
ls -1 lab/.backups
cp -r lab/.backups/<timestamp>/* lab/
```

The full per-run dirs (`lab/runs/<run_id>/`) are NOT backed up here — they
are committed to git. The backup is for the synthesized corpus surfaces that
the Curator rewrites in place.

---

## Costs

`claude -p '/iterate' --output-format json` returns `total_cost_usd` per
iteration. Aggregate:

```bash
grep -h '"total_cost_usd"' lab/iterations.log lab/iterations.log.* 2>/dev/null \
  | uv run python -c "
import json, sys
total = 0.0
for line in sys.stdin:
    try: total += json.loads(line.strip()).get('total_cost_usd', 0.0)
    except: pass
print(f'\${total:.2f}')
"
```

This is approximate (we strip JSON-only lines from the log). For exact
accounting, use the Anthropic console.

---

## When to halt manually

Write `lab/HALT_REQUESTED.md` (or run `make stop`) when:

- The corpus stats show mode-collapse persisting across 3+ iterations after
  the Researcher had access to `CORPUS_STATS.md`.
- All recent runs are `sanity-failed` and the same root cause keeps
  recurring.
- You're going to be unavailable for >24h and want to gate continuation on
  a manual review.
- A bug in the framework requires a fix that ripples through `train.py`
  generation.

The Curator may also halt autonomously — its prompt grants that authority.
Read `lab/HALT_REQUESTED.md` for the diagnosis when it does.

---

## Resuming after a long break

```bash
make status                          # corpus state
make stats                           # refresh CORPUS_STATS.md
make preflight                       # nothing leftover from a prior crash?
git status                           # any uncommitted ledger / threads / lessons?
git log --since='1 week ago' --oneline lab/   # what changed in the corpus?
make loop                            # resume
```

If the gap was >1 week, also re-read `docs/charter.md` and `docs/loop.md` —
the agents do this every iteration; you should too.
