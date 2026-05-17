#!/usr/bin/env bash
# One-screen ops dashboard for the rl-research loop.
#
# Prints (in order):
#   - loop process state (tmux session, claude -p PID if running)
#   - HALT/in-progress flags
#   - corpus snapshot (total runs, status histogram, pending verdicts)
#   - latest 5 ledger lines
#   - GPU state
#   - disk free on the lab/ partition
#   - iterations.log size + tail
#
# Usage:
#   scripts/status.sh
#   make status
#
# Read-only. Safe to run while the loop is active.

set -u
cd "$(dirname "$0")/.."

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }

bold "== rl-research status =="
date -Iseconds
echo

# Loop process state.
bold "Loop"
if command -v tmux >/dev/null 2>&1 && tmux has-session -t loop 2>/dev/null; then
  echo "  tmux: session 'loop' is RUNNING (attach with: tmux attach -t loop)"
else
  echo "  tmux: no 'loop' session"
fi
claude_pid=$(pgrep -af 'claude' 2>/dev/null | grep -E '(/iterate| -p )' | awk '{print $1}' | head -n1 || true)
if [[ -n $claude_pid ]]; then
  echo "  claude -p: PID=$claude_pid (iteration in flight)"
else
  echo "  claude -p: not running"
fi
train_pids=$(pgrep -af 'lab/runs/.*/train\.py' 2>/dev/null || true)
if [[ -n $train_pids ]]; then
  echo "  train.py:"
  echo "$train_pids" | sed 's/^/    /'
fi
if [[ -f lab/HALT_REQUESTED.md ]]; then
  echo "  HALT_REQUESTED: $(head -n1 lab/HALT_REQUESTED.md)"
fi
if [[ -f lab/.run_id.in_progress ]]; then
  echo "  in-progress run: $(cat lab/.run_id.in_progress)"
fi
if [[ -f lab/.heartbeat ]]; then
  echo "  heartbeat: $(cat lab/.heartbeat)"
fi
echo

# Corpus snapshot.
bold "Corpus"
if [[ -s lab/ledger.jsonl ]]; then
  python3 -c "
import json, collections
rows = []
with open('lab/ledger.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
n = len(rows)
print(f'  total runs: {n}')
print(f'  status: {dict(collections.Counter(r.get(\"status\",\"?\") for r in rows))}')
verdicts = collections.Counter((r.get('verdict_curator') or 'pending') for r in rows)
print(f'  verdict: {dict(verdicts)}')
threads = collections.Counter(r.get('thread','?') for r in rows[-20:])
print(f'  last 20 threads: {dict(threads)}')
" 2>/dev/null
else
  echo "  ledger empty"
fi
echo

# Latest ledger lines.
bold "Last 5 ledger entries"
if [[ -s lab/ledger.jsonl ]]; then
  tail -n5 lab/ledger.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: r = json.loads(line)
    except json.JSONDecodeError: continue
    rid = r.get('run_id', '?')
    st = r.get('status', '?')
    verdict = r.get('verdict_curator') or 'pending'
    wall = r.get('wallclock_s', 0) or 0
    print(f'  {rid:<24} status={st:<18} verdict={verdict:<12} wall={wall:.0f}s')
" 2>/dev/null
else
  echo "  (none)"
fi
echo

# GPU.
bold "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader \
    | sed 's/^/  /'
else
  echo "  nvidia-smi not available"
fi
echo

# Disk on the lab/ partition.
bold "Disk"
df -h lab/ | tail -n1 | awk '{ printf "  %s available / %s total (%s used) on %s\n", $4, $2, $5, $6 }'
du -sh lab/runs 2>/dev/null | awk '{ printf "  lab/runs: %s\n", $1 }'
du -sh lab/baselines 2>/dev/null | awk '{ printf "  lab/baselines: %s\n", $1 }'
du -sh lab/.backups 2>/dev/null | awk '{ printf "  lab/.backups: %s\n", $1 }'
echo

# Iterations log.
bold "iterations.log"
if [[ -f lab/iterations.log ]]; then
  size=$(stat -c %s lab/iterations.log)
  printf "  size: %s bytes\n" "$size"
  echo "  last 5 lines:"
  tail -n5 lab/iterations.log | sed 's/^/    /'
else
  echo "  (does not exist yet)"
fi
