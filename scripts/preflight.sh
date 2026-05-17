#!/usr/bin/env bash
# Pre-flight health check before each iteration.
#
# Refuses to start an iteration if any of the following are true:
#   - lab/HALT_REQUESTED.md exists
#   - GPU not available or already saturated
#   - Disk free < PREFLIGHT_MIN_GB on the partition holding lab/
#   - Inode usage > PREFLIGHT_MAX_INODE_PCT
#   - Available RAM < PREFLIGHT_MIN_MEM_MB
#   - claude CLI not on PATH or not authenticated
#   - Orphan train.py processes from a previous iteration are still alive
#   - Existing ledger.jsonl has malformed lines
#   - Required baseline files missing OR empty / not valid JSON
#
# Used by scripts/loop.sh and the /iterate command. Exit codes:
#   0 = all clear, OK to start an iteration
#   1 = blocking failure (must be resolved before next iteration)
#   2 = halt requested (the wrapper should stop entirely, not retry)
#
# Knobs:
#   PREFLIGHT_MIN_GB         default 20    — min free GB on the lab/ partition
#   PREFLIGHT_MAX_INODE_PCT  default 90    — fail if used inodes >= this %
#   PREFLIGHT_MIN_MEM_MB     default 2048  — min available memory in MB
#   PREFLIGHT_GPU_MAX_MEM    default 4096  — max MB of GPU memory in use to be considered idle
#   PREFLIGHT_SKIP_GPU       default 0     — set to 1 to skip the GPU check (CI/dry-run)
#   PREFLIGHT_SKIP_AUTH      default 0     — set to 1 to skip the claude-auth check
#   PREFLIGHT_SKIP_BASELINES default 0     — set to 1 to skip the baseline-presence check

set -u

cd "$(dirname "$0")/.."

PREFLIGHT_MIN_GB=${PREFLIGHT_MIN_GB:-20}
PREFLIGHT_MAX_INODE_PCT=${PREFLIGHT_MAX_INODE_PCT:-90}
PREFLIGHT_MIN_MEM_MB=${PREFLIGHT_MIN_MEM_MB:-2048}
PREFLIGHT_GPU_MAX_MEM=${PREFLIGHT_GPU_MAX_MEM:-4096}
PREFLIGHT_SKIP_GPU=${PREFLIGHT_SKIP_GPU:-0}
PREFLIGHT_SKIP_AUTH=${PREFLIGHT_SKIP_AUTH:-0}
PREFLIGHT_SKIP_BASELINES=${PREFLIGHT_SKIP_BASELINES:-0}

fail() {
  echo "preflight: FAIL — $*" >&2
  exit 1
}

halt() {
  echo "preflight: HALT — $*" >&2
  exit 2
}

# Pick a python interpreter that does NOT require uv. uv venv setup is heavy
# and breaks if uv.lock or .venv get corrupted; preflight must keep working
# even when uv is broken so we can diagnose the corpus.
PY=$(command -v python3 || command -v python || true)
[[ -n $PY ]] || fail "neither python3 nor python on PATH; cannot validate ledger"

# 1. Halt request takes priority over everything else.
if [[ -f lab/HALT_REQUESTED.md ]]; then
  halt "lab/HALT_REQUESTED.md present (first line: $(head -n1 lab/HALT_REQUESTED.md))"
fi

# 2. Disk space on the partition that holds lab/.
free_kb=$(df --output=avail lab/ | tail -n1 | tr -d ' ')
free_gb=$((free_kb / 1024 / 1024))
if [[ $free_gb -lt $PREFLIGHT_MIN_GB ]]; then
  fail "only ${free_gb} GB free on lab/ partition; need >= ${PREFLIGHT_MIN_GB} GB"
fi

# 3. Inode availability — disk-free can be fine while the partition is full of
#    tiny files (TB events, fix-N.md, etc.), at which point new files fail with
#    ENOSPC even though `df -h` looks healthy.
inode_pct_raw=$(df --output=ipcent lab/ | tail -n1 | tr -d '% ')
if [[ -n $inode_pct_raw && $inode_pct_raw =~ ^[0-9]+$ ]]; then
  if (( inode_pct_raw >= PREFLIGHT_MAX_INODE_PCT )); then
    fail "inodes ${inode_pct_raw}% used on lab/ partition; >= ${PREFLIGHT_MAX_INODE_PCT}% threshold"
  fi
fi

# 4. Available RAM. A loop running for weeks can leak (mostly via train.py
#    crashes); if the host has only a few hundred MB free, the next iteration's
#    Engineer will OOM mid-build.
mem_mb=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo "")
if [[ -n $mem_mb && $mem_mb =~ ^[0-9]+$ ]]; then
  if (( mem_mb < PREFLIGHT_MIN_MEM_MB )); then
    fail "only ${mem_mb} MB RAM available; need >= ${PREFLIGHT_MIN_MEM_MB} MB"
  fi
fi

# 5. GPU available and idle.
used_mb=""
if [[ $PREFLIGHT_SKIP_GPU -eq 0 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found; GPU required"
  fi
  used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if [[ -z $used_mb ]]; then
    fail "could not read GPU memory usage from nvidia-smi"
  fi
  if [[ $used_mb -gt $PREFLIGHT_GPU_MAX_MEM ]]; then
    fail "GPU memory in use is ${used_mb} MB > ${PREFLIGHT_GPU_MAX_MEM} MB; another process may be running"
  fi
fi

# 6. claude CLI present + authenticated. A weeks-long loop where auth has
#    silently expired wastes credits on every iteration; check up front.
if [[ $PREFLIGHT_SKIP_AUTH -eq 0 ]]; then
  if ! command -v claude >/dev/null 2>&1; then
    fail "claude CLI not on PATH; the loop wrapper cannot spawn iterations"
  fi
  # `claude --version` exits 0 even when not logged in, so we can't use it as
  # an auth probe. We trust that `claude -p` will surface auth errors via
  # non-zero exit, and the consecutive-failure halt in loop.sh will catch
  # repeated auth failures. We only check the CLI is callable here.
fi

# 7. Orphan train.py from a prior iteration. The Engineer never leaves
#    processes alive across iterations; if we see one, the prior iteration was
#    killed abruptly and we must clean up before starting a new one. We DO NOT
#    kill here — that's the loop wrapper's job; we only flag it.
orphans=$(pgrep -af 'lab/runs/.*/train\.py' || true)
if [[ -n $orphans ]]; then
  fail "orphan train.py processes detected (kill them or wait):
${orphans}"
fi

# 8. Ledger validates as JSONL.
if [[ -f lab/ledger.jsonl ]]; then
  bad_line=$("$PY" - <<'PYEOF' 2>&1
import json, sys
ok = True
with open('lab/ledger.jsonl') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'line {i}: {e}')
            ok = False
            break
sys.exit(0 if ok else 1)
PYEOF
  ) || fail "lab/ledger.jsonl has malformed lines:
${bad_line}"
fi

# 9. Required baselines present AND parse as JSON. The Engineer's sanity gate
#    compares against random.json; a 0-byte file from a partial write would
#    pass a simple existence test but break Stage A on every run.
if [[ $PREFLIGHT_SKIP_BASELINES -eq 0 ]]; then
  rj=lab/baselines/random.json
  [[ -f $rj ]] || fail "lab/baselines/random.json missing"
  [[ -s $rj ]] || fail "lab/baselines/random.json is empty"
  if ! "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$rj" >/dev/null 2>&1; then
    fail "lab/baselines/random.json is not valid JSON"
  fi
fi

# 10. Sanity: ensure the corpus directories exist.
mkdir -p lab/runs lab/threads lab/.backups

echo "preflight: OK (free=${free_gb}GB, inodes=${inode_pct_raw:-?}%, mem=${mem_mb:-?}MB, gpu_used=${used_mb:-skipped}MB)"
exit 0
