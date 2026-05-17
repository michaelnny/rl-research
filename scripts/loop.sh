#!/usr/bin/env bash
# Headless wrapper for the rl-research loop.
#
# Spawns `claude -p '/iterate'` in a sequential loop until any of:
#   - the daily wallclock cap (LOOP_DAILY_HOURS) is hit
#   - lab/HALT_REQUESTED.md exists (any agent — typically the Curator —
#     can write this when the loop has stopped producing signal)
#   - a preflight check returns exit 2 (halt requested) or fails persistently
#   - LOOP_MAX_CONSECUTIVE_FAILS iterations in a row have failed (bug in
#     framework or agent prompts — keep going burns credits for no signal)
#
# Usage:
#   tmux new -s loop 'cd ~/projects/rl-research && scripts/loop.sh'
#   ^B d                 # detach
#   tmux kill-session -t loop
#
# Per-iteration sequence:
#   1. Rotate iterations.log if it has grown beyond LOOP_LOG_MAX_MB
#   2. Run preflight; abort iter (or halt loop) if it fails
#   3. Backup the corpus to lab/.backups/
#   4. Spawn `claude -p '/iterate'` under `timeout LOOP_ITER_MAX_S` — a hung
#      iteration cannot freeze the loop forever
#   5. Cleanup: kill orphan train.py procs, refresh corpus_stats
#   6. Update lab/.heartbeat (so external watchdogs can detect a stuck loop)
#   7. Sleep LOOP_COOLDOWN_S, then repeat
#
# Knobs (env vars):
#   LOOP_DAILY_HOURS             default 16   — total wallclock cap before this script exits
#   LOOP_COOLDOWN_S              default 60   — pause between iterations
#   LOOP_ITER_MAX_S              default 14400 (4h) — hard kill any iteration that exceeds this
#   LOOP_LOG_MAX_MB              default 100  — rotate iterations.log when it exceeds this
#   LOOP_LOG_RETAIN_N            default 5    — number of rotated logs to keep
#   LOOP_PREFLIGHT_RETRIES       default 3    — preflight retry budget before declaring loop unhealthy
#   LOOP_PREFLIGHT_BACKOFF_S     default 300  — sleep between preflight retries
#   LOOP_MAX_CONSECUTIVE_FAILS   default 5    — auto-halt after N back-to-back failed iterations

set -uo pipefail

cd "$(dirname "$0")/.."

LOOP_DAILY_HOURS=${LOOP_DAILY_HOURS:-16}
LOOP_COOLDOWN_S=${LOOP_COOLDOWN_S:-60}
LOOP_ITER_MAX_S=${LOOP_ITER_MAX_S:-14400}
LOOP_LOG_MAX_MB=${LOOP_LOG_MAX_MB:-100}
LOOP_LOG_RETAIN_N=${LOOP_LOG_RETAIN_N:-5}
LOOP_PREFLIGHT_RETRIES=${LOOP_PREFLIGHT_RETRIES:-3}
LOOP_PREFLIGHT_BACKOFF_S=${LOOP_PREFLIGHT_BACKOFF_S:-300}
LOOP_MAX_CONSECUTIVE_FAILS=${LOOP_MAX_CONSECUTIVE_FAILS:-5}

LOG=lab/iterations.log
HEARTBEAT=lab/.heartbeat

mkdir -p lab

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG"
}

heartbeat() {
  # Atomic via tmp+rename so a crash mid-write never leaves a half-written file.
  local tmp="${HEARTBEAT}.tmp"
  printf '%s iter=%s status=%s consecutive_fails=%s\n' \
    "$(date -Iseconds)" "${1:-?}" "${2:-?}" "${3:-0}" > "$tmp"
  mv -f "$tmp" "$HEARTBEAT"
}

rotate_log() {
  [[ -f $LOG ]] || return 0
  local size_mb
  size_mb=$(du -m "$LOG" | awk '{print $1}')
  if (( size_mb >= LOOP_LOG_MAX_MB )); then
    local i=$LOOP_LOG_RETAIN_N
    rm -f "$LOG.$i"
    while (( i > 1 )); do
      [[ -f $LOG.$((i-1)) ]] && mv "$LOG.$((i-1))" "$LOG.$i"
      i=$((i-1))
    done
    mv "$LOG" "$LOG.1"
    log "rotated iterations.log (was ${size_mb} MB; retained ${LOOP_LOG_RETAIN_N})"
  fi
}

cleanup_orphans() {
  # Any train.py from a prior iteration that survived (e.g., grandchild
  # workers, dataloaders, the run_monitor sidecar) gets a SIGTERM and 5s grace,
  # then SIGKILL. We never do this *during* an iteration — only between, after
  # `claude -p` has returned.
  #
  # We match on the run dir prefix rather than the binary name so we also
  # catch sidecars, dataloader workers spawned via spawn(), and any helper
  # processes that import from lab/runs/<id>/.
  local pids
  pids=$(pgrep -f 'lab/runs/.*/(train\.py|run_monitor)' || true)
  pids+=$'\n'
  pids+=$(pgrep -f 'rl_research\.run_monitor' || true)
  pids=$(echo "$pids" | sort -u | xargs)
  if [[ -n "$pids" ]]; then
    log "cleanup: terminating orphan PIDs: $pids"
    # Kill the process *group* of each matching PID so dataloader workers and
    # CUDA helper threads die with the parent.
    for pid in $pids; do
      pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
      if [[ -n "$pgid" ]]; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
      else
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    sleep 5
    for pid in $pids; do
      pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
      if [[ -n "$pgid" ]]; then
        kill -KILL -- "-$pgid" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  fi
}

write_halt() {
  # Idempotent: don't clobber an existing halt file (Curator may have written
  # a richer diagnosis).
  if [[ ! -f lab/HALT_REQUESTED.md ]]; then
    cat > lab/HALT_REQUESTED.md <<EOF
auto-halt by scripts/loop.sh at $(date -Iseconds)
reason: $1
EOF
    log "wrote lab/HALT_REQUESTED.md (auto-halt: $1)"
  fi
}

start_ts=$(date +%s)
deadline=$((start_ts + LOOP_DAILY_HOURS * 3600))

# Trap SIGTERM/SIGINT (e.g., `tmux kill-session`) so we can clean up orphans
# before exiting. Without the trap, a kill leaves train.py / run_monitor
# orphaned and the next iteration's preflight refuses to start.
on_signal() {
  log "loop.sh received signal $1; cleaning up and exiting"
  cleanup_orphans
  heartbeat "$iter" signal-exit "$consecutive_fails"
  exit 130
}
iter=0
consecutive_fails=0
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGINT' INT
trap 'on_signal SIGHUP' HUP

log "loop.sh starting (daily_hours=$LOOP_DAILY_HOURS, iter_max_s=$LOOP_ITER_MAX_S, cooldown_s=$LOOP_COOLDOWN_S, log_max_mb=$LOOP_LOG_MAX_MB)"

heartbeat 0 starting "$consecutive_fails"

while [[ $(date +%s) -lt $deadline ]]; do
  iter=$((iter+1))
  rotate_log

  heartbeat "$iter" preflight "$consecutive_fails"

  # Preflight with retry. If it returns exit 2, that's a halt request —
  # break out entirely. If 1, retry up to N times before declaring loop dead.
  preflight_attempt=0
  preflight_ok=0
  while (( preflight_attempt < LOOP_PREFLIGHT_RETRIES )); do
    preflight_attempt=$((preflight_attempt+1))
    if bash scripts/preflight.sh 2>&1 | tee -a "$LOG"; then
      preflight_ok=1
      break
    fi
    rc=${PIPESTATUS[0]}
    if [[ $rc -eq 2 ]]; then
      log "preflight requested halt; exiting loop"
      heartbeat "$iter" halted "$consecutive_fails"
      exit 0
    fi
    log "preflight attempt $preflight_attempt/$LOOP_PREFLIGHT_RETRIES failed (rc=$rc); sleeping ${LOOP_PREFLIGHT_BACKOFF_S}s"
    sleep "$LOOP_PREFLIGHT_BACKOFF_S"
  done
  if (( preflight_ok == 0 )); then
    log "preflight failed $LOOP_PREFLIGHT_RETRIES times; aborting loop"
    heartbeat "$iter" preflight-dead "$consecutive_fails"
    exit 1
  fi

  bash scripts/ledger_backup.sh 2>&1 | tee -a "$LOG" || true

  log "iteration $iter start"
  heartbeat "$iter" running "$consecutive_fails"

  # `claude -p` does its own internal retry/auth/rate-limit handling; if it
  # exits non-zero, the iteration is over. We log and proceed.
  #
  # `timeout` enforces the hard cap. SIGTERM first, then SIGKILL after 30s,
  # so a half-decent claude can flush its output before we kill it.
  iter_rc=0
  timeout --kill-after=30s --signal=TERM "$LOOP_ITER_MAX_S" \
    claude -p '/iterate' --output-format json 2>&1 | tee -a "$LOG"
  iter_rc=${PIPESTATUS[0]}

  case "$iter_rc" in
    0)
      consecutive_fails=0
      log "iteration $iter done (rc=0)"
      ;;
    124|137)
      consecutive_fails=$((consecutive_fails+1))
      log "iteration $iter TIMEOUT after ${LOOP_ITER_MAX_S}s (rc=$iter_rc); consecutive_fails=$consecutive_fails"
      ;;
    *)
      consecutive_fails=$((consecutive_fails+1))
      log "/iterate exited non-zero (rc=$iter_rc); consecutive_fails=$consecutive_fails"
      ;;
  esac

  cleanup_orphans
  uv run python scripts/corpus_stats.py 2>&1 | tee -a "$LOG" || true
  # NB: do NOT remove lab/.run_id.in_progress here. The marker is the
  # signal to the next iteration's stale-state branch that this iteration
  # crashed mid-flight. /iterate clears it on a clean exit; if it's still
  # present after `claude -p` returns (timeout, crash, OOM-kill), the next
  # iteration writes a killed-error result for the abandoned run_id.

  if (( consecutive_fails >= LOOP_MAX_CONSECUTIVE_FAILS )); then
    write_halt "${consecutive_fails} consecutive iteration failures"
    heartbeat "$iter" auto-halt "$consecutive_fails"
    exit 1
  fi

  heartbeat "$iter" sleeping "$consecutive_fails"
  log "iteration $iter end; sleeping ${LOOP_COOLDOWN_S}s"
  sleep "$LOOP_COOLDOWN_S"
done

heartbeat "$iter" deadline "$consecutive_fails"
log "loop.sh exiting (deadline reached)"
