#!/usr/bin/env bash
# Lab loop runner. Runs forever (until SIGINT/SIGTERM or `pkill`).
#
# Per iteration:
#   1. Pick the next session number from docs/journal/.
#   2. Render the Claude prompt with that number; run `claude -p` at
#      max-effort Opus inside the repo. Claude writes
#      docs/journal/sessionNNNN-<slug>.md.
#   3. Find the journal entry Claude just created.
#   4. Render the Codex prompt with that path; run `codex exec -p jelly`
#      to append a `## Peer note`.
#   5. Commit whatever changed under a non-verdictive descriptive
#      message.
#   6. Sleep a small jitter so back-to-back failures don't burn API
#      tokens at full speed, and start the next iteration.
#
# Run with:
#     nohup bash lab/run_lab.sh > lab/logs/run.log 2>&1 &
# or in a tmux/screen session. Stop with Ctrl+C or `kill <pid>`.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LAB_DIR="$REPO_ROOT/lab"
JOURNAL_DIR="$REPO_ROOT/docs/journal"
LOG_DIR="$LAB_DIR/logs"
CLAUDE_PROMPT_TEMPLATE="$LAB_DIR/prompts/claude_session.md"
CODEX_PROMPT_TEMPLATE="$LAB_DIR/prompts/codex_peer.md"
# Watchdog upper bound for each agent call (seconds). Far above the normal
# wall-clock of a session — present only to recover from a stuck CLI process
# (e.g. the codex node wrapper failing to propagate native-helper exit). NOT a
# normal-case latency budget; a healthy session finishes in 1-5 minutes.
AGENT_WATCHDOG_SECS=1800

mkdir -p "$LOG_DIR" "$JOURNAL_DIR"

log_run="$LOG_DIR/run.log"

# Single-instance lock: refuse to start if another loop is already running.
# Two concurrent loops race on the journal + the working tree; never desired.
PID_FILE="$LOG_DIR/run.pid"
if [ -f "$PID_FILE" ]; then
  existing_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "lab loop already running as PID $existing_pid (see $PID_FILE). Exiting." >&2
    exit 1
  fi
  # Stale PID file (process died without cleanup) — remove and continue.
  rm -f "$PID_FILE"
fi
echo $$ > "$PID_FILE"

# Branch sanity: we don't want the loop committing to master by accident.
# If we're on master, branch to `lab/auto` once.
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" = "master" ] || [ "$current_branch" = "main" ]; then
  if git show-ref --verify --quiet refs/heads/lab/auto; then
    git checkout lab/auto
  else
    git checkout -b lab/auto
  fi
fi

# Hard-pin: from here on the loop refuses to run if the working tree drifts
# off lab/auto. We check on every iteration too, not just at startup.
expected_branch="lab/auto"
actual_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$actual_branch" != "$expected_branch" ]; then
  echo "lab loop refusing to start: expected branch $expected_branch, on $actual_branch. Exiting." >&2
  rm -f "$PID_FILE"
  exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop starting on branch $(git rev-parse --abbrev-ref HEAD) (pid $$)" >> "$log_run"

# ----- helpers ------------------------------------------------------------- #

next_session_number() {
  # Find the largest sessionNNNN-*.md filename and return NNNN + 1 zero-padded
  # to 4 digits. Returns 0000 if no sessions exist yet.
  local max
  max=$(
    ls "$JOURNAL_DIR" 2>/dev/null \
      | grep -E '^session[0-9]{4}-' \
      | sed -E 's/^session([0-9]{4}).*/\1/' \
      | sort -n \
      | tail -n 1
  )
  if [ -z "$max" ]; then
    printf '0000'
  else
    # Strip leading zeros for arithmetic, then re-pad.
    printf '%04d' $((10#$max + 1))
  fi
}

# Substitute __KEY__ placeholders in a template file. Reads from stdin if
# template path is `-`.
render_template() {
  local template="$1"
  shift
  local content
  content="$(cat "$template")"
  while [ $# -ge 2 ]; do
    local key="$1" val="$2"
    shift 2
    # Use python for portable, safe substitution (no sed-escaping headaches).
    content=$(KEY="$key" VAL="$val" CONTENT="$content" python3 -c '
import os
print(os.environ["CONTENT"].replace("__" + os.environ["KEY"] + "__", os.environ["VAL"]), end="")
')
  done
  printf '%s' "$content"
}

# Watchdog: run "$@" with a wall-clock ceiling of $1 seconds. Inline so we
# don't depend on GNU `timeout` (which is not in macOS base). On exceeded
# watchdog we SIGTERM, sleep 5s, then SIGKILL; the function exits with the
# child's actual exit code, or 124 (matching GNU timeout) if killed.
run_with_watchdog() {
  local secs=$1
  shift
  "$@" &
  local child=$!
  (
    sleep "$secs"
    if kill -0 "$child" 2>/dev/null; then
      kill -TERM "$child" 2>/dev/null
      sleep 5
      kill -KILL "$child" 2>/dev/null
    fi
  ) &
  local watchdog=$!
  local rc=0
  wait "$child" 2>/dev/null
  rc=$?
  # Watchdog still alive means child exited on its own — cancel it. If the
  # watchdog already fired (child killed), kill returns nonzero, which is fine.
  kill "$watchdog" 2>/dev/null
  wait "$watchdog" 2>/dev/null
  return "$rc"
}

# ----- main loop ----------------------------------------------------------- #

trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop stopping" >> "$log_run"; rm -f "$PID_FILE"; exit 0' INT TERM

while true; do
  # Per-iteration branch guard: if something switched the working tree off
  # lab/auto between iterations, stop loudly rather than commit to master.
  iter_branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$iter_branch" != "$expected_branch" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop aborting: branch drifted to $iter_branch (expected $expected_branch)" | tee -a "$log_run" >&2
    rm -f "$PID_FILE"
    exit 1
  fi

  session_num="$(next_session_number)"
  iter_dir="$LOG_DIR/iter-$session_num"
  mkdir -p "$iter_dir"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === session $session_num starting ===" | tee -a "$log_run"

  # 1) Claude session.
  claude_prompt="$(render_template "$CLAUDE_PROMPT_TEMPLATE" SESSION_NUMBER "$session_num")"
  printf '%s' "$claude_prompt" > "$iter_dir/claude_prompt.txt"

  # Take a snapshot of the journal before Claude runs so we can detect what
  # Claude added.
  before_entries="$(ls "$JOURNAL_DIR" 2>/dev/null | sort)"

  set +e
  # NOTE: --add-dir is variadic (<directories...>), so any positional that
  # follows it gets consumed as another directory and the prompt is lost.
  # Pipe the prompt on stdin instead. The wrapping watchdog is NOT a normal-
  # case budget — a healthy session is 1-5 min; it exists only to recover
  # from a stuck CLI process.
  printf '%s' "$claude_prompt" | run_with_watchdog "$AGENT_WATCHDOG_SECS" claude -p \
    --bare \
    --system-prompt-file "$LAB_DIR/prompts/claude_system.md" \
    --no-session-persistence \
    --model opus \
    --effort max \
    --permission-mode bypassPermissions \
    --add-dir "$REPO_ROOT/docs" \
    --add-dir "$REPO_ROOT/experiments" \
    --add-dir "$REPO_ROOT/lab" \
    --add-dir "$REPO_ROOT/src" \
    --add-dir "$REPO_ROOT/tests" \
    > "$iter_dir/claude_stdout.txt" 2> "$iter_dir/claude_stderr.txt"
  claude_exit=$?
  set -e
  echo "claude exit=$claude_exit" >> "$log_run"

  after_entries="$(ls "$JOURNAL_DIR" 2>/dev/null | sort)"
  new_entry="$(comm -13 <(echo "$before_entries") <(echo "$after_entries") | grep -E "^session${session_num}-.*\.md$" | head -n 1 || true)"

  if [ -z "$new_entry" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Claude did not create a journal entry; logging and moving on" | tee -a "$log_run"
    # Commit anything else Claude might have done so we don't carry it forward.
    git add -A
    if ! git diff --cached --quiet; then
      git commit -m "session $session_num: claude produced no journal entry, recording side effects" >> "$log_run" 2>&1 || true
    fi
    sleep 30
    continue
  fi

  entry_path="docs/journal/$new_entry"
  echo "session $session_num entry: $entry_path" | tee -a "$log_run"

  # 2) Codex peer note.
  codex_prompt="$(render_template "$CODEX_PROMPT_TEMPLATE" JOURNAL_ENTRY_PATH "$entry_path")"
  printf '%s' "$codex_prompt" > "$iter_dir/codex_prompt.txt"

  set +e
  # Watchdog (NOT a normal-case budget). Codex peer review typically completes
  # in 1-3 min; the long ceiling is only to recover from a stuck CLI process
  # (the node wrapper occasionally fails to propagate exit from the native
  # helper, leaving codex hung indefinitely).
  run_with_watchdog "$AGENT_WATCHDOG_SECS" codex exec \
    -p jelly \
    -c "model_instructions_file=\"$LAB_DIR/prompts/codex_system.md\"" \
    --sandbox workspace-write \
    --dangerously-bypass-approvals-and-sandbox \
    -C "$REPO_ROOT" \
    --skip-git-repo-check \
    "$codex_prompt" \
    < /dev/null \
    > "$iter_dir/codex_stdout.txt" 2> "$iter_dir/codex_stderr.txt"
  codex_exit=$?
  set -e
  echo "codex exit=$codex_exit" >> "$log_run"

  # 3) Commit. Descriptive, not verdictive. We pull a one-line summary from
  # the entry's first non-heading non-empty line, falling back to the slug.
  slug="$(echo "$new_entry" | sed -E "s/^session${session_num}-(.*)\.md$/\1/")"
  summary="$(awk '
    /^# / { next }
    /^date:/ { next }
    /^session kind:/ { next }
    /^## / { next }
    NF > 0 { print; exit }
  ' "$entry_path" | head -c 100)"
  if [ -z "$summary" ]; then
    summary="$slug"
  fi
  msg="session $session_num: $slug — $summary"

  git add -A
  if git diff --cached --quiet; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: nothing to commit (entry not written?)" | tee -a "$log_run"
  else
    git commit -m "$msg" >> "$log_run" 2>&1 || echo "commit failed" | tee -a "$log_run"
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === session $session_num done ===" | tee -a "$log_run"

  # 4) Small jitter so a failure mode doesn't hammer the API back-to-back.
  sleep $(( 5 + RANDOM % 20 ))
done
