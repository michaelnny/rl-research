#!/usr/bin/env bash
# Lab loop runner. Runs forever (until SIGINT/SIGTERM or `pkill`).
#
# Per iteration:
#   1. Pick the next session number from docs/journal/.
#   2. If enough Claude sessions have accumulated since the last Codex
#      steering memo, run `codex -a never exec -p hai` to write
#      docs/journal/sessionNNNN-codex-steering.md, commit it, and move on.
#   3. Otherwise render the Claude prompt with that number; run `claude -p`.
#      Claude writes docs/journal/sessionNNNN-<slug>.md.
#   4. Find the journal entry Claude just created.
#   5. Render the Codex peer prompt with that path; run
#      `codex -a never exec -p hai`
#      to append a `## Peer note`.
#   6. Commit whatever changed under a non-verdictive descriptive message.
#   7. Sleep a small jitter so back-to-back failures don't burn API tokens at
#      full speed, and start the next iteration.
#
# Run with:
#     nohup bash lab/run_lab.sh > lab/logs/console.log 2>&1 &
# or in a tmux/screen session. Stop with Ctrl+C or `kill <pid>`.

set +e
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LAB_DIR="$REPO_ROOT/lab"
JOURNAL_DIR="$REPO_ROOT/docs/journal"
LOG_DIR="$LAB_DIR/logs"
CLAUDE_PROMPT_TEMPLATE="$LAB_DIR/prompts/claude_session.md"
CODEX_PROMPT_TEMPLATE="$LAB_DIR/prompts/codex_peer.md"
CODEX_STEERING_PROMPT_TEMPLATE="$LAB_DIR/prompts/codex_steering.md"
CODEX_STEERING_SYSTEM_PROMPT="$LAB_DIR/prompts/codex_steering_system.md"
CODEX_PROFILE="${CODEX_PROFILE:-hai}"
CODEX_ENV_KEY="${CODEX_ENV_KEY:-HAI_OPENAI_API_KEY}"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
CODEX_PROFILE_CONFIG="$CODEX_HOME_DIR/${CODEX_PROFILE}.config.toml"
STEERING_INTERVAL="${STEERING_INTERVAL:-5}"
# Watchdog upper bound for each agent call (seconds). Far above the normal
# wall-clock of a session — present only to recover from a stuck CLI process
# (e.g. the codex node wrapper failing to propagate native-helper exit). NOT a
# normal-case latency budget; a healthy session finishes in 1-5 minutes.
AGENT_WATCHDOG_SECS=1800
ACTIVE_CHILD=""

mkdir -p "$LOG_DIR" "$JOURNAL_DIR"

log_run="$LOG_DIR/run.log"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "lab loop refusing to start: required command not found: $1" >&2
    exit 1
  fi
}

require_file() {
  if [ ! -r "$1" ]; then
    echo "lab loop refusing to start: required file is missing or unreadable: $1" >&2
    exit 1
  fi
}

worktree_status() {
  git status --porcelain --untracked-files=all
}

abort_if_dirty() {
  local reason dirty
  reason="$1"
  dirty="$(worktree_status)"
  if [ -n "$dirty" ]; then
    echo "$reason" >&2
    printf '%s\n' "$dirty" >&2
    exit 1
  fi
}

changed_paths() {
  {
    git diff --name-only
    git diff --cached --name-only
    git ls-files --others --exclude-standard
  } | sort -u
}

assert_no_substrate_changes() {
  local paths
  paths="$(changed_paths | grep -E '^src/rlh_bench(/|$)' || true)"
  if [ -n "$paths" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] refusing to commit substrate changes under src/rlh_bench:" | tee -a "$log_run" >&2
    printf '%s\n' "$paths" | tee -a "$log_run" >&2
    exit 1
  fi
}

assert_only_changed_path() {
  local allowed paths unexpected
  allowed="$1"
  paths="$(changed_paths)"
  unexpected="$(printf '%s\n' "$paths" | sed '/^$/d' | grep -vxF "$allowed" || true)"
  if [ -n "$unexpected" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] expected only $allowed to change, but saw:" | tee -a "$log_run" >&2
    printf '%s\n' "$unexpected" | tee -a "$log_run" >&2
    exit 1
  fi
}

assert_no_other_journal_changes() {
  local allowed paths
  allowed="$1"
  paths="$(changed_paths | grep -E '^docs/journal/' | grep -vxF "$allowed" || true)"
  if [ -n "$paths" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] refusing to commit edits to journal files other than $allowed:" | tee -a "$log_run" >&2
    printf '%s\n' "$paths" | tee -a "$log_run" >&2
    exit 1
  fi
}

for cmd in git claude codex python3 pgrep mktemp awk comm grep sed sort tail ps; do
  require_command "$cmd"
done

for required_file in \
  "$CLAUDE_PROMPT_TEMPLATE" \
  "$LAB_DIR/prompts/claude_system.md" \
  "$CODEX_PROMPT_TEMPLATE" \
  "$LAB_DIR/prompts/codex_system.md" \
  "$CODEX_STEERING_PROMPT_TEMPLATE" \
  "$CODEX_STEERING_SYSTEM_PROMPT"; do
  require_file "$required_file"
done

require_file "$CODEX_PROFILE_CONFIG"

if ! [[ "$CODEX_ENV_KEY" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "lab loop refusing to start: CODEX_ENV_KEY is not a valid environment variable name: $CODEX_ENV_KEY" >&2
  exit 1
fi

# Single-instance lock: refuse to start if another loop is already running.
# Two concurrent loops race on the journal + the working tree; never desired.
#
# The lock is process-name-aware: `kill -0 $pid` alone is not enough after a
# crash because PIDs get recycled. We only treat the PID file as live if the
# recorded PID still belongs to a run_lab.sh process.
PID_FILE="$LOG_DIR/run.pid"
pid_belongs_to_this_loop() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q "run_lab\.sh" || return 1
  return 0
}

if [ "${1:-}" = "--unlock" ]; then
  if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
    echo "removed $PID_FILE"
  else
    echo "no PID file at $PID_FILE"
  fi
  exit 0
fi

if [ -f "$PID_FILE" ]; then
  existing_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ "$existing_pid" = "$$" ]; then
    # Backward compatibility with the old README launch snippet, which wrote
    # `$!` to run.pid outside the script.
    :
  elif pid_belongs_to_this_loop "$existing_pid"; then
    echo "lab loop already running as PID $existing_pid (see $PID_FILE). Exiting." >&2
    echo "if you're sure it isn't, run: bash lab/run_lab.sh --unlock" >&2
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi
echo $$ > "$PID_FILE"
cleanup_pid_file() {
  if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

# Branch sanity: we don't want the loop committing to master by accident.
# If we're on master, branch to `lab/auto` once.
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" = "master" ] || [ "$current_branch" = "main" ]; then
  abort_if_dirty "lab loop refusing to auto-branch from $current_branch with uncommitted changes. Commit or stash them first."
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
  exit 1
fi

abort_if_dirty "lab loop refusing to start: working tree is not clean. Commit or stash human changes before launching the autonomous loop."

if [ -z "${!CODEX_ENV_KEY:-}" ]; then
  echo "lab loop refusing to start: missing required Codex API env var $CODEX_ENV_KEY for profile $CODEX_PROFILE" >&2
  exit 1
fi
if ! [[ "$STEERING_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "lab loop refusing to start: STEERING_INTERVAL must be a positive integer, got $STEERING_INTERVAL" >&2
  exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop starting on branch $(git rev-parse --abbrev-ref HEAD) (pid $$)" >> "$log_run"

legacy_peer_gaps=$(grep -Rsl '<!-- Codex appends here -->' "$JOURNAL_DIR"/session*.md 2>/dev/null | wc -l | tr -d ' ')
if [ "${legacy_peer_gaps:-0}" -gt 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] warning: $legacy_peer_gaps existing journal entries still contain Codex peer placeholders; new entries are enforced, old entries are not backfilled" | tee -a "$log_run" >&2
fi

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
      | tail -n 1 \
      || true
  )
  if [ -z "$max" ]; then
    printf '0000'
  else
    # Strip leading zeros for arithmetic, then re-pad.
    printf '%04d' $((10#$max + 1))
  fi
}

latest_steering_number() {
  local max
  max=$(
    ls "$JOURNAL_DIR" 2>/dev/null \
      | grep -E '^session[0-9]{4}-codex-steering\.md$' \
      | sed -E 's/^session([0-9]{4}).*/\1/' \
      | sort -n \
      | tail -n 1 \
      || true
  )
  if [ -z "$max" ]; then
    printf '%d' -1
  else
    printf '%d' $((10#$max))
  fi
}

regular_sessions_since_latest_steering() {
  local latest count
  latest="$(latest_steering_number)"
  count=0
  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    case "$entry" in
      *-codex-steering.md) continue ;;
    esac
    local n
    n="$(echo "$entry" | sed -E 's/^session([0-9]{4}).*/\1/')"
    if [ $((10#$n)) -gt "$latest" ]; then
      count=$((count + 1))
    fi
  done < <(ls "$JOURNAL_DIR" 2>/dev/null | grep -E '^session[0-9]{4}-.*\.md$' || true)
  printf '%d' "$count"
}

should_run_steering() {
  local since
  since="$(regular_sessions_since_latest_steering)"
  [ "$since" -ge "$STEERING_INTERVAL" ]
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

kill_tree() {
  local signal pid children child
  signal="$1"
  pid="$2"
  children="$(pgrep -P "$pid" 2>/dev/null || true)"
  for child in $children; do
    kill_tree "$signal" "$child"
  done
  kill "-$signal" "$pid" 2>/dev/null || true
}

# Watchdog: run "$@" with a wall-clock ceiling of $1 seconds. Inline so we
# don't depend on GNU `timeout` (which is not in macOS base). The function
# returns the command's actual exit code, or 124 (matching GNU timeout) when
# the watchdog had to kill it.
run_with_watchdog() {
  local secs=$1
  shift
  local child watchdog rc sentinel stdin_copy
  sentinel=$(mktemp -t rlh_wd_XXXXXX) || return 2
  stdin_copy=$(mktemp -t rlh_stdin_XXXXXX) || {
    rm -f "$sentinel"
    return 2
  }
  # Bash redirects stdin for background jobs to /dev/null in the non-job-control
  # mode used by scripts. Buffer the caller's stdin first so `claude -p` and
  # `codex exec` still receive their prompt while we keep a concrete child PID
  # for watchdog/signal cleanup.
  cat > "$stdin_copy"
  "$@" < "$stdin_copy" &
  child=$!
  ACTIVE_CHILD="$child"
  (
    sleep "$secs"
    if kill -0 "$child" 2>/dev/null; then
      touch "$sentinel.killed"
      kill_tree TERM "$child"
      sleep 5
      kill_tree KILL "$child"
    fi
  ) &
  watchdog=$!
  wait "$child" 2>/dev/null
  rc=$?
  if [ "${ACTIVE_CHILD:-}" = "$child" ]; then
    ACTIVE_CHILD=""
  fi
  kill "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  if [ -e "$sentinel.killed" ]; then
    rc=124
  fi
  rm -f "$sentinel" "$sentinel.killed" "$stdin_copy"
  return "$rc"
}

stop_active_child() {
  if [ -n "${ACTIVE_CHILD:-}" ] && kill -0 "$ACTIVE_CHILD" 2>/dev/null; then
    kill_tree TERM "$ACTIVE_CHILD"
    sleep 5
    kill_tree KILL "$ACTIVE_CHILD"
  fi
}

handle_signal() {
  local signal code
  signal="$1"
  code="$2"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop stopping after $signal" >> "$log_run"
  stop_active_child
  exit "$code"
}

# ----- main loop ----------------------------------------------------------- #

trap 'handle_signal SIGINT 130' INT
trap 'handle_signal SIGTERM 143' TERM

while true; do
  # Per-iteration branch guard: if something switched the working tree off
  # lab/auto between iterations, stop loudly rather than commit to master.
  iter_branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$iter_branch" != "$expected_branch" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop aborting: branch drifted to $iter_branch (expected $expected_branch)" | tee -a "$log_run" >&2
    exit 1
  fi

  abort_if_dirty "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop aborting: working tree became dirty before starting a new session. Resolve these changes first."

  session_num="$(next_session_number)"
  iter_dir="$LOG_DIR/iter-$session_num"
  mkdir -p "$iter_dir"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === session $session_num starting ===" | tee -a "$log_run"

  if should_run_steering; then
    steering_entry_path="docs/journal/session${session_num}-codex-steering.md"
    echo "session $session_num steering: $steering_entry_path" | tee -a "$log_run"
    steering_prompt="$(render_template "$CODEX_STEERING_PROMPT_TEMPLATE" SESSION_NUMBER "$session_num" JOURNAL_ENTRY_PATH "$steering_entry_path" STEERING_INTERVAL "$STEERING_INTERVAL")"
    printf '%s' "$steering_prompt" > "$iter_dir/codex_steering_prompt.txt"

    run_with_watchdog "$AGENT_WATCHDOG_SECS" codex -a never exec \
      -p "$CODEX_PROFILE" \
      -c "model_instructions_file=\"$CODEX_STEERING_SYSTEM_PROMPT\"" \
      --sandbox workspace-write \
      -C "$REPO_ROOT" \
      --skip-git-repo-check \
      --ephemeral \
      "$steering_prompt" \
      < /dev/null \
      > "$iter_dir/codex_steering_stdout.txt" 2> "$iter_dir/codex_steering_stderr.txt"
    steering_exit=$?
    echo "codex steering exit=$steering_exit" >> "$log_run"

    if [ "$steering_exit" -ne 0 ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Codex steering failed with exit $steering_exit; leaving changes uncommitted and stopping" | tee -a "$log_run" >&2
      exit 1
    fi
    if [ ! -s "$steering_entry_path" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Codex steering did not create $steering_entry_path; stopping" | tee -a "$log_run" >&2
      exit 1
    fi
    if ! grep -q '^session kind: steer$' "$steering_entry_path"; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: steering entry is missing 'session kind: steer'; stopping" | tee -a "$log_run" >&2
      exit 1
    fi
    if ! grep -q '^author: Codex$' "$steering_entry_path"; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: steering entry is missing 'author: Codex'; stopping" | tee -a "$log_run" >&2
      exit 1
    fi
    if grep -q '<!-- Codex appends here -->\|^## Peer note' "$steering_entry_path"; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: steering entry looks like a peer-reviewed Claude entry; stopping" | tee -a "$log_run" >&2
      exit 1
    fi
    assert_only_changed_path "$steering_entry_path"

    git add -A
    if git diff --cached --quiet; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: steering produced nothing to commit; stopping" | tee -a "$log_run" >&2
      exit 1
    else
      if ! git commit -m "session $session_num: codex steering memo" >> "$log_run" 2>&1; then
        echo "commit failed" | tee -a "$log_run" >&2
        exit 1
      fi
    fi
    abort_if_dirty "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: working tree still dirty after steering commit; stopping"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === session $session_num steering done ===" | tee -a "$log_run"
    sleep $(( 5 + RANDOM % 20 ))
    continue
  fi

  # 1) Claude session.
  claude_prompt="$(render_template "$CLAUDE_PROMPT_TEMPLATE" SESSION_NUMBER "$session_num")"
  printf '%s' "$claude_prompt" > "$iter_dir/claude_prompt.txt"

  # Take a snapshot of the journal before Claude runs so we can detect what
  # Claude added.
  before_entries="$(ls "$JOURNAL_DIR" 2>/dev/null | sort)"

  # NOTE: --add-dir is variadic (<directories...>), so any positional that
  # follows it gets consumed as another directory and the prompt is lost.
  # Feed the prompt on stdin from a file. Do not pipe into run_with_watchdog;
  # pipeline subshell behavior makes process supervision brittle on macOS bash.
  run_with_watchdog "$AGENT_WATCHDOG_SECS" claude -p \
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
    < "$iter_dir/claude_prompt.txt" \
    > "$iter_dir/claude_stdout.txt" 2> "$iter_dir/claude_stderr.txt"
  claude_exit=$?
  echo "claude exit=$claude_exit" >> "$log_run"

  after_entries="$(ls "$JOURNAL_DIR" 2>/dev/null | sort)"
  new_entry="$(comm -13 <(echo "$before_entries") <(echo "$after_entries") | grep -E "^session${session_num}-.*\.md$" | head -n 1 || true)"

  if [ -z "$new_entry" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Claude did not create a journal entry; leaving changes uncommitted and stopping" | tee -a "$log_run" >&2
    exit 1
  fi

  case "$new_entry" in
    *-codex-steering.md)
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Claude created reserved steering filename $new_entry; stopping" | tee -a "$log_run" >&2
      exit 1
      ;;
  esac

  entry_path="docs/journal/$new_entry"
  echo "session $session_num entry: $entry_path" | tee -a "$log_run"

  # 2) Codex peer note.
  codex_prompt="$(render_template "$CODEX_PROMPT_TEMPLATE" JOURNAL_ENTRY_PATH "$entry_path")"
  printf '%s' "$codex_prompt" > "$iter_dir/codex_prompt.txt"

  # Watchdog (NOT a normal-case budget). Codex peer review typically completes
  # in 1-3 min; the long ceiling is only to recover from a stuck CLI process
  # (the node wrapper occasionally fails to propagate exit from the native
  # helper, leaving codex hung indefinitely).
  run_with_watchdog "$AGENT_WATCHDOG_SECS" codex -a never exec \
    -p "$CODEX_PROFILE" \
    -c "model_instructions_file=\"$LAB_DIR/prompts/codex_system.md\"" \
    --sandbox workspace-write \
    -C "$REPO_ROOT" \
    --skip-git-repo-check \
    --ephemeral \
    "$codex_prompt" \
    < /dev/null \
    > "$iter_dir/codex_stdout.txt" 2> "$iter_dir/codex_stderr.txt"
  codex_exit=$?
  echo "codex exit=$codex_exit" >> "$log_run"

  if [ "$codex_exit" -ne 0 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Codex failed with exit $codex_exit; leaving changes uncommitted and stopping" | tee -a "$log_run" >&2
    exit 1
  fi

  if ! grep -q '^## Peer note' "$entry_path"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Codex/Claude entry has no ## Peer note section; leaving changes uncommitted and stopping" | tee -a "$log_run" >&2
    exit 1
  fi

  if grep -q '<!-- Codex appends here -->' "$entry_path"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: Codex left peer-note placeholder unchanged; leaving changes uncommitted and stopping" | tee -a "$log_run" >&2
    exit 1
  fi

  assert_no_substrate_changes
  assert_no_other_journal_changes "$entry_path"

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
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: nothing to commit (entry not written?); stopping" | tee -a "$log_run" >&2
    exit 1
  else
    if ! git commit -m "$msg" >> "$log_run" 2>&1; then
      echo "commit failed" | tee -a "$log_run" >&2
      exit 1
    fi
  fi
  abort_if_dirty "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session $session_num: working tree still dirty after commit; stopping"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === session $session_num done ===" | tee -a "$log_run"

  # 4) Small jitter so a failure mode doesn't hammer the API back-to-back.
  sleep $(( 5 + RANDOM % 20 ))
done
