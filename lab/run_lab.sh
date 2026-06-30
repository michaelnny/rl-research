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

mkdir -p "$LOG_DIR" "$JOURNAL_DIR"

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

log_run="$LOG_DIR/run.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop starting on branch $(git rev-parse --abbrev-ref HEAD)" >> "$log_run"

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

# ----- main loop ----------------------------------------------------------- #

trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lab loop stopping" >> "$log_run"; exit 0' INT TERM

while true; do
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
  claude -p \
    --model opus \
    --effort max \
    --permission-mode bypassPermissions \
    --append-system-prompt "You are in the rl-research lab. Follow lab/prompts/claude_session.md as your task for this session." \
    "$claude_prompt" \
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
  codex exec \
    -p jelly \
    --sandbox workspace-write \
    --dangerously-bypass-approvals-and-sandbox \
    -C "$REPO_ROOT" \
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
