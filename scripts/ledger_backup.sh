#!/usr/bin/env bash
# Ring-buffer backup of the curated corpus surface.
#
# Backs up under lab/.backups/<timestamp>/:
#   - ledger.jsonl  (the append-only research log)
#   - lessons.md    (Curator-distilled findings)
#   - threads/      (per-thread state)
#
# Retains the most recent N backups (default 14) and prunes the rest.
#
# Called from scripts/loop.sh between iterations and from .claude/commands/iterate.md
# at iteration end. Idempotent: a no-op when the corpus is empty.
#
# Knobs:
#   BACKUP_RETAIN_N   default 14 — number of backups to keep

set -u
cd "$(dirname "$0")/.."

BACKUP_RETAIN_N=${BACKUP_RETAIN_N:-14}
BACKUP_DIR=lab/.backups

mkdir -p "$BACKUP_DIR"

# Skip if there's nothing to back up.
if [[ ! -s lab/ledger.jsonl ]]; then
  echo "ledger_backup: ledger empty, skipping"
  exit 0
fi

# Use nanosecond precision so two backups in the same second never collide.
# Falls back to second precision + a random suffix if `date %N` is unavailable.
ts=$(date -u +%Y%m%dT%H%M%S%NZ 2>/dev/null)
if [[ -z $ts || $ts == *N* ]]; then
  ts=$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM
fi
target="$BACKUP_DIR/$ts"
if [[ -e $target ]]; then
  echo "ledger_backup: $target already exists, skipping" >&2
  exit 0
fi
mkdir -p "$target"

cp -p lab/ledger.jsonl "$target/" 2>/dev/null || true
cp -p lab/lessons.md "$target/" 2>/dev/null || true
if [[ -d lab/threads ]]; then
  cp -rp lab/threads "$target/" 2>/dev/null || true
fi

echo "ledger_backup: wrote $target"

# Prune all but the most recent N entries.
# `ls -1t` sorts by mtime descending; we tail the rest.
to_prune=$(ls -1t "$BACKUP_DIR" 2>/dev/null | tail -n +$((BACKUP_RETAIN_N + 1)) || true)
if [[ -n $to_prune ]]; then
  for d in $to_prune; do
    rm -rf "$BACKUP_DIR/$d"
  done
  echo "ledger_backup: pruned $(echo "$to_prune" | wc -l) old backup(s)"
fi
