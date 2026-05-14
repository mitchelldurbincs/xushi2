#!/usr/bin/env bash
# Codex Kanban Monitor — long-running autonomous task keeper
# Runs in a loop, monitoring the xushi2 kanban board and fixing blockers

set -euo pipefail

WORKDIR="/home/aspect/source/personal/xushi2"
LOGDIR="/home/aspect/.hermes/kanban/boards/xushi2/logs"
DB="/home/aspect/.hermes/kanban/boards/xushi2/kanban.db"
PROMPT_FILE="$WORKDIR/GOAL_INSTRUCTIONS.md"
MONITOR_LOG="$LOGDIR/codex-monitor.log"

mkdir -p "$LOGDIR"

echo "=== Codex Kanban Monitor started at $(date -Iseconds) ===" >> "$MONITOR_LOG"
echo "PID: $$ | Workdir: $WORKDIR | DB: $DB" >> "$MONITOR_LOG"

# Run in an infinite loop so if Codex exits, it restarts
while true; do
  echo "=== Starting codex exec loop at $(date -Iseconds) ===" >> "$MONITOR_LOG"

  codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    -C "$WORKDIR" \
    --sandbox danger-full-access \
    --ignore-user-config \
    --ephemeral \
    -- < "$PROMPT_FILE" 2>&1 | tee -a "$MONITOR_LOG"

  EXIT_CODE=${PIPESTATUS[0]}
  echo "=== Codex exec exited with $EXIT_CODE at $(date -Iseconds) ===" >> "$MONITOR_LOG"
  echo "Restarting in 60 seconds..." >> "$MONITOR_LOG"
  sleep 60
done
