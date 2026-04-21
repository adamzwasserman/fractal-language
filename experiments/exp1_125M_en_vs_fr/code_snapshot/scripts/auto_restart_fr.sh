#!/bin/bash
# Wait for EN to catch up to 66k, then auto-restart FR training

TARGET=66000
LOG_FILE="/tmp/claude/tasks/auto_restart_fr.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting FR auto-restart monitor (target: EN >= $TARGET)" | tee -a "$LOG_FILE"

while true; do
  en_step=$(cat "/Volumes/Misc Backup/fractal/checkpoints/en/125M/training_state.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['step'])" 2>/dev/null)

  if [ -z "$en_step" ]; then
    echo "$(date '+%H:%M:%S') - Could not read EN step, retrying..." | tee -a "$LOG_FILE"
    sleep 60
    continue
  fi

  echo "$(date '+%H:%M:%S') - EN step: $en_step (target: $TARGET)" | tee -a "$LOG_FILE"

  if [ "$en_step" -ge "$TARGET" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EN reached $en_step >= $TARGET!" | tee -a "$LOG_FILE"
    echo "Starting FR training..." | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # Start FR training in background with memory limits
    cd /Users/adam/dev/fractal-language
    env FRACTAL_MEMORY_LIMIT=8.0 nice -n 15 nohup uv run python -u scripts/training_with_eval.py fr 125M >> /tmp/claude/tasks/fr_training.log 2>&1 &
    FR_PID=$!

    echo "$(date '+%Y-%m-%d %H:%M:%S') - FR training started with PID: $FR_PID" | tee -a "$LOG_FILE"
    echo "Log file: /tmp/claude/tasks/fr_training.log" | tee -a "$LOG_FILE"

    # Verify it started
    sleep 5
    if ps -p $FR_PID > /dev/null 2>&1; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - FR training confirmed running" | tee -a "$LOG_FILE"
    else
      echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: FR training may have failed to start!" | tee -a "$LOG_FILE"
    fi

    exit 0
  fi

  sleep 300  # Check every 5 minutes
done
