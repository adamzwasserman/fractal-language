#!/bin/bash
# Wait for EN to catch up to FR (66k), then restart FR

TARGET=66000

while true; do
  en_step=$(cat "/Volumes/Misc Backup/fractal/checkpoints/en/125M/training_state.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['step'])" 2>/dev/null)

  echo "$(date '+%H:%M:%S') EN step: $en_step (target: $TARGET)"

  if [ "$en_step" -ge "$TARGET" ]; then
    echo "EN caught up to FR! Restarting FR training..."
    exit 0
  fi

  sleep 300  # Check every 5 minutes
done
