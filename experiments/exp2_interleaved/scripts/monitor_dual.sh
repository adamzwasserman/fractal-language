#!/bin/bash
# Monitor both EN and FR training progress until 200k steps

while true; do
  date_str=$(date '+%Y-%m-%d %H:%M:%S')

  fr_step=$(cat "/Volumes/Misc Backup/fractal/checkpoints/fr/125M/training_state.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['step'])" 2>/dev/null)
  en_step=$(cat "/Volumes/Misc Backup/fractal/checkpoints/en/125M/training_state.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['step'])" 2>/dev/null)

  # Check if processes are running
  fr_running=$(ps aux | grep "train.*fr.*125M" | grep -v grep | wc -l)
  en_running=$(ps aux | grep "train.*en.*125M" | grep -v grep | wc -l)

  echo "$date_str - FR: $fr_step (running: $fr_running), EN: $en_step (running: $en_running)"

  # Check for completion
  if [ "$fr_step" -ge 200000 ] && [ "$en_step" -ge 200000 ]; then
    echo "Both trainings complete!"
    exit 0
  fi

  # Check if either stopped unexpectedly
  if [ "$fr_running" -eq 0 ] && [ "$fr_step" -lt 200000 ]; then
    echo "WARNING: FR training stopped at step $fr_step"
  fi
  if [ "$en_running" -eq 0 ] && [ "$en_step" -lt 200000 ]; then
    echo "WARNING: EN training stopped at step $en_step"
  fi

  sleep 600  # Check every 10 minutes
done
