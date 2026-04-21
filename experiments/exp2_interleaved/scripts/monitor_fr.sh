#!/bin/bash
# Monitor FR training until it catches up to EN (step 29k)

while true; do
  step=$(cat "/Volumes/Misc Backup/fractal/checkpoints/fr/125M/training_state.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['step'])" 2>/dev/null)
  if [ -z "$step" ]; then
    echo "Could not read FR step"
    sleep 300
    continue
  fi
  echo "$(date '+%H:%M:%S') FR step: $step"
  if [ "$step" -ge 28000 ]; then
    echo "FR nearly caught up to EN (29k). Time to start EN training."
    exit 0
  fi
  sleep 300  # Check every 5 minutes
done
