#!/bin/bash
# Monitor for emergence detection (ppl < 150) and alert

THRESHOLD=150
LOG_FILE="/tmp/claude/tasks/emergence_alert.log"
ALERT_FILE="/tmp/claude/tasks/EMERGENCE_DETECTED.txt"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting emergence monitor (threshold: ppl < $THRESHOLD)" | tee -a "$LOG_FILE"

while true; do
  # Check EN training log for latest smoke test
  en_ppl=$(grep "Smoke test: ppl=" /tmp/claude/tasks/bbce869.output 2>/dev/null | tail -1 | sed 's/.*ppl=\([0-9.]*\).*/\1/')
  en_step=$(grep "Checkpoint saved: step" /tmp/claude/tasks/bbce869.output 2>/dev/null | tail -1 | sed 's/.*step \([0-9]*\).*/\1/')

  # Check FR training log for latest smoke test
  fr_ppl=$(grep "Smoke test: ppl=" /tmp/claude/tasks/b097d62.output 2>/dev/null | tail -1 | sed 's/.*ppl=\([0-9.]*\).*/\1/')
  fr_step=$(grep "Checkpoint saved: step" /tmp/claude/tasks/b097d62.output 2>/dev/null | tail -1 | sed 's/.*step \([0-9]*\).*/\1/')

  echo "$(date '+%H:%M:%S') - EN: step $en_step, ppl=$en_ppl | FR: step $fr_step, ppl=$fr_ppl" | tee -a "$LOG_FILE"

  # Check for EN emergence
  if [ -n "$en_ppl" ]; then
    en_emerged=$(python3 -c "print('yes' if float('$en_ppl') < $THRESHOLD else 'no')" 2>/dev/null)
    if [ "$en_emerged" = "yes" ]; then
      echo "" | tee -a "$LOG_FILE"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
      echo "🎉 EMERGENCE DETECTED: EN at step $en_step" | tee -a "$LOG_FILE"
      echo "   Perplexity: $en_ppl (threshold: $THRESHOLD)" | tee -a "$LOG_FILE"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
      echo "EN emerged at step $en_step with ppl=$en_ppl" >> "$ALERT_FILE"

      # macOS notification
      osascript -e "display notification \"EN emerged at step $en_step with ppl=$en_ppl\" with title \"EMERGENCE DETECTED\" sound name \"Glass\"" 2>/dev/null

      # Also beep
      afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &
    fi
  fi

  # Check for FR emergence
  if [ -n "$fr_ppl" ]; then
    fr_emerged=$(python3 -c "print('yes' if float('$fr_ppl') < $THRESHOLD else 'no')" 2>/dev/null)
    if [ "$fr_emerged" = "yes" ]; then
      echo "" | tee -a "$LOG_FILE"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
      echo "🎉 EMERGENCE DETECTED: FR at step $fr_step" | tee -a "$LOG_FILE"
      echo "   Perplexity: $fr_ppl (threshold: $THRESHOLD)" | tee -a "$LOG_FILE"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
      echo "FR emerged at step $fr_step with ppl=$fr_ppl" >> "$ALERT_FILE"

      # macOS notification
      osascript -e "display notification \"FR emerged at step $fr_step with ppl=$fr_ppl\" with title \"EMERGENCE DETECTED\" sound name \"Glass\"" 2>/dev/null

      # Also beep
      afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &
    fi
  fi

  # Check if both have emerged - if so, we can exit
  if [ -f "$ALERT_FILE" ]; then
    en_done=$(grep -c "EN emerged" "$ALERT_FILE" 2>/dev/null || echo 0)
    fr_done=$(grep -c "FR emerged" "$ALERT_FILE" 2>/dev/null || echo 0)
    if [ "$en_done" -gt 0 ] && [ "$fr_done" -gt 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Both models emerged! Monitor complete." | tee -a "$LOG_FILE"
      exit 0
    fi
  fi

  sleep 300  # Check every 5 minutes
done
