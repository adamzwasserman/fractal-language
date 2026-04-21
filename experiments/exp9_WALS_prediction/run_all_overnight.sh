#!/bin/bash
# Run all exp9 data preparation jobs with nohup

cd /Users/adam/dev/fractal-language

LOG_DIR="/Volumes/fractal/exp9_WALS_prediction/logs"
mkdir -p "$LOG_DIR"

echo "Starting all jobs at $(date)"

# EN chunks
nohup uv run python /Volumes/fractal/exp9_WALS_prediction/create_chunks_exp9.py en \
    > "$LOG_DIR/en_chunks.log" 2>&1 &
echo "EN chunks PID: $!"

# FR chunks
nohup uv run python /Volumes/fractal/exp9_WALS_prediction/create_chunks_exp9.py fr \
    > "$LOG_DIR/fr_chunks.log" 2>&1 &
echo "FR chunks PID: $!"

# Synth alpha corpus (all encoders)
nohup uv run python /Volumes/fractal/exp9_WALS_prediction/generate_synth_alpha_corpus.py --encoder all \
    > "$LOG_DIR/synth_alpha.log" 2>&1 &
echo "Synth alpha PID: $!"

echo ""
echo "All jobs started. Check progress with:"
echo "  tail -f $LOG_DIR/en_chunks.log"
echo "  tail -f $LOG_DIR/fr_chunks.log"
echo "  tail -f $LOG_DIR/synth_alpha.log"
echo ""
echo "Or check if running with:"
echo "  pgrep -fl 'create_chunks_exp9\|generate_synth_alpha'"
