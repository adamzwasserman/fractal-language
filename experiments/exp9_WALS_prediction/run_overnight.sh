#!/bin/bash
cd /Users/adam/dev/fractal-language
mkdir -p /Volumes/fractal/exp9_WALS_prediction/logs
echo "Starting jobs at $(date)"
nohup uv run python /Volumes/fractal/exp9_WALS_prediction/create_chunks_exp9.py en > /Volumes/fractal/exp9_WALS_prediction/logs/en_chunks.log 2>&1 &
echo "EN chunks PID: $!"
nohup uv run python /Volumes/fractal/exp9_WALS_prediction/create_chunks_exp9.py fr > /Volumes/fractal/exp9_WALS_prediction/logs/fr_chunks.log 2>&1 &
echo "FR chunks PID: $!"
nohup uv run python /Volumes/fractal/exp9_WALS_prediction/generate_synth_alpha_corpus.py --encoder all > /Volumes/fractal/exp9_WALS_prediction/logs/synth_alpha.log 2>&1 &
echo "Synth alpha PID: $!"
echo "All jobs started. Logs in /Volumes/fractal/exp9_WALS_prediction/logs/"
