#!/bin/bash
# Start EN training with 8GB memory limit
cd /Users/adam/dev/fractal-language
nice -n 15 FRACTAL_MEMORY_LIMIT=8.0 /Users/adam/.pyenv/versions/3.13.6/bin/uv run python -u scripts/training_with_eval.py en 125M >> "/Volumes/Misc Backup/fractal/logs/en_125M_eval.log" 2>&1
