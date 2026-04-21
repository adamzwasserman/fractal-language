#!/bin/bash
# Monitoring script - runs probes every N checkpoints and displays status
export DATA_PATH="/workspace/data"
cd /workspace/pytorch

PROBE_INTERVAL=5000  # Run probes every 5000 steps

last_en_step=0
last_fr_step=0

while true; do
    clear
    echo "=============================================="
    echo "  FRACTAL LANGUAGE TRAINING MONITOR"
    echo "  $(date)"
    echo "=============================================="
    echo ""

    # GPU Status
    echo "=== GPU STATUS ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "GPU query failed"
    echo ""

    # Chunk Status
    EN_CHUNKS=$(ls /workspace/data/chunks/en_chunk_* 2>/dev/null | wc -l)
    FR_CHUNKS=$(ls /workspace/data/chunks/fr_chunk_* 2>/dev/null | wc -l)
    echo "=== DATA STATUS ==="
    echo "EN chunks: $EN_CHUNKS"
    echo "FR chunks: $FR_CHUNKS"
    echo ""

    # Training Progress
    echo "=== TRAINING PROGRESS ==="

    # EN Progress
    if [ -f /workspace/data/logs/training_en.csv ]; then
        EN_LATEST=$(tail -1 /workspace/data/logs/training_en.csv 2>/dev/null)
        EN_STEP=$(echo $EN_LATEST | cut -d',' -f1)
        EN_LOSS=$(echo $EN_LATEST | cut -d',' -f2)
        EN_PPL=$(echo $EN_LATEST | cut -d',' -f3)
        echo "EN: step=$EN_STEP loss=$EN_LOSS ppl=$EN_PPL"
    else
        EN_STEP=0
        echo "EN: not started"
    fi

    # FR Progress
    if [ -f /workspace/data/logs/training_fr.csv ]; then
        FR_LATEST=$(tail -1 /workspace/data/logs/training_fr.csv 2>/dev/null)
        FR_STEP=$(echo $FR_LATEST | cut -d',' -f1)
        FR_LOSS=$(echo $FR_LATEST | cut -d',' -f2)
        FR_PPL=$(echo $FR_LATEST | cut -d',' -f3)
        echo "FR: step=$FR_STEP loss=$FR_LOSS ppl=$FR_PPL"
    else
        FR_STEP=0
        echo "FR: not started"
    fi
    echo ""

    # Grammar Probe Results
    echo "=== LATEST GRAMMAR PROBES ==="
    if [ -f /workspace/data/logs/grammar_probes_en.csv ]; then
        echo "EN (last 5):"
        tail -5 /workspace/data/logs/grammar_probes_en.csv | column -t -s','
    else
        echo "EN: no results yet"
    fi
    echo ""
    if [ -f /workspace/data/logs/grammar_probes_fr.csv ]; then
        echo "FR (last 5):"
        tail -5 /workspace/data/logs/grammar_probes_fr.csv | column -t -s','
    else
        echo "FR: no results yet"
    fi
    echo ""

    # Process Status
    echo "=== PROCESSES ==="
    ps aux | grep "python train.py" | grep -v grep || echo "No training processes"
    echo ""

    # ETA calculation
    if [ "$EN_STEP" -gt 0 ] 2>/dev/null; then
        REMAINING=$((300000 - EN_STEP))
        echo "=== ETA ==="
        echo "EN remaining: $REMAINING steps"
    fi

    echo ""
    echo "(refreshes every 30 seconds)"

    sleep 30
done
