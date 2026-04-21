#!/bin/bash
# Start all 12 languages on 12x RTX 4090
# Each language gets one GPU

set -e

export DATA_PATH=${DATA_PATH:-/workspace/exp8}
export BATCH_SIZE=${BATCH_SIZE:-4}  # Reduced for 250k vocab

LANGUAGES=(en fr es fi ru id vi zh synth_a synth_b synth_c synth_d)
LOG_DIR="$DATA_PATH/logs"
mkdir -p "$LOG_DIR"

echo "Starting Exp8 training on 12 languages"
echo "Data path: $DATA_PATH"
echo "Batch size: $BATCH_SIZE"
echo ""

# Start each language on its GPU
for i in "${!LANGUAGES[@]}"; do
    LANG=${LANGUAGES[$i]}
    GPU=$i
    LOG="$LOG_DIR/${LANG}_training.log"

    # Check if chunks exist for this language
    CHUNKS=$(ls "$DATA_PATH/chunks_xlmr/${LANG}_chunk_"*.npy 2>/dev/null | wc -l)
    if [ "$CHUNKS" -eq 0 ]; then
        echo "[$LANG] No chunks found, skipping"
        continue
    fi

    echo "[$LANG] GPU $GPU, $CHUNKS chunks available"

    # Start training in background (use full path)
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 /workspace/exp8/train_exp8_ddp.py "$LANG" \
        > "$LOG" 2>&1 &

    PID=$!
    echo "  PID: $PID -> $LOG"
done

echo ""
echo "All languages started. Monitor with:"
echo "  tail -f $LOG_DIR/*_training.log"
echo ""
echo "Or check progress:"
echo "  ls -la $DATA_PATH/checkpoints/*/125M/"
