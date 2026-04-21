#!/bin/bash
# Train EN and FR in parallel on multi-GPU system
# Each model gets its own GPU(s)

set -e

export DATA_PATH="/workspace/data"
STEPS=${1:-300000}

echo "=== Parallel Training: EN + FR to $STEPS steps ==="
echo "GPUs available:"
nvidia-smi -L

# Count GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Total GPUs: $NUM_GPUS"

if [ $NUM_GPUS -ge 2 ]; then
    echo "Running EN on GPU 0, FR on GPU 1"

    # Run EN on GPU 0
    CUDA_VISIBLE_DEVICES=0 python train.py en 125M $STEPS 2>&1 | tee /workspace/data/en_training.log &
    EN_PID=$!

    # Run FR on GPU 1
    CUDA_VISIBLE_DEVICES=1 python train.py fr 125M $STEPS 2>&1 | tee /workspace/data/fr_training.log &
    FR_PID=$!

    echo "EN training PID: $EN_PID"
    echo "FR training PID: $FR_PID"
    echo ""
    echo "Monitor with: tail -f /workspace/data/en_training.log"
    echo "Monitor with: tail -f /workspace/data/fr_training.log"

    # Wait for both
    wait $EN_PID $FR_PID
else
    echo "Only $NUM_GPUS GPU(s), running sequentially"
    python train.py en 125M $STEPS
    python train.py fr 125M $STEPS
fi

echo "=== All training complete ==="
