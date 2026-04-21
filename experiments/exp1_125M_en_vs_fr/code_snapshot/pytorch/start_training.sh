#!/bin/bash
# Start EN and FR training in parallel on 2x RTX 4090
# EN on GPU 0, FR on GPU 1

set -e

export DATA_PATH="/workspace/data"
cd /workspace/pytorch

# Update batch size for RTX 4090 (24GB VRAM)
sed -i 's/batch=8/batch=32/g' train.py

# Create log directory
mkdir -p /workspace/data/logs

echo "=== Starting Fractal Language Training ==="
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check chunk counts
EN_CHUNKS=$(ls /workspace/data/chunks/en_chunk_* 2>/dev/null | wc -l)
FR_CHUNKS=$(ls /workspace/data/chunks/fr_chunk_* 2>/dev/null | wc -l)
echo "EN chunks available: $EN_CHUNKS"
echo "FR chunks available: $FR_CHUNKS"
echo ""

# Start EN on GPU 0
if [ "$EN_CHUNKS" -gt 0 ]; then
    echo "Starting EN training on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 nohup python train.py en 125M 300000 > /workspace/data/logs/en_stdout.log 2>&1 &
    EN_PID=$!
    echo "EN PID: $EN_PID"
else
    echo "No EN chunks yet, skipping EN training"
    EN_PID=""
fi

# Start FR on GPU 1 (if chunks available)
if [ "$FR_CHUNKS" -gt 0 ]; then
    echo "Starting FR training on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 nohup python train.py fr 125M 300000 > /workspace/data/logs/fr_stdout.log 2>&1 &
    FR_PID=$!
    echo "FR PID: $FR_PID"
else
    echo "No FR chunks yet - start FR manually when available:"
    echo "  CUDA_VISIBLE_DEVICES=1 nohup python train.py fr 125M 300000 > /workspace/data/logs/fr_stdout.log 2>&1 &"
    FR_PID=""
fi

echo ""
echo "=== Training Started ==="
echo ""
echo "Monitor commands:"
echo "  GPU usage:     watch -n 2 nvidia-smi"
echo "  EN progress:   tail -f /workspace/data/logs/en_stdout.log"
echo "  FR progress:   tail -f /workspace/data/logs/fr_stdout.log"
echo "  Chunk sync:    watch -n 10 'ls /workspace/data/chunks/en_chunk_* | wc -l; ls /workspace/data/chunks/fr_chunk_* | wc -l'"
echo "  Training CSV:  tail -f /workspace/data/logs/training_en.csv"
echo "  Grammar CSV:   tail -f /workspace/data/logs/grammar_probes_en.csv"
echo ""
echo "To start FR later (when chunks sync):"
echo "  CUDA_VISIBLE_DEVICES=1 nohup python train.py fr 125M 300000 > /workspace/data/logs/fr_stdout.log 2>&1 &"
