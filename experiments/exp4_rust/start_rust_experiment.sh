#!/bin/bash
# Start Rust Experiment - both conditions in parallel

echo "=== Starting Rust Experiment ==="
echo "GPU 0: rust-only"
echo "GPU 1: rust+en"
echo ""

cd /workspace/pytorch

# Start rust-only on GPU 0
echo "Starting rust-only training on GPU 0..."
nohup python train_rust_ddp.py rust 0 125M 200000 > /workspace/logs/rust_training.log 2>&1 &
PID_RUST=$!
echo "rust PID: $PID_RUST"

# Start rust+en on GPU 1 (only if rust_en chunks exist)
if ls /workspace/data/chunks/rust_en_chunk_* 1> /dev/null 2>&1; then
    echo "Starting rust+en training on GPU 1..."
    nohup python train_rust_ddp.py rust_en 1 125M 200000 > /workspace/logs/rust_en_training.log 2>&1 &
    PID_RUST_EN=$!
    echo "rust_en PID: $PID_RUST_EN"
else
    echo "WARNING: No rust_en chunks found. Run rust+en data prep first."
    echo "  python scripts/rust_en_dataset_builder.py"
    echo "  python scripts/create_rust_chunks.py rust_en"
fi

echo ""
echo "Training started! Run monitor with:"
echo "  python /workspace/monitor_rust.py"
echo ""
echo "Or watch logs:"
echo "  tail -f /workspace/logs/rust_training.log"
