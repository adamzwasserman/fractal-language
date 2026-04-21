#!/bin/bash
# Run training on cloud GPU
# Usage: ./cloud_run.sh en 125M 300000
#        ./cloud_run.sh fr 125M 300000

set -e

LANG=${1:-en}
SIZE=${2:-125M}
STEPS=${3:-300000}

export DATA_PATH="/workspace/data"

echo "=== Starting $SIZE $LANG training to $STEPS steps ==="
echo "Data path: $DATA_PATH"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l)"

# Run training
python train.py $LANG $SIZE $STEPS

echo "=== Training complete ==="
