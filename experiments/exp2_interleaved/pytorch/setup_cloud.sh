#!/bin/bash
# Setup script for Vast.ai / RunPod
# Run this after SSH'ing into your instance

set -e

echo "=== Fractal Language Cloud Setup ==="

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Install dependencies
echo "Installing dependencies..."
pip install torch tokenizers safetensors numpy tqdm

# Create data directory
mkdir -p /workspace/data/checkpoints/en/125M
mkdir -p /workspace/data/checkpoints/fr/125M
mkdir -p /workspace/data/chunks
mkdir -p /workspace/data/results

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Upload your data:"
echo "   rsync -avP --progress /Volumes/Misc\\ Backup/fractal/chunks/ user@<IP>:/workspace/data/chunks/"
echo "   rsync -avP /Volumes/Misc\\ Backup/fractal/joint_tokenizer.json user@<IP>:/workspace/data/"
echo ""
echo "2. Run training (in separate tmux sessions):"
echo "   ./cloud_run.sh en 125M 300000"
echo "   ./cloud_run.sh fr 125M 300000"
echo ""
echo "3. Download results when done:"
echo "   rsync -avP user@<IP>:/workspace/data/checkpoints/ ./cloud_checkpoints/"
