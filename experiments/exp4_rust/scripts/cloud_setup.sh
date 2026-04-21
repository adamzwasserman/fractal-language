#!/bin/bash
# Cloud setup script for Vast.ai / RunPod
# Run this after SSH'ing into your instance

set -e

echo "=== Setting up Fractal Language Training ==="

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone or upload your repo (option 1: from GitHub)
# git clone https://github.com/YOUR_USERNAME/fractal-language.git
# cd fractal-language

# Option 2: You'll rsync the code (see instructions below)
cd /workspace/fractal-language

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install mlx mlx-lm tokenizers numpy tqdm

# Create directories
mkdir -p /workspace/data/checkpoints/en/125M
mkdir -p /workspace/data/checkpoints/fr/125M
mkdir -p /workspace/data/chunks
mkdir -p /workspace/data/logs
mkdir -p /workspace/data/results

echo "=== Setup complete ==="
echo "Next steps:"
echo "1. Upload tokenizer: rsync joint_tokenizer.json"
echo "2. Upload chunks: rsync chunks/*.npy"
echo "3. Run training: ./scripts/cloud_train.sh"
