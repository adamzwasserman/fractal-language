#!/bin/bash
set -e
cd /workspace/exp8

echo "$(date): Downloading CC-100 Indonesian (36GB compressed)..."
curl -o /tmp/id.txt.xz "https://data.statmt.org/cc-100/id.txt.xz"

echo "$(date): Extracting..."
xzcat /tmp/id.txt.xz | head -n 5000000 > raw_text_extended/id.txt
rm /tmp/id.txt.xz

echo "$(date): Checking quality..."
wc -l raw_text_extended/id.txt
head -5 raw_text_extended/id.txt

echo "$(date): Creating chunks..."
python3 << 'EOF'
from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

tokenizer = Tokenizer.from_file("/workspace/exp8/joint_tokenizer.json")
input_file = Path("/workspace/exp8/raw_text_extended/id.txt")
output_dir = Path("/workspace/exp8/chunks")
CHUNK_SIZE = 1_000_000
SEQ_LEN = 512
PAD_ID = tokenizer.token_to_id("<pad>")

all_tokens = []
chunk_idx = 0

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        text = line.strip()
        if len(text) < 50:
            continue
        ids = tokenizer.encode(text).ids
        all_tokens.extend(ids)
        
        while len(all_tokens) >= CHUNK_SIZE:
            chunk = np.array(all_tokens[:CHUNK_SIZE], dtype=np.uint32)
            np.save(output_dir / f"id_chunk_{chunk_idx:04d}.npy", chunk)
            print(f"  Saved id_chunk_{chunk_idx:04d}.npy")
            chunk_idx += 1
            all_tokens = all_tokens[CHUNK_SIZE:]
            
        if i % 100000 == 0:
            print(f"  Processed {i} lines, {chunk_idx} chunks...")

# Save final chunk with padding
if all_tokens:
    needed = CHUNK_SIZE - len(all_tokens)
    all_tokens.extend([PAD_ID] * needed)
    chunk = np.array(all_tokens[:CHUNK_SIZE], dtype=np.uint32)
    np.save(output_dir / f"id_chunk_{chunk_idx:04d}.npy", chunk)
    print(f"  Saved final id_chunk_{chunk_idx:04d}.npy (padded)")
    chunk_idx += 1

print(f"Total: {chunk_idx} chunks")
EOF

echo "$(date): Restarting ID training..."
CUDA_VISIBLE_DEVICES=5 nohup python3 /workspace/exp8/train_exp8_ddp.py id --max-steps 200000 > /workspace/exp8/logs/id_training.log 2>&1 &

echo "$(date): Done!"
