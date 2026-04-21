#!/usr/bin/env python
"""Create tokenized chunks for Rust training"""
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import gc
import sys

DATA_DIR = Path("/workspace/data")
CHUNK_DIR = DATA_DIR / "chunks"
CHUNK_DIR.mkdir(exist_ok=True)
TOKENS_PER_CHUNK = 1_000_000

def create_chunks(mode):
    print(f"Creating chunks for {mode}...")
    
    # Select tokenizer and text file based on mode
    if mode == "rust":
        tokenizer_path = DATA_DIR / "rust_tokenizer.json"
        text_file = DATA_DIR / "rust_texts.txt"
    else:  # rust_en
        tokenizer_path = DATA_DIR / "rust_en_tokenizer.json"
        text_file = DATA_DIR / "rust_en_texts.txt"
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    
    chunk_idx = 0
    token_buffer = []
    
    with open(text_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            text = line.strip().replace("\\\\n", "\\n").replace("\\\\r", "\\r")
            tokens = tokenizer.encode(text).ids + [eos_id]
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= TOKENS_PER_CHUNK:
                chunk = np.array(token_buffer[:TOKENS_PER_CHUNK], dtype=np.uint32)
                np.save(CHUNK_DIR / f"{mode}_chunk_{chunk_idx:06d}.npy", chunk)
                print(f"Saved {mode}_chunk_{chunk_idx:06d}.npy")
                token_buffer = token_buffer[TOKENS_PER_CHUNK:]
                chunk_idx += 1
                gc.collect()
    
    if token_buffer:
        chunk = np.array(token_buffer, dtype=np.uint32)
        np.save(CHUNK_DIR / f"{mode}_chunk_{chunk_idx:06d}.npy", chunk)
        print(f"Saved final chunk {chunk_idx}")
    
    print(f"Created {chunk_idx + 1} chunks for {mode}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rust"
    create_chunks(mode)
