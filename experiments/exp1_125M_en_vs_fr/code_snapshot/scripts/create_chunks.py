#!/usr/bin/env python
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import gc

# Configuration
CHUNK_DIR = Path("/Volumes/Misc Backup/fractal")
TOKENS_PER_CHUNK = 1_000_000  # 1M tokens per chunk = ~4MB
OUTPUT_CHUNKS_DIR = CHUNK_DIR / "chunks"
OUTPUT_CHUNKS_DIR.mkdir(exist_ok=True)

def create_chunks_for_language(lang: str):
    """Convert text file to efficient .npy chunks for training"""
    print(f"Creating chunks for {lang.upper()}...")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(CHUNK_DIR / "joint_tokenizer.json"))
    eos_id = tokenizer.token_to_id("<eos>")
    
    text_file = CHUNK_DIR / f"{lang}_texts.txt"
    chunk_idx = 0
    token_buffer = []
    
    with open(text_file, 'r', encoding='utf-8') as f:
        pbar = tqdm(desc=f"Processing {lang} texts", unit="doc")
        
        for line in f:
            # Unescape and tokenize
            text = line.strip().replace('\\n', '\n').replace('\\r', '\r')
            tokens = tokenizer.encode(text).ids + [eos_id]
            token_buffer.extend(tokens)
            pbar.update(1)
            
            # Save chunk when buffer is full
            if len(token_buffer) >= TOKENS_PER_CHUNK:
                # Extract exact chunk size
                chunk_tokens = token_buffer[:TOKENS_PER_CHUNK]
                token_buffer = token_buffer[TOKENS_PER_CHUNK:]
                
                # Save as uint32 numpy array (4 bytes per token)
                chunk_array = np.array(chunk_tokens, dtype=np.uint32)
                chunk_path = OUTPUT_CHUNKS_DIR / f"{lang}_chunk_{chunk_idx:06d}.npy"
                np.save(chunk_path, chunk_array)
                
                print(f"Saved chunk {chunk_idx}: {len(chunk_tokens):,} tokens → {chunk_path}")
                chunk_idx += 1
                gc.collect()
        
        # Save final chunk if any tokens remain
        if token_buffer:
            chunk_array = np.array(token_buffer, dtype=np.uint32)
            chunk_path = OUTPUT_CHUNKS_DIR / f"{lang}_chunk_{chunk_idx:06d}.npy"
            np.save(chunk_path, chunk_array)
            print(f"Saved final chunk {chunk_idx}: {len(token_buffer):,} tokens → {chunk_path}")
        
        pbar.close()
    
    print(f"{lang.upper()} chunking complete: {chunk_idx + 1} chunks created")
    return chunk_idx + 1

if __name__ == "__main__":
    # Create chunks for both languages
    en_chunks = create_chunks_for_language("en")
    fr_chunks = create_chunks_for_language("fr")
    
    # Save metadata
    metadata = {
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "en_chunks": en_chunks,
        "fr_chunks": fr_chunks,
        "total_chunks": en_chunks + fr_chunks,
        "chunk_size_mb": TOKENS_PER_CHUNK * 4 / (1024 * 1024),  # uint32 = 4 bytes
        "chunks_directory": str(OUTPUT_CHUNKS_DIR),
    }
    
    with open(OUTPUT_CHUNKS_DIR / "chunk_metadata.json", 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"\nChunking complete!")
    print(f"English: {en_chunks} chunks")
    print(f"French: {fr_chunks} chunks") 
    print(f"Total: {en_chunks + fr_chunks} chunks")
    print(f"Chunk size: {TOKENS_PER_CHUNK:,} tokens (~4MB each)")
    print(f"Chunks saved to: {OUTPUT_CHUNKS_DIR}")