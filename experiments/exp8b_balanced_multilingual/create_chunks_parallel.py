#!/usr/bin/env python3
"""
Parallel chunking script using HuggingFace tokenizer.
Parallelizes across languages and uses multiprocessing for speed.
"""
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import time

# Config
CHUNK_SIZE = 1_000_000  # 1M tokens per chunk
SEQ_LEN = 512
RAW_TEXT_DIR = Path("/workspace/exp8/raw_text")
EXTENDED_DIR = Path("/workspace/exp8/raw_text_extended")
CHUNKS_DIR = Path("/workspace/exp8/chunks")
TOKENIZER_PATH = Path("/workspace/exp8/joint_tokenizer.json")

LANGUAGES = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh",
             "synth_a", "synth_b", "synth_c", "synth_d"]

def chunk_language(lang: str) -> dict:
    """Process a single language - runs in separate process."""
    # Load tokenizer in each process
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("<eos>")

    # Find raw text file (prefer extended if exists)
    extended_path = EXTENDED_DIR / f"{lang}.txt"
    base_path = RAW_TEXT_DIR / f"{lang}.txt"

    if extended_path.exists():
        filepath = extended_path
    elif base_path.exists():
        filepath = base_path
    else:
        return {"lang": lang, "error": f"No raw text found", "chunks": 0, "tokens": 0}

    # Read and tokenize
    all_ids = []
    batch_texts = []
    batch_size = 1000  # Batch encode for speed

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            text = line.strip()
            if len(text) < 50:
                continue
            # Unescape
            text = text.replace('\\n', '\n').replace('\\r', '\r')
            batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                # Batch encode
                encodings = tokenizer.encode_batch(batch_texts)
                for enc in encodings:
                    all_ids.extend(enc.ids)
                    all_ids.append(eos_id)
                batch_texts = []

    # Process remaining
    if batch_texts:
        encodings = tokenizer.encode_batch(batch_texts)
        for enc in encodings:
            all_ids.extend(enc.ids)
            all_ids.append(eos_id)

    if not all_ids:
        return {"lang": lang, "error": "No tokens", "chunks": 0, "tokens": 0}

    # Convert to numpy and save chunks
    all_ids = np.array(all_ids, dtype=np.uint32)
    total_tokens = len(all_ids)

    # Pad to multiple of SEQ_LEN
    pad_len = (SEQ_LEN - (len(all_ids) % SEQ_LEN)) % SEQ_LEN
    if pad_len > 0:
        all_ids = np.concatenate([all_ids, np.zeros(pad_len, dtype=np.uint32)])

    # Split into chunks
    chunk_idx = 0
    for start in range(0, len(all_ids), CHUNK_SIZE):
        chunk = all_ids[start:start + CHUNK_SIZE]
        if len(chunk) < SEQ_LEN:
            continue

        # Pad chunk if needed
        if len(chunk) < CHUNK_SIZE:
            chunk = np.concatenate([chunk, np.zeros(CHUNK_SIZE - len(chunk), dtype=np.uint32)])

        chunk_path = CHUNKS_DIR / f"{lang}_chunk_{chunk_idx:04d}.npy"
        np.save(chunk_path, chunk)
        chunk_idx += 1

    return {"lang": lang, "chunks": chunk_idx, "tokens": total_tokens}

def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing chunks
    for f in CHUNKS_DIR.glob("*.npy"):
        f.unlink()

    print(f"Parallel chunking with {mp.cpu_count()} CPUs")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,} tokens")
    print()

    start = time.time()

    # Process all languages in parallel
    with ProcessPoolExecutor(max_workers=min(12, mp.cpu_count())) as executor:
        futures = {executor.submit(chunk_language, lang): lang for lang in LANGUAGES}

        for future in as_completed(futures):
            result = future.result()
            lang = result["lang"]
            if "error" in result:
                print(f"  {lang}: ERROR - {result['error']}")
            else:
                print(f"  {lang}: {result['chunks']} chunks, {result['tokens']:,} tokens")

    elapsed = time.time() - start

    # Summary
    chunks = list(CHUNKS_DIR.glob("*.npy"))
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total chunks: {len(chunks)}")

    # Verify max token ID
    max_id = 0
    for chunk_path in chunks[:5]:  # Sample first 5
        chunk = np.load(chunk_path)
        max_id = max(max_id, chunk.max())
    print(f"Max token ID (sample): {max_id}")

if __name__ == "__main__":
    main()
