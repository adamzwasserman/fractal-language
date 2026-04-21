#!/usr/bin/env python3
"""
Create tokenized chunks for exp8 multilingual experiment.

Uses the balanced 50k BPE tokenizer trained on equal samples from all 12 languages.
"""

import numpy as np
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
import gc
import json
from datetime import datetime

# Configuration
TOKENS_PER_CHUNK = 1_000_000  # 1M tokens per chunk = ~4MB
RAW_TEXT_DIR = Path('/workspace/exp8/raw_text')
OUTPUT_DIR = Path('/workspace/exp8/chunks')
TOKENIZER_PATH = Path('/workspace/exp8/balanced_tokenizer.model')

# All 12 languages
LANGUAGES = ['en', 'fr', 'es', 'fi', 'ru', 'id', 'vi', 'zh',
             'synth_a', 'synth_b', 'synth_c', 'synth_d']


def create_chunks_for_language(lang: str, sp: spm.SentencePieceProcessor):
    """Convert text file to efficient .npy chunks for training."""
    print(f"\n{'='*60}")
    print(f"Creating chunks for {lang.upper()}")
    print(f"{'='*60}")

    text_file = RAW_TEXT_DIR / f'{lang}.txt'
    if not text_file.exists():
        print(f"  WARNING: {text_file} not found!")
        return 0

    eos_id = sp.eos_id()  # Should be 3
    chunk_idx = 0
    token_buffer = []
    total_tokens = 0

    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  Processing {len(lines):,} lines...")

    for line in tqdm(lines, desc=f"  {lang}", unit="line"):
        text = line.strip()
        if not text:
            continue

        # Tokenize and add EOS
        tokens = sp.encode(text) + [eos_id]
        token_buffer.extend(tokens)
        total_tokens += len(tokens)

        # Save chunk when buffer is full
        while len(token_buffer) >= TOKENS_PER_CHUNK:
            chunk_tokens = token_buffer[:TOKENS_PER_CHUNK]
            token_buffer = token_buffer[TOKENS_PER_CHUNK:]

            # Save as uint32 numpy array
            chunk_array = np.array(chunk_tokens, dtype=np.uint32)
            chunk_path = OUTPUT_DIR / f"{lang}_chunk_{chunk_idx:06d}.npy"
            np.save(chunk_path, chunk_array)

            print(f"  Saved: {chunk_path.name} ({len(chunk_tokens):,} tokens)")
            chunk_idx += 1
            gc.collect()

    # Save final chunk if any tokens remain (pad if needed)
    if token_buffer:
        # Pad to chunk size with EOS tokens
        while len(token_buffer) < TOKENS_PER_CHUNK:
            token_buffer.append(eos_id)

        chunk_array = np.array(token_buffer[:TOKENS_PER_CHUNK], dtype=np.uint32)
        chunk_path = OUTPUT_DIR / f"{lang}_chunk_{chunk_idx:06d}.npy"
        np.save(chunk_path, chunk_array)
        print(f"  Saved: {chunk_path.name} (final, padded)")
        chunk_idx += 1

    print(f"  {lang.upper()}: {chunk_idx} chunks, {total_tokens:,} tokens total")
    return chunk_idx


def main():
    print("=" * 60)
    print("EXP8 CHUNK CREATION")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Tokens per chunk: {TOKENS_PER_CHUNK:,}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    print(f"  Vocab size: {sp.vocab_size()}")
    print(f"  EOS ID: {sp.eos_id()}")

    # Create chunks for each language
    stats = {}
    for lang in LANGUAGES:
        chunks = create_chunks_for_language(lang, sp)
        stats[lang] = {'chunks': chunks, 'tokens': chunks * TOKENS_PER_CHUNK}

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'tokenizer': str(TOKENIZER_PATH),
        'vocab_size': sp.vocab_size(),
        'tokens_per_chunk': TOKENS_PER_CHUNK,
        'languages': stats,
        'total_chunks': sum(s['chunks'] for s in stats.values()),
    }

    with open(OUTPUT_DIR / 'chunk_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Completed: {datetime.now().isoformat()}")
    for lang, s in stats.items():
        print(f"  {lang}: {s['chunks']} chunks")
    print(f"Total chunks: {metadata['total_chunks']}")
    print(f"Metadata: {OUTPUT_DIR / 'chunk_metadata.json'}")


if __name__ == '__main__':
    main()
