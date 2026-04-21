#!/usr/bin/env python3
"""
Create tokenized chunks for Exp9 training.

Uses the exp8 joint BPE tokenizer (50k vocab) to create 1M-token chunks.
For 2B tokens per language, we need ~2000 chunks per language.
"""

import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import json
import gc
import argparse

# Configuration
EXP9_DIR = Path("/Volumes/fractal/exp9_WALS_prediction")
TOKENIZER_PATH = Path("/Volumes/fractal/exp8b_balanced_multilingual/joint_tokenizer.json")
CHUNKS_DIR = EXP9_DIR / "chunks"
TOKENS_PER_CHUNK = 1_000_000  # 1M tokens per chunk = ~4MB
MAX_TOKENS = 2_000_000_000  # 2B tokens per language

# Language to source file mapping
LANG_FILES = {
    "en": EXP9_DIR / "en_texts.txt",
    "fr": EXP9_DIR / "fr_texts.txt",
    "synth_alpha_a": EXP9_DIR / "synth_alpha_a_texts.txt",
    "synth_alpha_b": EXP9_DIR / "synth_alpha_b_texts.txt",
    "synth_alpha_c": EXP9_DIR / "synth_alpha_c_texts.txt",
}


def create_chunks_for_language(lang: str, max_tokens: int = MAX_TOKENS):
    """Convert text file to efficient .npy chunks for training"""

    text_file = LANG_FILES.get(lang)
    if not text_file or not text_file.exists():
        print(f"ERROR: Source file not found for {lang}: {text_file}")
        return None

    print(f"\n{'='*70}")
    print(f"Creating chunks for {lang.upper()}")
    print(f"{'='*70}")
    print(f"Source: {text_file}")
    print(f"Target: {CHUNKS_DIR}/{lang}_chunk_*.npy")
    print(f"Max tokens: {max_tokens:,}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("<eos>")

    chunk_idx = 0
    token_buffer = []
    total_tokens = 0
    docs_processed = 0

    with open(text_file, 'r', encoding='utf-8') as f:
        pbar = tqdm(desc=f"{lang} chunks", unit="doc")

        for line in f:
            # Unescape and tokenize
            text = line.strip().replace('\\n', '\n').replace('\\r', '\r')
            if not text:
                continue

            tokens = tokenizer.encode(text).ids + [eos_id]
            token_buffer.extend(tokens)
            docs_processed += 1
            pbar.update(1)

            # Save chunk when buffer is full
            while len(token_buffer) >= TOKENS_PER_CHUNK:
                # Extract exact chunk size
                chunk_tokens = token_buffer[:TOKENS_PER_CHUNK]
                token_buffer = token_buffer[TOKENS_PER_CHUNK:]

                # Save as uint32 numpy array (4 bytes per token)
                chunk_array = np.array(chunk_tokens, dtype=np.uint32)
                chunk_path = CHUNKS_DIR / f"{lang}_chunk_{chunk_idx:06d}.npy"
                np.save(chunk_path, chunk_array)

                total_tokens += TOKENS_PER_CHUNK
                chunk_idx += 1

                if chunk_idx % 100 == 0:
                    print(f"  Saved chunk {chunk_idx}: {total_tokens:,} tokens so far")

                gc.collect()

                # Check if we've reached max tokens
                if total_tokens >= max_tokens:
                    print(f"  Reached {max_tokens:,} token limit")
                    break

            if total_tokens >= max_tokens:
                break

        # Save final chunk if any tokens remain and we haven't hit the limit
        if token_buffer and total_tokens < max_tokens:
            chunk_array = np.array(token_buffer, dtype=np.uint32)
            chunk_path = CHUNKS_DIR / f"{lang}_chunk_{chunk_idx:06d}.npy"
            np.save(chunk_path, chunk_array)
            total_tokens += len(token_buffer)
            chunk_idx += 1
            print(f"  Saved final chunk {chunk_idx}: {len(token_buffer):,} tokens")

        pbar.close()

    stats = {
        "language": lang,
        "source_file": str(text_file),
        "docs_processed": docs_processed,
        "total_tokens": total_tokens,
        "chunks_created": chunk_idx,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
    }

    print(f"\n{lang.upper()} complete:")
    print(f"  Documents: {docs_processed:,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Chunks: {chunk_idx}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Create chunks for Exp9')
    parser.add_argument('languages', nargs='*', default=['all'],
                       help='Languages to process (default: all)')
    parser.add_argument('--max-tokens', '-n', type=int, default=MAX_TOKENS,
                       help=f'Max tokens per language (default: {MAX_TOKENS:,})')
    args = parser.parse_args()

    # Create output directory
    CHUNKS_DIR.mkdir(exist_ok=True)

    # Verify tokenizer exists
    if not TOKENIZER_PATH.exists():
        print(f"ERROR: Tokenizer not found: {TOKENIZER_PATH}")
        return

    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Output directory: {CHUNKS_DIR}")

    # Determine which languages to process
    if 'all' in args.languages:
        languages = list(LANG_FILES.keys())
    else:
        languages = args.languages

    # Process each language
    all_stats = {}
    for lang in languages:
        if lang not in LANG_FILES:
            print(f"WARNING: Unknown language '{lang}', skipping")
            continue

        stats = create_chunks_for_language(lang, args.max_tokens)
        if stats:
            all_stats[lang] = stats

    # Save summary
    summary_path = CHUNKS_DIR / "chunk_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for lang, stats in all_stats.items():
        print(f"{lang}: {stats['chunks_created']} chunks, {stats['total_tokens']:,} tokens")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
