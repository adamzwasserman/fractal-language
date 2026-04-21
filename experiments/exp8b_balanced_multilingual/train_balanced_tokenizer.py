#!/usr/bin/env python3
"""
Train a balanced 12-language tokenizer using EXACT same methodology as exp1.

Uses HuggingFace tokenizers library with:
- BPE model with unk_token="<unk>"
- Whitespace pre-tokenizer
- vocab_size=50,000
- special_tokens=["<pad>", "<unk>", "<eos>"]

Balanced: equal samples from each language.
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import random

# Config - EXACT same as exp1
VOCAB_SIZE = 50_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>"]
SAMPLES_PER_LANG = 100_000  # ~1.2M total samples across 12 langs

# Paths
RAW_TEXT_DIR = Path("/workspace/exp8/raw_text")
OUTPUT_PATH = Path("/workspace/exp8/joint_tokenizer.json")

# All 12 languages
LANGUAGES = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh",
             "synth_a", "synth_b", "synth_c", "synth_d"]

def load_samples(lang: str, n_samples: int) -> list:
    """Load n_samples lines from a language's raw text file."""
    filepath = RAW_TEXT_DIR / f"{lang}.txt"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found, skipping {lang}")
        return []

    samples = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) > 50:  # Min length filter
                # Unescape newlines if present
                text = line.replace('\\n', '\n').replace('\\r', '\r')
                samples.append(text)
            if len(samples) >= n_samples:
                break

    print(f"  {lang}: loaded {len(samples):,} samples")
    return samples

def main():
    print(f"Training balanced 12-language tokenizer")
    print(f"Methodology: EXACT same as exp1 (HuggingFace BPE)")
    print(f"Vocab size: {VOCAB_SIZE:,}")
    print(f"Samples per language: {SAMPLES_PER_LANG:,}")
    print()

    # Collect balanced samples from all languages
    all_samples = []
    for lang in LANGUAGES:
        samples = load_samples(lang, SAMPLES_PER_LANG)
        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples):,}")

    # Shuffle to mix languages
    random.seed(42)
    random.shuffle(all_samples)

    # Train tokenizer - EXACT same methodology as exp1
    print(f"\nTraining tokenizer (same as exp1)...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    tokenizer.train_from_iterator(all_samples, trainer=trainer)

    # Save
    tokenizer.save(str(OUTPUT_PATH))
    print(f"\nTokenizer saved to {OUTPUT_PATH}")

    # Verify
    print(f"\nVerification:")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Test encode
    test = "The quick brown fox jumps over the lazy dog."
    enc = tokenizer.encode(test)
    print(f"  Test: '{test}'")
    print(f"  Tokens: {enc.tokens}")
    print(f"  IDs: {enc.ids}")

if __name__ == "__main__":
    main()
