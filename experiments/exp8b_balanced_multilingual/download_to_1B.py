#!/usr/bin/env python3
"""Download corpus until we have 1B tokens or a language runs out."""

import os
from pathlib import Path
from datetime import datetime

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets --quiet")
    from datasets import load_dataset

import sentencepiece as spm

OUTPUT_DIR = Path("/workspace/exp8/raw_text_extended")
TOKENIZER_PATH = Path("/workspace/exp8/balanced_tokenizer.model")
CHUNKS_DIR = Path("/workspace/exp8/chunks")
TARGET_TOKENS = 1_000_000_000

NATURAL_LANGS = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh"]

# allenai/c4 uses different naming
C4_NAMES = {
    "en": ("allenai/c4", "en"),
    "fr": ("allenai/c4", "fr"), 
    "es": ("allenai/c4", "es"),
    "fi": ("allenai/c4", "fi"),
    "ru": ("allenai/c4", "ru"),
    "id": ("allenai/c4", "id"),
    "vi": ("allenai/c4", "vi"),
    "zh": ("allenai/c4", "zh"),
}

def count_existing_tokens():
    total = 0
    for f in CHUNKS_DIR.glob("*.npy"):
        total += 1_000_000
    return total

def download_language(lang: str, chars_needed: int) -> tuple:
    output_file = OUTPUT_DIR / f"{lang}.txt"
    
    # Check if already have enough
    if output_file.exists():
        existing = output_file.stat().st_size
        if existing >= chars_needed:
            print(f"  {lang}: Already have {existing/1e6:.1f}M chars")
            return True, existing
    
    dataset_name, config = C4_NAMES[lang]
    print(f"\n  {lang}: Downloading from {dataset_name}/{config}...")
    
    try:
        ds = load_dataset(dataset_name, config, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  {lang}: Failed: {e}")
        return False, 0
    
    chars = 0
    docs = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in ds:
            text = doc["text"].strip()
            if text:
                f.write(text + "\n")
                chars += len(text)
                docs += 1
                
                if docs % 10000 == 0:
                    print(f"    {lang}: {docs:,} docs, {chars/1e6:.1f}M chars", flush=True)
                
                if chars >= chars_needed:
                    break
    
    if chars < chars_needed:
        print(f"  {lang}: EXHAUSTED at {chars/1e6:.1f}M chars")
        return False, chars
    
    print(f"  {lang}: Done - {chars/1e6:.1f}M chars")
    return True, chars

def main():
    print("=" * 60)
    print("DOWNLOAD TO 1B TOKENS")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Target: {TARGET_TOKENS/1e9:.1f}B tokens")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    print(f"Tokenizer vocab: {sp.vocab_size()}")
    
    existing = count_existing_tokens()
    print(f"Existing tokens: {existing/1e6:.0f}M")
    
    needed = TARGET_TOKENS - existing
    print(f"Tokens needed: {needed/1e6:.0f}M")
    
    if needed <= 0:
        print("Already have 1B tokens!")
        return
    
    # 8 natural languages, ~0.3 tokens/char
    tokens_per_lang = needed // 8
    chars_per_lang = int(tokens_per_lang / 0.3)
    
    print(f"Target per language: ~{tokens_per_lang/1e6:.0f}M tokens (~{chars_per_lang/1e6:.0f}M chars)")
    print(f"(Synth languages generated from EN during chunking)")
    
    for lang in NATURAL_LANGS:
        success, chars = download_language(lang, chars_per_lang)
        if not success and chars < chars_per_lang * 0.5:
            print(f"\n*** {lang} has insufficient data - stopping ***")
            break
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.glob("*.txt")):
        print(f"  {f.name}: {f.stat().st_size/1e6:.1f}MB")

if __name__ == "__main__":
    main()
