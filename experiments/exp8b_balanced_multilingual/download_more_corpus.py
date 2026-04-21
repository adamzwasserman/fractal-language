#!/usr/bin/env python3
"""Download additional corpus for exp8 (runs in background, CPU only)."""

import json
from pathlib import Path
from datetime import datetime
from datasets import load_dataset

CHARS_PER_LANG = 250_000_000  # 250M chars per language (5x original)
OUTPUT_DIR = Path("/workspace/exp8/raw_text_extended")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NATURAL_LANGUAGES = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh"]

def download_c4_text(lang: str, max_chars: int, start_offset: int = 0) -> str:
    """Download text from C4/mC4 corpus."""
    print(f"  Downloading C4 for {lang} (offset={start_offset})...")
    
    try:
        if lang == "en":
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        else:
            ds = load_dataset("mc4", lang, split="train", streaming=True)
    except Exception as e:
        print(f"    Error: {e}")
        return ""
    
    texts = []
    total_chars = 0
    skipped = 0
    
    for example in ds:
        # Skip to offset
        if skipped < start_offset:
            skipped += 1
            continue
            
        text = example.get("text", "")
        if text and len(text) > 100:
            texts.append(text)
            total_chars += len(text)
            
            if total_chars >= max_chars:
                break
            
            if len(texts) % 10000 == 0:
                print(f"    {len(texts)} docs, {total_chars/1e6:.1f}M chars...")
    
    print(f"    Done: {len(texts)} docs, {total_chars/1e6:.1f}M chars")
    return "\n".join(texts)

def main():
    print("=" * 60)
    print("DOWNLOADING EXTENDED CORPUS")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Target: {CHARS_PER_LANG/1e6:.0f}M chars per language")
    print()
    
    # Check existing data to determine offset
    existing_dir = Path("/workspace/exp8/raw_text")
    
    for lang in NATURAL_LANGUAGES:
        print(f"\n{lang.upper()}:")
        output_file = OUTPUT_DIR / f"{lang}.txt"
        
        if output_file.exists():
            print(f"  Already exists, skipping...")
            continue
        
        # Count existing chars to use as offset
        existing_file = existing_dir / f"{lang}.txt"
        offset = 0
        if existing_file.exists():
            with open(existing_file) as f:
                existing_chars = len(f.read())
            # Rough estimate: 1 doc ~ 2500 chars
            offset = int(existing_chars / 2500)
            print(f"  Existing: {existing_chars/1e6:.1f}M chars, offset={offset}")
        
        text = download_c4_text(lang, CHARS_PER_LANG, offset)
        if text:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  Saved: {len(text)/1e6:.1f}M chars")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
