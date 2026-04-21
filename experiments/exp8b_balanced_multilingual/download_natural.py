#!/usr/bin/env python3
"""Download natural languages from C4 to 1B tokens each."""
import os
from pathlib import Path
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import threading

OUTPUT_DIR = Path("/workspace/exp8/raw_text_extended")
OUTPUT_DIR.mkdir(exist_ok=True)

# Target: 1B tokens ≈ 4B chars (assuming ~4 chars/token avg)
TARGET_CHARS = 4_000_000_000

LANGS = {
    "en": "en",
    "fr": "fr", 
    "es": "es",
    "fi": "fi",
    "ru": "ru",
    "id": "id",  # Indonesian not in C4, will try multilingual
    "vi": "vi",
    "zh": "zh",
}

lock = threading.Lock()

def download_lang(lang, c4_name):
    filepath = OUTPUT_DIR / f"{lang}.txt"
    
    # Check existing size
    existing_chars = 0
    if filepath.exists():
        existing_chars = filepath.stat().st_size
        if existing_chars >= TARGET_CHARS:
            print(f"{lang}: Already have {existing_chars/1e9:.1f}B chars, skipping")
            return
    
    print(f"{lang}: Starting download (have {existing_chars/1e6:.0f}M chars)...")
    
    try:
        ds = load_dataset("allenai/c4", name=c4_name, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"{lang}: C4 failed ({e}), trying mc4...")
        try:
            ds = load_dataset("mc4", name=lang, split="train", streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"{lang}: mc4 also failed: {e2}")
            return
    
    total_chars = existing_chars
    with open(filepath, 'a', encoding='utf-8') as f:
        for i, example in enumerate(ds):
            text = example["text"].strip()
            if len(text) < 100:
                continue
            escaped = text.replace('\n', '\\n').replace('\r', '\\r')
            f.write(escaped + '\n')
            total_chars += len(text)
            
            if i % 10000 == 0:
                with lock:
                    print(f"  {lang}: {total_chars/1e6:.0f}M chars")
            
            if total_chars >= TARGET_CHARS:
                break
    
    print(f"{lang}: Done - {total_chars/1e9:.1f}B chars")

def main():
    print("Downloading natural languages to 1B tokens each...")
    print(f"Target: {TARGET_CHARS/1e9:.0f}B chars per language\n")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_lang, lang, c4_name) 
                   for lang, c4_name in LANGS.items()]
        for f in futures:
            f.result()
    
    print("\nAll downloads complete.")

if __name__ == "__main__":
    main()
